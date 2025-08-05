# pi0_discrete_flow.py

import dataclasses
import logging

from typing import Any

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils


logger = logging.getLogger("openpi")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision. (Unchanged from pi0.py)"""
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float = 4e-3, max_period: float = 4.0
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions. (Unchanged from pi0.py)"""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@dataclasses.dataclass(frozen=True)
class Pi0DiscreteFlowConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # action_dim, action_horizon, used for fast_tokenizer
    action_dim: int = 7
    action_horizon: int = 10
    # NOTE: max_token_len is now for the prefix (observation) only.
    # Action tokens will have their own sequence length derived from action_horizon.
    max_action_token_len: int = 48
    max_text_token_len: int = 64
    max_token_len: int = 120

    # Tokenizer for the fast model.
    fast_model_tokenizer: Any | None = None
    # Keyword arguments for the fast model tokenizer.
    fast_model_tokenizer_kwargs: dict[str, Any] | None = None

    # dirty way to get the value
    from .tokenizer import FASTTokenizer

    tokenizer = FASTTokenizer()
    pg_vocab_size: int = tokenizer._paligemma_tokenizer.vocab_size()
    pg_skip_tokens: int = tokenizer._fast_skip_tokens  # Should be 128
    action_vocab_size: int = tokenizer._fast_tokenizer.vocab_size  # THIS IS 2048
    
    # we use this to calculate loss over [mask] id!
    mask_token_id: int = pg_vocab_size - 1 - pg_skip_tokens - 306 # 306 is a local id for fast_tokenizer, and denote a sequence of 1024 0s.

    # # Config for the new Head
    # head_num_layers: int = 4
    # head_num_heads: int = 8
    # head_mlp_dim: int = 1024

    @property
    @override
    def model_type(self) -> _model.ModelType:
        # We might need to add a new model type later.
        return _model.ModelType.PI0_DFM

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0DiscreteFlow":
        return Pi0DiscreteFlow(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        # The external interface now expects pre-tokenized actions from data preprocessing.
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        # Actions are now pre-tokenized action tokens, not the original continuous actions
        action_spec = jax.ShapeDtypeStruct([batch_size, self.max_action_token_len], jnp.int32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        # Simplified freeze filter as we only have one LLM.
        filters = []
        if "lora" in self.paligemma_variant:
            filters.append(nnx_utils.PathRegex(".*llm.*"))
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))

        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0DiscreteFlow(_model.BaseModel):
    def __init__(self, config: Pi0DiscreteFlowConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)

        # --- PaliGemma Setup ---
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        
        # Now we read the constants from the config, may we find a way to do more cleaner.
        self.config = config
        self.tokenizer = config.tokenizer
        self.pg_vocab_size = config.pg_vocab_size
        self.pg_skip_tokens = config.pg_skip_tokens
        self.action_vocab_size = config.action_vocab_size
        self.mask_token_id = config.mask_token_id
        self.action_expert_width = action_expert_config.width
        self.paligemma_width = paligemma_config.width
        # --- VLM Setup (Simplified) ---
        # --- 1. 创建所有模块的蓝图 ---
        llm_module = _gemma.Module(
            configs=[paligemma_config, action_expert_config],
            embed_dtype=config.dtype
        )
        llm = nnx_bridge.ToNNX(llm_module)
        llm.lazy_init(rngs=rngs, method="init")
        
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
      
        self.suffix_in_proj = nnx.Linear(paligemma_config.width, action_expert_config.width, rngs=rngs)
        
        self.time_embed_mlp_in = nnx.Linear(paligemma_config.width, action_expert_config.width, rngs=rngs)
        self.time_embed_mlp_out = nnx.Linear(action_expert_config.width, paligemma_config.width, rngs=rngs)
        self.head_out_proj = nnx.Linear(action_expert_config.width, self.action_vocab_size, rngs=rngs)
        
    @at.typecheck
    def _local_action_indices_to_pg_tokens(self, indices: at.Int[at.Array | np.ndarray, "..."]) -> at.Int[at.Array, "..."]:
        """Maps local action indices [0, action_vocab_size-1] to global PaliGemma token IDs."""
        # This logic is correct.
        return self.pg_vocab_size - self.pg_skip_tokens - indices - 1
        
    @at.typecheck
    def _pg_tokens_to_local_action_indices(self, pg_tokens: at.Int[at.Array | np.ndarray, "..."]) -> at.Int[at.Array, "..."]:
        """Maps global PaliGemma action token IDs back to local action indices [0, action_vocab_size-1]."""
        # This logic is correct.
        return self.pg_vocab_size - self.pg_skip_tokens - pg_tokens - 1

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s_p emb"], at.Bool[at.Array, "b s_p"]]:
        """
        Encodes the observation (images, language) into prefix embeddings.
        This implementation is correct.
        """
        input_mask_list, tokens_list = [], []
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens_list.append(image_tokens)
            input_mask_list.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))

        if obs.tokenized_prompt is not None:
            # Using the LLM's own embedding layer is the correct and efficient approach.
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens_list.append(tokenized_inputs)
            input_mask_list.append(obs.tokenized_prompt_mask)

        elif obs.dfm_prefix_token is not None:
            dfm_prefix_inputs = self.PaliGemma.llm(obs.dfm_prefix_token, method="embed")
            tokens_list.append(dfm_prefix_inputs)
            input_mask_list.append(obs.dfm_prefix_mask)

        tokens = jnp.concatenate(tokens_list, axis=1)
        input_mask = jnp.concatenate(input_mask_list, axis=1)
        return tokens, input_mask

    @at.typecheck
    def embed_suffix(
        self,
        noisy_action_tokens: at.Int[at.Array, "b s_a"],
        time: at.Float[at.Array, " b"],
        action_mask: at.Bool[at.Array, "b s_a"],
    ) -> tuple[at.Float[at.Array, "b s_a emb"], at.Bool[at.Array, "b s_a"]]:
        """
        Embeds the masked action tokens and the timestep for the Action Expert.
        This implementation is excellent. Reusing the LLM's embedding layer is key.
        """
        # 1. Embed action tokens using the main LLM's embedding table.
        # This correctly handles regular tokens and the [MASK] token.
        action_embeds = self.PaliGemma.llm(noisy_action_tokens, method="embed") # paligemma
        action_time_embeds = action_embeds + posemb_sincos(time, self.paligemma_width)[:,None,:]
        suffix_tokens_embedded = self.time_embed_mlp_in(action_time_embeds)
        suffix_tokens_embedded = nnx.swish(suffix_tokens_embedded)
        suffix_tokens_embedded = self.time_embed_mlp_out(suffix_tokens_embedded)

        return suffix_tokens_embedded, action_mask

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b"]:
        """
        Computes the Discrete Flow Matching loss.
        """
        preprocess_rng, time_rng, mask_rng = jax.random.split(rng, 3)
        # only consider the valid action token
        action_mask = observation.dfm_action_mask

        # 1. Preprocess observations and use pre-tokenized actions.
        # process image
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        # x_1 are the ground truth tokens from data preprocessing (passed as actions), with global PaliGemma IDs.
        x_1 = observation.dfm_action_token  # Shape: (B, S_a) - now actions are pre-tokenized
        batch_size, seq_len = x_1.shape

        # For now, assume all tokens are valid (no padding mask)
        # TODO: Add proper action_token_mask support if needed
        # observation.dfm_action_mask show us the valid action tokens.

        # 2. DFM forward process: sample t and create masked input x_t.
        # Sample time 't' from Uniform(0, 1]. 't' represents the ratio of kept tokens.
        # TODO: beta_sampling
        time = jax.random.beta(time_rng, 1, 1.5, shape=(batch_size,)) * 0.999 + 0.001
        time = jnp.clip(time, 0., 1 - 1e-4)

        # Generate random noise for masking decision.
        rand_unif = jax.random.uniform(mask_rng, (batch_size, seq_len))

        # If random number < t, we keep the token. Otherwise, we mask it.
        # So t = 1 is the clean data, t = 0 is the noisy distribution, same as the flow matching
        tokens_to_keep = rand_unif < time[:, None]
        tokens_to_mask = ~tokens_to_keep

        # Create the masked input sequence x_t.
        x_t = jnp.where(tokens_to_mask, self.mask_token_id, x_1)
        # Ensure x_t has the correct type
        x_t = jnp.asarray(x_t, dtype=jnp.int32)

        # 3. Embed prefix and suffix.
        prefix_tokens_embedded, prefix_mask = self.embed_prefix(observation)
        # Note: The time passed to the model represents "corruption", so we use (1-t).
        suffix_tokens_embedded, suffix_mask = self.embed_suffix(x_t, 1.0 - time, action_mask)
        suffix_tokens_embedded = self.suffix_in_proj(suffix_tokens_embedded)

        # 4. Prepare for the dual-expert model pass.
        # Prefix part has bidirectional attention among its tokens.
        prefix_ar_mask = jnp.zeros(prefix_tokens_embedded.shape[1], dtype=jnp.bool_)
        # Suffix part attends to all of prefix, but has its own attention logic.
        # For DFM, we want each action token to attend to all others (and prefix).
        # Setting ar_mask=True for the first suffix token makes it a new causal block.
        # Setting subsequent ones to False allows bidirectional attention within suffix.
        # This is a key design choice. Let's assume bidirectional attention within suffix.
        suffix_ar_mask = jnp.zeros(suffix_tokens_embedded.shape[1], dtype=jnp.bool_)
        suffix_ar_mask = suffix_ar_mask.at[0].set(True)  # Start of a new context block.

        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        # 5. Forward pass through the dual-expert LLM.
        # _gemma.Module will route the first list element to LLM 0 and the second to LLM 1.
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens_embedded, suffix_tokens_embedded], mask=attn_mask, positions=positions
        ) # suffix_out.shape: bs, action_seq, 2048 (action_vocab_size) : TODO: check the shape of suffix_out

        # 6. Compute loss on the masked tokens.
        # Project output features to logits over the action vocabulary.
        logits = self.head_out_proj(suffix_out)  # Shape: (B, S_a, action_vocab_size)

        # Convert ground truth global PG tokens to local action indices for loss calculation.
        local_targets = self._pg_tokens_to_local_action_indices(x_1)
        
        # safe_local_targets = jnp.where(action_mask, local_targets, 0)

        # Calculate cross-entropy loss.
        token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, local_targets)

        # Only consider the loss for the tokens that were actually masked.
        # masked_loss = (token_loss * tokens_to_mask) * action_mask

        # Normalize the loss by the number of masked tokens per sequence.
        num_masked_tokens = jnp.sum(tokens_to_mask, axis=-1)
        # num_masked_tokens = jnp.sum(tokens_to_mask * action_mask, axis=-1)
        # Avoid division by zero if a sequence has no masked tokens (e.g., if t=1).
        
        # we can consider to uncomment the devisor: loss at beginning is just too big
        # in llada, it's been divided by all length
        sequence_loss = jnp.sum(token_loss * tokens_to_mask, axis=-1) / jnp.maximum(1.0, num_masked_tokens)
        sequence_loss = sequence_loss / (1 - time)

        # Return the mean loss over the batch.
        return jnp.mean(sequence_loss)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> at.Int[at.Array, "b s_a"]:
        """
        Samples actions using iterative, parallel decoding (MaskGIT-style).
        """
        decode_rng = rng
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]

        # 1. Prefill KV cache with the observation prefix using the Prefix Expert.
        prefix_tokens_embedded, prefix_mask = self.embed_prefix(observation)
        prefix_ar_mask = jnp.zeros(prefix_tokens_embedded.shape[1], dtype=jnp.bool_)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        # This call uses only the Prefix Expert (LLM 0) and populates the KV cache.
        # be same as the pi0
        _, kv_cache= self.PaliGemma.llm(
            [prefix_tokens_embedded, None], mask=prefix_attn_mask, positions=positions
        )

        # 2. Initialize for iterative decoding.
        # Use the max action token length from config
        action_seq_len = self.config.max_action_token_len

        # Start with a sequence of all [MASK] tokens.
        action_tokens = jnp.full((batch_size, action_seq_len), self.mask_token_id, dtype=jnp.int32)
        # Keep track of which tokens still need to be predicted.
        mask_to_be_predicted = jnp.ones_like(action_tokens, dtype=jnp.bool_)

        # 3. Iteratively decode the action sequence in a fori_loop.
        def loop_body(i, carry):
            action_tokens, mask_to_be_predicted, rng = carry

            # a. Calculate masking schedule.
            # it's wrong?
            completion_ratio = i / num_steps
            mask_ratio = jnp.cos(completion_ratio * jnp.pi / 2.0)

            # The model's time input is the "corruption level", which is the mask_ratio.
            corrupt_for_model = jnp.full((batch_size,), mask_ratio)

            # b. Embed current (partially masked) tokens and time.
            suffix_tokens_embedded, suffix_mask = self.embed_suffix(action_tokens, corrupt_for_model, mask_to_be_predicted)
            projected_suffix_embedded = self.suffix_in_proj(suffix_tokens_embedded)

            # c. Create attention mask for the Action Expert.
            # Suffix tokens attend to the full prefix (via KV cache) and each other.
            # This is simpler than in loss because we're not concatenating prefixes.
            # The KV cache handles the prefix context. We just need suffix-internal attention.
            suffix_ar_mask = jnp.zeros(action_seq_len, dtype=jnp.bool_)
            suffix_ar_mask = suffix_ar_mask.at[0].set(True)  # Bidirectional attention within suffix.
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)

            suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens_embedded.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)

            # d. Run the Action Expert part of the LLM.
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, projected_suffix_embedded], positions=suffix_positions, kv_cache=kv_cache, mask=full_attn_mask
            )
            logits = self.head_out_proj(suffix_out)  # Shape: (B, S_a, action_vocab_size), can be decoded by fasttoenizer

            # e. MaskGIT logic to update tokens.
            # Get predicted token IDs and their confidence scores.
            predicted_local_ids = jnp.argmax(logits, axis=-1)
            predicted_global_ids = self._local_action_indices_to_pg_tokens(predicted_local_ids)

            confidence = jnp.max(jax.nn.softmax(logits, axis=-1), axis=-1)
            # We only care about the confidence of currently masked tokens.
            confidence = jnp.where(mask_to_be_predicted, confidence, -1.0)

            # Determine how many tokens to unmask in this step.
            num_tokens_total = action_seq_len
            num_masked_currently = jnp.sum(mask_to_be_predicted, axis=-1)
            # Target number of masked tokens based on cosine schedule.
            num_masked_target = jnp.floor(num_tokens_total * mask_ratio).astype(jnp.int32)
            num_to_unmask = num_masked_currently - num_masked_target
            # Ensure we unmask at least one token to make progress, except at the start.
            num_to_unmask = jnp.maximum(1, num_to_unmask)

            # In the final step, unmask all remaining tokens.
            num_to_unmask = jnp.where(i == num_steps - 1, num_masked_currently, num_to_unmask)

            # Find the indices of the most confident predictions to unmask.
            indices_to_unmask = jnp.argsort(confidence, axis=-1)[:, ::-1]  # Sort descending

            # Create a mask for the tokens we will update in this step.
            # This requires careful batch-aware indexing.
            # A simple way is to create a grid and check if the column index is in the top_k.
            col_indices = jnp.arange(action_seq_len)
            unmask_update_mask = col_indices < num_to_unmask[:, None]
            unmask_update_mask = jnp.take_along_axis(
                unmask_update_mask, jnp.argsort(indices_to_unmask, axis=-1), axis=-1
            )

            # f. Update the action tokens and the mask of tokens to be predicted.
            new_action_tokens = jnp.where(unmask_update_mask, predicted_global_ids, action_tokens)
            new_mask_to_be_predicted = jnp.logical_and(mask_to_be_predicted, ~unmask_update_mask)

            return new_action_tokens, new_mask_to_be_predicted, rng

        # Run the decoding loop.
        final_tokens, _, _ = jax.lax.fori_loop(
            0, num_steps, loop_body, (action_tokens, mask_to_be_predicted, decode_rng)
        )
        
        return final_tokens
    