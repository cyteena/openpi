import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable


import flax.nnx as nnx # <-- Import nnx
import flax.traverse_util

import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)

# This is the key function to fix the problem.
def safe_flatten_dict(dct, sep="/"):
    """
    A variant of flax.traverse_util.flatten_dict that is robust to
    non-string keys and handles flax.nnx.Variable types as leaves.
    """
    def _flatten(xs, prefix):
        # A leaf is anything that is NOT a dictionary or is an empty dictionary.
        # CRUCIALLY, we also treat nnx.Variable as a leaf, to prevent recursing into it.
        if not isinstance(xs, dict) or not xs or isinstance(xs, nnx.Variable):
            # The key is to convert all path elements to string before joining.
            return {sep.join(map(str, prefix)): xs} if prefix else {}
        
        result = {}
        for key, value in xs.items():
            # Recursively build the path.
            path = prefix + (key,)
            result.update(_flatten(value, path))
        return result

    # The original implementation handles non-dict inputs, let's replicate that.
    if not isinstance(dct, dict):
         # This case should ideally not happen for model parameters, but for safety:
         return {None: dct}

    return _flatten(dct, ())


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")

# In src/openpi/training/weight_loaders.py

def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters."""
    # `params` is a PyTree whose leaves are ShapeDtypeStruct objects.
    # `loaded_params` is a PyTree of real numpy arrays.
    
    flat_ref = safe_flatten_dict(params, sep="/")
    flat_loaded = safe_flatten_dict(loaded_params, sep="/")

    result = {}
    
    # Process loaded parameters
    for k, v in flat_loaded.items():
        if k is None or k not in flat_ref:
            continue
        
        # --- START OF FIX ---
        # The leaf from the reference shape is the ShapeDtypeStruct itself.
        ref_leaf_shape = flat_ref[k]
        
        # Ensure we have a dtype to cast to.
        # ShapeDtypeStruct directly has a .dtype attribute.
        if hasattr(ref_leaf_shape, 'dtype'):
            result[k] = v.astype(ref_leaf_shape.dtype)
        # --- END OF FIX ---

    # Process missing parameters from the reference shape dict
    pattern = re.compile(missing_regex)
    for k, v in flat_ref.items():
        if k is None:
            continue
        
        # If the key is not in our loaded result, and it matches the regex,
        # we add the original reference leaf (which is the ShapeDtypeStruct).
        if k not in result and pattern.fullmatch(k):
            result[k] = v # v is the original ShapeDtypeStruct

    return flax.traverse_util.unflatten_dict(result, sep="/")