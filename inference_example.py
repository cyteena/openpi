import dataclasses

import jax

from pprint import pprint

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


config = _config.get_config("pi0_fast_droid")
pprint(f"{config=}")
checkpoint_dir = download.maybe_download("/inspire/hdd/project/embodied-multimodality/gongjingjing-25039/ytchen/openpi/data/openpi-assets/checkpoints/pi0_fast_droid")
# Create a trained policy.
print("we are making the policy!")
policy = _policy_config.create_trained_policy(config, checkpoint_dir)
pprint(f'{policy=}')

# Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
example = droid_policy.make_droid_example()
pprint(f'{example=}')
for key in example:
    print(f'{key=}')
result = policy.infer(example)
pprint(f'{result=}')

# Delete the policy to free up memory.
del policy

print("Actions shape:", result["actions"].shape)