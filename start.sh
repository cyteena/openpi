export OPENPI_DATA_HOME=./data
export HF_HOME=./data
export HF_LEROBOT_HOME=./data
export HF_HUB_OFFLINE=1
export MUJOCO_GL=egl
export WANDB_MODE=offline
# export JAX_DEBUG_NANS=True
export JAX_TRACEBACK_FILTERING=off

export UV_PYTHON_INSTALL_DIR=./uv_python

export PATH=~/.local/bin:$PATH

# conda deactivate
pip install uv
source .venv/bin/activate

git config --global user.name "Yitong Chen"
git config --global user.email "yitongchen719@gmail.com"

# # wandb sync
# find outputs -path "*/wandb/offline-*" | xargs -I {} wandb sync {}