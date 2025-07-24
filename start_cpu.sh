export OPENPI_DATA_HOME=./data
export HF_HOME=./data
export HF_LEROBOT_HOME=./data
export MUJOCO_GL=egl

export PATH=~/.local/bin:$PATH

# conda deactivate
source .venv/bin/activate

git config --global user.name "Yitong Chen"
git config --global user.email "yitongchen719@gmail.com"

# # wandb sync
# find outputs -path "*/wandb/offline-*" | xargs -I {} wandb sync {}