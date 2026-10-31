#!/bin/bash
# conda 環境 genesis: PyTorch (CUDA 12) + Genesis + rsl-rl。ログは install.log
set -x
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n genesis python=3.11
conda activate genesis
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install genesis-world "rsl-rl-lib==2.2.4" tensorboard
python -c "import torch, genesis as gs; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('genesis', gs.__version__)"
echo INSTALL_DONE
