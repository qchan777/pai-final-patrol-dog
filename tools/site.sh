#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh; conda activate genesis
cd ~/claude-playground/physical_ai/local/genesis/Genesis/examples/locomotion
CUDA_VISIBLE_DEVICES=1 python go2_train_site.py -e go2-site -b 4096 --max-iterations 1000 > ../../../site_train.log 2>&1
echo SITE_DONE
