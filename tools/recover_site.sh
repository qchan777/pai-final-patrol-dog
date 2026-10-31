#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh; conda activate genesis
cd ~/claude-playground/physical_ai/local/genesis/Genesis/examples/locomotion
CUDA_VISIBLE_DEVICES=1 python go2_train_recover_site.py -e go2-recover-site2 -b 4096 --max-iterations 600 > ../../../recover_site_train.log 2>&1
echo RS_DONE
