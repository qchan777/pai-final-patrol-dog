#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh; conda activate genesis
cd ~/claude-playground/physical_ai/local/genesis/Genesis/examples/locomotion
CUDA_VISIBLE_DEVICES=1 python go2_train_pg.py -e go2-pg -b 4096 --max-iterations 1000 > ../../../pg_train.log 2>&1
echo PGTRAIN_DONE
