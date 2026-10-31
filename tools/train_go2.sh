#!/bin/bash
# Go2 の歩行方策を学習 (Genesis の例題そのまま)。ログは train_go2.log、結果は Genesis/examples/locomotion/logs/go2-walking/
source ~/miniconda3/etc/profile.d/conda.sh; conda activate genesis
cd ~/claude-playground/physical_ai/local/genesis/Genesis/examples/locomotion
python go2_train.py -e go2-walking -b 4096 --max-iterations 101
echo TRAIN_DONE
