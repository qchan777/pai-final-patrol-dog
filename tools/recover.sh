#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh; conda activate genesis
cd ~/claude-playground/physical_ai/local/genesis/Genesis/examples/locomotion
python go2_train_recover.py -e go2-recover2 -b 4096 --max-iterations 500 > ../../../recover_train.log 2>&1
echo RTRAIN_DONE
ck=$(ls logs/go2-recover2 | grep -oE "model_[0-9]+" | sed 's/model_//' | sort -n | tail -1)
for s in 1 2 3 4; do python go2_video_recover.py -e go2-recover2 --ckpt $ck --seed $s --steps 250 -o /tmp/claude-1000/recover_$s.mp4 2>&1 | grep -E "RECOVER|Traceback|Error"; done
echo RECOVER_DONE
