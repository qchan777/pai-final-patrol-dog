#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh; conda activate genesis
cd ~/claude-playground/physical_ai/local/genesis/Genesis/examples/locomotion
CUDA_VISIBLE_DEVICES=1 python go2_train_terrain.py -e go2-terrain -b 4096 --max-iterations 800 > ../../../terrain_train.log 2>&1
echo TTRAIN_DONE
ck=$(ls logs/go2-terrain | grep -oE "model_[0-9]+" | sed 's/model_//' | sort -n | tail -1)
CUDA_VISIBLE_DEVICES=1 python go2_video_terrain.py -e go2-terrain --ckpt $ck --steps 750 --vx 0.5 -o /tmp/claude-1000/terrain_trained.mp4 2>&1 | grep -E "TERRAIN|Traceback|Error"
echo TERRAIN_DONE
