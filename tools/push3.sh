#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh; conda activate genesis
cd ~/claude-playground/physical_ai/local/genesis/Genesis/examples/locomotion
python go2_train_push3.py -e go2-push3 -b 4096 --max-iterations 600 > ../../../push3_train.log 2>&1
echo TRAIN3_DONE
ck=$(ls logs/go2-push3 | grep -oE "model_[0-9]+" | sed 's/model_//' | sort -n | tail -1)
for v in 1.0 2.0 2.5 3.0 3.5; do
  python go2_video_push.py -e go2-push3 --ckpt $ck --push-vel $v --push-interval 2.0 --steps 750 --seed 11 -o /tmp/claude-1000/sweep_go2-push3_${v}.mp4 2>&1 | grep EVAL | sed 's/pushes_at=.*//'
done
echo SWEEP3_DONE
