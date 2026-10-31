#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh; conda activate genesis
cd ~/claude-playground/physical_ai/local/genesis/Genesis/examples/locomotion
python go2_train_push2.py -e go2-push2 -b 4096 --max-iterations 300 > ../../../push2_train.log 2>&1
echo TRAIN2_DONE
for e in go2-walking:100 go2-push:100 go2-push2:300; do exp=${e%%:*}; ck=${e##*:}
  for v in 1.0 1.5 2.0 2.5 3.0; do
    python go2_video_push.py -e $exp --ckpt $ck --push-vel $v --push-interval 2.0 --steps 750 --seed 11 -o /tmp/claude-1000/sweep_${exp}_${v}.mp4 2>&1 | grep EVAL
  done
done
echo SWEEP_DONE
