#!/bin/bash
# 成長過程の動画: 10 反復ごとに保存して学習し直し、各段階を 6 秒ずつ描いて繋ぐ
source ~/miniconda3/etc/profile.d/conda.sh; conda activate genesis
cd ~/claude-playground/physical_ai/local/genesis/Genesis/examples/locomotion
python go2_train_growth.py -e go2-growth -b 4096 --max-iterations 101 > ../../../growth_train.log 2>&1
ls logs/go2-growth | grep model_ | tr "\n" " "; echo
for c in 0 10 20 30 50 100; do
  python go2_video.py -e go2-growth --ckpt $c --steps 300 --vx 0.6 -o ~/claude-playground/physical_ai/local/genesis/growth_$c.mp4 2>&1 | grep VIDEO
done
echo GROWTH_DONE
