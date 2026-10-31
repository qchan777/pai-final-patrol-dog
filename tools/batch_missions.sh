#!/bin/bash
# 見回り任務の信頼性: seed を変えて 8 回。1 回 1 行の要約を batch4_missions.log に
source ~/miniconda3/etc/profile.d/conda.sh; conda activate genesis
cd ~/claude-playground/physical_ai/local/genesis/Genesis/examples/locomotion
for s in 1 2 3 4 5 6 7 8; do
  CUDA_VISIBLE_DEVICES=1 python mission.py --walk go2-site:999 --recover go2-recover-site2:599 --kick-at 40 --max-s 320 --vlm --answer はい --seed $s -o /tmp/claude-1000/batch_s$s.mp4 > /tmp/claude-1000/batch_s$s.raw 2>&1
  python3 - "$s" <<'PY' >> ~/claude-playground/physical_ai/local/genesis/batch4_missions.log
import json, re, sys
s = sys.argv[1]; raw = open(f'/tmp/claude-1000/batch_s{s}.raw', encoding='utf-8', errors='ignore').read()
m = re.search(r'MISSION (done|incomplete at wp \d+) t=([\d.]+)s', raw)
ev = json.load(open(f'/tmp/claude-1000/batch_s{s}.mp4.events.json', encoding='utf-8')) if m else []
falls = sum('転倒。' in x[1] for x in ev); found = sum('要救助者らしきものを発見' in x[1] for x in ev); far = sum('遠方に' in x[1] for x in ev); det = sum('迂回' in x[1] for x in ev)
print(f'seed={s} result={m.group(1) if m else "crash"} t={m.group(2) if m else "-"} falls={falls} found={found} far={far} detours={det}')
PY
done
echo BATCH_DONE >> ~/claude-playground/physical_ai/local/genesis/batch4_missions.log
