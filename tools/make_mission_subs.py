#!/usr/bin/env python3
"""mission.py の events.json から字幕 (SRT) を作る。使い方: make_mission_subs.py events.json out.srt [1 件の表示秒 (既定 4)]"""
import json, sys
ev = json.load(open(sys.argv[1], encoding='utf-8')); out = sys.argv[2]; show = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
def ts(t):
    t = max(0.0, t); h, m = divmod(int(t), 3600); m, s = divmod(m, 60)
    return f'{h:02d}:{m:02d}:{s:02d},{int((t % 1) * 1000):03d}'
with open(out, 'w', encoding='utf-8') as f:
    for k, (t, msg) in enumerate(ev, 1):
        end = min(t + show, ev[k][0] - 0.1) if k < len(ev) else t + show
        f.write(f'{k}\n{ts(t)} --> {ts(max(end, t + 0.8))}\n{msg}\n\n')
print(f'{len(ev)} cues')
