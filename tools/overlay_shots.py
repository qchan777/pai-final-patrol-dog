#!/usr/bin/env python3
"""見回り動画に「犬っこが見た写真 (頭のカメラ) と VLM の答え」の小窓を重ねる ffmpeg の指示を作る。
使い方: overlay_shots.py events.json shots_dir prefix out_filter.txt
  到着の出来事 (写真を判定) ごとに、prefix_wp{N}_{k}.png を 4.5 秒表示。「はい」なら最後の写真 (見回しを打ち切った向き)、「いいえ」なら正面。"""
import json, os, re, sys
ev = json.load(open(sys.argv[1], encoding='utf-8')); shots_dir, prefix, out = sys.argv[2], sys.argv[3], sys.argv[4]
shots_dir_in_ffmpeg = sys.argv[5] if len(sys.argv) > 5 else shots_dir   # ffmpeg を別の場所 (コンテナ) で動かすときの写真の置き場
def esc(s):
    return s.replace('\\', '\\\\').replace(':', '\\:').replace(',', '\\,').replace("'", '').replace('%', '\\%')
items = []; wp = 0
for t, msg in ev:
    m = re.search(r'写真を判定 \([\d.]+ 秒\): (.*)$', msg)
    if not m:
        continue
    ans = m.group(1).strip()
    ks = sorted(int(re.search(r'_(\d+)\.png$', f).group(1)) for f in os.listdir(shots_dir) if f.startswith(f'{prefix}_wp{wp}_'))
    if ks:
        k = ks[-1] if ans.startswith('はい') else 0
        label = ('発見! ' if ans.startswith('はい') else '異常なし ') + ans[:34]
        items.append((t, f'{shots_dir_in_ffmpeg}/{prefix}_wp{wp}_{k}.png', label))
    wp += 1
inputs = ''.join(f' -loop 1 -i "{p}"' for _, p, _ in items)
chain = '[0:v]'
for i, (t, p, label) in enumerate(items, 1):
    chain += f'[{i}:v]scale=320:180,drawbox=x=0:y=0:w=iw:h=ih:color=0xffcc00@0.9:t=3[s{i}];'
    chain = chain.replace(f'[{i}:v]scale', f'[{i}:v]scale', 1)
# overlay を順に繋ぐ
filt = ''
prev = '[0:v]'
for i, (t, p, label) in enumerate(items, 1):
    filt += f'[{i}:v]scale=320:180,drawbox=x=0:y=0:w=iw:h=ih:color=0xffcc00@0.9:t=3[s{i}];'
    filt += f"{prev}[s{i}]overlay=12:H-h-12:enable='between(t,{t:.1f},{t+4.5:.1f})'[o{i}];"
    filt += f"[o{i}]drawtext=fontfile=/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc:text='犬っこの目 → VLM\\: {esc(label)}':fontsize=18:fontcolor=white:box=1:boxcolor=0x000000aa:boxborderw=6:x=12:y=H-h-12-30:enable='between(t,{t:.1f},{t+4.5:.1f})'[v{i}];"
    prev = f'[v{i}]'
open(out, 'w', encoding='utf-8').write(inputs + '\n' + filt.rstrip(';') + '\n' + prev + '\n')
print(len(items), 'shots:', [(round(t, 1), os.path.basename(p)) for t, p, _ in items])
