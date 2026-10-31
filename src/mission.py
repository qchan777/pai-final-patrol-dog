"""災害時の見回り (自律): GPS もどき (1 Hz、±1.5 m のばらつき) で経由地を順に回る。転んだら起き上がり方策、立てたら歩行方策に戻す。
進めないときは向きを変えて迂回する。人の指示は最初の任務だけ。
python mission.py --walk go2-pg:999 --recover go2-recover2:499 --kick-at 20 -o mission.mp4"""
import argparse, math, os, pickle, random, torch, genesis as gs
from rsl_rl.runners import OnPolicyRunner
from go2_env_site import Go2EnvSite
from disaster_site import WAYPOINTS as _WP, PERSON, OBSTACLES
from planner import Planner
from PIL import Image
WAYPOINTS = []
for _n, _p in _WP:
    WAYPOINTS.append((_n, _p))
p = argparse.ArgumentParser()
p.add_argument('--walk', default='go2-pg:999'); p.add_argument('--recover', default='go2-recover2:499')
p.add_argument('--kick-at', type=float, default=-1); p.add_argument('--kick', type=float, default=2.5)
p.add_argument('--gps-noise', type=float, default=1.5); p.add_argument('--max-s', type=float, default=120)
p.add_argument('--seed', type=int, default=0); p.add_argument('-o', '--out', default='mission.mp4')
p.add_argument('--vlm', action='store_true', help='到着時に写真を VLM で判定する'); p.add_argument('--answer', default='はい', help='人の返事 (はい/いいえ)。音声入力は後で')
a = p.parse_args()
gs.init(logging_level='warning'); torch.manual_seed(a.seed); random.seed(a.seed)

def load(spec, env):
    exp, ck = spec.split(':'); log_dir = f'logs/{exp}'
    cfg = pickle.load(open(f'{log_dir}/cfgs.pkl', 'rb'))
    r = OnPolicyRunner(env, cfg[4], log_dir, device=gs.device); r.load(os.path.join(log_dir, f'model_{ck}.pt'))
    return r.get_inference_policy(device=gs.device)

exp = a.walk.split(':')[0]
env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f'logs/{exp}/cfgs.pkl', 'rb'))
reward_cfg['reward_scales'] = {}; env_cfg['push_vel'] = 0.0; env_cfg['episode_length_s'] = 1e6
env_cfg['termination_if_pitch_greater_than'] = 1e6; env_cfg['termination_if_roll_greater_than'] = 1e6; env_cfg['obstacles'] = True
env = Go2EnvSite(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=False)
walk = load(a.walk, env); recover = load(a.recover, env)
vlm = None
if a.vlm:
    from vlm_check import VLM
    vlm = VLM(); print(f'VLM loaded in {vlm.load_s:.0f}s', flush=True)
obs = env.reset()
env.robot.set_qpos(torch.cat([torch.tensor([[0.0, 0.0, 0.42]], device=gs.device), torch.tensor([[1.0, 0, 0, 0.0]], device=gs.device), env.init_dof_pos.reshape(1, -1)], dim=1), zero_velocity=True)
obs = env.get_observations()
dt = env.dt; fps = int(round(1 / dt))
env.cam.start_recording(save_to_filename=a.out, fps=fps); env.cam2.start_recording(save_to_filename=a.out.replace('.mp4', '_over.mp4'), fps=fps)

# --- GPS もどき: 1 秒に 1 回、真の位置にばらつきを足したものが届く。直近 3 回の平均で使う ---
gps_hist = []
def gps_read(true_xy):
    gps_hist.append((true_xy[0] + random.gauss(0, a.gps_noise / 2), true_xy[1] + random.gauss(0, a.gps_noise / 2)))
    del gps_hist[:-3]
    return (sum(g[0] for g in gps_hist) / len(gps_hist), sum(g[1] for g in gps_hist) / len(gps_hist))

mode, stand_t, wp_i, events = 'walk', 0, 0, []
planner = Planner([(ox, oy, sx, sy, sz) for _, (ox, oy), (sx, sy, sz) in OBSTACLES]); path, path_goal = [], None
extra_wp, approach_done, approached = None, False, set()
fall_t, resume_t = 0, -1.0
recover_since, wriggle_until = -1.0, -1.0
retreat_until, falls_recent = -1.0, []
est = (0.0, 0.0); last_prog_t, last_prog_d, detour_until, detour_sign = 0.0, 1e9, -1.0, 1.0
ARRIVE = 1.2
BLOCK_H = 0.25                      # これより高い段は越えられないとみなす
def scan_blocked(yaw_c):
    """高さスキャナ (LiDAR の見立て): 進行方向 yaw_c の前方 0.4〜1.8 m、幅 ±0.35 m の格子で、足元より BLOCK_H 以上高い所があるか"""
    base = env.base_pos[0, :2]; g0 = env.terrain_height(env.base_pos[:, :2])[0]
    pts = []
    for r in (0.4, 0.75, 1.1, 1.45, 1.8):
        for l in (-0.35, 0.0, 0.35):
            pts.append([base[0].item() + r * math.cos(yaw_c) - l * math.sin(yaw_c), base[1].item() + r * math.sin(yaw_c) + l * math.cos(yaw_c)])
    hs = (env.terrain_height(torch.tensor(pts, device=gs.device, dtype=gs.tc_float)) - g0).reshape(5, 3)
    rise = hs[1:] - hs[:-1]                                                   # 隣の行との段差 (階段なら 1 段ぶん)
    scan_blocked.last = f'手前 {hs[0].max().item()*100:.0f} cm / 段差 {rise.max().item()*100:.0f} cm / 最大 {hs.max().item()*100:.0f} cm'
    return bool((hs[0] > BLOCK_H).any() or (rise > 0.18).any() or (hs > 0.6).any())   # 一段で 18 cm 超 = 越えられない
scan_until, scan_yaw, scan_side, scan_last = -1.0, 0.0, 1, -10.0
stuck_log, escape_until = [], -1.0
turn_dir, thrash_until, immobile_n = 0, -1.0, 0
last_move_t, last_move_pos, dbg_t = 0.0, (0.0, 0.0), -1.0
def log(t, msg):
    events.append((round(t, 1), msg)); print(f'[{t:6.1f}s] {msg}', flush=True)
log(0.0, f'任務受領: {len(WAYPOINTS)} 地点を見回る (人の指示はここだけ)')
with torch.no_grad():
    for i in range(int(a.max_s / dt)):
        t = i * dt
        bp = env.base_pos[0].tolist(); upright = -env.projected_gravity[0, 2].item(); h = bp[2] - env.ground_height(env.base_pos[:, :2])[0].item()
        if i % fps == 0:
            est = gps_read(bp)
        yaw_now = math.radians(env.base_euler[0, 2].item())
        name, (wx, wy) = WAYPOINTS[wp_i]
        goal = extra_wp if extra_wp is not None else (wx, wy)
        if goal != path_goal or i % (5 * fps) == 0:                 # 目標が変わるか 5 秒ごとに計画し直す (GPS 推定位置から)
            path, path_goal = planner.plan(est, goal), goal
        la = planner.lookahead(path, est, 1.3)
        d = math.hypot(wx - est[0], wy - est[1])
        # --- 到着判定と次の経由地 ---
        if a.kick_at >= 0 and i == int(round(a.kick_at / dt)):      # 余震 (到着判定より前に処理)
            v = env.robot.get_dofs_velocity(dofs_idx_local=[0, 1]); v = v + torch.tensor([[0.0, a.kick]], device=v.device, dtype=v.dtype)
            env.robot.set_dofs_velocity(v, dofs_idx_local=[0, 1]); log(t, f'余震 (横から {a.kick} m/s の衝撃)')
        if extra_wp is not None and mode == 'walk':
            ex, ey = extra_wp
            if math.hypot(ex - bp[0], ey - bp[1]) < 1.2:            # 近づくときは GPS でなく足元 (実機なら近距離センサ) で判定
                log(t, f'要救助者のそばに到着。位置を通報 (GPS {est[0]:+.1f}, {est[1]:+.1f})。来た道を戻って見回りを続行')
                extra_wp = None; approach_done = True; last_prog_t, last_prog_d = t + 6.0, 1e9
                WAYPOINTS.insert(wp_i, ('合流点 (' + WAYPOINTS[wp_i - 1][0] + ')', WAYPOINTS[wp_i - 1][1]))
            else:
                d = math.hypot(ex - est[0], ey - est[1])
        if d < ARRIVE and mode == 'walk' and extra_wp is None:
            msg = f'{name} に到着 (GPS 推定の誤差 {math.hypot(bp[0]-est[0], bp[1]-est[1]):.1f} m)'
            found = False
            if vlm is not None:
                shot = a.out.replace('.mp4', f'_wp{wp_i}.png')
                ans, dt_v = '', 0.0
                for k_, ddeg in enumerate((0, -45, 45, -90, 90, -135, 135, 180)):   # 360° 見回す (8 方向)、「はい」で打ち切り
                    yk = yaw_now + math.radians(ddeg); hx_, hy_ = math.cos(yk), math.sin(yk)
                    env.cam3.set_pose(pos=(bp[0] + 0.3 * hx_, bp[1] + 0.3 * hy_, bp[2] + 0.15), lookat=(bp[0] + 2.5 * hx_, bp[1] + 2.5 * hy_, bp[2] - 0.25), up=(0.0, 0.0, 1.0))
                    shot_k = shot.replace('.png', f'_{k_}.png'); Image.fromarray(env.cam3.render()[0]).save(shot_k)
                    ans_k, dt_k = vlm.ask(shot_k, '災害現場のシミュレーション画像です (物は単純な形で描かれています)。地面に横たわる「人の形をした物体」(倒れている人、または救助訓練用のマネキン。オレンジの上着に青いズボンのことが多い) は写っていますか? 最初に「はい」か「いいえ」、そのあと一言で状況を。')
                    dt_v += dt_k
                    if ans_k.startswith('はい') or not ans:
                        ans = ans_k
                    if ans_k.startswith('はい'):
                        break
                found = ans.startswith('はい') and wp_i not in approached
                log(t, msg + f'。写真を判定 ({dt_v:.1f} 秒): ' + ans.replace(chr(10), ' ')[:60])
                if found and math.hypot(PERSON['pos'][0] - bp[0], PERSON['pos'][1] - bp[1]) > 4.0:
                    log(t, '遠方に要救助者らしきものを発見。位置を記録し、近い経由地で確認する')
                    found = False
                if found:
                    log(t, '要救助者らしきものを発見。人に確認: 「近づいて状況を確認しますか?」')
                    log(t + 0.5, f'人の返事: 「{a.answer}」')
                    if a.answer.startswith('はい'):
                        extra_wp = PERSON['pos']; approached.add(wp_i)
                    else:
                        log(t + 0.6, '了解。位置だけ通報して見回りを続行')
            else:
                log(t, msg + '。異常なし')
            wp_i += 1; last_prog_t, last_prog_d = t, 1e9
            if wp_i >= len(WAYPOINTS):
                log(t, '任務完了。待機します'); break
            continue
        # --- 転倒の検知と復帰 ---
        fall_t = fall_t + 1 if (upright < 0.5 or h < 0.10) else 0
        if mode == 'walk' and fall_t >= 15:                       # 0.3 秒続いたら転倒とみなす (段差での一瞬の傾きは無視)
            mode = 'recover'; stand_t = 0; recover_since = t; log(t, '転倒。起き上がり方策に切り替え')
        elif mode == 'recover' and t - recover_since > 8.0 and t >= wriggle_until:
            wriggle_until = t + 1.0; recover_since = t; log(t, '8 秒たっても立てない。体勢を変えてやり直し')
        elif mode == 'recover':
            stand_t = stand_t + 1 if (upright > 0.95 and h > 0.22) else 0
            if stand_t >= 25:
                mode = 'walk'; resume_t = t; last_prog_t, last_prog_d = t, 1e9
                falls_recent = [ft for ft in falls_recent if t - ft < 20.0] + [t]
                if len(falls_recent) >= 3:
                    detour_until = t + 5.0; detour_sign = -detour_sign; falls_recent = []
                    log(t, '起き上がり完了。同じ所で転び続けるので大きく迂回')
                else:
                    retreat_until = t + 2.0; log(t, '起き上がり完了。少し戻ってから見回りを再開')
        # --- 外部イベント (余震や落下物の見立て) ---
        # --- 進めないときの迂回 ---
        err_now = (math.atan2(la[1] - est[1], la[0] - est[0]) - yaw_now + math.pi) % (2 * math.pi) - math.pi
        if mode == 'walk':
            if d < last_prog_d - 0.3 or abs(err_now) > 0.6:         # 向きを変えている間は進めなくて当然
                last_prog_d, last_prog_t = d, t
            elif t - last_prog_t > 4.0 and t > detour_until:
                detour_until = t + 2.5; detour_sign = -detour_sign; last_prog_t = t
                log(t, '進めない。向きを変えて迂回')
        # --- 本当に動いていないときの見張り (向きも位置も変わらない = 何かにはまっている) ---
        if math.hypot(bp[0] - last_move_pos[0], bp[1] - last_move_pos[1]) > 0.25:
            last_move_t, last_move_pos = t, (bp[0], bp[1])
        elif mode == 'walk' and t - last_move_t > 8.0 and t >= escape_until and t >= thrash_until:
            immobile_n += 1; last_move_t = t
            if immobile_n % 2 == 1:
                escape_until = t + 2.5; log(t, '8 秒間その場から動けない。前へ押し出して立て直す')
            else:
                thrash_until = t + 2.0; log(t, 'まだ動けない。起き上がり方策で体勢を崩して脱出')
        if math.hypot(bp[0] - last_move_pos[0], bp[1] - last_move_pos[1]) > 0.25:
            immobile_n = 0
        if t - dbg_t >= 5.0:
            dbg_t = t
            print(f'    dbg t={t:.0f} mode={mode} pos=({bp[0]:.2f},{bp[1]:.2f}) est=({est[0]:.2f},{est[1]:.2f}) goal={goal} la=({la[0]:.2f},{la[1]:.2f}) yaw={math.degrees(yaw_now):.0f} err={math.degrees(err_now):.0f} up={upright:.2f} h={h:.2f}', flush=True)
        # --- 指令 ---
        yaw_now = math.radians(env.base_euler[0, 2].item())
        target_yaw = math.atan2(la[1] - est[1], la[0] - est[0]) + (detour_sign * 1.2 if t < detour_until else 0.0)
        if t < retreat_until or t < escape_until:                 # 戻る/脱出: 中央の方を向いて歩く
            target_yaw = math.atan2(0.0 - est[1], 0.0 - est[0])
        # 高さスキャナ: 5 Hz で前方を見て、越えられない物があれば空いている向きへ舵を切る
        if mode == 'walk' and i % 10 == 0 and t >= escape_until:
            if t >= scan_until:
                if scan_blocked(target_yaw):
                    blocked_info = scan_blocked.last
                    if t - scan_last > 5.0:
                        scan_side = 1 if scan_blocked(target_yaw - math.radians(50)) else -1   # 右が塞がっていれば左
                    scan_last = t
                    for ddeg in [scan_side * d for d in (35, 70, 105, 150)] + [-scan_side * d for d in (35, 70, 105, 150)] + [180]:
                        cand = target_yaw + math.radians(ddeg)
                        if not scan_blocked(cand):
                            scan_yaw, scan_until = cand, t + 1.5
                            log(t, f'前方に越えられない障害物 ({blocked_info})。{"左" if ddeg > 0 else "右"}へ迂回')
                            stuck_log = [x for x in stuck_log if t - x < 20.0] + [t]
                            if len(stuck_log) >= 8 and t - last_prog_t > 3.0:
                                escape_until = t + 5.0; scan_until = -1.0; stuck_log = []
                                log(t, '行き詰まり。いったん中央へ向かって脱出')
                            break
        if t < scan_until:
            target_yaw = scan_yaw
        err = (target_yaw - yaw_now + math.pi) % (2 * math.pi) - math.pi
        if abs(err) > math.radians(150) and turn_dir == 0:
            turn_dir = 1 if err > 0 else -1                            # 回る向きを決めて固定 (GPS のばらつきで左右に迷わない)
        elif abs(err) < math.radians(90):
            turn_dir = 0
        if turn_dir != 0:
            err = turn_dir * abs(err)
        if mode == 'walk' and t < escape_until and escape_until - t <= 2.5:
            env.commands[:, 0] = 0.5; env.commands[:, 1] = 0.0; env.commands[:, 2] = 0.0
        elif mode == 'walk':
            slow = t - resume_t < 1.5                                 # 起き上がり直後はゆっくり
            env.commands[:, 0] = (0.3 if slow else 0.7) if abs(err) < 0.8 else 0.25; env.commands[:, 1] = 0.0; env.commands[:, 2] = max(-0.8, min(0.8, 2.0 * err))
        else:
            env.commands[:, :] = 0.0
        actions = (walk if (mode == 'walk' or t < wriggle_until) and t >= thrash_until else recover)(obs)
        obs, rews, dones, infos = env.step(actions)
        bp = env.base_pos[0].tolist()
        gz = env.ground_height(env.base_pos[:, :2])[0].item()                    # カメラは地面基準で水平を保つ
        env.cam.set_pose(pos=(bp[0] - 2.4 * math.cos(yaw_now) + 0.9 * math.sin(yaw_now), bp[1] - 2.4 * math.sin(yaw_now) - 0.9 * math.cos(yaw_now), gz + 1.4), lookat=(bp[0] + 0.5 * math.cos(yaw_now), bp[1] + 0.5 * math.sin(yaw_now), gz + 0.3), up=(0.0, 0.0, 1.0))
        env.cam.render(); env.cam2.render()
env.cam.stop_recording(); env.cam2.stop_recording()
import json; json.dump(events, open(a.out + '.events.json', 'w'), ensure_ascii=False)
print('MISSION', 'done' if wp_i >= len(WAYPOINTS) else f'incomplete at wp {wp_i}', f't={t:.1f}s')
