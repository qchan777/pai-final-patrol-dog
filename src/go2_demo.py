"""司令役つきデモ: 歩行方策で歩く → 強い蹴りで転ぶ → 起き上がり方策に切り替え → 立てたら歩行方策に戻す。
python go2_demo.py --walk go2-push3:599 --recover go2-recover3:499 --kick 4.0 --kick-at 3.0 --steps 600 -o demo.mp4"""
import argparse, math, os, pickle, torch, genesis as gs
from rsl_rl.runners import OnPolicyRunner
from go2_env_recover import Go2EnvRecover
p = argparse.ArgumentParser()
p.add_argument('--walk', default='go2-push3:599'); p.add_argument('--recover', default='go2-recover3:499')
p.add_argument('--kick', type=float, default=4.0); p.add_argument('--kick-at', type=float, default=3.0)
p.add_argument('--steps', type=int, default=600); p.add_argument('--vx', type=float, default=0.6); p.add_argument('-o', '--out', default='demo.mp4')
a = p.parse_args()
gs.init(logging_level='warning'); torch.manual_seed(5)

def load(spec, env):
    exp, ck = spec.split(':'); log_dir = f'logs/{exp}'
    cfg = pickle.load(open(f'{log_dir}/cfgs.pkl', 'rb'))
    runner = OnPolicyRunner(env, cfg[4], log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f'model_{ck}.pt'))
    return runner.get_inference_policy(device=gs.device)

# 環境は歩行用の設定 (立った姿勢から開始、倒れても終了しない、こちらで蹴る)
exp = a.walk.split(':')[0]
env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f'logs/{exp}/cfgs.pkl', 'rb'))
reward_cfg['reward_scales'] = {}; env_cfg['push_vel'] = 0.0; env_cfg['episode_length_s'] = 1000.0
env = Go2EnvRecover(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=False)
walk = load(a.walk, env); recover = load(a.recover, env)
obs = env.reset()
# 立った姿勢に置き直す (recover 環境のリセットは倒れた姿勢なので)
env.robot.set_qpos(env.init_qpos, zero_velocity=True); obs = env.get_observations()
env.cam.start_recording(save_to_filename=a.out, fps=int(round(1 / env.dt)))
mode, stand_t, log = 'walk', 0, []
with torch.no_grad():
    for i in range(a.steps):
        t = i * env.dt
        upright = -env.projected_gravity[0, 2].item(); h = env.base_pos[0, 2].item()
        fallen = upright < 0.6 or h < 0.15
        if mode == 'walk' and fallen:
            mode = 'recover'; stand_t = 0; log.append((t, 'recover'))
        elif mode == 'recover':
            stand_t = stand_t + 1 if (upright > 0.95 and h > 0.25) else 0
            if stand_t >= 25:                      # 0.5 秒立てていたら歩行へ
                mode = 'walk'; log.append((t, 'walk'))
        if abs(t - a.kick_at) < env.dt / 2:
            v = env.robot.get_dofs_velocity(dofs_idx_local=[0, 1])
            v = v + torch.tensor([[0.0, a.kick]], device=v.device, dtype=v.dtype)
            env.robot.set_dofs_velocity(v, dofs_idx_local=[0, 1]); log.append((t, 'kick'))
        env.commands[:, 0] = a.vx if mode == 'walk' else 0.0; env.commands[:, 1] = 0.0; env.commands[:, 2] = 0.0
        actions = (walk if mode == 'walk' else recover)(obs)
        obs, rews, dones, infos = env.step(actions)
        bp = env.base_pos[0].tolist()
        env.cam.set_pose(pos=(bp[0] + 1.8, bp[1] - 1.8, 1.1), lookat=(bp[0], bp[1], 0.3))
        env.cam.render()
env.cam.stop_recording()
print('DEMO', [(round(t, 2), m) for t, m in log], f'final_x={bp[0]:.2f}')
