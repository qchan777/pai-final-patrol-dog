"""学習済み方策を、決まった蹴りの列 (seed 固定) にさらして評価し、動画を書く。
python go2_video_push.py -e go2-walking --ckpt 100 --push-vel 1.0 --push-interval 2.0 --steps 500 -o out.mp4"""
import argparse, math, os, pickle, torch, genesis as gs
from rsl_rl.runners import OnPolicyRunner
from go2_env_push import Go2EnvPush
p = argparse.ArgumentParser()
p.add_argument('-e', '--exp_name', default='go2-walking'); p.add_argument('--ckpt', type=int, default=100)
p.add_argument('--steps', type=int, default=500); p.add_argument('--vx', type=float, default=0.6)
p.add_argument('--push-vel', type=float, default=1.0); p.add_argument('--push-interval', type=float, default=2.0)
p.add_argument('--seed', type=int, default=7); p.add_argument('-o', '--out', default='push_eval.mp4')
a = p.parse_args()
gs.init(logging_level='warning'); torch.manual_seed(a.seed)
log_dir = f'logs/{a.exp_name}'
env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f'{log_dir}/cfgs.pkl', 'rb'))
reward_cfg['reward_scales'] = {}
env_cfg['push_vel'] = 0.0                      # 自動の小突きは切り、こちらで決めた列を与える
env = Go2EnvPush(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=False)
runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
runner.load(os.path.join(log_dir, f'model_{a.ckpt}.pt'))
policy = runner.get_inference_policy(device=gs.device)
obs = env.reset()
env.cam.start_recording(save_to_filename=a.out, fps=int(round(1 / env.dt)))
g = torch.Generator(device='cpu').manual_seed(a.seed)
every = int(a.push_interval / env.dt); falls = 0; pushes = []
with torch.no_grad():
    for i in range(a.steps):
        env.commands[:, 0] = a.vx; env.commands[:, 1] = 0.0; env.commands[:, 2] = 0.0
        if i > 0 and i % every == 0:
            ang = torch.rand(1, generator=g).item() * 2 * math.pi; mag = a.push_vel
            v = env.robot.get_dofs_velocity(dofs_idx_local=[0, 1])
            v = v + torch.tensor([[mag * math.cos(ang), mag * math.sin(ang)]], device=v.device, dtype=v.dtype)
            env.robot.set_dofs_velocity(v, dofs_idx_local=[0, 1]); pushes.append(i * env.dt)
        actions = policy(obs)
        obs, rews, dones, infos = env.step(actions)
        if bool(dones[0]) and i < a.steps - 1:
            falls += 1
        bp = env.base_pos[0].tolist()
        env.cam.set_pose(pos=(bp[0] + 1.8, bp[1] - 1.8, 1.1), lookat=(bp[0], bp[1], 0.3))
        env.cam.render()
env.cam.stop_recording()
print(f'EVAL exp={a.exp_name} ckpt={a.ckpt} push_vel={a.push_vel} every={a.push_interval}s pushes={len(pushes)} falls={falls} x_travel={bp[0]:.2f} m pushes_at={[round(t,1) for t in pushes]}')
