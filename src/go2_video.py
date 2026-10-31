"""学習した Go2 の歩行方策を 1 体で動かし、追従カメラで mp4 に書き出す (ヘッドレス)。
python go2_video.py -e go2-walking --ckpt 100 --steps 500 --vx 0.6 -o go2_walk.mp4"""
import argparse, os, pickle, torch, genesis as gs
from rsl_rl.runners import OnPolicyRunner
from go2_env_cam import Go2EnvCam

p = argparse.ArgumentParser()
p.add_argument('-e', '--exp_name', default='go2-walking'); p.add_argument('--ckpt', type=int, default=100)
p.add_argument('--steps', type=int, default=500); p.add_argument('--vx', type=float, default=0.6); p.add_argument('--wz', type=float, default=0.0)
p.add_argument('-o', '--out', default='go2_walk.mp4')
a = p.parse_args()
gs.init(logging_level='warning')
log_dir = f'logs/{a.exp_name}'
env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f'{log_dir}/cfgs.pkl', 'rb'))
reward_cfg['reward_scales'] = {}
env = Go2EnvCam(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=False)
runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
runner.load(os.path.join(log_dir, f'model_{a.ckpt}.pt'))
policy = runner.get_inference_policy(device=gs.device)
obs = env.reset()
env.cam.start_recording(save_to_filename=a.out, fps=int(round(1 / env.dt)))
dist = 0.0
with torch.no_grad():
    for i in range(a.steps):
        if hasattr(env, 'commands'):
            env.commands[:, 0] = a.vx; env.commands[:, 1] = 0.0; env.commands[:, 2] = a.wz
        actions = policy(obs)
        obs, rews, dones, infos = env.step(actions)
        bp = env.base_pos[0].tolist()
        env.cam.set_pose(pos=(bp[0] + 1.8, bp[1] - 1.8, 1.1), lookat=(bp[0], bp[1], 0.3))
        env.cam.render()
        if i == a.steps - 1:
            dist = bp[0]
env.cam.stop_recording()
print(f'VIDEO {a.out} steps={a.steps} x_travel={dist:.2f} m')
