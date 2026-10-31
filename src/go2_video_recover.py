"""倒れた姿勢から起き上がる方策の動画。python go2_video_recover.py -e go2-recover --ckpt 299 --seed 3 --steps 300 -o out.mp4"""
import argparse, os, pickle, torch, genesis as gs
from rsl_rl.runners import OnPolicyRunner
from go2_env_recover import Go2EnvRecover
p = argparse.ArgumentParser()
p.add_argument('-e', '--exp_name', default='go2-recover'); p.add_argument('--ckpt', type=int, default=299)
p.add_argument('--steps', type=int, default=300); p.add_argument('--seed', type=int, default=3); p.add_argument('-o', '--out', default='recover.mp4')
a = p.parse_args()
gs.init(logging_level='warning'); torch.manual_seed(a.seed)
log_dir = f'logs/{a.exp_name}'
env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f'{log_dir}/cfgs.pkl', 'rb'))
reward_cfg['reward_scales'] = {}; env_cfg['episode_length_s'] = 1000.0
env = Go2EnvRecover(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=False)
runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
runner.load(os.path.join(log_dir, f'model_{a.ckpt}.pt'))
policy = runner.get_inference_policy(device=gs.device)
obs = env.reset()
env.cam.start_recording(save_to_filename=a.out, fps=int(round(1 / env.dt)))
up_t = None
with torch.no_grad():
    for i in range(a.steps):
        actions = policy(obs)
        obs, rews, dones, infos = env.step(actions)
        upright = -env.projected_gravity[0, 2].item(); h = env.base_pos[0, 2].item()
        if up_t is None and upright > 0.95 and h > 0.25:
            up_t = i * env.dt
        bp = env.base_pos[0].tolist()
        env.cam.set_pose(pos=(bp[0] + 1.6, bp[1] - 1.6, 1.0), lookat=(bp[0], bp[1], 0.25))
        env.cam.render()
env.cam.stop_recording()
print(f'RECOVER seed={a.seed} stood_up_at={up_t} final_upright={upright:.2f} final_height={h:.2f}')
