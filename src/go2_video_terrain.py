"""地形 (階段) の上で方策を評価して動画にする。+x (上り階段) に向けて歩かせる。
python go2_video_terrain.py -e go2-terrain --ckpt 799 --steps 750 --vx 0.5 -o out.mp4"""
import argparse, math, os, pickle, torch, genesis as gs
from rsl_rl.runners import OnPolicyRunner
from go2_env_terrain import Go2EnvTerrain
from terrain_field import step_height, FLAT, STEP_W
p = argparse.ArgumentParser()
p.add_argument('-e', '--exp_name', default='go2-terrain'); p.add_argument('--ckpt', type=int, default=799)
p.add_argument('--steps', type=int, default=750); p.add_argument('--vx', type=float, default=0.5); p.add_argument('--heading', type=float, default=0.0)
p.add_argument('-o', '--out', default='terrain_eval.mp4')
a = p.parse_args()
gs.init(logging_level='warning'); torch.manual_seed(1)
log_dir = f'logs/{a.exp_name}'
env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f'{log_dir}/cfgs.pkl', 'rb'))
reward_cfg['reward_scales'] = {}; env_cfg['push_vel'] = 0.0; env_cfg['episode_length_s'] = 1000.0; env_cfg['terrain_kind'] = 'stairs'
env = Go2EnvTerrain(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=False)
runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
runner.load(os.path.join(log_dir, f'model_{a.ckpt}.pt'))
policy = runner.get_inference_policy(device=gs.device)
obs = env.reset()
yaw = a.heading; q = torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]], device=gs.device)
env.robot.set_qpos(torch.cat([torch.tensor([[0.0, 0.0, 0.42]], device=gs.device), q, env.init_dof_pos.reshape(1, -1)], dim=1), zero_velocity=True)
obs = env.get_observations()
env.cam.start_recording(save_to_filename=a.out, fps=int(round(1 / env.dt)))
falls = 0; max_x = 0.0; max_d = 0.0; hx, hy = math.cos(a.heading), math.sin(a.heading)
with torch.no_grad():
    for i in range(a.steps):
        env.commands[:, 0] = a.vx; env.commands[:, 1] = 0.0
        # 進行方向 +x を保つ (ヨー誤差に比例した旋回指令)
        yaw_now = math.radians(env.base_euler[0, 2].item()); err = (a.heading - yaw_now + math.pi) % (2 * math.pi) - math.pi
        env.commands[:, 2] = max(-0.8, min(0.8, 1.5 * err))
        actions = policy(obs)
        obs, rews, dones, infos = env.step(actions)
        if bool(dones[0]) and i < a.steps - 1:
            falls += 1
        bp = env.base_pos[0].tolist(); max_x = max(max_x, bp[0]); max_d = max(max_d, bp[0] * hx + bp[1] * hy)
        env.cam.set_pose(pos=(bp[0] + 1.6, bp[1] - 2.2, bp[2] + 1.0), lookat=(bp[0] + 0.3, bp[1], bp[2]))
        env.cam.render()
env.cam.stop_recording()
k = max(0, int((max_x - FLAT) // STEP_W) + 1) if max_x > FLAT else 0
print(f'TERRAIN exp={a.exp_name} ckpt={a.ckpt} heading={math.degrees(a.heading):.0f} falls={falls} dist={max_d:.2f} m max_x={max_x:.2f} m steps_climbed={k} top_step_height={step_height(k) if k else 0:.2f} m final_z={bp[2]:.2f}')
