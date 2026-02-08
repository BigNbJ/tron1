# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from isaacgym.torch_utils import *
from legged_gym.envs import *
from legged_gym.utils import (
    get_args,
    export_policy_as_jit,
    export_mlp_as_onnx,
    task_registry,
    Logger,
)
from legged_gym.utils.RecordVideoWrapper import RecordVideoWrapper

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import time

class LivePlotter:
    def __init__(self, max_len=200, dt=0.02, refresh_rate=5, action_scale=1.0):
        self.max_len = max_len
        self.dt = dt
        self.refresh_rate = refresh_rate
        self.action_scale = action_scale
        self.counter = 0
        
        # Data buffers
        self.time_buf = deque(maxlen=max_len)
        # Forces
        self.force_x_L = deque(maxlen=max_len)
        self.force_x_R = deque(maxlen=max_len)
        self.force_y_L = deque(maxlen=max_len)
        self.force_y_R = deque(maxlen=max_len)
        self.force_z_L = deque(maxlen=max_len)
        self.force_z_R = deque(maxlen=max_len)
        
        # Trigger
        self.trigger_L = deque(maxlen=max_len)
        self.trigger_R = deque(maxlen=max_len)
        
        # Actions (Policy)
        self.pol_hip_L = deque(maxlen=max_len)
        self.pol_hip_R = deque(maxlen=max_len)
        self.pol_knee_L = deque(maxlen=max_len)
        self.pol_knee_R = deque(maxlen=max_len)
        
        # Actions (Feedforward)
        self.ff_hip_L = deque(maxlen=max_len)
        self.ff_hip_R = deque(maxlen=max_len)
        self.ff_knee_L = deque(maxlen=max_len)
        self.ff_knee_R = deque(maxlen=max_len)
        
        # Initialize time
        self.current_time = 0.0
        
        # Setup plot: 2 Cols x 3 Rows
        plt.ion()
        self.fig, self.axs = plt.subplots(3, 2, sharex=True, figsize=(14, 10))
        self.fig.canvas.manager.set_window_title('Live Oscilloscope')
        
        # --- Column 1: Contact Forces ---
        # Row 0: Force X
        self.ax_fx = self.axs[0, 0]
        self.line_fx_L, = self.ax_fx.plot([], [], label='Fx L', color='b')
        self.line_fx_R, = self.ax_fx.plot([], [], label='Fx R', color='r')
        self.ax_fx.set_ylabel('Force X [N]')
        self.ax_fx.legend(loc='upper right', fontsize='small')
        self.ax_fx.grid(True)
        
        # Row 1: Force Y
        self.ax_fy = self.axs[1, 0]
        self.line_fy_L, = self.ax_fy.plot([], [], label='Fy L', color='b')
        self.line_fy_R, = self.ax_fy.plot([], [], label='Fy R', color='r')
        self.ax_fy.set_ylabel('Force Y [N]')
        self.ax_fy.legend(loc='upper right', fontsize='small')
        self.ax_fy.grid(True)
        
        # Row 2: Force Z
        self.ax_fz = self.axs[2, 0]
        self.line_fz_L, = self.ax_fz.plot([], [], label='Fz L', color='b')
        self.line_fz_R, = self.ax_fz.plot([], [], label='Fz R', color='r')
        self.ax_fz.set_ylabel('Force Z [N]')
        self.ax_fz.set_xlabel('Time [s]')
        self.ax_fz.legend(loc='upper right', fontsize='small')
        self.ax_fz.grid(True)
        
        # --- Column 2: Trigger & Actions ---
        # Row 0: Trigger
        self.ax_trig = self.axs[0, 1]
        self.line_trig_L, = self.ax_trig.plot([], [], label='Trigger L', color='b', alpha=0.7)
        self.line_trig_R, = self.ax_trig.plot([], [], label='Trigger R', color='r', alpha=0.7)
        self.ax_trig.set_ylabel('Trigger')
        self.ax_trig.set_ylim(-0.1, 1.1)
        self.ax_trig.legend(loc='upper right', fontsize='small')
        self.ax_trig.grid(True)
        
        # Row 1: Hip Actions
        self.ax_hip = self.axs[1, 1]
        self.line_pol_hip_L, = self.ax_hip.plot([], [], label='Pol Hip L', color='b', linestyle='--')
        self.line_pol_hip_R, = self.ax_hip.plot([], [], label='Pol Hip R', color='r', linestyle='--')
        self.line_ff_hip_L, = self.ax_hip.plot([], [], label='FF Hip L', color='b')
        self.line_ff_hip_R, = self.ax_hip.plot([], [], label='FF Hip R', color='r')
        self.ax_hip.set_ylabel('Hip Action [rad]')
        self.ax_hip.legend(loc='upper right', fontsize='small', ncol=2)
        self.ax_hip.grid(True)

        # Row 2: Knee Actions
        self.ax_knee = self.axs[2, 1]
        self.line_pol_knee_L, = self.ax_knee.plot([], [], label='Pol Knee L', color='b', linestyle='--')
        self.line_pol_knee_R, = self.ax_knee.plot([], [], label='Pol Knee R', color='r', linestyle='--')
        self.line_ff_knee_L, = self.ax_knee.plot([], [], label='FF Knee L', color='b')
        self.line_ff_knee_R, = self.ax_knee.plot([], [], label='FF Knee R', color='r')
        self.ax_knee.set_ylabel('Knee Action [rad]')
        self.ax_knee.set_xlabel('Time [s]')
        self.ax_knee.legend(loc='upper right', fontsize='small', ncol=2)
        self.ax_knee.grid(True)
        
    def update(self, 
               force_x_l, force_x_r, force_y_l, force_y_r, force_z_l, force_z_r,
               trigger_l, trigger_r, 
               pol_hip_l, pol_hip_r, pol_knee_l, pol_knee_r,
               ff_hip_l, ff_hip_r, ff_knee_l, ff_knee_r):
        self.current_time += self.dt
        
        self.time_buf.append(self.current_time)
        
        # Forces
        self.force_x_L.append(force_x_l)
        self.force_x_R.append(force_x_r)
        self.force_y_L.append(force_y_l)
        self.force_y_R.append(force_y_r)
        self.force_z_L.append(force_z_l)
        self.force_z_R.append(force_z_r)
        
        # Trigger
        self.trigger_L.append(trigger_l)
        self.trigger_R.append(trigger_r)
        
        # Actions (Scale to Radians)
        s = self.action_scale
        self.pol_hip_L.append(pol_hip_l * s)
        self.pol_hip_R.append(pol_hip_r * s)
        self.pol_knee_L.append(pol_knee_l * s)
        self.pol_knee_R.append(pol_knee_r * s)
        
        self.ff_hip_L.append(ff_hip_l * s)
        self.ff_hip_R.append(ff_hip_r * s)
        self.ff_knee_L.append(ff_knee_l * s)
        self.ff_knee_R.append(ff_knee_r * s)
        
        self.counter += 1
        if self.counter % self.refresh_rate == 0:
            self._draw()
            
    def _draw(self):
        t = list(self.time_buf)
        
        # Forces
        self.line_fx_L.set_data(t, list(self.force_x_L))
        self.line_fx_R.set_data(t, list(self.force_x_R))
        self.line_fy_L.set_data(t, list(self.force_y_L))
        self.line_fy_R.set_data(t, list(self.force_y_R))
        self.line_fz_L.set_data(t, list(self.force_z_L))
        self.line_fz_R.set_data(t, list(self.force_z_R))
        
        # Trigger
        self.line_trig_L.set_data(t, list(self.trigger_L))
        self.line_trig_R.set_data(t, list(self.trigger_R))
        
        # Actions
        self.line_pol_hip_L.set_data(t, list(self.pol_hip_L))
        self.line_pol_hip_R.set_data(t, list(self.pol_hip_R))
        self.line_ff_hip_L.set_data(t, list(self.ff_hip_L))
        self.line_ff_hip_R.set_data(t, list(self.ff_hip_R))
        
        self.line_pol_knee_L.set_data(t, list(self.pol_knee_L))
        self.line_pol_knee_R.set_data(t, list(self.pol_knee_R))
        self.line_ff_knee_L.set_data(t, list(self.ff_knee_L))
        self.line_ff_knee_R.set_data(t, list(self.ff_knee_R))
        
        # Rescale axes
        if len(t) > 0:
            self.ax_fz.set_xlim(min(t), max(t) + self.dt)
            # ax_knee shares X, so no need to set it explicitly
            
            # Auto-scale Forces
            # FX
            all_fx = list(self.force_x_L) + list(self.force_x_R)
            if all_fx:
                min_v, max_v = min(all_fx), max(all_fx)
                span = max(1.0, max_v - min_v)
                self.ax_fx.set_ylim(min_v - 0.1*span, max_v + 0.1*span)
            
            # FY
            all_fy = list(self.force_y_L) + list(self.force_y_R)
            if all_fy:
                min_v, max_v = min(all_fy), max(all_fy)
                span = max(1.0, max_v - min_v)
                self.ax_fy.set_ylim(min_v - 0.1*span, max_v + 0.1*span)
                
            # FZ
            all_fz = list(self.force_z_L) + list(self.force_z_R)
            if all_fz:
                min_v, max_v = min(all_fz), max(all_fz)
                span = max(1.0, max_v - min_v)
                self.ax_fz.set_ylim(min_v - 0.1*span, max_v + 0.1*span)

            # Auto-scale Actions
            all_hips = list(self.pol_hip_L) + list(self.pol_hip_R) + list(self.ff_hip_L) + list(self.ff_hip_R)
            if all_hips:
                min_h, max_h = min(all_hips), max(all_hips)
                span = max(0.1, max_h - min_h)
                self.ax_hip.set_ylim(min_h - 0.1*span, max_h + 0.1*span)
                
            all_knees = list(self.pol_knee_L) + list(self.pol_knee_R) + list(self.ff_knee_L) + list(self.ff_knee_R)
            if all_knees:
                min_k, max_k = min(all_knees), max(all_knees)
                span = max(0.1, max_k - min_k)
                self.ax_knee.set_ylim(min_k - 0.1*span, max_k + 0.1*span)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # plt.pause(0.001) # This might block, but usually needed for updates

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.episode_length_s = 30
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)

    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 20
    # env_cfg.terrain.terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]
    env_cfg.terrain.max_init_terrain_level = 9
    env_cfg.terrain.curriculum = True
    env_cfg.noise.add_noise = True
    env_cfg.noise.noise_level = 0.5
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.push_robots = False

    if RECORD_VIDEO:
        env_cfg.env.enable_camera_sensors = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env = RecordVideoWrapper(env)
    # get robot_type
    robot_type = os.getenv("ROBOT_TYPE", "")
    commands_val = torch.zeros(env.cfg.commands.num_commands, device=env.device)
    
    if robot_type.startswith("PF"):
        commands_val[0] = 0.5
    elif robot_type == "WF_TRON1A":
        commands_val[0] = 2.0
    else:
        commands_val[0] = 1.5
    
    # commands_val = to_torch([0.5, 0.0, 0, 0], device=env.device) if robot_type.startswith("PF")\
    #     else to_torch([1.0, 0.0, 0.0], device=env.device) if robot_type == "WF_TRON1A" else to_torch([1.5, 0.0, 0.0, 0.0, 0.0])
    action_scale = env.cfg.control.action_scale_pos if robot_type == "WF_TRON1A"\
        else env.cfg.control.action_scale
    obs, obs_history, commands, _ = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    # train_cfg.runner.checkpoint = -1

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    encoder = ppo_runner.get_inference_encoder(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            args.task,
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)
        export_mlp_as_onnx(
            ppo_runner.alg.actor_critic.actor,
            path,
            "policy",
            ppo_runner.alg.actor_critic.num_actor_obs,
        )
        export_mlp_as_onnx(
            ppo_runner.alg.encoder,
            path,
            "encoder",
            ppo_runner.alg.encoder.num_input_dim,
        )

    logger = Logger(env.dt)
    robot_index = 5  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 300  # number of steps before plotting states
    stop_rew_log = (
        env.max_episode_length + 1
    )  # number of steps before print average episode rewards
    # camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    # camera_vel = np.array([1.0, 1.0, 0.0])
    # camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    est = None
    
    # Initialize Live Plotter
    live_plotter = LivePlotter(max_len=600, dt=env.dt, refresh_rate=5, action_scale=action_scale)
    
    if RECORD_VIDEO:
        env.set_camera_video_props(frame_size=(720, 480), camera_offset=(0.0, 2.5, 0.5), camera_rotation=(0., 0., -90.0), env_idx=robot_index, actor_idx=0, rigid_body_idx=0, fps=50)
        env.start_recording_video()

    for i in range(10 * int(env.max_episode_length)):
        # Trigger jump every 200 steps for 10 steps
        # if env.cfg.commands.num_commands > 4:
        #     if i % 200 >= 100 and i % 200 < 110:
        #         commands_val[4] = 0.3  # Set jump height
        #     else:
        #         commands_val[4] = 0.0

        est = encoder(obs_history)
        actions = policy(torch.cat((est, obs, commands), dim=-1).detach())

        env.commands[:, :] = commands_val

        obs, rews, dones, infos, obs_history, commands, _ = env.step(
            actions.detach()
        )
        
        # Update Live Plotter
        if "trigger_mask" in infos and "ff_actions" in infos:
            # Get Contact Forces
            # contact_forces shape: (num_envs, num_bodies, 3)
            # feet_indices: [left_wheel_idx, right_wheel_idx]
            forces = env.contact_forces[robot_index, env.feet_indices, :].cpu()
            
            # Get Actions
            # actions shape: (num_envs, 12)
            # Indices: Left Hip=1, Left Knee=2, Right Hip=5, Right Knee=6
            pol_actions = actions[robot_index].cpu()
            
            live_plotter.update(
                # Forces (L/R)
                force_x_l=forces[0, 0].item(), force_x_r=forces[1, 0].item(),
                force_y_l=forces[0, 1].item(), force_y_r=forces[1, 1].item(),
                force_z_l=forces[0, 2].item(), force_z_r=forces[1, 2].item(),
                
                # Trigger
                trigger_l=infos["trigger_mask"][robot_index, 0].item(),
                trigger_r=infos["trigger_mask"][robot_index, 1].item(),
                
                # Policy Actions
                pol_hip_l=pol_actions[1].item(), pol_hip_r=pol_actions[5].item(),
                pol_knee_l=pol_actions[2].item(), pol_knee_r=pol_actions[6].item(),
                
                # FF Actions
                ff_hip_l=infos["ff_actions"][robot_index, 1].item(),
                ff_hip_r=infos["ff_actions"][robot_index, 5].item(),
                ff_knee_l=infos["ff_actions"][robot_index, 2].item(),
                ff_knee_r=infos["ff_actions"][robot_index, 6].item()
            )

        if RECORD_VIDEO and i == 500:
            video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
            env.end_and_save_recording_video(video_path=video_dir, filename=f"video_{args.task}.mp4")
            print(f"Video saved to {video_dir}/video_{args.task}.mp4")
            
            # Save the live plotter figure with complete 500 frames
            plot_path = os.path.join(video_dir, f"video_{args.task}_plot.png")
            live_plotter.fig.savefig(plot_path)
            print(f"Data plot saved to {plot_path}")

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(
                    LEGGED_GYM_ROOT_DIR,
                    "logs",
                    train_cfg.runner.experiment_name,
                    "exported",
                    "frames",
                    f"{img_idx}.png",
                )
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_offset = np.array(env_cfg.viewer.pos)
            target_position = np.array(
                env.base_position[robot_index, :].to(device="cpu")
            )
            target_position[2] = 0
            camera_position = target_position + camera_offset
            # env.set_camera(camera_position, target_position)

        if i < stop_state_log:
            logger.log_states(
                {
                    "dof_pos_target": actions[robot_index, joint_index].item() * action_scale,
                    "dof_pos": (
                        env.dof_pos[robot_index, joint_index]
                        - env.raw_default_dof_pos[joint_index]
                    ).item(),
                    "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.torques[robot_index, joint_index].item(),
                    "command_x": env.commands[robot_index, 0].item(),
                    "command_y": env.commands[robot_index, 1].item(),
                    "command_yaw": env.commands[robot_index, 2].item(),
                    "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                    "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                    "power": torch.sum(env.power[robot_index, :]).item(),
                    "contact_forces_z": env.contact_forces[
                        robot_index, env.feet_indices, 2
                    ]
                    .cpu()
                    .numpy(),
                    "wheel_height_L": env.foot_positions[robot_index, 0, 2].item(),
                    "wheel_height_R": env.foot_positions[robot_index, 1, 2].item(),
                    "wheel_force_x_L": env.contact_forces[robot_index, env.feet_indices[0], 0].item(),
                    "wheel_force_y_L": env.contact_forces[robot_index, env.feet_indices[0], 1].item(),
                    "wheel_force_z_L": env.contact_forces[robot_index, env.feet_indices[0], 2].item(),
                    "wheel_force_x_R": env.contact_forces[robot_index, env.feet_indices[1], 0].item(),
                    "wheel_force_y_R": env.contact_forces[robot_index, env.feet_indices[1], 1].item(),
                    "wheel_force_z_R": env.contact_forces[robot_index, env.feet_indices[1], 2].item(),
                    "dof_pos_hip_L": env.dof_pos[robot_index, 1].item(),
                    "dof_pos_knee_L": env.dof_pos[robot_index, 2].item(),
                    "dof_pos_hip_R": env.dof_pos[robot_index, 5].item(),
                    "dof_pos_knee_R": env.dof_pos[robot_index, 6].item(),
                    "trigger_mask_L": infos["trigger_mask"][robot_index, 0].item() if "trigger_mask" in infos else 0,
                    "trigger_mask_R": infos["trigger_mask"][robot_index, 1].item() if "trigger_mask" in infos else 0,
                    "ff_action_hip_L": infos["ff_actions"][robot_index, 1].item() if "ff_actions" in infos else 0,
                    "ff_action_knee_L": infos["ff_actions"][robot_index, 2].item() if "ff_actions" in infos else 0,
                    "ff_action_hip_R": infos["ff_actions"][robot_index, 5].item() if "ff_actions" in infos else 0,
                    "ff_action_knee_R": infos["ff_actions"][robot_index, 6].item() if "ff_actions" in infos else 0,
                }
            )
            # print(torch.sum(env.power[robot_index, :]).item())
            if est != None:
                logger.log_states(
                    {
                        "est_lin_vel_x": est[robot_index, 0].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_y": est[robot_index, 1].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_z": est[robot_index, 2].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                    }
                )
        elif i == stop_state_log:
            # logger.plot_states()
            pass

        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == "__main__":
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    RECORD_VIDEO = True
    MOVE_CAMERA = True
    args = get_args()
    play(args)
