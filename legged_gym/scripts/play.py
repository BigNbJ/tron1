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
import csv

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

        # Heights
        self.base_height = deque(maxlen=max_len)
        self.measured_heights = deque(maxlen=max_len)
        
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
        
        # Actions (Fused)
        self.fused_hip_L = deque(maxlen=max_len)
        self.fused_hip_R = deque(maxlen=max_len)
        self.fused_knee_L = deque(maxlen=max_len)
        self.fused_knee_R = deque(maxlen=max_len)
        
        # Joint Positions
        self.hip_pos_L = deque(maxlen=max_len)
        self.hip_pos_R = deque(maxlen=max_len)
        self.knee_pos_L = deque(maxlen=max_len)
        self.knee_pos_R = deque(maxlen=max_len)
        
        # Initialize time
        self.current_time = 0.0
        
        # Setup plot: 2 Cols x 4 Rows
        plt.ion()
        self.fig, self.axs = plt.subplots(4, 2, sharex=True, figsize=(14, 12))
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
        
        # Row 2: Heights
        self.ax_fz = self.axs[2, 0]
        self.line_base_h, = self.ax_fz.plot([], [], label='Base Height', color='b')
        self.line_measured_h, = self.ax_fz.plot([], [], label='Measured Height', color='r')
        self.ax_fz.set_ylabel('Height [m]')
        self.ax_fz.legend(loc='upper right', fontsize='small')
        self.ax_fz.grid(True)
        
        # Row 3: Trigger
        self.ax_trig = self.axs[3, 0]
        self.line_trig_L, = self.ax_trig.plot([], [], label='Trigger L', color='b', alpha=0.7)
        self.line_trig_R, = self.ax_trig.plot([], [], label='Trigger R', color='r', alpha=0.7)
        self.ax_trig.set_ylabel('Trigger')
        self.ax_trig.set_xlabel('Time [s]')
        self.ax_trig.set_ylim(-0.1, 1.1)
        self.ax_trig.legend(loc='upper right', fontsize='small')
        self.ax_trig.grid(True)
        
        # --- Column 2: Actions ---
        self.ax_hip_L = self.axs[0, 1]
        self.line_pol_hip_L, = self.ax_hip_L.plot([], [], label='Pol Hip L', color='#1f77b4', linestyle='-')
        self.line_ff_hip_L, = self.ax_hip_L.plot([], [], label='FF Hip L', color='#ff7f0e', linestyle='--')
        self.line_fused_hip_L, = self.ax_hip_L.plot([], [], label='Fused Hip L', color='#9467bd', linestyle='-.')
        self.line_hip_pos_L, = self.ax_hip_L.plot([], [], label='Hip Pos L', color='#2ca02c', linestyle=':')
        self.ax_hip_L.set_ylabel('Hip L [rad]')
        self.ax_hip_L.legend(loc='upper right', fontsize='small')
        self.ax_hip_L.grid(True)

        self.ax_hip_R = self.axs[1, 1]
        self.line_pol_hip_R, = self.ax_hip_R.plot([], [], label='Pol Hip R', color='#1f77b4', linestyle='-')
        self.line_ff_hip_R, = self.ax_hip_R.plot([], [], label='FF Hip R', color='#ff7f0e', linestyle='--')
        self.line_fused_hip_R, = self.ax_hip_R.plot([], [], label='Fused Hip R', color='#9467bd', linestyle='-.')
        self.line_hip_pos_R, = self.ax_hip_R.plot([], [], label='Hip Pos R', color='#2ca02c', linestyle=':')
        self.ax_hip_R.set_ylabel('Hip R [rad]')
        self.ax_hip_R.legend(loc='upper right', fontsize='small')
        self.ax_hip_R.grid(True)

        self.ax_knee_L = self.axs[2, 1]
        self.line_pol_knee_L, = self.ax_knee_L.plot([], [], label='Pol Knee L', color='#1f77b4', linestyle='-')
        self.line_ff_knee_L, = self.ax_knee_L.plot([], [], label='FF Knee L', color='#ff7f0e', linestyle='--')
        self.line_fused_knee_L, = self.ax_knee_L.plot([], [], label='Fused Knee L', color='#9467bd', linestyle='-.')
        self.line_knee_pos_L, = self.ax_knee_L.plot([], [], label='Knee Pos L', color='#2ca02c', linestyle=':')
        self.ax_knee_L.set_ylabel('Knee L [rad]')
        self.ax_knee_L.legend(loc='upper right', fontsize='small')
        self.ax_knee_L.grid(True)

        self.ax_knee_R = self.axs[3, 1]
        self.line_pol_knee_R, = self.ax_knee_R.plot([], [], label='Pol Knee R', color='#1f77b4', linestyle='-')
        self.line_ff_knee_R, = self.ax_knee_R.plot([], [], label='FF Knee R', color='#ff7f0e', linestyle='--')
        self.line_fused_knee_R, = self.ax_knee_R.plot([], [], label='Fused Knee R', color='#9467bd', linestyle='-.')
        self.line_knee_pos_R, = self.ax_knee_R.plot([], [], label='Knee Pos R', color='#2ca02c', linestyle=':')
        self.ax_knee_R.set_ylabel('Knee R [rad]')
        self.ax_knee_R.set_xlabel('Time [s]')
        self.ax_knee_R.legend(loc='upper right', fontsize='small')
        self.ax_knee_R.grid(True)
        
    def update(self, 
               force_x_l, force_x_r, force_y_l, force_y_r,
               base_height, measured_height,
               trigger_l, trigger_r, 
               pol_hip_l, pol_hip_r, pol_knee_l, pol_knee_r,
               ff_hip_l, ff_hip_r, ff_knee_l, ff_knee_r,
               fused_hip_l, fused_hip_r, fused_knee_l, fused_knee_r,
               hip_pos_l, hip_pos_r, knee_pos_l, knee_pos_r):
        self.current_time += self.dt
        
        self.time_buf.append(self.current_time)
        
        # Forces
        self.force_x_L.append(force_x_l)
        self.force_x_R.append(force_x_r)
        self.force_y_L.append(force_y_l)
        self.force_y_R.append(force_y_r)

        # Heights
        self.base_height.append(base_height)
        self.measured_heights.append(measured_height)
        
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

        self.fused_hip_L.append(fused_hip_l * s)
        self.fused_hip_R.append(fused_hip_r * s)
        self.fused_knee_L.append(fused_knee_l * s)
        self.fused_knee_R.append(fused_knee_r * s)

        self.hip_pos_L.append(hip_pos_l)
        self.hip_pos_R.append(hip_pos_r)
        self.knee_pos_L.append(knee_pos_l)
        self.knee_pos_R.append(knee_pos_r)
        
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

        # Heights
        self.line_base_h.set_data(t, list(self.base_height))
        self.line_measured_h.set_data(t, list(self.measured_heights))
        
        # Trigger
        self.line_trig_L.set_data(t, list(self.trigger_L))
        self.line_trig_R.set_data(t, list(self.trigger_R))
        
        # Actions
        self.line_pol_hip_L.set_data(t, list(self.pol_hip_L))
        self.line_pol_hip_R.set_data(t, list(self.pol_hip_R))
        self.line_ff_hip_L.set_data(t, list(self.ff_hip_L))
        self.line_ff_hip_R.set_data(t, list(self.ff_hip_R))
        self.line_fused_hip_L.set_data(t, list(self.fused_hip_L))
        self.line_fused_hip_R.set_data(t, list(self.fused_hip_R))
        self.line_hip_pos_L.set_data(t, list(self.hip_pos_L))
        self.line_hip_pos_R.set_data(t, list(self.hip_pos_R))
        
        self.line_pol_knee_L.set_data(t, list(self.pol_knee_L))
        self.line_pol_knee_R.set_data(t, list(self.pol_knee_R))
        self.line_ff_knee_L.set_data(t, list(self.ff_knee_L))
        self.line_ff_knee_R.set_data(t, list(self.ff_knee_R))
        self.line_fused_knee_L.set_data(t, list(self.fused_knee_L))
        self.line_fused_knee_R.set_data(t, list(self.fused_knee_R))
        self.line_knee_pos_L.set_data(t, list(self.knee_pos_L))
        self.line_knee_pos_R.set_data(t, list(self.knee_pos_R))
        
        # Rescale axes
        if len(t) > 0:
            self.ax_trig.set_xlim(min(t), max(t) + self.dt)
            
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
                
            # Height
            all_h = list(self.base_height) + list(self.measured_heights)
            if all_h:
                min_v, max_v = min(all_h), max(all_h)
                span = max(1.0, max_v - min_v)
                self.ax_fz.set_ylim(min_v - 0.1*span, max_v + 0.1*span)

            # Auto-scale Actions
            all_hip_l = list(self.pol_hip_L) + list(self.ff_hip_L) + list(self.fused_hip_L) + list(self.hip_pos_L)
            if all_hip_l:
                min_h, max_h = min(all_hip_l), max(all_hip_l)
                span = max(0.1, max_h - min_h)
                self.ax_hip_L.set_ylim(min_h - 0.1*span, max_h + 0.1*span)

            all_hip_r = list(self.pol_hip_R) + list(self.ff_hip_R) + list(self.fused_hip_R) + list(self.hip_pos_R)
            if all_hip_r:
                min_h, max_h = min(all_hip_r), max(all_hip_r)
                span = max(0.1, max_h - min_h)
                self.ax_hip_R.set_ylim(min_h - 0.1*span, max_h + 0.1*span)
                
            all_knee_l = list(self.pol_knee_L) + list(self.ff_knee_L) + list(self.fused_knee_L) + list(self.knee_pos_L)
            if all_knee_l:
                min_k, max_k = min(all_knee_l), max(all_knee_l)
                span = max(0.1, max_k - min_k)
                self.ax_knee_L.set_ylim(min_k - 0.1*span, max_k + 0.1*span)

            all_knee_r = list(self.pol_knee_R) + list(self.ff_knee_R) + list(self.fused_knee_R) + list(self.knee_pos_R)
            if all_knee_r:
                min_k, max_k = min(all_knee_r), max(all_knee_r)
                span = max(0.1, max_k - min_k)
                self.ax_knee_R.set_ylim(min_k - 0.1*span, max_k + 0.1*span)

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

    env_cfg.terrain.mesh_type = "plane" if env_cfg.terrain.mesh_type == "plane" else ("heightfield" if args.headless else "trimesh")

    if RECORD_VIDEO:
        env_cfg.env.enable_camera_sensors = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env = RecordVideoWrapper(env)
    # get robot_type
    robot_type = os.getenv("ROBOT_TYPE", "")
    num_commands = env.cfg.commands.num_commands
    if env.cfg.commands.heading_command:
        num_commands += 1
    commands_val = torch.zeros(num_commands, device=env.device)
    
    if robot_type.startswith("PF"):
        commands_val[0] = 0.5
    elif robot_type == "WF_TRON1A":
        commands_val[0] = 0.2
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
    robot_index = 44  # which robot is used for logging
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
    csv_rows = []
    csv_fieldnames = [
        "frame",
        "force_x_l",
        "force_x_r",
        "force_y_l",
        "force_y_r",
        "base_height",
        "measured_height",
        "trigger_l",
        "trigger_r",
        "pol_hip_l",
        "pol_hip_r",
        "pol_knee_l",
        "pol_knee_r",
        "ff_hip_l",
        "ff_hip_r",
        "ff_knee_l",
        "ff_knee_r",
        "fused_hip_l",
        "fused_hip_r",
        "fused_knee_l",
        "fused_knee_r",
        "hip_pos_l",
        "hip_pos_r",
        "knee_pos_l",
        "knee_pos_r",
    ]
    
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
            force_x_l = forces[0, 0].item()
            force_x_r = forces[1, 0].item()
            force_y_l = forces[0, 1].item()
            force_y_r = forces[1, 1].item()
            base_height = env.base_height[robot_index].item()
            measured_height = torch.mean(env.measured_heights[robot_index]).item()
            trigger_l = infos["trigger_mask"][robot_index, 0].item()
            trigger_r = infos["trigger_mask"][robot_index, 1].item()
            pol_hip_l = pol_actions[1].item()
            pol_hip_r = pol_actions[5].item()
            pol_knee_l = pol_actions[2].item()
            pol_knee_r = pol_actions[6].item()
            ff_hip_l = infos["ff_actions"][robot_index, 1].item()
            ff_hip_r = infos["ff_actions"][robot_index, 5].item()
            ff_knee_l = infos["ff_actions"][robot_index, 2].item()
            ff_knee_r = infos["ff_actions"][robot_index, 6].item()
            fused_hip_l = env.actions[robot_index, 1].item()
            fused_hip_r = env.actions[robot_index, 5].item()
            fused_knee_l = env.actions[robot_index, 2].item()
            fused_knee_r = env.actions[robot_index, 6].item()
            hip_pos_l = (env.dof_pos[robot_index, 1] - env.raw_default_dof_pos[1]).item()
            hip_pos_r = (env.dof_pos[robot_index, 5] - env.raw_default_dof_pos[5]).item()
            knee_pos_l = (env.dof_pos[robot_index, 2] - env.raw_default_dof_pos[2]).item()
            knee_pos_r = (env.dof_pos[robot_index, 6] - env.raw_default_dof_pos[6]).item()
            
            live_plotter.update(
                # Forces (L/R)
                force_x_l=force_x_l, force_x_r=force_x_r,
                force_y_l=force_y_l, force_y_r=force_y_r,
                
                # Heights
                base_height=base_height,
                measured_height=measured_height,
                
                # Trigger
                trigger_l=trigger_l,
                trigger_r=trigger_r,
                
                # Policy Actions
                pol_hip_l=pol_hip_l, pol_hip_r=pol_hip_r,
                pol_knee_l=pol_knee_l, pol_knee_r=pol_knee_r,
                
                # FF Actions
                ff_hip_l=ff_hip_l,
                ff_hip_r=ff_hip_r,
                ff_knee_l=ff_knee_l,
                ff_knee_r=ff_knee_r,
                
                fused_hip_l=fused_hip_l,
                fused_hip_r=fused_hip_r,
                fused_knee_l=fused_knee_l,
                fused_knee_r=fused_knee_r,
                
                # Joint Positions
                hip_pos_l=hip_pos_l,
                hip_pos_r=hip_pos_r,
                knee_pos_l=knee_pos_l,
                knee_pos_r=knee_pos_r,
            )
            if len(csv_rows) < 500:
                csv_rows.append(
                    {
                        "frame": i,
                        "force_x_l": force_x_l,
                        "force_x_r": force_x_r,
                        "force_y_l": force_y_l,
                        "force_y_r": force_y_r,
                        "base_height": base_height,
                        "measured_height": measured_height,
                        "trigger_l": trigger_l,
                        "trigger_r": trigger_r,
                        "pol_hip_l": pol_hip_l,
                        "pol_hip_r": pol_hip_r,
                        "pol_knee_l": pol_knee_l,
                        "pol_knee_r": pol_knee_r,
                        "ff_hip_l": ff_hip_l,
                        "ff_hip_r": ff_hip_r,
                        "ff_knee_l": ff_knee_l,
                        "ff_knee_r": ff_knee_r,
                        "fused_hip_l": fused_hip_l,
                        "fused_hip_r": fused_hip_r,
                        "fused_knee_l": fused_knee_l,
                        "fused_knee_r": fused_knee_r,
                        "hip_pos_l": hip_pos_l,
                        "hip_pos_r": hip_pos_r,
                        "knee_pos_l": knee_pos_l,
                        "knee_pos_r": knee_pos_r,
                    }
                )

        if RECORD_VIDEO and i == 500:
            video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
            os.makedirs(video_dir, exist_ok=True)
            env.end_and_save_recording_video(video_path=video_dir, filename=f"video_{args.task}.mp4")
            print(f"Video saved to {video_dir}/video_{args.task}.mp4")
            
            # Save the live plotter figure with complete 500 frames
            plot_path = os.path.join(video_dir, f"video_{args.task}_plot.png")
            live_plotter.fig.savefig(plot_path)
            print(f"Data plot saved to {plot_path}")
            
            csv_path = os.path.join(video_dir, f"video_{args.task}_plot.csv")
            with open(csv_path, "w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"Data csv saved to {csv_path}")

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
