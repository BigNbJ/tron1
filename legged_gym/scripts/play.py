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

import numpy as np
import torch
import matplotlib.pyplot as plt
import csv
from datetime import datetime

def sphere2cart(sphere_coords):
    l = sphere_coords[..., 0]
    p = sphere_coords[..., 1]
    y = sphere_coords[..., 2]
    
    pitch_sin = torch.sin(p)
    pitch_cos = torch.cos(p)
    yaw_sin = torch.sin(y)
    yaw_cos = torch.cos(y)
    proj_len = l * pitch_cos
    
    cart = torch.zeros_like(sphere_coords)
    cart[..., 0] = proj_len * yaw_cos
    cart[..., 1] = proj_len * yaw_sin
    cart[..., 2] = l * pitch_sin
    return cart

def plot_custom_states(log, dt):
    # Plot Base Height
    if "base_height" in log:
        time = np.linspace(0, len(log["base_height"])*dt, len(log["base_height"]))
        plt.figure()
        plt.plot(time, log["base_height"], label='Base Height')
        plt.xlabel('Time [s]')
        plt.ylabel('Height [m]')
        plt.title('Base Height')
        plt.legend()
    
    # Plot Torques
    if "dof_torques" in log:
        torques = np.array(log["dof_torques"])
        num_dof = torques.shape[1]
        time = np.linspace(0, len(torques)*dt, len(torques))
        plt.figure()
        for i in range(num_dof):
            plt.plot(time, torques[:, i], label=f'Joint {i}')
        plt.xlabel('Time [s]')
        plt.ylabel('Torque [Nm]')
        plt.title('Joint Torques')
        plt.legend()

    # Plot EE Position
    if "ee_target_x" in log:
        time = np.linspace(0, len(log["ee_target_x"])*dt, len(log["ee_target_x"]))
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        # X
        axs[0].plot(time, log["ee_target_x"], label='Target X')
        if "ee_meas_x" in log: axs[0].plot(time, log["ee_meas_x"], label='Measured X')
        axs[0].set_ylabel('X [m]')
        axs[0].legend()
        axs[0].set_title('EE Position Tracking')
        # Y
        axs[1].plot(time, log["ee_target_y"], label='Target Y')
        if "ee_meas_y" in log: axs[1].plot(time, log["ee_meas_y"], label='Measured Y')
        axs[1].set_ylabel('Y [m]')
        axs[1].legend()
        # Z
        axs[2].plot(time, log["ee_target_z"], label='Target Z')
        if "ee_meas_z" in log: axs[2].plot(time, log["ee_meas_z"], label='Measured Z')
        axs[2].set_ylabel('Z [m]')
        axs[2].set_xlabel('Time [s]')
        axs[2].legend()
        
        plt.tight_layout()
        plt.show()

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.episode_length_s = 30
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)

    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 20
    env_cfg.terrain.terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
    env_cfg.terrain.max_init_terrain_level = 4
    env_cfg.terrain.curriculum = True
    env_cfg.noise.add_noise = True
    env_cfg.noise.noise_level = 0.5
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 3
    env_cfg.domain_rand.randomize_Kp = False
    env_cfg.domain_rand.randomize_Kd = False
    env_cfg.domain_rand.randomize_motor_torque = False
    env_cfg.domain_rand.randomize_default_dof_pos = False
    env_cfg.domain_rand.randomize_action_delay = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # get robot_type
    robot_type = os.getenv("ROBOT_TYPE")
    commands_val = to_torch([0.5, 0.0, 0, 0], device=env.device) if robot_type.startswith("PF")\
        else to_torch([1.0, 0.0, 0.0], device=env.device) if robot_type == "WF_TRON1A" else to_torch([1.5, 0.0, 0.0, 0.0, 0.0])
    
    # Define fixed EE target in Base Frame (Length, Pitch, Yaw)
    # ee_target_sphere = to_torch([0.7, 1.2, -0.5], device=env.device) 
    # ee_target_cart = sphere2cart(ee_target_sphere)
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
    stop_state_log = 500  # number of steps before plotting states
    stop_rew_log = (
        env.max_episode_length + 1
    )  # number of steps before print average episode rewards
    # camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    # camera_vel = np.array([1.0, 1.0, 0.0])
    # camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    est = None
    for i in range(10 * int(env.max_episode_length)):
        est = encoder(obs_history)
        actions = policy(torch.cat((est, obs, commands), dim=-1).detach())

        env.commands[:, :] = commands_val

        # Update EE target to follow a trajectory (Sinusoidal Yaw)
        t = i * env.dt
        traj_freq = 0.5  # Hz
        traj_amp_yaw = 0.8  # rad
        
        # Calculate new target (LPY)
        # Keep Length and Pitch constant, vary Yaw
        new_yaw = traj_amp_yaw * np.sin(2 * np.pi * traj_freq * t)
        new_yaw = 0
        ee_target_sphere = to_torch([0.44, 1.18, new_yaw], device=env.device)
        ee_target_cart = sphere2cart(ee_target_sphere)

        # Override EE goal if environment supports it
        if hasattr(env, 'curr_ee_goal_sphere'):
            env.curr_ee_goal_sphere[:] = ee_target_sphere
            env.ee_goal_sphere[:] = ee_target_sphere
            env.ee_start_sphere[:] = ee_target_sphere
            if hasattr(env, 'curr_ee_goal_cart'):
                env.curr_ee_goal_cart[:] = ee_target_cart
                env.ee_goal_cart[:] = ee_target_cart
            if hasattr(env, 'traj_total_timesteps'):
                env.traj_total_timesteps[:] = 10000.0 # Prevent resampling
            if hasattr(env, 'goal_timer'):
                env.goal_timer[:] = 0.0

        obs, rews, dones, infos, obs_history, commands, _ = env.step(
            actions.detach()
        )
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
            # Calculate EE target in Base frame
            ee_goal_sphere = env.curr_ee_goal_sphere[robot_index, :].unsqueeze(0)
            ee_goal_cart_base = sphere2cart(ee_goal_sphere)
            
            # Calculate EE measured in Base frame
            # 1. Get world EE pos
            ee_meas_world = env.ee_pos[robot_index, :].unsqueeze(0)
            # 2. Subtract base pos -> relative vector in world frame
            rel_pos_world = ee_meas_world - env.root_states[robot_index, :3].unsqueeze(0)
            # 3. Rotate by inverse base quat -> relative vector in base frame
            base_quat = env.base_quat[robot_index, :].unsqueeze(0)
            ee_meas_base = quat_rotate_inverse(base_quat, rel_pos_world)
            
            base_height_val = env.base_height[robot_index].item() if hasattr(env, 'base_height') else env.root_states[robot_index, 2].item()
            
            # Calculate Base Yaw
            quat = env.base_quat[robot_index, :].unsqueeze(0)
            _, _, yaw_val = get_euler_xyz(quat)

            logger.log_states(
                {
                    "base_pos_x": env.root_states[robot_index, 0].item(),
                    "base_pos_y": env.root_states[robot_index, 1].item(),
                    "base_pos_z": env.root_states[robot_index, 2].item(),
                    "base_yaw": yaw_val.item(),
                    "dof_pos_target": actions[robot_index, joint_index].item() * action_scale,
                    "dof_pos": (
                        env.dof_pos[robot_index, joint_index]
                        - env.raw_default_dof_pos[joint_index]
                    ).item(),
                    "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.torques[robot_index, joint_index].item(),
                    "dof_torques": env.torques[robot_index, :].detach().cpu().numpy(),
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
                    "base_height": base_height_val,
                    "ee_target_x": ee_goal_cart_base[0, 0].item(),
                    "ee_target_y": ee_goal_cart_base[0, 1].item(),
                    "ee_target_z": ee_goal_cart_base[0, 2].item(),
                    "ee_meas_x": ee_meas_base[0, 0].item(),
                    "ee_meas_y": ee_meas_base[0, 1].item(),
                    "ee_meas_z": ee_meas_base[0, 2].item(),
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
            logger.plot_states()
            plot_custom_states(logger.state_log, env.dt)

            # Save data for scientific plotting
            log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.task, train_cfg.runner.experiment_name)
            os.makedirs(log_dir, exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            save_path = os.path.join(log_dir, f"play_log_{timestamp}.npz")
            # Convert lists to numpy arrays
            data_to_save = {k: np.array(v) for k, v in logger.state_log.items()}
            np.savez(save_path, **data_to_save)
            print(f"Logged states saved to {save_path}")

            # Save as CSV for Origin
            csv_path = os.path.join(log_dir, f"play_log_{timestamp}.csv")
            
            # Flatten dictionary for CSV (handle array columns)
            flat_data = {}
            row_count = 0
            
            # First pass: determine structure and length
            for key, value in logger.state_log.items():
                if len(value) == 0: continue
                row_count = len(value)
                
                # Check type of first element to decide if flattening is needed
                first_elem = value[0]
                if hasattr(first_elem, '__len__') and not isinstance(first_elem, str):
                    # It's an array/list (e.g. torques, forces)
                    dim = len(first_elem)
                    for j in range(dim):
                        flat_data[f"{key}_{j}"] = [v[j] for v in value]
                else:
                    # Scalar
                    flat_data[key] = value

            # Write to CSV
            if row_count > 0:
                # Add Time column
                flat_data["Time"] = [k * env.dt for k in range(row_count)]
                
                # Sort keys but keep Time first
                keys = sorted([k for k in flat_data.keys() if k != "Time"])
                keys.insert(0, "Time")
                
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(keys) # Header
                    for i in range(row_count):
                        row = [flat_data[k][i] for k in keys]
                        writer.writerow(row)
                print(f"Logged states saved to {csv_path} (Origin compatible)")

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
    MOVE_CAMERA = True
    args = get_args()
    play(args)
