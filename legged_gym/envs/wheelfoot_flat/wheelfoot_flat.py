import math
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import random

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float,
)
from .wheelfoot_flat_config import BipedCfgWF
from legged_gym.utils.helpers import class_to_dict

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

class BipedWF(BaseTask):
    def __init__(
        self, cfg: BipedCfgWF, sim_params, physics_engine, sim_device, headless
    ):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None

        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.pi = torch.acos(torch.zeros(1, device=self.device)) * 2
        self.group_idx = torch.arange(0, self.cfg.env.num_envs)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

        print("关节名称顺序:", self.dof_names)
        print("关节力矩限幅:", self.torque_limits)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum:
            # self.update_command_curriculum(time_out_env_ids) # This logic is usually outside per-env reset
            pass

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self._resample_ee_goal(env_ids, is_init=True)
        # self._resample_gaits(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_base_position[env_ids] = self.base_position[env_ids]
        self.last_foot_positions[env_ids] = self.foot_positions[env_ids]
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.envs_steps_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history[env_ids] = 0
        self.goal_timer[env_ids] = 0.
        
        obs_buf, _ = self.compute_group_observations()
        self.obs_history[env_ids] = obs_buf[env_ids].repeat(1, self.obs_history_length)
        self.gait_indices[env_ids] = 0
        self.fail_buf[env_ids] = 0
        self.action_fifo[env_ids] = 0
        self.dof_pos_int[env_ids] = 0
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["group_terrain_level"] = torch.mean(
                self.terrain_levels[self.group_idx].float()
            )
            self.extras["episode"]["group_terrain_level_stair_up"] = torch.mean(
                self.terrain_levels[self.stair_up_idx].float()
            )
        if self.cfg.terrain.curriculum and self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.smooth_slope_idx, 1].float()
            )
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf | self.edge_reset_buf

    def step(self, actions):
        self._action_clip(actions)
        # step physics and render each frame
        self.render()
        self.pre_physics_step()
        for _ in range(self.cfg.control.decimation):
            self.action_fifo = torch.cat(
                (self.actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1
            )
            self.envs_steps_buf += 1
            self.torques = self._compute_torques(
                self.action_fifo[torch.arange(self.num_envs), self.action_delay_idx, :]
            ).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.compute_dof_vel()
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        return (
            self.obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.obs_history,
            self.commands[:, :3] * self.commands_scale,
            self.critic_obs_buf # make sure critic_obs update in every for loop
        )
        
    def _action_clip(self, actions):
        self.actions = actions
        
    def _compute_torques(self, actions):
        pos_action = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        vel_action = torch.zeros(self.num_envs, self.num_dof, device=self.device)

        arm_actions = actions[:, :6]
        base_actions = actions[:, 6:]

        pos_action[:, self.arm_dof_indices] = arm_actions * self.cfg.control.action_scale_pos
        vel_action[:, self.arm_dof_indices] = arm_actions * self.cfg.control.action_scale_vel

        base_leg_pos = torch.cat((base_actions[:, 0:3], base_actions[:, 4:7]), dim=1)
        pos_action[:, self.base_leg_pos_dof_indices] = base_leg_pos * self.cfg.control.action_scale_pos

        base_wheel_vel = torch.stack((base_actions[:, 3], base_actions[:, 7]), dim=1)
        vel_action[:, self.base_wheel_dof_indices] = base_wheel_vel * self.cfg.control.action_scale_vel_wheel

        # pd controller
        torques = self.p_gains * (pos_action + self.default_dof_pos - self.dof_pos) + self.d_gains * (vel_action - self.dof_vel)
        torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
        return torques

    def post_physics_step(self):
        super().post_physics_step()
        self.wheel_lin_vel = self.foot_velocities[:, 0, :] + self.foot_velocities[:, 1, :]

        if self.viewer and self.enable_viewer_sync:
            self.gym.clear_lines(self.viewer)
            self._draw_ee_goal()

    def _draw_ee_goal(self):
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 8, 8, None, color=(1, 0, 0))
        final_goal_geom = gymutil.WireframeSphereGeometry(0.05, 8, 8, None, color=(0, 1, 0)) # Green for final goal
        start_goal_geom = gymutil.WireframeSphereGeometry(0.04, 8, 8, None, color=(0, 0, 1)) # Blue for start point
        
        # Interpolate between start and goal for visualization trail
        # ee_start_sphere: (N, 3), ee_goal_sphere: (N, 3)
        # t: (1, 1, 1, 10)
        # unsqueeze for broadcasting: (N, 3, 1)
        t = torch.linspace(0, 1, 10, device=self.device).reshape(1, 1, 10)
        ee_target_all_sphere = self.ee_start_sphere.unsqueeze(2) + (self.ee_goal_sphere.unsqueeze(2) - self.ee_start_sphere.unsqueeze(2)) * t
        # ee_target_all_sphere shape: (N, 3, 10)

        ee_target_all_cart_world = torch.zeros(self.num_envs, 3, 10, device=self.device)
        
        for i in range(10):
            # Get i-th point for all envs
            p_sphere = ee_target_all_sphere[:, :, i] # (N, 3)
            p_cart_base = sphere2cart(p_sphere) # (N, 3)
            
            # Rotate to world (using base_quat as goal is in base frame)
            p_cart_world = quat_apply(self.base_quat, p_cart_base)
            
            # Add base position
            p_cart_world += self.root_states[:, :3]
            
            ee_target_all_cart_world[:, :, i] = p_cart_world
            
        # Draw for all environments
        draw_env_ids = range(self.num_envs)
        
        for i in draw_env_ids:
            # Draw trajectory trail
            for j in range(10):
                 pose = gymapi.Transform(gymapi.Vec3(ee_target_all_cart_world[i, 0, j], ee_target_all_cart_world[i, 1, j], ee_target_all_cart_world[i, 2, j]), r=None)
                 gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)
            
            # Draw start point separately
            p_start_sphere = self.ee_start_sphere[i]
            p_start_cart_base = sphere2cart(p_start_sphere)
            p_start_cart_world = quat_apply(self.base_quat[i], p_start_cart_base) + self.root_states[i, :3]
            pose_start = gymapi.Transform(gymapi.Vec3(p_start_cart_world[0], p_start_cart_world[1], p_start_cart_world[2]), r=None)
            gymutil.draw_lines(start_goal_geom, self.gym, self.viewer, self.envs[i], pose_start)

            # Draw final goal separately (last point in trajectory)
            # Or use self.ee_goal_sphere directly
            p_goal_sphere = self.ee_goal_sphere[i]
            p_goal_cart_base = sphere2cart(p_goal_sphere)
            p_goal_cart_world = quat_apply(self.base_quat[i], p_goal_cart_base) + self.root_states[i, :3]
            
            pose_final = gymapi.Transform(gymapi.Vec3(p_goal_cart_world[0], p_goal_cart_world[1], p_goal_cart_world[2]), r=None)
            gymutil.draw_lines(final_goal_geom, self.gym, self.viewer, self.envs[i], pose_final)

    def compute_group_observations(self):
        # note that observation noise need to modified accordingly !!!
        dof_pos = (self.dof_pos - self.default_dof_pos)[:, self.base_leg_pos_dof_indices]
        
        arm_dof_pos = (self.dof_pos - self.default_dof_pos)[:, self.arm_dof_indices]
        arm_dof_vel = self.dof_vel[:, self.arm_dof_indices] * self.obs_scales.dof_vel
        
        # Calculate EE position error (in base frame, spherical)
        # 1. Get EE pos in base frame (Cartesian)
        rel_pos = self.ee_pos - self.root_states[:, :3]
        rel_pos = quat_rotate_inverse(self.base_quat, rel_pos)
        
        # 2. Convert to spherical
        l = torch.norm(rel_pos, dim=1)
        p = torch.asin(torch.clamp(rel_pos[:, 2] / (l + 1e-6), -1.0, 1.0))
        y = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])
        
        current_sphere = torch.stack([l, p, y], dim=1)
        
        # 3. Compute error
        ee_pos_error = (self.curr_ee_goal_sphere - current_sphere)

        obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel, # 3
                self.projected_gravity, # 3
                dof_pos * self.obs_scales.dof_pos, # 6
                self.dof_vel[:, self.base_dof_indices] * self.obs_scales.dof_vel,
                self.actions[:, 6:], # 8 (Base actions are last 8)
                # Arm obs
                self.curr_ee_goal_sphere, # 3
                ee_pos_error, # 3
                self.ee_goal_delta_orn_euler, # 3
                arm_dof_pos, # 6
                arm_dof_vel, # 6
                self.actions[:, :6] # 6 (Arm actions are first 6)
            ),
            dim=-1,
        )
        critic_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel, self.obs_buf), dim=-1)
        return obs_buf, critic_obs_buf
    
    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (
            (
                self.episode_length_buf
                % int(self.cfg.commands.resampling_time / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(env_ids)
        
        # Update curriculum
        if self.cfg.commands.curriculum:
            self.update_command_curriculum(env_ids)
            
        # Update EE Goal
        self.update_curr_ee_goal()

        # self._resample_gaits(env_ids)
        # self._step_contact_targets()

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = 0.1 * wrap_to_pi(self.commands[:, 3] - heading)

        if self.cfg.terrain.measure_heights or self.cfg.terrain.critic_measure_heights:
            self.measured_heights = self._get_heights()

        self.base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        # Helper to safely get float
        def get_range_val(ranges, idx):
            val = ranges[idx]
            if hasattr(val, 'item'):
                return val.item()
            return val

        self.commands[env_ids, 0] = torch_rand_float(
            float(get_range_val(self.lin_vel_x_ranges, 0)), float(get_range_val(self.lin_vel_x_ranges, 1)), 
            (len(env_ids), 1), device=self.device).squeeze(1)
        
        self.commands[env_ids, 1] = torch_rand_float(
            float(get_range_val(self.lin_vel_y_ranges, 0)), float(get_range_val(self.lin_vel_y_ranges, 1)), 
            (len(env_ids), 1), device=self.device).squeeze(1)
            
        self.commands[env_ids, 2] = torch_rand_float(
            float(get_range_val(self.ang_vel_yaw_ranges, 0)), float(get_range_val(self.ang_vel_yaw_ranges, 1)), 
            (len(env_ids), 1), device=self.device).squeeze(1)

        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                float(get_range_val(self.heading_ranges, 0)),
                float(get_range_val(self.heading_ranges, 1)),
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        #set 50% of resample to go straight
        resample_nums = len(env_ids)
        env_list = list(range(resample_nums))
        half_env_list = random.sample(env_list, resample_nums // 2)
        # forward = quat_apply(self.base_quat[env_ids[half_env_list]], \
        #                      self.forward_vec[env_ids[half_env_list]])
        # heading = torch.atan2(forward[:,1], forward[:,0])
        # self.commands[env_ids[half_env_list], 3] = heading
        
        # set 20% of the rest 50% to be stand still
        rest_env_list = list(set(env_list) - set(half_env_list))
        zero_cmd_env_idx_ = random.sample(rest_env_list, resample_nums // 2 // 5)

        self.commands[env_ids[zero_cmd_env_idx_], 0] = 0.0
        self.commands[env_ids[zero_cmd_env_idx_], 1] = 0.0
        self.commands[env_ids[zero_cmd_env_idx_], 2] = 0.0
        #use heading
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat[env_ids[zero_cmd_env_idx_]], \
                                 self.forward_vec[env_ids[zero_cmd_env_idx_]])
            heading = torch.atan2(forward[:,1], forward[:,0])
            self.commands[env_ids[zero_cmd_env_idx_], 3] = heading
            
    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = (
            noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        )
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:12] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[12:20] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[20:] = 0.0  # previous actions
        
        # Arm obs
        # 37:43 arm_dof_pos
        noise_vec[37:43] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        # 43:49 arm_dof_vel
        noise_vec[43:49] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        
        return noise_vec

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.arm_reward_scales = class_to_dict(self.cfg.rewards.arm_scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges) # Ensure this is available

        # Curriculum commands
        self.lin_vel_x_schedule = cfg.commands.lin_vel_x_schedule
        self.ang_vel_yaw_schedule = cfg.commands.ang_vel_yaw_schedule
        self.tracking_ang_vel_yaw_schedule = cfg.commands.tracking_ang_vel_yaw_schedule
        
        self.init_lin_vel_x_ranges = cfg.commands.ranges.init_lin_vel_x
        self.final_lin_vel_x_ranges = cfg.commands.ranges.final_lin_vel_x
        self.init_ang_vel_yaw_ranges = cfg.commands.ranges.init_ang_vel_yaw
        self.final_ang_vel_yaw_ranges = cfg.commands.ranges.final_ang_vel_yaw
        self.final_tracking_ang_vel_yaw_exp = cfg.commands.ranges.final_tracking_ang_vel_yaw_exp
        
        # EE Goal
        self.goal_ee_l_schedule = cfg.goal_ee.l_schedule
        self.goal_ee_p_schedule = cfg.goal_ee.p_schedule
        self.goal_ee_y_schedule = cfg.goal_ee.y_schedule
        self.tracking_ee_reward_schedule = cfg.goal_ee.tracking_ee_reward_schedule
        
        self.init_goal_ee_l_ranges = cfg.goal_ee.ranges.init_pos_l
        self.final_goal_ee_l_ranges = cfg.goal_ee.ranges.final_pos_l
        self.init_goal_ee_p_ranges = cfg.goal_ee.ranges.init_pos_p
        self.final_goal_ee_p_ranges = cfg.goal_ee.ranges.final_pos_p
        self.init_goal_ee_y_ranges = cfg.goal_ee.ranges.init_pos_y
        self.final_goal_ee_y_ranges = cfg.goal_ee.ranges.final_pos_y
        self.final_tracking_ee_reward = cfg.goal_ee.ranges.final_tracking_ee_reward

    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        # Add arm reward functions
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.arm_reward_scales.keys()):
            scale = self.arm_reward_scales[key]
            print(f"[DEBUG] Scaling arm reward {key} by {self.dt}")
            if scale==0:
                self.arm_reward_scales.pop(key) 
            else:
                print(f"[DEBUG] Scaling arm reward {key} by {self.dt}")
                self.arm_reward_scales[key] *= self.dt
        # prepare list of functions
        self.arm_reward_functions = []
        self.arm_reward_names = []
        for name, scale in self.arm_reward_scales.items():
            if name=="termination":
                continue
            self.arm_reward_names.append(name)
            name = '_reward_' + name
            self.arm_reward_functions.append(getattr(self, name))

        # reward episode sums - merge existing keys with new arm keys
        # The base class already initialized episode_sums with reward_scales keys.
        # We need to add arm keys.
        for name in self.arm_reward_scales.keys():
            self.episode_sums[name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def compute_reward(self):
        super().compute_reward()
        # Compute arm rewards and add to rew_buf
        arm_rew_buf = torch.zeros_like(self.rew_buf)
        for i in range(len(self.arm_reward_functions)):
            name = self.arm_reward_names[i]
            rew = self.arm_reward_functions[i]() * self.arm_reward_scales[name]
            rew = torch.clip(rew, -self.cfg.rewards.clip_single_reward, self.cfg.rewards.clip_single_reward)
            arm_rew_buf += rew
            self.episode_sums[name] += rew
        
        if self.cfg.rewards.only_positive_rewards:
            arm_rew_buf[:] = torch.clip(arm_rew_buf[:], min=0.)
            
        # add termination reward after clipping if needed
        if "termination" in self.arm_reward_scales:
            rew = self._reward_termination() * self.arm_reward_scales["termination"]
            arm_rew_buf += rew
            self.episode_sums["termination"] += rew
            
        self.rew_buf += arm_rew_buf
        
        self.rew_buf[:] = torch.clip(self.rew_buf[:], -self.cfg.rewards.clip_reward, self.cfg.rewards.clip_reward)

    def _init_buffers(self):
        super()._init_buffers()
        
        # Initialize curriculum and EE variables
        self.update_counter = 0
        self.sphere_error_scale = torch.tensor(self.cfg.goal_ee.sphere_error_scale, device=self.device)
        self.orn_error_scale = torch.tensor(self.cfg.goal_ee.orn_error_scale, device=self.device)
        self.traj_timesteps = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.traj_total_timesteps = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        self.init_lin_vel_x_ranges = np.array(self.init_lin_vel_x_ranges)
        self.final_lin_vel_x_ranges = np.array(self.final_lin_vel_x_ranges)
        self.init_ang_vel_yaw_ranges = np.array(self.init_ang_vel_yaw_ranges)
        self.final_ang_vel_yaw_ranges = np.array(self.final_ang_vel_yaw_ranges)
        
        self.init_goal_ee_l_ranges = np.array(self.init_goal_ee_l_ranges)
        self.final_goal_ee_l_ranges = np.array(self.final_goal_ee_l_ranges)
        self.init_goal_ee_p_ranges = np.array(self.init_goal_ee_p_ranges)
        self.final_goal_ee_p_ranges = np.array(self.final_goal_ee_p_ranges)
        self.init_goal_ee_y_ranges = np.array(self.init_goal_ee_y_ranges)
        self.final_goal_ee_y_ranges = np.array(self.final_goal_ee_y_ranges)

        self.lin_vel_x_ranges = self.init_lin_vel_x_ranges
        self.ang_vel_yaw_ranges = self.init_ang_vel_yaw_ranges
        self.goal_ee_l_ranges = self.init_goal_ee_l_ranges
        self.goal_ee_p_ranges = self.init_goal_ee_p_ranges
        self.goal_ee_y_ranges = self.init_goal_ee_y_ranges
        
        self.lin_vel_y_ranges = np.array(self.cfg.commands.ranges.lin_vel_y)
        self.heading_ranges = np.array(self.cfg.commands.ranges.heading)

        self.wheel_lin_vel = torch.zeros_like(self.foot_velocities)
        self.wheel_ang_vel = torch.zeros_like(self.base_ang_vel)
        
        # Override self.torques to match num_dof (14) instead of num_actions (8)
        self.torques = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.arm_dof_names = ["J1", "J2", "J3", "J4", "J5", "J6"]
        self.base_dof_names = [
            "abad_L_Joint",
            "hip_L_Joint",
            "knee_L_Joint",
            "wheel_L_Joint",
            "abad_R_Joint",
            "hip_R_Joint",
            "knee_R_Joint",
            "wheel_R_Joint",
        ]
        self.base_leg_pos_dof_names = [
            "abad_L_Joint",
            "hip_L_Joint",
            "knee_L_Joint",
            "abad_R_Joint",
            "hip_R_Joint",
            "knee_R_Joint",
        ]
        self.base_wheel_dof_names = ["wheel_L_Joint", "wheel_R_Joint"]

        self.arm_dof_indices = torch.tensor(
            [self.dof_names.index(name) for name in self.arm_dof_names],
            device=self.device,
            dtype=torch.long,
        )
        self.base_dof_indices = torch.tensor(
            [self.dof_names.index(name) for name in self.base_dof_names],
            device=self.device,
            dtype=torch.long,
        )
        self.base_leg_pos_dof_indices = torch.tensor(
            [self.dof_names.index(name) for name in self.base_leg_pos_dof_names],
            device=self.device,
            dtype=torch.long,
        )
        self.base_wheel_dof_indices = torch.tensor(
            [self.dof_names.index(name) for name in self.base_wheel_dof_names],
            device=self.device,
            dtype=torch.long,
        )

        # Ensure p_gains and d_gains are set (if not set by base class)
        for i, dof_name in enumerate(self.dof_names):
            # defaults
            self.p_gains[:, i] = 0.0
            self.d_gains[:, i] = 0.0
            # set from config
            if dof_name in self.cfg.control.stiffness:
                self.p_gains[:, i] = self.cfg.control.stiffness[dof_name]
            if dof_name in self.cfg.control.damping:
                self.d_gains[:, i] = self.cfg.control.damping[dof_name]
        arm_override = torch.tensor([18.0, 18.0, 18.0, 8.0, 8.0, 8.0], device=self.device)
        self.torque_limits[self.arm_dof_indices] = torch.max(self.torque_limits[self.arm_dof_indices], arm_override)
        
        # Initialize EE buffers
        # Assuming the last link is the end-effector or we need to find it by name
        # For TRON1 with arm, let's assume the gripper or last link index.
        # We need to find the body index for the end-effector.
        # Based on config or URDF, let's try to find a link name containing "J6" or similar if explicit name not provided.
        # However, rigid_body_state gives us all bodies.
        # Let's assume the last body is the EE for now, or search for it.
        # Ideally, we should add `ee_name` to config. For now, let's try to find "J6" or use the last one.
        
        self.rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_state = gymtorch.wrap_tensor(self.rigid_body_state).view(self.num_envs, -1, 13)
        

        # Find EE index
        self.ee_idx = self.num_bodies - 1 # Default to last body
        for i in range(self.num_bodies):
            name = self.gym.get_actor_rigid_body_names(self.envs[0], 0)[i]
            nl = name.lower()
            if ("j6" in nl) or ("link6" in nl):
                self.ee_idx = i
                print(f"End-effector index found: {self.ee_idx} with name: {name}")
                break
        
        self.ee_pos = self.rigid_body_state[:, self.ee_idx, :3]
        self.ee_orn = self.rigid_body_state[:, self.ee_idx, 3:7]
        # self.ee_vel = self.rigid_body_state[:, self.ee_idx, 7:] # If needed

        # Initialize EE Goal Buffers
        self.ee_start_sphere = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.curr_ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        
        self.ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.curr_ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        
        self.ee_goal_delta_orn_euler = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        
        self.goal_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

    def _get_curriculum_value(self, schedule, init_range, final_range, counter):
        return np.clip((counter - schedule[0]) / (schedule[1] - schedule[0]), 0, 1) * (final_range - init_range) + init_range

    def update_command_curriculum(self, env_ids):
        self.update_counter += 1

        self.lin_vel_x_ranges = self._get_curriculum_value(self.lin_vel_x_schedule, self.init_lin_vel_x_ranges, self.final_lin_vel_x_ranges, self.update_counter)
        self.ang_vel_yaw_ranges = self._get_curriculum_value(self.ang_vel_yaw_schedule, self.init_ang_vel_yaw_ranges, self.final_ang_vel_yaw_ranges, self.update_counter)
        
        self.goal_ee_l_ranges = self._get_curriculum_value(self.goal_ee_l_schedule, self.init_goal_ee_l_ranges, self.final_goal_ee_l_ranges, self.update_counter)
        self.goal_ee_p_ranges = self._get_curriculum_value(self.goal_ee_p_schedule, self.init_goal_ee_p_ranges, self.final_goal_ee_p_ranges, self.update_counter)
        self.goal_ee_y_ranges = self._get_curriculum_value(self.goal_ee_y_schedule, self.init_goal_ee_y_ranges, self.final_goal_ee_y_ranges, self.update_counter)
        
        self.arm_reward_scales['tracking_ee_sphere'] = self._get_curriculum_value(self.tracking_ee_reward_schedule, 0, self.final_tracking_ee_reward, self.update_counter) * self.dt
        self.reward_scales['tracking_ang_vel_yaw_exp'] = self._get_curriculum_value(self.tracking_ang_vel_yaw_schedule, 0, self.final_tracking_ang_vel_yaw_exp, self.update_counter) * self.dt

    def _resample_ee_goal(self, env_ids, is_init=False):
        self.ee_start_sphere[env_ids] = self.curr_ee_goal_sphere[env_ids]
        
        # Sample new goal in sphere coordinates
        goal_l = torch_rand_float(self.goal_ee_l_ranges[0], self.goal_ee_l_ranges[1], (len(env_ids), 1), device=self.device).squeeze(1)
        goal_p = torch_rand_float(self.goal_ee_p_ranges[0], self.goal_ee_p_ranges[1], (len(env_ids), 1), device=self.device).squeeze(1)
        goal_y = torch_rand_float(self.goal_ee_y_ranges[0], self.goal_ee_y_ranges[1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        self.ee_goal_sphere[env_ids, 0] = goal_l
        self.ee_goal_sphere[env_ids, 1] = goal_p
        self.ee_goal_sphere[env_ids, 2] = goal_y
        
        # Convert to cartesian for visualization or other uses if needed
        # x = l * cos(p) * cos(y)
        # y = l * cos(p) * sin(y)
        # z = l * sin(p)
        # Note: definition of p (pitch) and y (yaw) depends on coordinate system conventions. 
        # Assuming standard spherical: p from xy plane, y around z axis.
        # Actually in WidowGo1 implementation:
        # pitch_sin = torch.sin(target_ee_pitch)
        # pitch_cos = torch.cos(target_ee_pitch)
        # yaw_sin = torch.sin(target_ee_yaw)
        # yaw_cos = torch.cos(target_ee_yaw)
        # proj_len = target_ee_len * pitch_cos
        # self.target_ee[env_ids, 0] = proj_len * yaw_cos
        # self.target_ee[env_ids, 1] = proj_len * yaw_sin
        # self.target_ee[env_ids, 2] = target_ee_len * pitch_sin
        
        pitch_sin = torch.sin(goal_p)
        pitch_cos = torch.cos(goal_p)
        yaw_sin = torch.sin(goal_y)
        yaw_cos = torch.cos(goal_y)
        proj_len = goal_l * pitch_cos
        
        self.ee_goal_cart[env_ids, 0] = proj_len * yaw_cos
        self.ee_goal_cart[env_ids, 1] = proj_len * yaw_sin
        self.ee_goal_cart[env_ids, 2] = goal_l * pitch_sin
        
        # Sample trajectory time and hold time
        traj_time = torch_rand_float(self.cfg.goal_ee.traj_time[0], self.cfg.goal_ee.traj_time[1], (len(env_ids), 1), device=self.device).squeeze(1)
        hold_time = torch_rand_float(self.cfg.goal_ee.hold_time[0], self.cfg.goal_ee.hold_time[1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        self.traj_timesteps[env_ids] = traj_time
        self.traj_total_timesteps[env_ids] = traj_time + hold_time

        # Reset timers
        self.goal_timer[env_ids] = 0.
        if is_init:
             self.curr_ee_goal_sphere[env_ids] = self.ee_goal_sphere[env_ids]
             self.curr_ee_goal_cart[env_ids] = self.ee_goal_cart[env_ids]
             self.ee_start_sphere[env_ids] = self.ee_goal_sphere[env_ids]
             self.traj_total_timesteps[env_ids] = 0. # Force immediate resample next step? No, keep hold time or init with 0 wait? 
             # Actually if is_init, we want to stay there for a bit or start moving immediately?
             # Let's respect the sampled hold time. But since start=goal, moving phase is effectively holding.
             # So total time is valid.
    
    def update_curr_ee_goal(self):
        self.goal_timer += self.dt
        # Linear interpolation
        # Avoid division by zero if traj_timesteps is very small
        ratio = torch.clip(self.goal_timer / (self.traj_timesteps + 1e-6), 0, 1).unsqueeze(1)
        self.curr_ee_goal_sphere = self.ee_start_sphere + (self.ee_goal_sphere - self.ee_start_sphere) * ratio
        
        # Update cartesian current goal for viz (optional) or consistency
        # Recompute cartesian from interpolated spherical
        l = self.curr_ee_goal_sphere[:, 0]
        p = self.curr_ee_goal_sphere[:, 1]
        y = self.curr_ee_goal_sphere[:, 2]
        
        pitch_sin = torch.sin(p)
        pitch_cos = torch.cos(p)
        yaw_sin = torch.sin(y)
        yaw_cos = torch.cos(y)
        proj_len = l * pitch_cos
        
        self.curr_ee_goal_cart[:, 0] = proj_len * yaw_cos
        self.curr_ee_goal_cart[:, 1] = proj_len * yaw_sin
        self.curr_ee_goal_cart[:, 2] = l * pitch_sin
        
        # Check if new goal needed
        env_ids = (self.goal_timer > self.traj_total_timesteps).nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._resample_ee_goal(env_ids)

    def _reward_tracking_ee_sphere(self):
        # Calculate current EE pos in base frame (spherical)
        # 1. Get EE pos in base frame
        # ee_pos is in world frame. base_pos is in world frame.
        # rel_pos = ee_pos - base_pos. Rotate by base_quat inverse.
        rel_pos = self.ee_pos - self.root_states[:, :3]
        rel_pos = quat_rotate_inverse(self.base_quat, rel_pos)
        
        # 2. Convert to spherical
        # x, y, z = rel_pos[:, 0], rel_pos[:, 1], rel_pos[:, 2]
        # l = sqrt(x^2 + y^2 + z^2)
        # p = asin(z / l)
        # y = atan2(y, x)
        l = torch.norm(rel_pos, dim=1)
        p = torch.asin(torch.clamp(rel_pos[:, 2] / (l + 1e-6), -1.0, 1.0))
        y = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])
        
        current_sphere = torch.stack([l, p, y], dim=1)
        
        # 3. Compute error
        error = torch.sum(torch.square((current_sphere - self.curr_ee_goal_sphere) * self.sphere_error_scale), dim=1)
        return torch.exp(-error / self.cfg.rewards.tracking_ee_sigma)

    def _reward_tracking_ee_orn(self):
        # Placeholder for orientation reward
        # Assuming we want end-effector to be level or some specific orientation
        # For now, let's say we want it to align with base orientation (keep relative orientation zero)
        # or specific target.
        # WidowGo1 uses tracking_ee_orn.
        return 0.0 # To be implemented if specific orientation target is defined

    def _reward_arm_energy_abs_sum(self):
        # Penalize energy consumption of the arm
        # Arm DOFs are first 6
        return torch.sum(torch.abs(self.torques[:, :6] * self.dof_vel[:, :6]), dim=1)

    def _reward_tracking_ee_cart(self):
        # Placeholder for cartesian tracking reward
        return 0.0

    def _reward_arm_orientation(self):
        # Placeholder
        return 0.0

    def _reward_tracking_ee_orn_ry(self):
        # Placeholder
        return 0.0

    # ------------ reward functions----------------

    def _reward_feet_distance(self):
        # Penalize base height away from target
        feet_distance = torch.norm(
            self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1
        )
        reward = torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1) + \
                 torch.clip(feet_distance - self.cfg.rewards.max_feet_distance, 0, 1)
        return reward

    def _reward_collision(self):
        return torch.sum(
            torch.norm(
                self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 1.0, dim=1)

    def _reward_nominal_foot_position(self):
        #1. calculate foot postion wrt base in base frame  
        nominal_base_height = -(self.cfg.rewards.base_height_target- self.cfg.asset.foot_radius)
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        reward = 0
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
            height_error = nominal_base_height - foot_positions_base[:, i, 2]
            reward += torch.exp(-(height_error ** 2)/ self.cfg.rewards.nominal_foot_position_tracking_sigma)
        vel_cmd_norm = torch.norm(self.commands[:, :3], dim=1)
        return reward / len(self.feet_indices)*torch.exp(-(vel_cmd_norm ** 2)/self.cfg.rewards.nominal_foot_position_tracking_sigma_wrt_v)
    
    def _reward_same_foot_z_position(self):
        reward = 0
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        foot_z_position_err = foot_positions_base[:,0,2] - foot_positions_base[:,1,2]
        return foot_z_position_err ** 2

    def _reward_leg_symmetry(self):
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        leg_symmetry_err = (abs(foot_positions_base[:,0,1])-abs(foot_positions_base[:,1,1]))
        return torch.exp(-(leg_symmetry_err ** 2)/ self.cfg.rewards.leg_symmetry_tracking_sigma)

    def _reward_same_foot_x_position(self):
        reward = 0
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        foot_x_position_err = foot_positions_base[:,0,0] - foot_positions_base[:,1,0]
        # reward = torch.exp(-(foot_x_position_err ** 2)/ self.cfg.rewards.foot_x_position_sigma)
        reward = torch.abs(foot_x_position_err)
        return reward

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        reward = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return reward

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques[:, 6:]), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(self.dof_acc[:, 6:]), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.actions - self.last_actions[:, :, 0]), dim=1)

    def _reward_action_smooth(self):
        # Penalize changes in actions
        return torch.sum(
            torch.square(
                self.actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1)

    def _reward_keep_balance(self):
        return torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_pb(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_lin_vel() - self.rwd_linVelTrackPrev)
        # return ang_vel_error
        return delta_phi / self.dt

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.ang_tracking_sigma)

    def _reward_tracking_ang_vel_pb(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_ang_vel() - self.rwd_angVelTrackPrev)
        # return ang_vel_error
        return delta_phi / self.dt
    
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.abs(base_height - self.cfg.rewards.base_height_target)

    def _reward_tracking_lin_vel_x_exp(self):
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel_yaw_exp(self):
        return self._reward_tracking_ang_vel()
