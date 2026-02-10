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
import torch.nn.functional as F
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

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum:
            time_out_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
            self.update_command_curriculum(time_out_env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
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
        obs_buf, _ = self.compute_group_observations()
        self.obs_history[env_ids] = obs_buf[env_ids].repeat(1, self.obs_history_length)
        self.gait_indices[env_ids] = 0
        self.fail_buf[env_ids] = 0
        self.action_fifo[env_ids] = 0
        self.dof_pos_int[env_ids] = 0
        
        # Reset contact force history and feedforward timers
        self.contact_force_history[env_ids] = 0.0
        self.ff_timers[env_ids] = -1.0

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

        # --- [NEW] Contact Trigger & Feedforward Logic ---

        # 1. 检测触发
        trigger_mask = self.check_contact_trigger(
            trigger_threshold=self.cfg.env.contact_trigger_threshold
        )
        
        # 2. 计算前馈 (使用修改后的函数)
        ff_actions = self._compute_feedforward_action(trigger_mask)
        
        self.extras["trigger_mask"] = trigger_mask
        self.extras["ff_actions"] = ff_actions
        
        # 3. 融合
        self.actions = self.k_pf * actions + self.k_ff * ff_actions
        # -------------------------------------------------

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
            self.commands[:, :self.cfg.commands.num_commands] * self.commands_scale,
            self.critic_obs_buf # make sure critic_obs update in every for loop
        )
        
    def _action_clip(self, actions):
        self.actions = actions
        
    def _compute_torques(self, actions):
        pos_action = (
            torch.cat(
                (
                    actions[:, 0:3], torch.zeros_like(actions[:, 0]).view(self.num_envs, 1),
                    actions[:, 4:7], torch.zeros_like(actions[:, 0]).view(self.num_envs, 1),
                ),
                axis=1,
            )
            * self.cfg.control.action_scale_pos
        )
        vel_action = (
            torch.cat(
                (
                    torch.zeros_like(actions[:, 0:3]), actions[:, 3].view(self.num_envs, 1),
                    torch.zeros_like(actions[:, 0:3]), actions[:, 7].view(self.num_envs, 1),
                ),
                axis=1,
            )
            * self.cfg.control.action_scale_vel
        )
        # pd controller
        torques = self.p_gains * (pos_action + self.default_dof_pos - self.dof_pos) + self.d_gains * (vel_action - self.dof_vel)
        torques = torch.clip(torques, -self.torque_limits, self.torque_limits )  # torque limit is lower than the torque-requiring lower bound
        return torques * self.torques_scale #notice that even send torque at torque limit , real motor may generate bigger torque that limit!!!!!!!!!!

    def post_physics_step(self):
        super().post_physics_step()
        self.wheel_lin_vel = self.foot_velocities[:, 0, :] + self.foot_velocities[:, 1, :]

    def compute_group_observations(self):
        # ---------------------------------------------------------
        # 1. Update Critic Contact History (Sliding Window)
        # ---------------------------------------------------------
        # 获取当前帧的 3D 接触力 (num_envs, 2, 3)
        current_feet_forces = self.contact_forces[:, self.feet_indices, :]
        
        # 更新历史: 丢弃最旧的一帧 (index 0), 拼接最新的一帧
        # self.critic_contact_history shape: (num_envs, 2, 3, 3)
        self.critic_contact_history = torch.cat(
            [self.critic_contact_history[..., 1:], current_feet_forces.unsqueeze(-1)], 
            dim=-1
        )
        
        # 计算平均值 (Avg contact forces)
        # mean over the last dim (window size), result shape: (num_envs, 2, 3)
        avg_contact_forces = torch.mean(self.critic_contact_history, dim=-1)
        
        # 展平为 (num_envs, 6)
        avg_contact_forces_flat = avg_contact_forces.view(self.num_envs, 6)

        # note that observation noise need to modified accordingly !!!
        dof_list = [0,1,2,4,5,6]
        dof_pos = (self.dof_pos - self.default_dof_pos)[:,dof_list]
        # dof_pos = torch.remainder(dof_pos + self.pi, 2 * self.pi) - self.pi

        obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                dof_pos * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                # self.clock_inputs_sin.view(self.num_envs, 1),
                # self.clock_inputs_cos.view(self.num_envs, 1),
                # self.gaits,
            ),
            dim=-1,
        )

        height_obs = self.measured_heights * self.obs_scales.height_measurements

        critic_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.obs_buf,
            avg_contact_forces_flat * self.obs_scales.contact_forces,
            height_obs,
        ), dim=-1)
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

        # === 新增：专门负责更新 feet_air_time 状态 ===
        # 1. 获取接触状态 (建议使用论文提到的滑动窗口滤波后的接触)
        contact = self._get_wheel_contacts() 
        
        # 2. 捕捉“刚落地”瞬间用于给奖励 (存入 buffer 供奖励函数读取)
        #    逻辑：如果在空中(time>0) 且 现在接触了 => 刚落地
        self.first_contact = (self.feet_air_time > 0.) & contact
        
        # 3. 记录“本帧结算的滞空时间”供奖励函数使用
        #    (必须在重置前记录，否则奖励函数读到的都是0)
        self.last_air_time = self.feet_air_time.clone()
        
        # 4. 更新计时器 (状态转移)
        self.feet_air_time += self.dt
        self.feet_air_time[contact] = 0. # 接触时清零

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = (
            self.command_ranges["lin_vel_x"][env_ids, 1]
            - self.command_ranges["lin_vel_x"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_x"
        ][
            env_ids, 0
        ]
        self.commands[env_ids, 1] = (
            self.command_ranges["lin_vel_y"][env_ids, 1]
            - self.command_ranges["lin_vel_y"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_y"
        ][
            env_ids, 0
        ]
        self.commands[env_ids, 2] = (
            self.command_ranges["ang_vel_yaw"][env_ids, 1]
            - self.command_ranges["ang_vel_yaw"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "ang_vel_yaw"
        ][
            env_ids, 0
        ]
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
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
            
        # Jump command sampling
        if self.cfg.commands.USE_JUMP:
            # Default to 0
            self.commands[env_ids, 4] = 0.0

            # Get terrain info to restrict jumping to smooth slopes
            is_smooth_slope = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
            if hasattr(self, "terrain_types"):
                terrain_col_indices = self.terrain_types[env_ids]
                terrain_proportions = torch.tensor(
                    self.cfg.terrain.terrain_proportions, device=self.device
                )
                terrain_thresholds = torch.cumsum(terrain_proportions, dim=0)
                normalized_indices = (
                    terrain_col_indices.float() + 0.001
                ) / self.cfg.terrain.num_cols
                # Smooth slope is the first category
                is_smooth_slope = normalized_indices < terrain_thresholds[0]
            
            # 20% chance to jump, ONLY on smooth slopes
            jump_prob = 0.2
            jump_mask = (torch.rand(len(env_ids), device=self.device) < jump_prob) & is_smooth_slope
            jump_indices = env_ids[jump_mask]
            
            if len(jump_indices) > 0:
                # Set jump height (delta)
                self.commands[jump_indices, 4] = (
                    self.command_ranges["jump_height"][1]
                    - self.command_ranges["jump_height"][0]
                ) * torch.rand(len(jump_indices), device=self.device) + self.command_ranges["jump_height"][0]
                
                # Enforce forward acceleration for jumpers
                # Set v_x to max
                self.commands[jump_indices, 0] = self.command_ranges["lin_vel_x"][jump_indices, 1]
                # Reset lateral/angular velocity for stability
                self.commands[jump_indices, 1] = 0.0
                self.commands[jump_indices, 2] = 0.0

            
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
        return noise_vec

    def _init_buffers(self):
        super()._init_buffers()
        self.wheel_lin_vel = torch.zeros_like(self.foot_velocities)
        self.wheel_ang_vel = torch.zeros_like(self.base_ang_vel)
        # History buffer for contact trigger mechanism: (num_envs, 2, 3)
        self.contact_force_history = torch.zeros(self.num_envs, 2, 1, device=self.device, dtype=torch.float)
        # [NEW] Critic 用的 3D 向量历史 (存 Fx, Fy, Fz)
        # Shape: (num_envs, 2, 3, 3) -> (环境数, 左右脚, 3维力, 历史长度3)
        self.critic_contact_history = torch.zeros(self.num_envs, 2, 3, 1, device=self.device, dtype=torch.float)
        
        # [NEW] Buffers for Potential-Based (PB) rewards splitting
        # Used to store previous error for tracking_lin_vel_x_pb and y_pb
        self.last_lin_vel_error_x = 0.0
        self.last_lin_vel_error_y = 0.0
        self.last_ang_vel_error = 0.0
        
        # --- [NEW] Feedforward Variables ---
        self.ff_timers = torch.full((self.num_envs, 2), -1.0, device=self.device, dtype=torch.float)
        
        # 参数设置
        self.ff_duration = 0.4  # 周期 T
        self.k_pf = 1.0
        self.k_ff = 0.7         # 权重 TODO 1.0->2.0
        
        # 定义幅度 (Magnitudes)，均为正数
        # 具体的正负号 (+/-) 在 _compute_feedforward_action 中根据左右腿施加
        self.ff_amp_hip = 0.5   # 髋关节抬起幅度 (根据需要调整大小)
        self.ff_amp_knee = 1.0  # 膝关节弯曲幅度 (通常是髋的2倍)
        # -----------------------------------

        if self.cfg.terrain.measure_heights or self.cfg.terrain.critic_measure_heights:
            self.measured_heights = torch.zeros(self.num_envs, self.cfg.env.num_height_samples, device=self.device, requires_grad=False)

        # Update commands_scale to match num_commands
        if self.cfg.commands.num_commands > 3:
            # Re-create commands_scale with appropriate size
            scales = [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel]
            
            # Fill remaining scales with 1.0 (or specific values if needed)
            # Index 3: Heading (if used)
            # Index 4: Jump height
            for i in range(3, self.cfg.commands.num_commands):
                scales.append(1.0)
                
            self.commands_scale = torch.tensor(
                scales,
                device=self.device,
                requires_grad=False,
            )

    def get_observations(self):
        return (
            self.obs_buf,
            self.obs_history,
            self.commands[:, :self.cfg.commands.num_commands] * self.commands_scale,
            self.critic_obs_buf
        )

    def check_contact_trigger(self, trigger_threshold=50.0):
        """
        Contact-Triggered Mechanism:
        1. Calculate horizontal force Fxy.
        2. Update sliding window (history of length 3).
        3. Determine stable contact (all 3 frames > threshold).
        4. Determine trigger mask based on priority.
        
        Returns:
            trigger_mask (torch.Tensor): Shape (num_envs, 2), Boolean mask indicating which leg to trigger.
        """
        # 1. Horizontal Force Calculation
        # self.contact_forces: (num_envs, num_bodies, 3)
        # self.feet_indices: (2,)
        feet_contact_forces = self.contact_forces[:, self.feet_indices, :] # (num_envs, 2, 3)
        f_xy = torch.norm(feet_contact_forces[:, :, :2], dim=-1) # (num_envs, 2)

        # 2. Sliding Window Update
        # Shift history: remove oldest, add new
        # self.contact_force_history: (num_envs, 2, 3)
        self.contact_force_history = torch.cat([self.contact_force_history[:, :, 1:], f_xy.unsqueeze(-1)], dim=-1)

        # 3. Stable Contact Judgment
        # Check if all 3 frames in history > threshold
        stable_contact = torch.all(self.contact_force_history > trigger_threshold, dim=-1) # (num_envs, 2)

        # --- [新增] 起步保护逻辑 ---
        # 设定保护时间，例如 50 步 (假设 dt=0.02s，即 1秒)
        startup_steps = 50 
        # 创建掩码：只有 episode 长度大于 startup_steps 的环境才允许触发
        is_warmed_up = self.episode_length_buf > startup_steps
        
        # 将掩码应用到 stable_contact 上 (广播机制: (num_envs,) & (num_envs, 2))
        stable_contact = stable_contact & is_warmed_up.unsqueeze(-1)
        # -------------------------
 
        # 4. Priority Determination
        trigger_mask = stable_contact.clone()
        
        # Identify envs where both are stable
        both_stable = torch.all(stable_contact, dim=-1) # (num_envs,)
        
        # Handle "Both Stable" case:
        if torch.any(both_stable):
            # Get current Fxy for both feet in these envs
            current_f_xy = f_xy[both_stable] # (N_both, 2)
            
            # Find which foot has larger force
            larger_idx = torch.argmax(current_f_xy, dim=-1) # (N_both,)
            
            # Create a mask for these envs
            resolved_mask = torch.nn.functional.one_hot(larger_idx, num_classes=2).bool()
            
            # Assign back to trigger_mask
            trigger_mask[both_stable] = resolved_mask
            
        return trigger_mask


    def _compute_feedforward_action(self, trigger_mask):
        """
        计算前馈动作
        规则：
        - 左腿 (Left): Hip idx=1, Knee idx=2. 符号为正 (+)
        - 右腿 (Right): Hip idx=5, Knee idx=6. 符号为负 (-)
        """
        # 1. 更新计时器 (与之前逻辑相同)
        active_mask = self.ff_timers >= 0
        self.ff_timers[active_mask] += self.dt

        # === [修复] 添加互斥锁逻辑 ===
        # 检查是否有【任何一条腿】正在执行任务
        # shape: (num_envs,) 
        any_leg_active = torch.any(self.ff_timers >= 0, dim=1)
        
        # 只有在【没有任何腿在忙】的情况下，才允许接受新的触发
        # 广播 active_mask: (num_envs,) -> (num_envs, 2)
        allow_trigger = ~any_leg_active.unsqueeze(-1)
        
        # 更新 trigger 条件：必须是 Trigger有效 且 Timer闲置 且 互斥锁允许
        new_trigger = trigger_mask & (self.ff_timers < 0) & allow_trigger        
        # 启动计时器
        self.ff_timers[new_trigger] = 0.0
        
        done_mask = self.ff_timers > self.ff_duration
        self.ff_timers[done_mask] = -1.0 
        
        # 2. 生成 0~1 的余弦波轨迹
        traj_val = torch.zeros_like(self.ff_timers)
        active_now = self.ff_timers >= 0
        if torch.any(active_now):
            t = self.ff_timers[active_now]
            phase = (2 * torch.pi * t) / self.ff_duration
            traj_val[active_now] = 0.5 * (1 - torch.cos(phase))
            
        # 3. 映射关节 (关键修改部分)
        ff_action = torch.zeros_like(self.actions)
        scale_pos = self.cfg.control.action_scale_pos
        
        # === 左腿 (Left Leg) ===
        # 索引: 0, [1], [2], 3
        # 符号: 正 (+)
        # traj_val[:, 0] 对应左腿计时器的值
        left_val = traj_val[:, 0] / scale_pos
        ff_action[:, 1] = left_val * self.ff_amp_hip   # Left Hip (+)
        ff_action[:, 2] = left_val * self.ff_amp_knee  # Left Knee (+)
        
        # === 右腿 (Right Leg) ===
        # 索引: 4, [5], [6], 7
        # 符号: 负 (-)
        # traj_val[:, 1] 对应右腿计时器的值
        right_val = traj_val[:, 1] / scale_pos
        ff_action[:, 5] = right_val * -self.ff_amp_hip  # Right Hip (-)
        ff_action[:, 6] = right_val * -self.ff_amp_knee # Right Knee (-)
        
        return ff_action
    # ------------ reward functions----------------

    def _reward_feet_distance(self):
        # Penalize base height away from target
        feet_distance = torch.norm(
            self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1
        )
        reward = torch.clamp(self.cfg.rewards.min_feet_distance - feet_distance, min=0.0) + \
                 torch.clamp(feet_distance - self.cfg.rewards.max_feet_distance, min=0.0)
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

    def _get_wheel_contacts(self):
        contact_forces = torch.norm(
            self.contact_forces[:, self.feet_indices, :], dim=-1
        )
        return contact_forces > 1.0

    def _reward_jump(self):
        # Heuristic based jump reward
        jump_cmd = self.commands[:, 4]
        is_jumping = jump_cmd > 0.05
        
        if not torch.any(is_jumping):
            return torch.zeros(self.num_envs, device=self.device)
            
        contacts = self._get_wheel_contacts() # shape (num_envs, num_feet)
        in_air = torch.all(~contacts, dim=1)
        on_ground = torch.any(contacts, dim=1)
        
        target_h = self.cfg.rewards.base_height_target + jump_cmd
        
        # Reward 1: Push off (Ground & Moving Up)
        push_reward = torch.zeros_like(jump_cmd)
        push_cond = is_jumping & on_ground & (self.base_lin_vel[:, 2] > 0.1)
        push_reward[push_cond] = self.base_lin_vel[push_cond, 2]
        
        # Reward 2: Flight Height (Air)
        flight_reward = torch.zeros_like(jump_cmd)
        height_error = target_h - self.base_height
        flight_cond = is_jumping & in_air
        flight_reward[flight_cond] = torch.exp(-torch.square(height_error[flight_cond]) / 0.05)
        
        return push_reward + flight_reward

    def _reward_jump_height_tracking(self):
        jump_cmd = self.commands[:, 4]
        is_jumping = jump_cmd > 0.05
        
        if not torch.any(is_jumping):
            return torch.zeros(self.num_envs, device=self.device)

        target_h = self.cfg.rewards.base_height_target + jump_cmd
        error = (self.base_height - target_h)
        
        reward = torch.exp(-torch.square(error) / 0.1)
        reward[~is_jumping] = 0.0
        return reward

    def _reward_leg_retraction(self):
        # 1) 判定跳跃与阶段
        if self.cfg.commands.num_commands <= 4:
            # 没有jump维度，就只维持nominal
            jump_cmd = torch.zeros(self.num_envs, device=self.device)
        else:
            jump_cmd = self.commands[:, 4]

        is_jumping = jump_cmd > 0.05

        contacts = self._get_wheel_contacts()
        in_air = torch.all(~contacts, dim=1)
        on_ground = torch.any(contacts, dim=1)
        air_ratio = (~contacts).float().mean(dim=1)

        foot_positions_base = self.foot_positions - (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :])
        foot_z_rel = foot_positions_base[:, :, 2]

        target_nominal = -0.6
        target_extend = -0.8
        target_retract = -0.45

        target = torch.full((self.num_envs,), target_nominal, device=self.device)

        # 收到jump且还在地面（准备/起跳）：伸长腿
        target[is_jumping & on_ground] = target_extend

        # 收到jump且在空中：缩短腿
        target[is_jumping & in_air] = target_retract

        err = foot_z_rel - target.unsqueeze(1)
        per_foot_reward = torch.exp(-torch.square(err) / self.cfg.rewards.leg_retraction_tracking_sigma)
        reward = per_foot_reward.min(dim=1).values

        vel_cmd_norm = torch.norm(self.commands[:, :3], dim=1)
        vel_weight = torch.exp(-(vel_cmd_norm ** 2) / self.cfg.rewards.nominal_foot_position_tracking_sigma_wrt_v)
        # vel_weight = 1
        stage_weight = torch.where(
            is_jumping,
            torch.where(in_air, air_ratio, 1.0 - air_ratio),
            torch.ones_like(air_ratio),
        )
        reward = reward * vel_weight * stage_weight
        return reward


    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        reward = torch.square(self.base_lin_vel[:, 2])
        if self.cfg.commands.num_commands > 4:
            jump_cmd = self.commands[:, 4]
            is_jumping = jump_cmd > 0.05
            reward[is_jumping] = 0.0
        return reward

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        reward = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return reward

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(self.dof_acc), dim=1)

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
        reward = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        if self.cfg.commands.num_commands > 4:
            jump_cmd = self.commands[:, 4]
            is_jumping = jump_cmd > 0.05
            # Don't penalize tracking error during jump (give max reward)
            reward[is_jumping] = 1.0
        return reward

    def _reward_tracking_lin_vel_pb(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_lin_vel() - self.rwd_linVelTrackPrev)
        # return ang_vel_error
        return delta_phi / self.dt

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        reward = torch.exp(-ang_vel_error / self.cfg.rewards.ang_tracking_sigma)
        if self.cfg.commands.num_commands > 4:
            jump_cmd = self.commands[:, 4]
            is_jumping = jump_cmd > 0.05
            reward[is_jumping] = 1.0
        return reward

    def _reward_tracking_ang_vel_pb(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_ang_vel() - self.rwd_angVelTrackPrev)
        # return ang_vel_error
        return delta_phi / self.dt
    
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        reward = torch.abs(base_height - self.cfg.rewards.base_height_target)
        if self.cfg.commands.num_commands > 4:
            jump_cmd = self.commands[:, 4]
            is_jumping = jump_cmd > 0.05
            reward[is_jumping] = 0.0
        return reward


    # ----------------------------------------------------------------
    # [NEW] Missing Rewards from CTBC Paper Table II
    # ----------------------------------------------------------------

    def _reward_feet_clearance(self):
        """
        [Paper] Feet clearance: Encourages swing foot to be within [h_min, h_max].
        Only active when the robot is INTENDED to swing (Triggered).
        """
        target_height_min = 0.10 
        target_height_max = 0.20
        
        # 1. 关键修改：使用“期望状态”而非“物理状态”
        # 只有在触发了前馈轨迹（ff_timers >= 0）时，才要求通过 Feet Clearance 奖励来引导抬腿高度
        # 如果平时(timer < 0)抬腿，不给这个奖励，防止它为了刷分而在平地乱抬腿
        is_commanded_swing = (self.ff_timers >= 0)
        
        # 2. 计算相对地形的足端高度
        foot_heights = self._get_foot_heights() # 地形高度
        # foot_positions[:, :, 2] 是足端的世界 Z 坐标
        foot_z = self.foot_positions[:, :, 2] - foot_heights - self.cfg.asset.foot_radius
        
        # 3. 判定高度是否在区间内
        in_range = (foot_z > target_height_min) & (foot_z < target_height_max)
        
        # 4. 计算奖励
        # 只有在【应该摆动】且【高度达标】时才给分
        reward = torch.sum(is_commanded_swing.float() * in_range.float(), dim=1)
        
        return reward

    def _reward_feet_air_time(self):
        """
        [Paper] Feet air time: Encourages longer steps.
        Reward is given when the foot first touches the ground after a swing phase.
        """
        # 直接读取我们在 post_physics_step 里算好的“结算时间”和“落地标记”
        rew_airTime = torch.sum(
            torch.clamp(self.last_air_time, max=0.3) * self.first_contact.float(), 
            dim=1
        )
        
        return rew_airTime

    def _reward_feet_contact_number(self):
        """
        [Paper] Feet contact number: 
        Reward matching the desired contact state defined by the trigger.
        Formula: I(contact == stance) - 1.3 * I(contact != stance)
        """
        # 1. 获取实际物理接触 (Actual Physics State)
        # 只要总接触力 > 1.0 就认为接触了 (不管是踩在平地还是踢到台阶)
        # shape: (num_envs, 2)
        actual_contact = self._get_wheel_contacts() 
        
        # 2. 获取期望接触状态 (Desired State from Trigger)
        # 逻辑：
        # - 如果 timer >= 0: 说明 Fxy 触发了，正在执行前馈抬腿，所以期望是【悬空/Swing】(False)
        # - 如果 timer < 0:  说明没触发，正常跑，所以期望是【接触/Stance】(True)
        # 注意：这里直接用了 step() 里计算好的 ff_timers，它包含了 Fxy 的判断结果
        desired_contact = (self.ff_timers < 0)
        
        # 3. 比较两者是否一致
        is_match = (actual_contact == desired_contact)
        
        # 4. 计算奖惩 (Paper Table II)
        # Match: +1.0
        # Mismatch: -1.3 (惩罚更重，强迫机器人听指挥)
        reward = is_match.float() * 1.0 - (~is_match).float() * 1.3
        
        return torch.sum(reward, dim=1)

    def _reward_wheel_zero_velocity(self):
        """
        [Paper] Wheel zero velocity: Penalize wheel rotation when leg is in swing phase.
        Prevents dangerous spinning in air.
        """
        contact = self._get_wheel_contacts()
        is_swing = ~contact
        
        # 获取轮子关节速度 (假设轮子索引是 3 和 7，请根据你的 cfg 调整)
        # 左轮: actions index 3 (vel control?), 右轮: actions index 7
        # 对应 dof_vel 索引
        wheel_indices = [3, 7] # 请确认你的 DOF 顺序！
        
        wheel_vel = self.dof_vel[:, wheel_indices]
        
        # 惩罚: exp(- sum( is_swing * vel^2 ))
        reward = torch.exp(-torch.sum(is_swing.float() * torch.square(wheel_vel), dim=1))
        return reward

    def _reward_wheel_spin(self):
        """
        [Paper] Wheel spin: Regularization to prevent wheel slipping on ground.
        Logic: If wheel linear vel >> foot linear vel
        """
        # 轮子半径
        r = self.cfg.asset.foot_radius # 0.06 or similar
        
        # 轮子角速度
        wheel_indices = [3, 7]
        wheel_omega = self.dof_vel[:, wheel_indices]
        
        # 轮子线速度 (r * omega)
        v_wheel = torch.abs(r * wheel_omega)
        
        # 足端实际线速度 (世界坐标系下的绝对速度)
        v_foot = torch.norm(self.foot_velocities[:, :, :2], dim=-1) # 只看水平速度
        
        # 误差: 0.8 * v_wheel - v_foot - 0.1 (阈值)
        spin_error = 0.8 * v_wheel - v_foot - 0.1
        
        reward = torch.sum(torch.clamp(spin_error, min=0.0), dim=1)
        return reward

    def _reward_default_pose(self):
        """
        [Paper] Default pose: Penalize deviation from default joint positions.
        """
        # 排除轮子关节，只计算腿部关节
        # 假设前3个是左腿，中间1个轮子(idx3)，后3个右腿，最后1个轮子(idx7)
        leg_indices = [0, 1, 2, 4, 5, 6] 
        
        diff = self.dof_pos[:, leg_indices] - self.default_dof_pos[:, leg_indices]
        
        # --- [NEW] 智能屏蔽逻辑 ---
        # 1. 获取触发状态 (ff_timers >= 0 表示正在执行抬腿)
        # ff_timers shape: (num_envs, 2) -> 扩展到关节维度
        # 假设关节顺序: [L_Abad, L_Hip, L_Knee, R_Abad, R_Hip, R_Knee]
        
        # 简单处理：如果该环境有任何腿在触发，暂时减弱该环境的 default_pose 惩罚
        is_triggered = torch.any(self.ff_timers >= 0, dim=1) # (num_envs,)
        
        penalty = torch.sum(torch.abs(diff), dim=1)
        
        # 如果触发了，惩罚系数乘 0.1 (几乎忽略)，否则乘 1.0 (正常惩罚)
        scale = torch.where(is_triggered, 0.1, 1.0)
        
        return penalty * scale
        
    def _reward_opposite_base_vel(self):
        """
        [Paper] Opposite base vel: Penalize moving backwards when commanded forwards.
        """
        v_cmd_x = self.commands[:, 0]
        v_base_x = self.base_lin_vel[:, 0]
        
        # 只有当命令非零时才计算
        # Logic: max(0, -sgn(v_cmd) * v_base)
        penalty = torch.clamp(-torch.sign(v_cmd_x) * v_base_x, min=0.0)

        # [关键修改] 爬楼梯时允许短暂的速度反向（因为可能被台阶弹回来）
        is_triggered = torch.any(self.ff_timers >= 0, dim=1)
        penalty[is_triggered] = 0.0

        return penalty

    def _reward_feet_contact_forces(self):
        """
        [Paper] Feet contact forces: Penalize high impact forces.
        """
        max_force = 300.0 # 假设阈值，需调整
        force_norms = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        return torch.sum(torch.clamp(force_norms - max_force, min=0.0), dim=1)


    # ----------------------------------------------------------------
    # [Table II] Task Rewards (Strict Implementation)
    # ----------------------------------------------------------------

    def _reward_tracking_lin_vel_x(self):
        # Formula: exp(-20 * (v_cmd_x - v_base_x)^2)
        is_triggered = torch.any(self.ff_timers >= 0, dim=1)
        # 如果有触发，期望的跟踪线速度变为0
        v_cmd_x = torch.where(is_triggered, torch.zeros_like(self.commands[:, 0]), self.commands[:, 0])
        lin_vel_error_x = torch.square(v_cmd_x - self.base_lin_vel[:, 0])
        reward = torch.exp(-20.0 * lin_vel_error_x)
        return reward

    def _reward_tracking_lin_vel_y(self):
        # Formula: exp(-20 * (v_cmd_y - v_base_y)^2)
        lin_vel_error_y = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-20.0 * lin_vel_error_y)

    def _reward_tracking_lin_vel_x_pb(self):
        # Potential-based reward for X
        current_reward = self._reward_tracking_lin_vel_x()
        delta = current_reward - self.last_lin_vel_error_x
        # Update history (Hack: updating state inside reward function)
        self.last_lin_vel_error_x = current_reward.detach() 
        return delta / self.dt

    def _reward_tracking_lin_vel_y_pb(self):
        # Potential-based reward for Y
        current_reward = self._reward_tracking_lin_vel_y()
        delta = current_reward - self.last_lin_vel_error_y
        self.last_lin_vel_error_y = current_reward.detach()
        return delta / self.dt


    def _reward_tracking_target_pos(self):
        """
        [NEW] Tracking target pos
        Formula: exp(-2 ||q - q_target||) - 0.2 ||q - q_target||
        """
        # Calculate q_target
        # target = default + scale * action
        # Only apply to leg joints (indices 0,1,2, 4,5,6), exclude wheels (3,7)
        leg_indices = [0, 1, 2, 4, 5, 6]
        
        q = self.dof_pos[:, leg_indices]
        q_des = self.default_dof_pos[:, leg_indices] + \
                self.cfg.control.action_scale_pos * self.actions[:, leg_indices]
        
        error = torch.norm(q - q_des, dim=1)
        
        # exp(-2 * error) - 0.2 * error (Assuming L2 norm inside exp based on Table II syntax ||...||)
        # Note: Formula in image is exp(-2||err||), not squared.
        return torch.exp(-2.0 * error) - 0.2 * error

    # ----------------------------------------------------------------
    # [Table II] Regularization Rewards (New additions)
    # ----------------------------------------------------------------

    def _reward_opposite_wheel_vel(self):
        """
        [NEW] Opposite wheel vel
        Formula: sum_j max(0, -sgn(v_cmd) * theta_dot_j)
        Penalize wheels spinning opposite to command.
        """
        wheel_indices = [3, 7]
        wheel_vel = self.dof_vel[:, wheel_indices] # (num_envs, 2)
        v_cmd_x = self.commands[:, 0].unsqueeze(1) # (num_envs, 1)
        
        # Only penalize if command is significant
        reward = torch.sum(torch.clamp(-torch.sign(v_cmd_x) * wheel_vel, min=0.0), dim=1)
        return reward

    def _reward_dof_vel(self):
        """
        [NEW] Dof vel
        Formula: sum(q_dot^2)
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)
