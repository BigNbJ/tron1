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
        # ctbc:接触触发
        self.xy_force_history[env_ids] = 0.0
        self.filtered_xy_contact[env_ids] = False
        self.contact_forces_feet_ema[env_ids] = 0.0
        self.contact_forces_history[env_ids] = 0.0

        self.is_lifting[env_ids] = False
        self.ff_phase[env_ids] = 0.0

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

    def _apply_feedforward(self, actions):
        """
        Apply feedforward trajectory (cosine wave) to left leg based on contact trigger.
        Fused with network actions using linear annealing.
        """
        # 1. 计算退火权重 k_ff
        # 从 1.0 线性衰减到 0.0
        # global_step_counter 是每步 +1
        k_ff = max(0.0, 1.0 - self.global_step_counter / self.cfg.ctbc.anneal_steps)
        
        # 如果退火结束，直接返回原始动作（节省计算）
        if k_ff <= 0.0:
            return actions

        # 2. 计算前馈轨迹
        # 公式: a_ff(t) = (A/2) * (1 - cos(2*pi*t/T))
        T = self.cfg.ctbc.ff_period
        A = self.cfg.ctbc.ff_amplitude
        
        # 仅对处于抬腿状态的环境计算
        # 注意: ff_phase 会在下面更新, 这里先用当前值计算
        
        # 相位比例 t/T
        phase_ratio = self.ff_phase / T
        
        # 计算轨迹值 (标量/向量)
        ff_val = (A / 2.0) * (1.0 - torch.cos(2 * math.pi * phase_ratio))
        
        # 3. 创建 ff_actions 张量并赋值
        ff_actions = torch.zeros_like(actions)
        
        # 强制左腿优先 (Hack): 注入到左腿髋关节 (index 1) 和 左腿膝关节 (index 2)
        # 比例 1:2
        # 注意符号: 
        #   Hip Flexion通常为正 -> 抬腿
        #   Knee Flexion通常为正 (根据用户设定) -> 抬小腿
        
        # 仅对 is_lifting 为 True 的行生效
        lifting_env_ids = self.is_lifting.nonzero(as_tuple=False).flatten()
        
        if len(lifting_env_ids) > 0:
            # 赋值: Hip = 1 * val, Knee = 2 * val (with sign)
            # 这里的 index 1 和 2 对应 hip_L 和 knee_L
            ff_actions[lifting_env_ids, 1] = ff_val[lifting_env_ids] * 1.0
            ff_actions[lifting_env_ids, 2] = ff_val[lifting_env_ids] * 2.0 
            
        # 4. 融合动作
        # a_t = a_pi + k_ff * a_ff
        fused_actions = actions + k_ff * ff_actions
        
        # 5. 更新状态
        # 更新相位
        self.ff_phase[self.is_lifting] += self.dt
        
        # 判断结束: 如果 ff_phase >= T, 结束抬腿
        finished = self.ff_phase >= T
        self.is_lifting[finished] = False
        
        return fused_actions

    def step(self, actions):
        actions = self._apply_feedforward(actions)
        self.global_step_counter += 1
        
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
        # 【新增的硬编码拦截】: 强制屏蔽网络的侧摆动作输出
        # 将左腿侧摆 (index 0) 和 右腿侧摆 (index 4) 的动作指令清零
        actions[:, 0] = 0.0
        actions[:, 4] = 0.0
        
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
        # note that observation noise need to modified accordingly !!!
        dof_list = [0,1,2,4,5,6]
        dof_pos = (self.dof_pos - self.default_dof_pos)[:,dof_list]
        # dof_pos = torch.remainder(dof_pos + self.pi, 2 * self.pi) - self.pi

        # 1. 计算缩放后的脚部力并拉平 [num_envs, 6]
        # 使用 3 帧平均值替代 EMA
        contact_forces_avg = torch.mean(self.contact_forces_history, dim=-1)
        contact_forces_flattened = (contact_forces_avg * self.cfg.normalization.obs_scales.contact_forces).view(self.num_envs, -1)
        
        # 2. 接触信号 (bool -> float) [num_envs, 2]
        contact_signal = self.filtered_xy_contact.float()

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
                torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements,
                contact_forces_flattened, # [num_envs, 6]
                contact_signal,           # [num_envs, 2]
            ),
            dim=-1,
        )

        critic_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel, self.obs_buf,
                # contact_forces_flattened, # already in obs_buf
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

        self._check_contact_trigger()

        current_feet_forces = self.contact_forces[:, self.feet_indices, :]
        self.contact_forces_feet_ema = 0.2 * current_feet_forces + (1 - 0.2) * self.contact_forces_feet_ema
       
        # Update 3-frame history for contact forces
        self.contact_forces_history = torch.roll(self.contact_forces_history, shifts=1, dims=-1)
        self.contact_forces_history[..., 0] = current_feet_forces

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
        noise_vec[20:28] = 0.0  # previous actions
        noise_vec[28:-8] = ( # height measurements
            noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        )
        noise_vec[-8:-2] = 0.1 * noise_level * self.cfg.normalization.obs_scales.contact_forces # contact forces noise
        noise_vec[-2:] = 0.0 # contact signal (bool) no noise
        return noise_vec

    def _init_buffers(self):
        super()._init_buffers()
        self.wheel_lin_vel = torch.zeros_like(self.foot_velocities)
        self.wheel_ang_vel = torch.zeros_like(self.base_ang_vel)

        # 加入这段代码，确保在第一次 reset 之前正确初始化高度测量张量
        if self.cfg.terrain.measure_heights or self.cfg.terrain.critic_measure_heights:
            self.measured_heights = torch.zeros(
                self.num_envs, 
                self.cfg.env.num_height_samples, 
                dtype=torch.float, 
                device=self.device, 
                requires_grad=False
            )
          
        # 初始化 XY 受力历史缓冲区
        # 形状: [num_envs, num_feet, history_length]
        # dtype 使用 float，因为我们要真实存储力的大小
        self.xy_force_history = torch.zeros(
            self.num_envs, len(self.feet_indices), self.cfg.ctbc.history_length,
            dtype=torch.float, 
            device=self.device, 
            requires_grad=False
        )

        self.filtered_xy_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)
        
        self.contact_forces_feet_ema = torch.zeros(
            self.num_envs, len(self.feet_indices), 3,
            dtype=torch.float, 
            device=self.device, 
            requires_grad=False
        )
        
        # New: Store 3 frames of contact forces history
        self.contact_forces_history = torch.zeros(
            self.num_envs, len(self.feet_indices), 3, 3, # [num_envs, num_feet, 3(xyz), 3(frames)]
            dtype=torch.float, 
            device=self.device, 
            requires_grad=False
        )

        # 初始化前馈控制相关的张量
        self.ff_phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.is_lifting = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.global_step_counter = 0

    # ------------ Contact Trigger----------------


    def _check_contact_trigger(self):
        xy_forces = self.contact_forces[:, self.feet_indices, 0:2]
        current_xy_norm = torch.norm(xy_forces, dim=2)
        
        self.xy_force_history = torch.roll(self.xy_force_history, shifts=1, dims=-1)
        self.xy_force_history[:, :, 0] = current_xy_norm
        
        force_threshold = self.cfg.ctbc.force_threshold
        is_high_force_history = self.xy_force_history > force_threshold
        
        # 将结果保存到类的属性中，供所有 Reward 共享读取
        self.filtered_xy_contact = torch.all(is_high_force_history, dim=-1)

        # ----------------------------------------
        # 新增: 触发前馈指令
        # ----------------------------------------
        # 1. 任意一只脚接触 (any_contact)
        any_contact = torch.any(self.filtered_xy_contact, dim=1)
        
        # 2. 当前不在抬腿状态 (not is_lifting)
        trigger = any_contact & (~self.is_lifting)
        
        # 3. 触发: 开启抬腿, 重置相位
        if torch.any(trigger):
            self.is_lifting[trigger] = True
            self.ff_phase[trigger] = 0.0

    # ------------ reward functions----------------

    def _reward_feet_distance(self):
        # Penalize base height away from target
        feet_distance = torch.norm(
            self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1
        )
        reward = torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1) + \
                 torch.clip(feet_distance - self.cfg.rewards.max_feet_distance, 0, 1)
        
        # If any foot is in contact (climbing), relax the feet distance constraint
        any_contact = torch.any(self.filtered_xy_contact, dim=1)
        reward = reward * (~any_contact).float()
        
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
            
            # Original reward term
            term = torch.exp(-(height_error ** 2)/ self.cfg.rewards.nominal_foot_position_tracking_sigma)
            
            # If filtered_xy_contact is True for this foot, we release the penalty (give full reward)
            # This allows the foot to lift without losing the nominal position reward
            is_contact = self.filtered_xy_contact[:, i]
            term = torch.where(is_contact, torch.ones_like(term), term)
            
            reward += term
            
        vel_cmd_norm = torch.norm(self.commands[:, :3], dim=1)
        return reward / len(self.feet_indices)*torch.exp(-(vel_cmd_norm ** 2)/self.cfg.rewards.nominal_foot_position_tracking_sigma_wrt_v)
    
    def _reward_same_foot_z_position(self):
        reward = 0
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        foot_z_position_err = foot_positions_base[:,0,2] - foot_positions_base[:,1,2]
        
        cost = foot_z_position_err ** 2
        
        # Use filtered_xy_contact to check for contact
        # If filtered_xy_contact is True for any foot, release the penalty
        any_contact = torch.any(self.filtered_xy_contact, dim=1)
        
        return cost * (~any_contact).float()

    def _reward_leg_symmetry(self):
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        leg_symmetry_err = (abs(foot_positions_base[:,0,1])-abs(foot_positions_base[:,1,1]))
        reward = torch.exp(-(leg_symmetry_err ** 2)/ self.cfg.rewards.leg_symmetry_tracking_sigma)
        
        # If any foot is in contact, relax symmetry constraint as legs might be in different phases
        any_contact = torch.any(self.filtered_xy_contact, dim=1)
        # However, leg symmetry is mainly about Y position (width), which might still be relevant.
        # But during climbing, body might tilt or shift weight, so relaxing is safer.
        # We use torch.where to set reward to 1.0 (max reward) when contact happens
        reward = torch.where(any_contact, torch.ones_like(reward), reward)
        
        return reward

    def _reward_same_foot_x_position(self):
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        foot_x_position_err = foot_positions_base[:,0,0] - foot_positions_base[:,1,0]
        # reward = torch.exp(-(foot_x_position_err ** 2)/ self.cfg.rewards.foot_x_position_sigma)
        
        cost = torch.abs(foot_x_position_err)
        
        # Use filtered_xy_contact to check for contact
        # If filtered_xy_contact is True for any foot, release the penalty
        any_contact = torch.any(self.filtered_xy_contact, dim=1)
        
        return cost * (~any_contact).float()

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        reward = torch.square(self.base_lin_vel[:, 2])
        
        # If any foot is in contact, allow z velocity (jumping/lifting)
        any_contact = torch.any(self.filtered_xy_contact, dim=1)
        reward = reward * (~any_contact).float()
        
        return reward

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        reward = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        
        # If any foot is in contact, allow angular velocity (tilt adjustment)
        any_contact = torch.any(self.filtered_xy_contact, dim=1)
        reward = reward * (~any_contact).float()
        
        return reward

    def _reward_orientation(self):
        # Penalize non flat base orientation
        reward = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        
        # If any foot is in contact, allow orientation tilt (pitch/roll)
        any_contact = torch.any(self.filtered_xy_contact, dim=1)
        reward = reward * (~any_contact).float()
        
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

    # ------------ Contact Trigger----------------
    def _reward_encourage_wheel_up(self):
        """
        [平稳抬腿奖励 - 绝对速度版 + 高度保持] 
        逻辑：当左轮受到水平冲击时，鼓励左轮在世界坐标系下产生真实的向上速度，并保持一定高度。
        """
        # 1. 确定左轮索引 (双轮足通常左轮为 0)
        left_idx = 0
        
        # 2. 获取世界坐标系下的左轮 Z 轴绝对速度
        world_vel_z = self.foot_velocities[:, left_idx, 2]
        
        # 3. 获取我们之前在 post_physics_step 中更新好的 XY 接触掩码
        left_contact_mask = self.filtered_xy_contact[:, left_idx].float()
        
        # 4. 提取向上速度：只奖励正值（向上收缩），不奖励向下伸展
        upward_vel = torch.clamp(world_vel_z, min=0.0, max=1.0)
        
        # 5. 添加高度奖励：鼓励在接触时抬高脚
        # 计算相对于基座的脚高度 (Z轴)
        foot_pos_z = self.foot_positions[:, left_idx, 2]
        base_pos_z = self.base_position[:, 2]
        rel_z = foot_pos_z - base_pos_z
        
        # 标称高度 (负值)
        nominal_h = -(self.cfg.rewards.base_height_target - self.cfg.asset.foot_radius)
        
        # 计算抬升量 (相对于标称位置)
        lift_amount = rel_z - nominal_h 
        # 限制奖励范围，避免过度抬升，假设抬升 20-30cm 足够
        lift_reward = torch.clamp(lift_amount, min=0.0, max=0.15)
        
        # 6. 组合奖励
        # 增加高度奖励的权重 (例如 5.0，使得 0.1m 的抬升相当于 0.5 的速度奖励)
        total_reward = (upward_vel + 5.0 * lift_reward) * left_contact_mask
        
        return total_reward

    def _reward_tracking_target_pos(self):
        # 1. 指定需要追踪的核心抬腿关节索引：左腿 hip(1), knee(2)；右腿 hip(5), knee(6)
        track_indices = [1, 2, 5, 6]
        
        # 2. 计算目标位置 q_target (默认位置 + 网络动作 * 动作缩放比例)
        q_target = self.default_dof_pos[:, track_indices] + self.actions[:, track_indices] * self.cfg.control.action_scale_pos
        
        # 3. 获取当前实际关节位置 q_current
        q_current = self.dof_pos[:, track_indices]
        
        # 4. 计算欧氏距离误差范数 ||q - q_target||
        pos_error = torch.norm(q_current - q_target, dim=1)
        
        # 5. 套用论文公式
        reward = torch.exp(-2.0 * pos_error) - 0.2 * pos_error
        
        # 6. 条件奖励掩码：仅在任意一脚检测到接触（触发抬腿）时激活
        any_contact = torch.any(self.filtered_xy_contact, dim=-1)
        
        return reward * any_contact.float()