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

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value


class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if "rew" in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def _plot(self):
        nb_rows = 3
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log
        # plot joint targets and measured positions
        a = axs[1, 0]
        if log["dof_pos"]:
            a.plot(time, log["dof_pos"], label="measured")
        if log["dof_pos_target"]:
            a.plot(time, log["dof_pos_target"], label="target")
        a.set(xlabel="time [s]", ylabel="Position [rad]", title="DOF Position")
        a.legend()
        # plot joint velocity
        a = axs[1, 1]
        if log["dof_vel"]:
            a.plot(time, log["dof_vel"], label="measured")
        if log["dof_vel_target"]:
            a.plot(time, log["dof_vel_target"], label="target")
        a.set(xlabel="time [s]", ylabel="Velocity [rad/s]", title="Joint Velocity")
        a.legend()
        # plot base vel x
        a = axs[0, 0]
        if log["base_vel_x"]:
            a.plot(time, log["base_vel_x"], label="measured")
        if log["command_x"]:
            a.plot(time, log["command_x"], label="commanded")
        if log["est_lin_vel_x"]:
            a.plot(time, log["est_lin_vel_x"], label="est")
        a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity x")
        a.legend()
        # plot base vel y
        a = axs[0, 1]
        if log["base_vel_y"]:
            a.plot(time, log["base_vel_y"], label="measured")
        if log["command_y"]:
            a.plot(time, log["command_y"], label="commanded")
        if log["est_lin_vel_y"]:
            a.plot(time, log["est_lin_vel_y"], label="est")
        a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity y")
        a.legend()
        # plot base vel yaw
        a = axs[0, 2]
        if log["base_vel_yaw"]:
            a.plot(time, log["base_vel_yaw"], label="measured")
        if log["command_yaw"]:
            a.plot(time, log["command_yaw"], label="commanded")
        a.set(
            xlabel="time [s]", ylabel="base ang vel [rad/s]", title="Base velocity yaw"
        )
        a.legend()
        # plot base vel z
        a = axs[1, 2]
        if log["base_vel_z"]:
            a.plot(time, log["base_vel_z"], label="measured")
        a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity z")
        a.legend()
        # plot contact forces
        a = axs[2, 0]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f"force {i}")
        a.set(xlabel="time [s]", ylabel="Forces z [N]", title="Vertical Contact forces")
        a.legend()
        # plot torque/vel curves
        a = axs[2, 1]
        # if log["dof_vel"] != [] and log["dof_torque"] != []:
        #     a.plot(log["dof_vel"], log["dof_torque"], "x", label="measured")
        # a.set(
        #     xlabel="Joint vel [rad/s]",
        #     ylabel="Joint Torque [Nm]",
        #     title="Torque/velocity curves",
        # )
        if log["power"]:
            a.plot(time, log["power"])
        a.set(xlabel="time [s]", ylabel="power [w]", title="Total Power")
        a.legend()
        # plot torques
        a = axs[2, 2]
        if log["dof_torque"] != []:
            a.plot(time, log["dof_torque"], label="measured")
        a.set(xlabel="time [s]", ylabel="Joint Torque [Nm]", title="Torque")
        a.legend()
        
        self._plot_custom_metrics(log, time)
        
        plt.show()

    def _plot_custom_metrics(self, state_log, time):
        # Plot Contact Forces (Separated X, Y, Z)
        fig1, (ax_x, ax_y, ax_z) = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
        
        # Force X
        if "wheel_force_x_L" in state_log:
            ax_x.plot(time, state_log["wheel_force_x_L"], label="Left Wheel Fx")
        if "wheel_force_x_R" in state_log:
            ax_x.plot(time, state_log["wheel_force_x_R"], label="Right Wheel Fx")
        ax_x.set(ylabel="Force [N]", title="Contact Force X")
        ax_x.legend()
        
        # Force Y
        if "wheel_force_y_L" in state_log:
            ax_y.plot(time, state_log["wheel_force_y_L"], label="Left Wheel Fy")
        if "wheel_force_y_R" in state_log:
            ax_y.plot(time, state_log["wheel_force_y_R"], label="Right Wheel Fy")
        ax_y.set(ylabel="Force [N]", title="Contact Force Y")
        ax_y.legend()
        
        # Force Z
        if "wheel_force_z_L" in state_log:
            ax_z.plot(time, state_log["wheel_force_z_L"], label="Left Wheel Fz")
        if "wheel_force_z_R" in state_log:
            ax_z.plot(time, state_log["wheel_force_z_R"], label="Right Wheel Fz")
        ax_z.set(xlabel="time [s]", ylabel="Force [N]", title="Contact Force Z")
        ax_z.legend()
        
        # Plot Joint Positions (Hip & Knee)
        fig2, (ax2_l, ax2_r) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
        
        # Left Leg
        if "dof_pos_hip_L" in state_log:
            ax2_l.plot(time, state_log["dof_pos_hip_L"], label="Hip L")
        if "dof_pos_knee_L" in state_log:
            ax2_l.plot(time, state_log["dof_pos_knee_L"], label="Knee L")
        ax2_l.set(ylabel="Position [rad]", title="Left Leg Joint Positions")
        ax2_l.legend()
        
        # Right Leg
        if "dof_pos_hip_R" in state_log:
            ax2_r.plot(time, state_log["dof_pos_hip_R"], label="Hip R")
        if "dof_pos_knee_R" in state_log:
            ax2_r.plot(time, state_log["dof_pos_knee_R"], label="Knee R")
        ax2_r.set(xlabel="time [s]", ylabel="Position [rad]", title="Right Leg Joint Positions")
        ax2_r.legend()

        # Plot Trigger & Feedforward Actions
        fig3, (ax3_trig, ax3_ff_hip, ax3_ff_knee) = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
        
        # Trigger Mask
        if "trigger_mask_L" in state_log:
            ax3_trig.plot(time, state_log["trigger_mask_L"], label="Trigger L", alpha=0.7)
        if "trigger_mask_R" in state_log:
            ax3_trig.plot(time, state_log["trigger_mask_R"], label="Trigger R", alpha=0.7)
        ax3_trig.set(ylabel="Trigger [bool]", title="Contact Trigger Mask")
        ax3_trig.legend()
        
        # FF Actions Hip
        if "ff_action_hip_L" in state_log:
            ax3_ff_hip.plot(time, state_log["ff_action_hip_L"], label="FF Hip L")
        if "ff_action_hip_R" in state_log:
            ax3_ff_hip.plot(time, state_log["ff_action_hip_R"], label="FF Hip R")
        ax3_ff_hip.set(ylabel="Action [rad]", title="Feedforward Hip Actions")
        ax3_ff_hip.legend()

        # FF Actions Knee
        if "ff_action_knee_L" in state_log:
            ax3_ff_knee.plot(time, state_log["ff_action_knee_L"], label="FF Knee L")
        if "ff_action_knee_R" in state_log:
            ax3_ff_knee.plot(time, state_log["ff_action_knee_R"], label="FF Knee R")
        ax3_ff_knee.set(xlabel="time [s]", ylabel="Action [rad]", title="Feedforward Knee Actions")
        ax3_ff_knee.legend()

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()
