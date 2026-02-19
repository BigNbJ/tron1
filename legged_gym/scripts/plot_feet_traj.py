import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 配置参数 (CONFIG)
# ==========================================
DURATION = 0.2       
DT = 0.005           

SCALES = 0.6
AMP_HIP = 0.5  *  SCALES      
AMP_KNEE = 1.0  * SCALES     
RETRACT_AMP = 1.3

DEFAULT_HIP = 0.0
DEFAULT_KNEE = 0.0

# ==========================================
# 2. 机器人运动学参数 (Vectors)
# ==========================================
OFFSET_HIP_KNEE = np.array([-0.1500, -0.0205, -0.25981])
OFFSET_KNEE_WHEEL = np.array([0.150e-0, 43.5e-3, -259.81e-3])

def get_rotation_y(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c,  0, s],
        [0,  1, 0],
        [-s, 0, c]
    ])

# ==========================================
# 3. 生成数据
# ==========================================
time_steps = np.arange(0, DURATION + DT, DT)

knee_x, knee_z = [], []
foot_x, foot_z = [], []

key_frames = []
mid_idx = int(len(time_steps) / 2)
key_indices = [0, mid_idx]

for i, t in enumerate(time_steps):
    phase = (2 * np.pi * t) / DURATION
    traj_val = 0.5 * (1 - np.cos(phase))
    
    # 核心差异点：这里加上了 Retract
    q_hip = DEFAULT_HIP + traj_val * AMP_HIP * RETRACT_AMP
    q_knee = DEFAULT_KNEE + traj_val * AMP_KNEE

    # FK 计算
    R_hip = get_rotation_y(q_hip)
    p_knee = np.dot(R_hip, OFFSET_HIP_KNEE)
    
    R_knee = get_rotation_y(-q_knee) 
    vec_wheel_local = np.dot(R_knee, OFFSET_KNEE_WHEEL)
    
    vec_wheel_global = np.dot(R_hip, vec_wheel_local)
    p_foot = p_knee + vec_wheel_global

    knee_x.append(p_knee[0])
    knee_z.append(p_knee[2])
    foot_x.append(p_foot[0])
    foot_z.append(p_foot[2])

    if i in key_indices:
        key_frames.append((np.array([0,0,0]), p_knee, p_foot))

# ==========================================
# 4. 绘图
# ==========================================
plt.figure(figsize=(10, 10))
plt.title(f'XZ Kinematics Trajectory (With Retraction)\nAmp_Hip={AMP_HIP}, Amp_Knee={AMP_KNEE}, Retract={RETRACT_AMP}')

# 绘制 Hip 原点
plt.plot(0, 0, 'ko', markersize=8, label='Hip Joint (Fixed)')

# 绘制 Knee 轨迹
plt.plot(knee_x, knee_z, 'g--', linewidth=2, label='Knee Trajectory')

# 绘制 Foot 轨迹 (蓝色实线，应该显示出后撤效果)
plt.plot(foot_x, foot_z, 'b-', linewidth=3, label='Foot Trajectory')

# 绘制连杆结构
colors = ['gray', 'orange'] 
labels = ['Start/End Pose', 'Peak Lift Pose']

for i, (ph, pk, pf) in enumerate(key_frames):
    plt.plot([ph[0], pk[0]], [ph[2], pk[2]], color=colors[i], linestyle='-', linewidth=4, alpha=0.6)
    plt.plot([pk[0], pf[0]], [pk[2], pf[2]], color=colors[i], linestyle='-', linewidth=4, alpha=0.6)
    plt.plot(pk[0], pk[2], 'o', color=colors[i], markersize=6)
    plt.plot(pf[0], pf[2], 'o', color=colors[i], markersize=6, label=labels[i])

plt.xlabel('X (Forward) [m]')
plt.ylabel('Z (Up) [m]')
plt.axis('equal') 
plt.grid(True)
plt.legend()

filename = 'leg_kinematics_xz_retract.png'
plt.savefig(filename)
print(f"修正后的图像已保存为: {filename}")