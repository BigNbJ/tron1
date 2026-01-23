import pybullet as p
import pybullet_data
import time
import numpy as np
import os

# ================= 配置区域 =================
# 1. 请修改为你的 URDF 文件的绝对路径或相对路径
# 参考你的 Config，路径可能类似： "resources/robots/WF_TRON1A/urdf/robot_with_arm.urdf"
URDF_PATH = "/home/server/workspace/WLG/Deep-Whole-Body-Control/tron1-rl-isaacgym/resources/robots/WF_TRON1A/urdf/robot_with_arm.urdf" 

# 2. 如果你知道末端 Link 的名字，请填在这里（例如 "Link6", "hand", "J6" 等）
# 如果留空 ""，脚本会自动尝试使用最后一个 Link
EE_LINK_NAME = "J6" 
# ===========================================

def cart2sph(x, y, z):
    l = np.sqrt(x**2 + y**2 + z**2)
    if l < 1e-6:
        return 0.0, 0.0, 0.0
    # Pitch: asin(z / l)
    p = np.arcsin(np.clip(z / l, -1.0, 1.0))
    # Yaw: atan2(y, x)
    y_ang = np.arctan2(y, x)
    return l, p, y_ang

def main():
    # 启动 PyBullet GUI
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    
    # 加载地面
    p.loadURDF("plane.urdf")

    # 检查 URDF 是否存在
    if not os.path.exists(URDF_PATH):
        print(f"错误: 找不到文件 {URDF_PATH}")
        print("请在脚本开头修改 URDF_PATH 为正确路径！")
        return

    # 加载机器人 (固定基座，方便调试手臂)
    try:
        robot_id = p.loadURDF(URDF_PATH, [0, 0, 0.5], useFixedBase=True)
    except Exception as e:
        print(f"加载 URDF 失败: {e}")
        return

    # 获取关节信息
    num_joints = p.getNumJoints(robot_id)
    joint_indices = []
    joint_names = []
    sliders = []
    
    # 查找末端执行器索引
    ee_idx = num_joints - 1 # 默认最后一个
    print(f"总关节数: {num_joints}")
    print("关节列表:")
    
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        j_name = info[1].decode("utf-8")
        l_name = info[12].decode("utf-8")
        j_type = info[2]
        
        print(f"ID: {i}, Joint: {j_name}, Link: {l_name}")
        
        # 如果指定了名字，匹配它
        if EE_LINK_NAME and EE_LINK_NAME in l_name:
            ee_idx = i
            
        # 为可动关节添加滑块 (Revolute 或 Prismatic)
        if j_type == p.JOINT_REVOLUTE or j_type == p.JOINT_PRISMATIC:
            joint_indices.append(i)
            joint_names.append(j_name)
            
            # 获取关节限制
            lower_limit = info[8]
            upper_limit = info[9]
            
            # 如果限制无效(lower >= upper)或者范围过大(如轮子连续旋转)，则限制滑块范围以便调试
            # URDF中 wheel_L_Joint 限制是 +/- 100000
            if lower_limit >= upper_limit:
                lower_limit = -3.14
                upper_limit = 3.14
            elif upper_limit - lower_limit > 4 * np.pi: # 如果范围超过 2圈 (约12.5)，限制一下，避免滑块无法微调
                # 保持中心，或者直接设为 -2pi ~ 2pi
                # 这里为了保留原始 limit 的意图，如果它只是有点大但合理，就不动。
                # 如果非常大(>20)，就 clamp
                 if upper_limit - lower_limit > 20:
                     lower_limit = -6.28
                     upper_limit = 6.28
            
            # 确保初始值 0 在范围内
            start_pos = 0.0
            if start_pos < lower_limit: start_pos = lower_limit
            if start_pos > upper_limit: start_pos = upper_limit

            # 添加调试滑块
            print(f"  -> {j_name} Limits: [{lower_limit:.3f}, {upper_limit:.3f}]")
            sid = p.addUserDebugParameter(j_name, lower_limit, upper_limit, start_pos)
            sliders.append(sid)

    print(f"\n选定的末端执行器 Link ID: {ee_idx}")
    print("开始运行... 按 Ctrl+C 退出")

    # 辅助线 ID
    line_id = -1
    text_id = -1

    while True:
        # 1. 读取滑块并控制关节
        for i, j_idx in enumerate(joint_indices):
            target_pos = p.readUserDebugParameter(sliders[i])
            p.setJointMotorControl2(robot_id, j_idx, p.POSITION_CONTROL, targetPosition=target_pos)

        p.stepSimulation()

        # 2. 获取位置
        # 基座位置 (虽然固定在 0,0,0.5，但为了通用性还是读取一下)
        base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
        # 末端位置
        ee_state = p.getLinkState(robot_id, ee_idx)
        ee_pos_world = ee_state[0] # 世界坐标系下的 XYZ
        
        # 3. 计算相对位置
        # 注意：你需要的是“相对于基座”的坐标
        # 将世界坐标差值 转换到 基座局部坐标系
        # (因为你的代码里用了 quat_rotate_inverse)
        
        # 世界坐标差
        diff_world = [ee_pos_world[0] - base_pos[0], 
                      ee_pos_world[1] - base_pos[1], 
                      ee_pos_world[2] - base_pos[2]]
        
        # 旋转到基座坐标系 (乘以基座逆姿态)
        # PyBullet 的 multiplyTransforms 可以处理
        # invert base transform
        inv_base_pos, inv_base_orn = p.invertTransform(base_pos, base_orn)
        # 这里的 rel_pos 就是在基座 Frame 下的 XYZ
        rel_pos, _ = p.multiplyTransforms(inv_base_pos, inv_base_orn, ee_pos_world, [0,0,0,1])
        
        # 4. 计算 L, P, Y
        l, pitch, yaw = cart2sph(rel_pos[0], rel_pos[1], rel_pos[2])

        # 5. 可视化
        # 在屏幕上打印文字
        msg = f"L: {l:.3f}\nP: {pitch:.3f}\nY: {yaw:.3f}"
        # 移除旧文字
        # PyBullet 的 debug text 更新比较闪烁，通常用 replaceItem
        if text_id < 0:
            text_id = p.addUserDebugText(msg, [0, 0, 1.0], textColorRGB=[1, 0, 0], textSize=1.5)
        else:
            text_id = p.addUserDebugText(msg, [0, 0, 1.0], textColorRGB=[1, 0, 0], textSize=1.5, replaceItemUniqueId=text_id)

        # 画线 (从基座到末端)
        if line_id < 0:
            line_id = p.addUserDebugLine(base_pos, ee_pos_world, [1, 1, 0], lineWidth=3)
        else:
            line_id = p.addUserDebugLine(base_pos, ee_pos_world, [1, 1, 0], lineWidth=3, replaceItemUniqueId=line_id)

        time.sleep(1./30.)

if __name__ == "__main__":
    main()