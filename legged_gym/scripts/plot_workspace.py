import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
import ikpy.chain
import math
import os
import xml.etree.ElementTree as ET
import trimesh

# --- 1. URDF 剪枝 (保持不变) ---
def prune_urdf_for_chain(urdf_path, root_link, tip_link, output_path):
    print(f"\n[1/5] 解析 URDF 结构: {urdf_path}")
    if not os.path.exists(urdf_path):
        print(f"[错误] URDF 文件不存在: {urdf_path}")
        return False
        
    tree = ET.parse(urdf_path)
    robot_root = tree.getroot()
    parent_map = {}
    joints = []
    
    for joint in robot_root.findall('joint'):
        parent = joint.find('parent').attrib['link']
        child = joint.find('child').attrib['link']
        parent_map[child] = (parent, joint)
        joints.append(joint)

    chain_links = set([tip_link])
    chain_joints = []
    curr_link = tip_link
    print(f"      回溯路径: {tip_link}", end="")
    while True:
        if curr_link == root_link:
            print(f" -> {root_link} (Root)")
            break
        if curr_link not in parent_map:
            print(f"\n[错误] 路径中断！找不到 {curr_link} 的父节点。")
            return False
        parent_link, joint_node = parent_map[curr_link]
        chain_links.add(parent_link)
        chain_joints.append(joint_node)
        print(f" -> {parent_link}", end="")
        curr_link = parent_link

    new_root = ET.Element('robot', name='temp_arm_chain')
    for link in robot_root.findall('link'):
        if link.attrib['name'] in chain_links:
            new_root.append(link)
    for joint in chain_joints: new_root.append(joint)
    
    ET.ElementTree(new_root).write(output_path)
    print(f"[2/5] 生成临时 URDF: {output_path}")
    return True

# --- 2. 增强版 Mesh 加载与绘制 (修复了崩溃问题) ---
def plot_robot_meshes(ax, chain, urdf_path, target_pose_angles=None, base_link_name_in_urdf="base_Link"):
    print(f"\n[4/5] 开始加载机器人模型...")
    if target_pose_angles is None:
        target_pose_angles = [0] * len(chain.links)

    transforms = chain.forward_kinematics(target_pose_angles, full_kinematics=True)
    
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # === 构建查找字典 ===
    link_mesh_map = {}
    for link in root.findall('link'):
        name = link.attrib['name']
        visual = link.find('visual')
        if visual is not None:
            geom = visual.find('geometry')
            if geom is not None:
                mesh_node = geom.find('mesh')
                if mesh_node is not None:
                    raw_filename = mesh_node.attrib['filename']
                    # 路径处理
                    if raw_filename.startswith("package://"):
                         # 简单回退处理，如果失败请手动修改这里
                         pass 
                    full_path = os.path.normpath(os.path.join(urdf_dir, raw_filename))
                    link_mesh_map[name] = full_path

    # 关节 -> 子连杆 映射
    joint_child_map = {}
    for joint in root.findall('joint'):
        j_name = joint.attrib['name']
        child_link = joint.find('child').attrib['link']
        joint_child_map[j_name] = child_link

    success_count = 0
    
    # === 遍历链条并绘图 ===
    for i, link in enumerate(chain.links):
        ikpy_name = link.name
        if i >= len(transforms): break
        tf_matrix = transforms[i]
        
        target_link_name = None
        
        # --- 匹配逻辑 ---
        if ikpy_name in link_mesh_map:
            target_link_name = ikpy_name
            
        elif ikpy_name in joint_child_map:
            child_name = joint_child_map[ikpy_name]
            if child_name in link_mesh_map:
                target_link_name = child_name
                print(f"      [映射] 关节 '{ikpy_name}' -> 连杆 '{child_name}'")

        # --- 强力修复 Base 匹配 ---
        # 如果是链条第0个节点，强制尝试使用用户提供的 base_link 名字
        elif i == 0:
            if base_link_name_in_urdf in link_mesh_map:
                target_link_name = base_link_name_in_urdf
                print(f"      [映射] 链条起点 (Index 0) -> 强制指定 '{target_link_name}'")
            else:
                print(f"      [警告] 无法找到 Base Mesh: {base_link_name_in_urdf}")

        # --- 开始加载 ---
        if target_link_name and target_link_name in link_mesh_map:
            mesh_path = link_mesh_map[target_link_name]
            
            if not os.path.exists(mesh_path):
                print(f"      ❌ 文件不存在: {mesh_path}")
                continue
                
            try:
                mesh = trimesh.load(mesh_path)
                
                # 单位处理
                extents = mesh.bounding_box.extents
                if np.max(extents) > 5.0: 
                    mesh.apply_scale(0.001)
                
                # --- 修复降采样崩溃 ---
                # 加一个 try-except 块，如果不支持降采样就直接跳过
                try:
                    if mesh.faces.shape[0] > 600:
                        # 尝试不同的简化方法名称，或者如果报错则捕获
                        if hasattr(mesh, 'simplify_quadratic_decimation'):
                            mesh = mesh.simplify_quadratic_decimation(600)
                        elif hasattr(mesh, 'simplify_quadric_decimation'):
                            mesh = mesh.simplify_quadric_decimation(600)
                        else:
                            # 简单的顶点切片（粗暴降采样，防止不支持高级算法）
                            # mesh = mesh.subdivide() # 不，这会增加面数
                            # 如果没有方法，就用原图
                            print(f"        (跳过降采样)")
                except Exception as simple_err:
                    print(f"        (降采样失败，使用原模: {simple_err})")
                
                mesh.apply_transform(tf_matrix)
                
                # 颜色
                color = [0.7, 0.7, 0.7, 0.4] 
                if "base" in target_link_name.lower(): color = [0.3, 0.3, 0.3, 0.5]
                if "L" in target_link_name or "link" in target_link_name.lower(): color = [0.9, 0.6, 0.2, 0.6]
                
                poly = art3d.Poly3DCollection(mesh.vectors)
                poly.set_alpha(color[3])
                poly.set_facecolor(color[:3])
                poly.set_edgecolor('k')
                poly.set_linewidth(0.05)
                ax.add_collection3d(poly)
                success_count += 1
                print(f"      ✅ 加载成功: {os.path.basename(mesh_path)}")
                
            except Exception as e:
                print(f"      ❌ 加载出错: {e}")
        else:
            if i > 0: 
                print(f"      跳过节点: {ikpy_name} (无 Mesh)")

    print(f"[完成] 成功渲染了 {success_count} 个部件。")

# --- 主逻辑 ---
def plot_workspace_with_robot_safe(urdf_path, base_link, tip_link, num_samples=3000):
    temp_urdf = "temp_arm_mesh_safe.urdf"
    if not prune_urdf_for_chain(urdf_path, base_link, tip_link, temp_urdf): return

    print(f"[3/5] 加载运动学链...")
    chain = ikpy.chain.Chain.from_urdf_file(temp_urdf, base_elements=[base_link], name="arm")
    
    print(f"      采样 {num_samples} 个点...")
    points = []
    for _ in range(num_samples):
        cfg = [0] 
        for i, link in enumerate(chain.links):
            if i==0: continue
            l, u = link.bounds
            if l is None or math.isinf(l): l, u = -np.pi, np.pi
            cfg.append(np.random.uniform(l, u))
        points.append(chain.forward_kinematics(cfg)[:3, 3])
    data = np.array(points)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. 画机器人 (传入 base_link 名字用于强力匹配)
    plot_robot_meshes(ax, chain, urdf_path, target_pose_angles=[0]*len(chain.links), base_link_name_in_urdf=base_link)
    
    # 2. 画点云
    sc = ax.scatter(data[:,0], data[:,1], data[:,2], s=2, c=data[:,2], cmap='viridis', alpha=0.3, label='Workspace')
    
    # 3. 自动视角
    # 将 0,0,0 加入范围计算
    all_points = np.vstack([data, [0,0,0]])
    
    max_range = np.array([all_points[:,0].max()-all_points[:,0].min(), 
                          all_points[:,1].max()-all_points[:,1].min(), 
                          all_points[:,2].max()-all_points[:,2].min()]).max() / 2.0
    mid_x = (all_points[:,0].max()+all_points[:,0].min()) * 0.5
    mid_y = (all_points[:,1].max()+all_points[:,1].min()) * 0.5
    mid_z = (all_points[:,2].max()+all_points[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("Reachable Workspace with Robot Mesh (Safe Mode)")
    plt.colorbar(sc, label="Z Height")
    plt.show()
    
    if os.path.exists(temp_urdf): os.remove(temp_urdf)

if __name__ == "__main__":
    urdf_file = "/home/server/workspace/WLG/Deep-Whole-Body-Control/tron1-rl-isaacgym/resources/robots/WF_TRON1A/urdf/robot_with_arm.urdf"
    
    # 确保这里是 URDF 里真实的基座和末端名称
    plot_workspace_with_robot_safe(urdf_file, "base_Link", "link6", num_samples=5000)