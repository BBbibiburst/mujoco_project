"""
圆周轨迹跟踪与实时可视化仿真模块.

该模块在基础抓取环境之上，实现了复杂的运动控制与调试可视化功能：
1. **轨迹生成**：控制机械臂末端沿指定参数的圆周轨迹运动（包含位置与姿态计算）。
2. **混合控制**：同时演示末端执行器的笛卡尔空间轨迹跟踪与机械手的关节空间随机动作。
3. **实时可视化**：利用 MuJoCo 的用户几何体（User Geom）接口，实时绘制末端实际位置与目标位置的对比。

设计要点：
- **参数配置化**：将圆周运动参数、颜色样式、控制频率提取为配置类，提高可维护性。
- **可视化工具类**：封装 `TrajectoryVisualizer` 以管理调试几何体的生命周期，避免缓冲区溢出。
- **时间管理**：实现基于固定时间步长的主循环，确保轨迹速度的物理真实性。

依赖：
    - mujoco: 物理仿真与渲染核心。
    - numpy: 数值计算与几何变换。
    - robot_arm_system: 机器人模型组装。
    - position_controller: 运动控制算法（OSC/IK）。
"""

import mujoco
import numpy as np
import time
from typing import Tuple, Optional
from dataclasses import dataclass, field
from src.robot.robot_arm_system import get_combined_spec
from src.controllers.position_controller import OSC_PositionController
from src.controllers.hand_arm_controller import HandArmController


# ====================== 配置数据类 ======================

@dataclass
class CircleTrajectoryConfig:
    """圆周轨迹运动参数配置.
    
    Attributes:
        center: 轨迹圆心在世界坐标系的位置 [x, y, z] (米)。
        radius: 轨迹半径 (米)。
        speed: 旋转角速度 (弧度/秒)。
        z_offset: 轨迹平面的 Z 轴高度修正 (米)。
    """
    center: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.0, 0.4]))
    radius: float = 0.15
    speed: float = 1.5
    z_offset: float = 0.0  # 如果需要 Z 轴变化，可在此配置


@dataclass
class VisualStyle:
    """调试可视化样式配置.
    
    Attributes:
        actual_rgba: 实际位置球体的颜色 (青蓝色)。
        target_rgba: 目标位置球体的颜色 (红色，半透明)。
        actual_size: 实际位置球体半径 (米)。
        target_size: 目标位置球体半径 (米)。
    """
    actual_rgba: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 1.0, 0.8]))  # 青色
    target_rgba: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.4]))  # 红色
    actual_size: float = 0.005  # 5mm
    target_size: float = 0.015  # 15mm


@dataclass
class RuntimeConfig:
    """运行时逻辑配置.
    
    Attributes:
        hand_random_interval: 手部随机姿态更新的时间间隔 (秒)。
        target_pos: 场景中静态目标物体的位置 (用于环境构建)。
    """
    hand_random_interval: float = 1.0
    target_pos: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.0, 0.025]))


# ====================== 可视化工具类 ======================

class TrajectoryVisualizer:
    """轨迹调试几何体绘制工具.
    
    利用 MuJoCo 的 `user_scn` 接口在仿真 Viewer 中绘制自定义几何体。
    该类专门用于管理末端执行器实际位置的动态标记。
    
    设计策略：
        - **缓冲区安全**：在 `draw` 方法中检查 `ngeom`，防止几何体数量超过 MuJoCo 缓冲区上限。
        - **状态保持**：持有当前要绘制的位置状态，允许在控制循环中更新，在渲染阶段绘制。
    """
    
    def __init__(self, style: VisualStyle):
        """初始化可视化工具.
        
        Args:
            style: 可视化样式配置对象。
        """
        self.current_pos: Optional[np.ndarray] = None
        self.style = style

    def update_point(self, pos: np.ndarray) -> None:
        """更新当前要绘制的末端位置.
        
        Args:
            pos: 3D 坐标位置向量。
        """
        self.current_pos = pos.copy()

    def draw(self, viewer: mujoco.viewer) -> None:
        """将当前点渲染到 Viewer 场景中.
        
        注意：
            必须在每帧循环开始时重置 `viewer.user_scn.ngeom`，
            否则几何体会累积导致画面混乱。
        
        Args:
            viewer: MuJoCo Viewer 实例。
        """
        if self.current_pos is None:
            return
            
        # 安全检查：防止超出几何体缓冲区上限 (通常为 1000)
        if viewer.user_scn.ngeom < 990:  # 预留一些空间
            geom_id = viewer.user_scn.ngeom
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[geom_id],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[self.style.actual_size, 0, 0],  # 球体仅需第一个参数
                pos=self.current_pos,
                mat=np.eye(3).flatten(),  # 单位旋转矩阵
                rgba=self.style.actual_rgba
            )
            viewer.user_scn.ngeom += 1


# ====================== 环境构建 ======================

def build_custom_grasp_environment(cfg_runtime: RuntimeConfig) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """构建包含机械臂、静态目标物体及光源的基础环境.
    
    Args:
        cfg_runtime: 运行时配置，包含静态物体位置。
    
    Returns:
        Tuple[mujoco.MjModel, mujoco.MjData]: 仿真模型与数据实例。
    """
    spec, _ = get_combined_spec(
        rot_xyz_deg=(-90, 0, 0), 
        attach_point_name="right_hand"
    )
    worldbody = spec.worldbody

    # 顶部平行光，提供均匀照明
    worldbody.add_light(
        name="top_light",
        pos=[0.0, 0.0, 2.0],
        dir=[0.0, 0.0, -1.0],
        diffuse=[1.0, 1.0, 1.0]
    )

    # 静态目标物体（立方体）
    obj_body = worldbody.add_body(name="target_box", pos=cfg_runtime.target_pos)
    obj_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.025, 0.025, 0.025],
        rgba=[1.0, 0.2, 0.2, 1.0],
        mass=0.2
    )
    obj_body.add_joint(name="box_free", type=mujoco.mjtJoint.mjJNT_FREE)

    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


# ====================== 主程序 ======================

def main():
    """圆周轨迹跟踪演示主循环.
    
    流程：
        1. 初始化：加载配置，构建环境，实例化控制器与可视化工具。
        2. 主循环：
            a. 轨迹生成：基于时间 `t` 计算圆周位置 `x(t)`。
            b. 混合控制：下发末端位姿目标 + 随机手部关节目标。
            c. 可视化：绘制实际位置（青色球）与目标位置（红色球）。
            d. 步进：物理仿真与渲染同步。
    """
    # 加载配置
    cfg_traj = CircleTrajectoryConfig()
    cfg_style = VisualStyle()
    cfg_runtime = RuntimeConfig()

    try:
        # ===== 1. 初始化系统 =====
        model, data = build_custom_grasp_environment(cfg_runtime)
        
        # 控制器栈初始化
        hardware_interface = HandArmController(model)
        pos_controller = OSC_PositionController(hardware_interface, model)

        # 可视化工具初始化
        traj_vis = TrajectoryVisualizer(cfg_style)

        # 启动 Viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("=== [Simulation] 开始运行：青色点=实际位置，红色点=目标位置 ===")
            
            # 运行时状态变量
            sim_step = 0
            last_hand_update = -cfg_runtime.hand_random_interval
            hand_target = np.zeros(pos_controller.base.HAND_DOF)  # 初始化手部目标

            # ===== 2. 主仿真循环 =====
            while viewer.is_running():
                step_start = time.time()

                # --- A. 渲染管线重置 ---
                # 关键：每帧必须重置几何体计数，否则旧图形会残留
                viewer.user_scn.ngeom = 0

                # --- B. 轨迹规划 (圆周运动) ---
                # 计算当前时间的角度 theta = w * t
                angle = sim_step * model.opt.timestep * cfg_traj.speed
                
                # 计算目标位置：圆心 + 半径 * (cos(theta), sin(theta), 0)
                ee_target_pos = cfg_traj.center + np.array([
                    cfg_traj.radius * np.cos(angle),
                    cfg_traj.radius * np.sin(angle),
                    cfg_traj.z_offset
                ])
                
                # 固定的末端朝向（四元数）
                ee_target_quat = np.array([1.0, 0.0, 0.0, 0.0])

                # --- C. 随机手部动作 ---
                current_time = data.time
                if current_time - last_hand_update > cfg_runtime.hand_random_interval:
                    # 在物理限位范围内生成随机手部姿态
                    low, high = pos_controller.hand_range[:, 0], pos_controller.hand_range[:, 1]
                    hand_target = np.random.uniform(low, high)
                    last_hand_update = current_time

                # --- D. 下发控制指令 ---
                # 使用 OSC 控制器进行末端位姿跟踪
                pos_controller.set_ee_target(
                    data=data,
                    ee_pos_target=ee_target_pos,
                    ee_quat_target=ee_target_quat,
                    hand_target=hand_target
                )

                # --- E. 物理步进 ---
                mujoco.mj_step(model, data)

                # --- F. 调试可视化绘制 ---
                # 1. 更新并绘制实际位置 (青色)
                traj_vis.update_point(data.site_xpos[pos_controller.ee_id])
                traj_vis.draw(viewer)

                # 2. 绘制目标位置 (红色大球)
                # 用于直观对比“目标”与“实际”的跟踪误差
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[cfg_style.target_size, 0, 0],
                    pos=ee_target_pos,
                    mat=np.eye(3).flatten(),
                    rgba=cfg_style.target_rgba
                )
                viewer.user_scn.ngeom += 1

                # --- G. 渲染同步 ---
                viewer.sync()

                # --- H. 时间管理 (固定步长) ---
                sim_step += 1
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    except Exception as e:
        print(f"\n[致命错误] 仿真异常终止: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n=== [Cleanup] 仿真结束，资源已释放 ===")


if __name__ == "__main__":
    main()