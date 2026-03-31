"""
圆周轨迹跟踪与实时可视化仿真模块.

该模块在之前的抓取环境基础上，增加了复杂的轨迹控制逻辑与实时调试可视化功能：
1. 轨迹生成：控制机械臂末端沿指定的圆周轨迹运动。
2. 实时可视化：利用 MuJoCo 的用户几何体接口，在 Viewer 中实时绘制末端实际位置与目标位置。
3. 混合控制：同时演示末端轨迹跟踪与手部随机动作。
"""
import mujoco
import mujoco.viewer
from mujoco import mjtGeom, mjtJoint
import numpy as np
import time
from typing import Tuple
from robot_arm_system import get_combined_spec, PhysicsConfig, JointPhysicsConfig
from position_controller import OSC_PositionController
from hand_arm_controller import HandArmController

# ====================== 轨迹可视化辅助类 ======================
class TrajectoryVisualizer:
    """
    调试几何体绘制工具类。
    
    利用 MuJoCo 的 `user_scn` 接口在仿真画面中绘制自定义几何体。
    该类专门用于绘制单个动态球体，用于标记关键点（如末端执行器位置）。
    """
    def __init__(self, rgba=[0, 0.7, 1, 0.8], size=0.005):
        """
        Args:
            rgba: 颜色与透明度配置 [R, G, B, Alpha]。
            size: 球体半径 (米)。
        """
        self.current_pos = None
        self.rgba = rgba
        self.size = size

    def update_point(self, pos):
        """更新当前要绘制的位置坐标"""
        self.current_pos = pos.copy()

    def draw(self, viewer):
        """
        将当前点渲染到 Viewer 场景中。
        
        注意：
        - 使用 `viewer.user_scn.ngeom` 管理几何体数量。
        - 必须在每帧循环开始时重置 `ngeom`，否则几何体会累积。
        """
        if self.current_pos is not None:
            # 安全检查：防止超出几何体缓冲区上限
            if viewer.user_scn.ngeom < 1000:
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[self.size, 0, 0],  # MuJoCo 球体只需第一个参数作为半径
                    pos=self.current_pos,
                    mat=np.eye(3).flatten(), # 单位旋转矩阵
                    rgba=self.rgba
                )
                viewer.user_scn.ngeom += 1

# ====================== 仿真配置 ======================
# 圆周运动参数
TARGET_CENTER = np.array([0.5, 0.0, 0.4])  # 圆周运动的圆心坐标 [x, y, z]
CIRCLE_RADIUS = 0.15                       # 圆周半径 (米)
CIRCLE_SPEED  = 1.5                        # 旋转角速度 (rad/s)

# 控制与场景参数
HAND_RANDOM_INTERVAL = 1.0  # 手部动作随机变化的间隔时间 (秒)
TARGET_POS = [0.4, 0.0, 0.025] # 静态物体位置


def build_custom_grasp_environment() -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """构建基础仿真环境（复用之前的逻辑）"""
    spec = get_combined_spec(rot_xyz_deg=(-90, 0, 0), attach_point_name="right_hand")
    worldbody = spec.worldbody
    worldbody.add_light(name="top_light", pos=[0.0, 0.0, 2.0], dir=[0.0, 0.0, -1.0], diffuse=[1.0, 1.0, 1.0])

    obj_body = worldbody.add_body(name="target_box", pos=TARGET_POS)
    obj_body.add_geom(type=mjtGeom.mjGEOM_BOX, size=[0.025, 0.025, 0.025], rgba=[1.0, 0.2, 0.2, 1.0], mass=0.2)
    obj_body.add_joint(name="box_free", type=mjtJoint.mjJNT_FREE)

    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data

def main():
    try:
        # ===== 1. 系统初始化 =====
        model, data = build_custom_grasp_environment()
        base_controller = HandArmController(model)
        pos_controller = OSC_PositionController(base_controller, model)

        # 初始化可视化工具：用于绘制末端实际位置
        # 颜色为青蓝色，半径 5mm
        traj_vis = TrajectoryVisualizer(rgba=[0, 1, 1, 0.8], size=0.005)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            sim_time = 0.0
            last_hand_update = -HAND_RANDOM_INTERVAL
            hand_target = np.zeros(6)

            print("=== 仿真开始：青色点展示末端当前位置，红色点展示目标位置 ===")

            # ======================
            # 主仿真循环
            # ======================
            while viewer.is_running():
                step_start = time.time()
                
                # --- A. 渲染管线重置 ---
                # 关键步骤：每帧开始前必须将用户几何体计数清零。
                # 否则上一帧绘制的球体会保留，导致画面出现拖尾或重叠。
                viewer.user_scn.ngeom = 0

                # --- B. 轨迹规划 (目标生成) ---
                # 计算当前时刻的角度
                angle = sim_time * CIRCLE_SPEED
                # 基于三角函数生成圆周轨迹坐标
                ee_target_pos = TARGET_CENTER + np.array([
                    CIRCLE_RADIUS * np.cos(angle),
                    CIRCLE_RADIUS * np.sin(angle),
                    0 
                ])
                # 固定的末端朝向（四元数：[w, x, y, z]）
                ee_target_quat = np.array([1.0, 0.0, 0.0, 0.0]) 

                # --- C. 随机手部动作逻辑 ---
                if sim_time - last_hand_update > HAND_RANDOM_INTERVAL:
                    # 在物理限位范围内生成随机手部姿态
                    low, high = pos_controller.hand_range[:, 0], pos_controller.hand_range[:, 1]
                    hand_target = np.random.uniform(low, high)
                    last_hand_update = sim_time

                # --- D. 下发控制指令 ---
                # 调用 IK 控制器计算关节目标，并经由 PD 控制器下发力矩
                pos_controller.set_ee_target(data, ee_pos_target=ee_target_pos, 
                                           ee_quat_target=ee_target_quat, hand_target=hand_target)

                # --- E. 物理步进 ---
                mujoco.mj_step(model, data)

                # --- F. 调试可视化绘制 ---
                # 1. 更新可视化工具的坐标为当前末端实际位置
                traj_vis.update_point(data.site_xpos[pos_controller.ee_id])
                
                # 2. 绘制当前位置点 (青色小球)
                traj_vis.draw(viewer)

                # 3. 绘制目标位置点 (红色大球，半透明)
                # 用于直观对比“目标”与“实际”的跟踪误差
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.015, 0, 0], # 半径 1.5cm，比实际位置点稍大
                    pos=ee_target_pos,
                    mat=np.eye(3).flatten(),
                    rgba=[1, 0, 0, 0.4] # 红色，透明度 0.4
                )
                viewer.user_scn.ngeom += 1

                # 同步渲染到画面
                viewer.sync()

                # --- G. 时间管理 (固定步长) ---
                sim_time += model.opt.timestep
                # 计算剩余睡眠时间，保持仿真节奏
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback; traceback.print_exc()
    finally:
        print("\n=== 仿真结束 ===")

if __name__ == "__main__":
    main()