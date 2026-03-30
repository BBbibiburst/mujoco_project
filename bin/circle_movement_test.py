import mujoco
import mujoco.viewer
from mujoco import mjtGeom, mjtJoint
import numpy as np
import time
from typing import Tuple

# ====================== 轨迹可视化辅助类 ======================
class TrajectoryVisualizer:
    def __init__(self, rgba=[0, 0.7, 1, 0.8], size=0.005):
        """
        用于在 Viewer 中绘制单个调试小球
        rgba: 小球颜色 [R, G, B, Alpha]
        size: 小球半径
        """
        self.current_pos = None
        self.rgba = rgba
        self.size = size

    def update_point(self, pos):
        """更新当前位置"""
        self.current_pos = pos.copy()

    def draw(self, viewer):
        """只渲染当前的一个点"""
        if self.current_pos is not None:
            if viewer.user_scn.ngeom < 1000:
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[self.size, 0, 0],
                    pos=self.current_pos,
                    mat=np.eye(3).flatten(),
                    rgba=self.rgba
                )
                viewer.user_scn.ngeom += 1

# ====================== 仿真配置 ======================
TARGET_CENTER = np.array([0.5, 0.0, 0.4])  # 圆周运动的圆心
CIRCLE_RADIUS = 0.15                       # 圆周半径
CIRCLE_SPEED  = 1.5                        # 旋转角速度
HAND_RANDOM_INTERVAL = 1.0 
TARGET_POS = [0.4, 0.0, 0.025]

# --- 导入模块 (请确保这些文件存在) ---
from robot_arm_system import get_combined_spec, PhysicsConfig, JointPhysicsConfig
from position_controller import PositionController
from hand_arm_controller import HandArmController

DEFAULT_GRASP_PHYSICS = PhysicsConfig(
    arm_defaults=JointPhysicsConfig(damping=100.0, frictionloss=0.1, armature=0.01),
    hand_defaults=JointPhysicsConfig(damping=0.01, frictionloss=0.01, armature=0.01),
    per_joint_overrides={"thumb_rotate_act_push_j": JointPhysicsConfig(damping=10.0)}
)

def build_custom_grasp_environment(physics: PhysicsConfig = DEFAULT_GRASP_PHYSICS) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    spec = get_combined_spec(rot_xyz_deg=(-90, 0, 0), attach_point_name="right_hand", physics=physics)
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
        model, data = build_custom_grasp_environment()
        base_controller = HandArmController(model)
        pos_controller = PositionController(base_controller, model)

        # 初始化可视化工具：只显示一个点
        # 颜色为青蓝色，半径 5mm
        traj_vis = TrajectoryVisualizer(rgba=[0, 1, 1, 0.8], size=0.005)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            sim_time = 0.0
            last_hand_update = -HAND_RANDOM_INTERVAL
            hand_target = np.zeros(6)

            print("=== 仿真开始：青色点展示末端当前位置，红色点展示目标位置 ===")

            while viewer.is_running():
                step_start = time.time()
                
                # --- 关键修复：每帧开始前清空 user_scn 的几何体计数 ---
                viewer.user_scn.ngeom = 0

                # --- A. 计算圆周目标 ---
                angle = sim_time * CIRCLE_SPEED
                ee_target_pos = TARGET_CENTER + np.array([
                    CIRCLE_RADIUS * np.cos(angle),
                    CIRCLE_RADIUS * np.sin(angle),
                    0 
                ])
                # 固定的末端朝向（四元数：[w, x, y, z]）
                ee_target_quat = np.array([1.0, 0.0, 0.0, 0.0]) 

                # --- B. 随机手部动作 ---
                if sim_time - last_hand_update > HAND_RANDOM_INTERVAL:
                    low, high = pos_controller.hand_range[:, 0], pos_controller.hand_range[:, 1]
                    hand_target = np.random.uniform(low, high)
                    last_hand_update = sim_time

                # --- C. 控制逻辑 ---
                pos_controller.set_ee_target(data, ee_pos_target=ee_target_pos, 
                                           ee_quat_target=ee_target_quat, hand_target=hand_target)

                # --- D. 物理步进 ---
                mujoco.mj_step(model, data)

                # --- E. 调试可视化 ---
                # 1. 更新末端实际位置 (每一帧都更新，保证点跟随机械臂)
                traj_vis.update_point(data.site_xpos[pos_controller.ee_id])
                
                # 2. 绘制当前位置点
                traj_vis.draw(viewer)

                # 3. 额外绘制：当前目标点 (大一些的红色半透明球)
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.015, 0, 0],
                    pos=ee_target_pos,
                    mat=np.eye(3).flatten(),
                    rgba=[1, 0, 0, 0.4]
                )
                viewer.user_scn.ngeom += 1

                # 同步到画面
                viewer.sync()

                # --- 时间管理 ---
                sim_time += model.opt.timestep
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