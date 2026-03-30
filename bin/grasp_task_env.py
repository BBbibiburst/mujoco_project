"""
自定义抓取环境构建模块.

该模块集成了机械臂与机械手模型，并在此基础上构建了一个包含动态物体、光照及自定义相机的
仿真场景。模块通过 OpenCV 实现了相机视角的实时渲染展示，并提供了演示用的控制逻辑。

核心流程：
1. 模型组装：通过 `get_combined_spec` 加载机器人 URDF/XML。
2. 场景增强：添加光源、目标物体（立方体）和自动计算的俯视相机。
3. 控制演示：在主循环中演示基于 PD 控制的手爪开合与机械臂复位逻辑。
"""

import mujoco
from mujoco import mjtGeom, mjtJoint
from robot_arm_system import get_combined_spec, PhysicsConfig, JointPhysicsConfig
from position_controller import PositionController
from typing import Tuple
import numpy as np

# 内部模块导入
from hand_arm_controller import HandArmController

# ====================== 仿真常量配置 ======================

# 相机分辨率配置 (宽 x 高)，影响渲染质量和计算开销
CAM_WIDTH  = 320   # 相机图像宽度（像素）
CAM_HEIGHT = 240   # 相机图像高度（像素）

# 目标物体初始位置 [x, y, z]（单位：米）
# 位于机械臂前方0.4米，高度0.025米处
TARGET_POS = [0.4, 0.0, 0.025]


# ====================== 环境构建 ======================

def build_custom_grasp_environment() -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    构建并编译抓取仿真环境，添加外部环境物体与光照.
    
    Returns:
        Tuple[mujoco.MjModel, mujoco.MjData]: 
            - model: 编译后的 MuJoCo 模型对象
            - data:  对应的仿真数据对象，包含状态信息
    
    环境组成：
        1. 机械臂+机械手组合体（通过 get_combined_spec 导入）
        2. 顶部定向光源照明
        3. 可抓取的红色立方体目标物体
        4. 俯视全局相机，自动计算视野角度
    """
    print("=== [EnvBuilder] 开始构建自定义抓取环境 ===")

    # 获取组合机器人模型规格，机械手安装在"right_hand"连接点
    # rot_xyz_deg: 机械手相对机械臂的旋转角度（度）
    spec = get_combined_spec(
        rot_xyz_deg=(-90, 0, 0),           # 绕X轴旋转-90度，调整机械手姿态
        attach_point_name="right_hand",     # 机械臂腕部连接点名称
    )
    worldbody = spec.worldbody

    # ----- 配置环境光照 -----
    # 添加顶部平行光源，提供均匀照明
    worldbody.add_light(
        name="top_light",                   # 光源标识名称
        pos=[0.0, 0.0, 2.0],               # 光源位置：场景正上方2米处
        dir=[0.0, 0.0, -1.0],              # 照射方向：垂直向下
        diffuse=[1.0, 1.0, 1.0],           # 漫反射颜色：白色
    )

    # ----- 添加抓取目标物体 -----
    # 创建可自由移动的立方体作为抓取目标
    obj_body = worldbody.add_body(
        name="target_box",                  # 物体名称
        pos=TARGET_POS,                     # 初始位置
    )
    # 添加立方体几何形状和物理属性
    obj_body.add_geom(
        type=mjtGeom.mjGEOM_BOX,           # 几何类型：立方体
        size=[0.025, 0.025, 0.025],        # 半边长：5cm立方体
        rgba=[1.0, 0.2, 0.2, 1.0],         # 颜色：红色不透明
        mass=0.2,                          # 质量：200克
    )
    # 添加6自由度自由关节，允许物体在空间中自由移动和旋转
    obj_body.add_joint(
        name="box_free", 
        type=mjtJoint.mjJNT_FREE           # 自由关节类型：6DOF
    )

    # ----- 自动定位俯视相机 -----
    # 计算机械臂基座与目标物体的中点位置
    base_pos   = np.array([0.0, 0.0, 0.0])  # 机械臂基座位置（假设在原点）
    target_pos = np.array(TARGET_POS)
    mid_point  = (base_pos + target_pos) / 2.0  # 场景中心点
    
    # 相机高度设置
    cam_height = 3.0  # 相机距离地面3米

    # 根据场景几何自动计算合适的视野角度(FOVY)
    # 确保相机能同时看到机械臂基座和目标物体
    dist_to_cover = np.linalg.norm(target_pos - base_pos)  # 基座到目标的水平距离
    # 三角函数计算：fovy = 2 * arctan(覆盖范围 / 相机高度)
    calc_fovy = 2 * np.degrees(np.arctan2((dist_to_cover / 2) * 2.0, cam_height))

    # 添加俯视相机到场景
    worldbody.add_camera(
        name="downward_cam",                # 相机名称
        pos=[mid_point[0], mid_point[1], cam_height],  # 位置：中点正上方
        xyaxes=[0, 1, 0, -1, 0, 0],        # 相机坐标系朝向：俯视-Z轴
        fovy=calc_fovy,                     # 垂直视野角度（自动计算）
    )

    print("[EnvBuilder] 模型构建完成，正在编译并生成仿真对象...")
    # 编译模型规格，生成可仿真的模型对象
    model = spec.compile()
    # 创建对应的仿真数据对象，存储状态变量
    data  = mujoco.MjData(model)
    return model, data


# ====================== 主程序 ======================

def main():
    try:
        # ===== 1. 初始化 =====
        model, data = build_custom_grasp_environment()

        # ⭐ 底层控制器
        controller = HandArmController(model)

        # ⭐ 新增：位置控制器
        pos_controller = PositionController(controller, model)

        # ===== 2. viewer =====
        with mujoco.viewer.launch_passive(model, data) as viewer:

            sim_time = 0.0
            circle_hand_time = 4

            # ⭐ 初始目标（机械臂不动）
            arm_target_origin = data.qpos[pos_controller.arm_qpos_ids].copy()

            while viewer.is_running():
                sim_time += model.opt.timestep

                # =========================
                # ⭐ 手爪目标（位置控制！）
                # =========================
                # 逻辑：计算当前处于第几个时间段
                current_period = int(sim_time / circle_hand_time)

                if current_period % 2 == 0:
                    # 偶数时间段 -> 闭合
                    hand_target = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.0])
                    arm_target = np.array([1, 1, 1, 1, 1, 1, 1])
                else:
                    # 奇数时间段 -> 张开
                    hand_target = np.zeros(6)
                    arm_target = arm_target_origin

                # =========================
                # ⭐ 使用 PD 控制器（关键）
                # =========================
                pos_controller.set_target(
                    data,
                    arm_target=arm_target,
                    hand_target=hand_target
                )

                # ===== 物理步进 =====
                mujoco.mj_step(model, data)
                viewer.sync()

                # ===== 调试输出 =====
                # 打印真实 torque（验证 gear 是否正确）
                ctrl = data.ctrl[controller.all_indices]
                torque = ctrl * controller.gear[controller.all_indices]

                print(
                    f"\r[Sim {sim_time:.2f}s] hand={hand_target.round(4)}",
                    end=" " * 20,   # 清除残留字符
                    flush=True
                )

    except Exception as e:
        print(f"\n[致命错误] {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n=== 仿真结束 ===")

if __name__ == "__main__":
    main()