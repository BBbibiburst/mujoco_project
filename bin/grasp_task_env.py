"""
自定义抓取环境构建模块.

该模块集成了机械臂与机械手模型，并在此基础上构建了一个包含动态物体、光照及自定义相机的
仿真场景。模块通过 OpenCV 实现了相机视角的实时渲染展示，并提供了演示用的控制逻辑。
"""

import mujoco
from mujoco import mjtGeom, mjtJoint
from robot_arm_system import get_combined_spec, PhysicsConfig, JointPhysicsConfig
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


# ====================== 物理参数配置 ======================

DEFAULT_GRASP_PHYSICS = PhysicsConfig(
    # 机械臂默认物理参数：较高的阻尼确保运动平稳
    arm_defaults=JointPhysicsConfig(
        damping=10.0,        # 关节阻尼系数，抑制振荡
        frictionloss=1,      # 摩擦损耗，模拟关节摩擦
        armature=0.01,       # 电机惯量，影响动态响应
    ),
    # 机械手默认物理参数：低阻尼以实现灵活抓取
    hand_defaults=JointPhysicsConfig(
        damping=0.01,        # 手指关节低阻尼，保证灵活性
        frictionloss=0.01,   # 手指摩擦系数
        armature=0.01,       # 手指电机惯量
    ),
    # 特定关节参数覆盖：拇指旋转关节需要更高阻尼以保持稳定性
    per_joint_overrides={
        "thumb_rotate_act_push_j": JointPhysicsConfig(damping=10.0),
    }
)


# ====================== 环境构建 ======================

def build_custom_grasp_environment(
    physics: PhysicsConfig = DEFAULT_GRASP_PHYSICS,
) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    构建并编译抓取仿真环境，添加外部环境物体与光照.
    
    Args:
        physics: 物理参数配置对象，包含机械臂和机械手的关节物理属性。
                默认为 DEFAULT_GRASP_PHYSICS。
    
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
        physics=physics,                    # 应用物理参数配置
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
    """
    演示主程序：运行抓取仿真并异步显示相机画面.
    
    仿真流程：
        1. 初始化仿真环境和控制器
        2. 启动被动式可视化窗口
        3. 进入实时仿真循环：
           - 前0.5秒：手爪保持张开（零点命令）
           - 0.5秒后：手爪闭合（施加抓取力）
        4. 同步更新物理状态和可视化渲染
    
    """
    try:
        # ----- 1. 环境与控制器初始化 -----
        # 构建仿真环境，获取模型和数据对象
        model, data = build_custom_grasp_environment()
        # 初始化手-臂协调控制器，绑定到当前模型
        controller  = HandArmController(model)

        # ----- 2. 启动可视化与仿真循环 -----
        # 使用被动模式启动MuJoCo查看器，允许外部控制循环
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # 仿真状态变量初始化
            sim_time   = 0.0        # 累计仿真时间（秒）
            step_count = 0          # 物理步进计数器
            close_hand_time = 0.5   # 手爪闭合触发时间（秒）

            # ----- 3. 主仿真循环 -----
            while viewer.is_running():
                # 更新仿真时间
                sim_time   += model.opt.timestep
                step_count += 1

                # --- 控制策略计算 ---
                # 机械臂保持零力矩（静止或重力补偿模式）
                arm_torques = np.zeros(7)  # 7自由度机械臂
                
                # 手爪控制命令：时间触发式闭合策略
                # 6维向量：[食指弯曲, 中指弯曲, 无名指弯曲, 小指弯曲, 拇指弯曲, 拇指旋转]
                hand_commands = (
                    np.array([300.0, 300.0, 300.0, 300.0, 300.0, 0.0])  # 闭合命令（除拇指旋转）
                    if sim_time > close_hand_time
                    else np.zeros(6)  # 张开状态（零点）
                )

                # --- 物理更新 ---
                # 应用控制命令到仿真数据
                controller.apply_control(data, arm_torques, hand_commands)
                # 执行单步物理仿真（积分动力学方程）
                mujoco.mj_step(model, data)
                # 同步更新可视化窗口
                viewer.sync()
                
                # --- 状态输出 ---
                # 实时打印仿真进度（不换行，覆盖输出）
                print(f"\r[Sim] 时间: {sim_time:.2f}s | 手爪命令: {hand_commands} ", end="", flush=True)

    # ----- 异常处理与资源清理 -----
    except Exception as e:
        # 捕获并显示所有未处理异常
        print(f"\n[致命错误] 仿真运行中断: {e}")
        import traceback
        traceback.print_exc()  # 打印完整堆栈跟踪
    finally:
        # 条件化资源清理：根据实际分配的资源进行释放
        # 注意：当前版本无显式资源需要清理（相机渲染已注释）
        # 如需添加相机线程，应在此处设置停止标志并等待线程结束
        print("\n=== [System] 资源已释放，仿真结束 ===")


if __name__ == "__main__":
    main()