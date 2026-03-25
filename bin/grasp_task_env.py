"""
自定义抓取环境构建模块.

该模块集成了机械臂与机械手模型，并在此基础上构建了一个包含动态物体、光照及自定义相机的
仿真场景。模块通过 Matplotlib 实现了相机视角的实时渲染展示，并提供了演示用的控制逻辑。
"""

import mujoco
from mujoco import mjtGeom, mjtJoint
from robot_arm_system import get_combined_spec, PhysicsConfig, JointPhysicsConfig  # 新增导入
from typing import Tuple, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

# 内部模块导入
from hand_arm_controller import HandArmController

# ====================== 仿真常量配置 ======================
CAM_WIDTH = 320
CAM_HEIGHT = 240
TARGET_POS = [0.4, 0.0, 0.025]  # 目标物体初始位置

# ====================== 物理参数配置 ======================
# 机械臂：高刚度 + 高阻尼，使各关节在无控制器输出时抵抗重力下垂
# 机械手：适中刚度与阻尼，保持手指自然张开状态
DEFAULT_GRASP_PHYSICS = PhysicsConfig(
    arm_defaults=JointPhysicsConfig(
        stiffness=500.0,   # 较高弹性刚度，关节趋向 ref=0 位置，抵抗重力
        damping=50.0,      # 较高阻尼，抑制关节震荡
        armature=0.01,     # 增加等效惯量，提升数值稳定性
        frictionloss=0.5,  # 轻微摩擦，辅助保持静止位姿
    ),
    hand_defaults=JointPhysicsConfig(
        stiffness=1.0,     # 手指轻微刚度，自然张开
        damping=0.1,       # 适中阻尼，防止手指过度震荡
        armature=0.001,    # 手指质量较小，armature 设为较小值以提升响应速度
        frictionloss=0.1,  # 轻微摩擦，辅助保持手指位置稳定
    ),
)


def build_custom_grasp_environment(
    physics: PhysicsConfig = DEFAULT_GRASP_PHYSICS,  # 默认使用抗重力参数
) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    构建并编译抓取仿真环境，添加外部环境物体与光照.

    该函数在合并好的机械臂模型基础上，手动操作 MjSpec 对象添加环境元素。
    这种方式允许在模型正式编译 (Compile) 前灵活注入物理实体。

    Args:
        physics: 物理参数配置对象，默认使用 DEFAULT_GRASP_PHYSICS（抗重力下垂）。
                 传入 None 可退回 XML 原始物理参数；传入自定义 PhysicsConfig 可精细调节。

    Returns:
        tuple: (model, data) 编译后的 MuJoCo 模型实例与仿真数据对象。
    
    Notes:
        - 挂载点默认为 'right_hand'。
        - 物体添加了 FreeJoint 以模拟真实的自由落体与碰撞物理特性。
        - 抗重力下垂的原理：stiffness 使关节趋向 ref 角度（默认 0），
          damping 抑制震荡，两者共同替代显式重力补偿控制器的作用。
    """
    print("=== [EnvBuilder] 开始构建自定义抓取环境 ===")

    # 1. 获取合并了机械手的机械臂 Spec 对象，同时注入物理参数
    spec = get_combined_spec(
        rot_xyz_deg=(-90, 0, 0),
        attach_point_name="right_hand",
        physics=physics,          # 传入物理参数
    )
    worldbody = spec.worldbody

    # 2. 配置环境光照 (提供阴影与深度感)
    worldbody.add_light(
        name="top_light",
        pos=[0.0, 0.0, 2.0],
        dir=[0.0, 0.0, -1.0],
        diffuse=[1.0, 1.0, 1.0],
    )

    # 3. 添加抓取目标物体 (红色立方体)
    obj_body = worldbody.add_body(
        name="target_box",
        pos=TARGET_POS
    )
    obj_body.add_geom(
        type=mjtGeom.mjGEOM_BOX,
        size=[0.025, 0.025, 0.025],
        rgba=[1.0, 0.2, 0.2, 1.0],
        mass=0.2,
    )
    # 为物体添加自由接头，使其受重力影响并可被碰撞拨动
    obj_body.add_joint(
        name="box_free",
        type=mjtJoint.mjJNT_FREE
    )

    # 4. 自动定位逻辑
    base_pos = np.array([0.0, 0.0, 0.0])
    target_pos = np.array(TARGET_POS)
    
    mid_point = (base_pos + target_pos) / 2.0
    cam_height = 3
    
    dist_to_cover = np.linalg.norm(target_pos - base_pos)
    calc_fovy = 2 * np.degrees(np.arctan2((dist_to_cover / 2) * 2.0, cam_height))

    # 5. 添加动态配置的相机
    worldbody.add_camera(
        name="downward_cam",
        pos=[mid_point[0], mid_point[1], cam_height],
        xyaxes=[0, 1, 0, -1, 0, 0], 
        fovy=calc_fovy
    )

    print("[EnvBuilder] 模型构建完成，正在编译并生成仿真对象...")
    model = spec.compile()
    data = mujoco.MjData(model)

    return model, data


def main():
    """
    演示主程序：运行抓取仿真并实时显示相机画面.
    """
    try:
        # 1. 环境与控制器初始化（使用默认抗重力物理参数）
        model, data = build_custom_grasp_environment()
        controller = HandArmController(model)
        
        # 2. 离屏渲染上下文准备 (Off-screen Rendering)
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "downward_cam")
        gl_ctx = mujoco.GLContext(CAM_WIDTH, CAM_HEIGHT)
        gl_ctx.make_current()
        
        scn = mujoco.MjvScene(model, maxgeom=100)
        ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        # 3. Matplotlib 交互式窗口初始化
        plt.ion()  
        fig, ax = plt.subplots(figsize=(5, 4))
        im_plot = ax.imshow(np.zeros((CAM_HEIGHT, CAM_WIDTH, 3)))
        ax.set_title("Robot Eye View: downward_cam")
        plt.axis('off')
        plt.show(block=False)

        # 4. 进入仿真循环
        with mujoco.viewer.launch_passive(model, data) as viewer:
            sim_time = 0.0
            close_hand_time = 1.0
            
            while viewer.is_running():
                sim_time += model.opt.timestep
                
                # --- 控制策略计算 ---
                arm_torques = np.zeros(7)
                if sim_time > close_hand_time:
                    hand_commands = np.array([3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 0.0])
                else:
                    hand_commands = np.zeros(6)

                controller.apply_control(data, arm_torques, hand_commands)
                print(f"\r[Sim] 时间: {sim_time:.2f}s | 手爪命令: {hand_commands} ", end="")
                mujoco.mj_step(model, data)
                viewer.sync()

                # --- 图像渲染与采集逻辑 ---
                cam = mujoco.MjvCamera()
                cam.fixedcamid = cam_id
                cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

                mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, 
                                       cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
                mujoco.mjr_render(mujoco.MjrRect(0, 0, CAM_WIDTH, CAM_HEIGHT), scn, ctx)
                
                rgb_buffer = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
                mujoco.mjr_readPixels(rgb_buffer, None, mujoco.MjrRect(0, 0, CAM_WIDTH, CAM_HEIGHT), ctx)
                
                image = np.flipud(rgb_buffer)
                im_plot.set_data(image)
                
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)

    except Exception as e:
        print(f"\n[致命错误] 仿真运行中断: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close('all')
        print("=== [System] 资源已释放，仿真结束 ===")


if __name__ == "__main__":
    main()