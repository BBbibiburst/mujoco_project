"""
自定义抓取环境构建模块.

该模块集成了机械臂与机械手模型，并在此基础上构建了一个包含动态物体、光照及自定义相机的
仿真场景。模块通过 Matplotlib 实现了相机视角的实时渲染展示，并提供了演示用的控制逻辑。
"""

import mujoco
from mujoco import mjtGeom, mjtJoint
from robot_arm_system import get_combined_spec
from typing import Tuple, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

# 内部模块导入
from hand_arm_controller import HandArmController

# ====================== 仿真常量配置 ======================
CAM_WIDTH = 320
CAM_HEIGHT = 240
TARGET_POS = [0.4, 0.0, 0.025]  # 目标物体初始位置


def build_custom_grasp_environment() -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    构建并编译抓取仿真环境，添加外部环境物体与光照.

    该函数在合并好的机械臂模型基础上，手动操作 MjSpec 对象添加环境元素。
    这种方式允许在模型正式编译 (Compile) 前灵活注入物理实体。

    Returns:
        tuple: (model, data) 编译后的 MuJoCo 模型实例与仿真数据对象。
    
    Notes:
        - 挂载点默认为 'right_hand'。
        - 物体添加了 FreeJoint 以模拟真实的自由落体与碰撞物理特性。
    """
    print("=== [EnvBuilder] 开始构建自定义抓取环境 ===")

    # 1. 获取合并了机械手的机械臂 Spec 对象
    spec = get_combined_spec(
        rot_xyz_deg=(-90, 0, 0),
        attach_point_name="right_hand"
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

    # 1. 自动定位逻辑
    # 读取机械臂底座位置与目标位置，计算相机的最佳位置与视野覆盖范围
    base_pos = np.array([0.0, 0.0, 0.0])
    target_pos = np.array(TARGET_POS)
    
    # 计算中点
    mid_point = (base_pos + target_pos) / 2.0
    
    # 设定相机高度 (高度决定了视野覆盖范围)
    cam_height = 3
    
    # 2. 计算视野 (fovy)
    # 确保从底座到目标的距离 (0.4m) 刚好能被看到
    dist_to_cover = np.linalg.norm(target_pos - base_pos)
    # 角度 = 2 * arctan( 覆盖范围的一半 / 高度 )
    # 再乘一个系数 (例如 1.2) 留出白边，防止机械臂贴边太死
    calc_fovy = 2 * np.degrees(np.arctan2((dist_to_cover / 2) * 2.0, cam_height))

    # 3. 添加动态配置的相机
    worldbody.add_camera(
        name="downward_cam",
        pos=[mid_point[0], mid_point[1], cam_height],
        # xyaxes 的含义：[右方向向量, 上方向向量]
        # [1, 0, 0] 表示图像右侧指向世界 X 正方向
        # [0, 1, 0] 表示图像上方指向世界 Y 正方向
        # 因为相机默认看 -Z，这样设置后：
        # 目标 (x=0.4) 会在画面右侧，底座 (x=0) 会在画面左侧。
        # 如果你想让底座在“底部”，我们需要交换 X 和 Y 的指向：
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
        # 1. 环境与控制器初始化
        model, data = build_custom_grasp_environment()
        controller = HandArmController(model)
        
        # 2. 离屏渲染上下文准备 (Off-screen Rendering)
        # 获取相机 ID 并创建 GL 上下文以进行像素读取
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "downward_cam")
        gl_ctx = mujoco.GLContext(CAM_WIDTH, CAM_HEIGHT)
        gl_ctx.make_current()
        
        # 初始化 MuJoCo 渲染组件
        scn = mujoco.MjvScene(model, maxgeom=100)
        ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        # 3. Matplotlib 交互式窗口初始化 (替代 OpenCV 显示画面)
        plt.ion()  
        fig, ax = plt.subplots(figsize=(5, 4))
        im_plot = ax.imshow(np.zeros((CAM_HEIGHT, CAM_WIDTH, 3)))
        ax.set_title("Robot Eye View: downward_cam")
        plt.axis('off')
        plt.show(block=False)

        # 4. 进入仿真循环
        with mujoco.viewer.launch_passive(model, data) as viewer:
            sim_time = 0.0
            close_hand_time = 1.0  # 定义何时开始合拢机械手
            
            while viewer.is_running():
                sim_time += model.opt.timestep
                
                # --- 控制策略计算 ---
                arm_torques = np.zeros(7) # 保持臂部静止或应用平衡重力
                if sim_time > close_hand_time:
                    # 给定合拢指令，模拟抓取动作
                    hand_commands = np.array([300.0, 300.0, 300.0, 300.0, 300.0, 0.0])
                else:
                    hand_commands = np.zeros(6)

                # 应用控制并更新物理状态
                controller.apply_control(data, arm_torques, hand_commands)
                mujoco.mj_step(model, data)
                viewer.sync()

                # --- 图像渲染与采集逻辑 ---
                cam = mujoco.MjvCamera()
                cam.fixedcamid = cam_id
                cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

                # 更新场景并执行离屏渲染
                mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, 
                                       cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
                mujoco.mjr_render(mujoco.MjrRect(0, 0, CAM_WIDTH, CAM_HEIGHT), scn, ctx)
                
                # 从显存读取像素：MuJoCo 的坐标系原点在左下角
                rgb_buffer = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
                mujoco.mjr_readPixels(rgb_buffer, None, mujoco.MjrRect(0, 0, CAM_WIDTH, CAM_HEIGHT), ctx)
                
                # 图像处理：上下翻转以符合常规显示习惯
                image = np.flipud(rgb_buffer)
                im_plot.set_data(image)
                
                # 刷新 Matplotlib 画布，实现“实时”预览
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)  # 释放控制权给 GUI 线程

    except Exception as e:
        print(f"\n[致命错误] 仿真运行中断: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close('all')
        print("=== [System] 资源已释放，仿真结束 ===")


if __name__ == "__main__":
    main()