"""
自定义抓取环境构建模块.

该模块集成了机械臂与机械手模型，并在此基础上构建了一个包含动态物体、光照及自定义相机的
仿真场景。模块通过 OpenCV 实现了相机视角的实时渲染展示，并提供了演示用的控制逻辑。
渲染与仿真解耦，cv2.imshow 在独立线程中异步刷新，不阻塞物理步进。

[修改点]: 新增 ENABLE_CAMERA_RENDERING 常量，用于一键开关相机渲染功能。
          当设置为 False 时，完全跳过渲染线程启动、OpenGL 上下文创建及图像采集步骤，
          显著提升仿真运行速度（适用于纯数据采集或无头模式）。
"""

import threading
import time
import queue
import mujoco
from mujoco import mjtGeom, mjtJoint
from robot_arm_system import get_combined_spec, PhysicsConfig, JointPhysicsConfig
from typing import Tuple, Optional
import numpy as np
import cv2

# 内部模块导入
from hand_arm_controller import HandArmController

# ====================== 仿真常量配置 ======================
CAM_WIDTH  = 320
CAM_HEIGHT = 240
TARGET_POS = [0.4, 0.0, 0.025]  # 目标物体初始位置

# 【核心修改】相机渲染开关常量
# 设置为 False 将完全禁用相机渲染线程和图像采集，大幅提升仿真速度
ENABLE_CAMERA_RENDERING = False  

# 每 N 个物理步采集一次图像写入共享缓冲区
# 仅在 ENABLE_CAMERA_RENDERING=True 时生效
RENDER_EVERY_N_STEPS = 10

# ====================== 物理参数配置 ======================
DEFAULT_GRASP_PHYSICS = PhysicsConfig(
    arm_defaults=JointPhysicsConfig(
        damping=10.0,        
        frictionloss=1,      
        armature=0.01,       
    ),
    hand_defaults=JointPhysicsConfig(
        damping=0.01,      
        frictionloss=0.01, 
        armature=0.01,     
    ),
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
    (此处逻辑未变，相机定义始终存在于模型中，只是是否被读取渲染由主程序控制)
    """
    print("=== [EnvBuilder] 开始构建自定义抓取环境 ===")

    spec = get_combined_spec(
        rot_xyz_deg=(-90, 0, 0),
        attach_point_name="right_hand",
        physics=physics,
    )
    worldbody = spec.worldbody

    # 配置环境光照
    worldbody.add_light(
        name="top_light",
        pos=[0.0, 0.0, 2.0],
        dir=[0.0, 0.0, -1.0],
        diffuse=[1.0, 1.0, 1.0],
    )

    # 添加抓取目标物体
    obj_body = worldbody.add_body(name="target_box", pos=TARGET_POS)
    obj_body.add_geom(
        type=mjtGeom.mjGEOM_BOX,
        size=[0.025, 0.025, 0.025],
        rgba=[1.0, 0.2, 0.2, 1.0],
        mass=0.2,
    )
    obj_body.add_joint(name="box_free", type=mjtJoint.mjJNT_FREE)

    # 自动定位逻辑
    base_pos   = np.array([0.0, 0.0, 0.0])
    target_pos = np.array(TARGET_POS)
    mid_point  = (base_pos + target_pos) / 2.0
    cam_height = 3.0

    # 计算视野
    dist_to_cover = np.linalg.norm(target_pos - base_pos)
    calc_fovy = 2 * np.degrees(np.arctan2((dist_to_cover / 2) * 2.0, cam_height))

    # 添加动态配置的相机
    worldbody.add_camera(
        name="downward_cam",
        pos=[mid_point[0], mid_point[1], cam_height],
        xyaxes=[0, 1, 0, -1, 0, 0],
        fovy=calc_fovy,
    )

    print("[EnvBuilder] 模型构建完成，正在编译并生成仿真对象...")
    model = spec.compile()
    data  = mujoco.MjData(model)
    return model, data


# ====================== 异步渲染线程（cv2 版本）======================

class AsyncCameraRenderer:
    """
    在独立线程中驱动 cv2 窗口，消费主线程写入的图像帧.
    """

    def __init__(self, width: int, height: int, target_fps: float = 30.0):
        self.width    = width
        self.height   = height
        self.interval = 1.0 / target_fps

        self._queue      = queue.Queue(maxsize=1)
        self._stop_flag  = threading.Event()
        self._thread     = threading.Thread(target=self._render_loop, daemon=True)
        self.quit_requested = threading.Event()

    def start(self):
        self._thread.start()

    def stop(self):
        """通知渲染线程退出并等待窗口关闭."""
        self._stop_flag.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    def push_frame(self, frame: np.ndarray):
        """非阻塞地推送最新帧."""
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass

    def _render_loop(self):
        win_name = "Robot Eye View: downward_cam"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, self.width * 2, self.height * 2)

        blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.imshow(win_name, blank)

        while not self._stop_flag.is_set():
            try:
                frame = self._queue.get(timeout=self.interval)
            except queue.Empty:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.quit_requested.set()
                continue

            if frame is None:
                break

            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(win_name, bgr_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.quit_requested.set()

        cv2.destroyAllWindows()


# ====================== 主程序 ======================

def main():
    """
    演示主程序：运行抓取仿真并异步显示相机画面.
    """
    renderer: Optional[AsyncCameraRenderer] = None
    
    # 渲染相关的变量预声明
    gl_ctx = None
    scn = None
    ctx = None
    cam = None
    rgb_buffer = None
    flipped_buffer = None
    rect = None
    cam_id = -1

    try:
        # 1. 环境与控制器初始化
        model, data = build_custom_grasp_environment()
        controller  = HandArmController(model)

        # 2. 【条件化】离屏渲染上下文准备
        if ENABLE_CAMERA_RENDERING:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "downward_cam")
            if cam_id == -1:
                raise ValueError("Camera 'downward_cam' not found in model!")
            
            gl_ctx = mujoco.GLContext(CAM_WIDTH, CAM_HEIGHT)
            gl_ctx.make_current()

            scn = mujoco.MjvScene(model, maxgeom=100)
            ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

            cam = mujoco.MjvCamera()
            cam.fixedcamid = cam_id
            cam.type       = mujoco.mjtCamera.mjCAMERA_FIXED

            rgb_buffer     = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
            flipped_buffer = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
            rect           = mujoco.MjrRect(0, 0, CAM_WIDTH, CAM_HEIGHT)

            # 3. 【条件化】启动异步渲染线程
            renderer = AsyncCameraRenderer(CAM_WIDTH, CAM_HEIGHT, target_fps=30.0)
            renderer.start()
            print("[Sim] 相机渲染已启用 (OpenCV 异步线程运行中)")
        else:
            print("[Sim] 相机渲染已禁用 (纯仿真模式，性能最大化)")

        # 4. 进入仿真主循环
        with mujoco.viewer.launch_passive(model, data) as viewer:
            sim_time   = 0.0
            step_count = 0
            close_hand_time = 0.5 

            while viewer.is_running():
                # 【条件化】检查渲染窗口是否收到 'q' 退出请求
                if ENABLE_CAMERA_RENDERING and renderer and renderer.quit_requested.is_set():
                    print("\n[Sim] 收到退出请求，停止仿真...")
                    break

                sim_time   += model.opt.timestep
                step_count += 1

                # --- 控制策略计算 ---
                arm_torques = np.zeros(7)
                hand_commands = (
                    np.array([300.0, 300.0, 300.0, 300.0, 300.0, 0.0])
                    if sim_time > close_hand_time
                    else np.zeros(6)
                )

                # 应用控制并更新物理状态
                controller.apply_control(data, arm_torques, hand_commands)
                mujoco.mj_step(model, data)
                viewer.sync()

                # --- 【条件化】降频图像采集 ---
                if ENABLE_CAMERA_RENDERING and renderer:
                    if step_count % RENDER_EVERY_N_STEPS == 0:
                        # 确保 OpenGL 上下文当前有效 (虽然在单线程循环中通常不需要重复 make_current，但为了安全)
                        # gl_ctx.make_current() 
                        
                        mujoco.mjv_updateScene(
                            model, data, mujoco.MjvOption(), None,
                            cam, mujoco.mjtCatBit.mjCAT_ALL, scn,
                        )
                        mujoco.mjr_render(rect, scn, ctx)
                        mujoco.mjr_readPixels(rgb_buffer, None, rect, ctx)

                        np.copyto(flipped_buffer, rgb_buffer[::-1])
                        renderer.push_frame(flipped_buffer.copy())

                # 简单的进度打印
                print(f"\r[Sim] 时间: {sim_time:.2f}s | 手爪命令: {hand_commands} ", end="", flush=True)

    except Exception as e:
        print(f"\n[致命错误] 仿真运行中断: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 【条件化】资源清理
        if renderer is not None:
            renderer.stop()
        
        if gl_ctx is not None:
            # GLContext 通常在销毁时自动清理，但显式释放是个好习惯
            pass 
            
        print("\n=== [System] 资源已释放，仿真结束 ===")


if __name__ == "__main__":
    main()