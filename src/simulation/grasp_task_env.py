"""
自定义抓取环境构建与演示模块.

该模块实现了一个完整的 MuJoCo 仿真抓取环境，包含以下核心组件：
- 环境构建器：`build_custom_grasp_environment`，负责组装机器人、目标物体、光源及相机。
- 演示控制器：`main`，集成了 OSC 控制器与预录的手部轨迹数据，实现闭环控制演示。

设计要点：
- **场景配置化**：将相机参数与物体位置提取为配置常量，便于调整。
- **资源管理**：使用上下文管理器（with statement）确保 Viewer 资源正确释放。
- **数据预处理**：在加载阶段对 CSV 轨迹数据进行归一化与广播处理，避免运行时计算开销。
- **鲁棒性**：包含全局异常捕获，防止仿真崩溃导致无法查看错误信息。

依赖：
    mujoco: 物理引擎核心。
    numpy: 数值计算与数组操作。
    robot_arm_system: 机器人模型组装工具。
    position_controller: 运动控制算法实现。
"""

import csv
from pathlib import Path
import mujoco
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass, field
from src.simulation.robot_arm_system import get_combined_spec
from src.controllers.position_controller import OSC_PositionController
from src.controllers.hand_arm_controller import HandArmController


# ====================== 配置数据类 ======================

@dataclass
class CameraConfig:
    """相机硬件与渲染参数配置.
    
    Attributes:
        width: 图像缓冲区宽度（像素）。
        height: 图像缓冲区高度（像素）。
        cam_height: 俯视相机距离地面的高度 [m]。
        base_to_target_dist_scale: 相机视野计算的缩放因子，控制视野覆盖范围。
    """
    width: int = 320
    height: int = 240
    cam_height: float = 3.0
    base_to_target_dist_scale: float = 2.0  # 覆盖基座到目标距离的2倍


@dataclass
class ObjectConfig:
    """目标物体物理属性配置.
    
    Attributes:
        pos: 物体初始位置 [x, y, z] (米)。
        size: 立方体边长的一半 (米)，几何尺寸。
        mass: 物体质量 (千克)。
        color: RGBA 颜色向量，此处为醒目红色。
    """
    pos: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.0, 0.025]))
    size: np.ndarray = field(default_factory=lambda: np.array([0.025, 0.025, 0.025]))
    mass: float = 0.1
    color: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 1.0]))


# ====================== 环境构建 ======================

def build_custom_grasp_environment() -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """构建并编译包含机械臂、目标物体及视觉系统的抓取仿真环境.

    该函数执行以下步骤：
    1. 调用 `get_combined_spec` 组装机器人模型（机械臂+手）。
    2. 修改世界实体（Worldbody）：添加顶部光源、目标立方体及俯视相机。
    3. 编译模型并初始化数据对象。

    Returns:
        Tuple[mujoco.MjModel, mujoco.MjData]: 
            - model: 编译后的 MuJoCo 模型实例。
            - data:  初始化的仿真数据实例。

    环境几何逻辑：
        俯视相机位置 = (机械臂基座与目标物体的中点) + (0, 0, cam_height)
        相机视野(FOVY) = 2 * arctan(覆盖半径 / 相机高度)
        覆盖半径 = ||目标位置 - 基座位置|| * base_to_target_dist_scale / 2
    """
    print("=== [EnvBuilder] 开始构建自定义抓取环境 ===")
    cfg_cam = CameraConfig()
    cfg_obj = ObjectConfig()

    # ----- 1. 机器人模型组装 -----
    # 使用组合规格加载机器人，机械手安装于 "right_hand" 接口
    # 旋转配置：绕 X 轴旋转 -90 度，使机械手手掌朝前
    spec, _ = get_combined_spec(
        rot_xyz_deg=(-90, 0, 0),
        attach_point_name="right_hand",
    )
    worldbody = spec.worldbody

    # ----- 2. 环境光照配置 -----
    # 添加定向平行光，模拟顶光照明，减少阴影干扰
    worldbody.add_light(
        name="top_light",
        pos=[0.0, 0.0, 2.0],       # 光源高度 2m
        dir=[0.0, 0.0, -1.0],      # 垂直向下照射
        diffuse=[0.8, 0.8, 0.8],   # 柔和白光
        ambient=[0.3, 0.3, 0.3],   # 提供基础环境光
    )

    # ----- 3. 相机视角计算与添加 -----
    base_pos = np.array([0.0, 0.0, 0.0])  # 假设基座位于原点
    target_pos = cfg_obj.pos

    # 计算场景中心点（基座与目标的中点）
    mid_point = (base_pos + target_pos) / 2.0

    # 计算相机视野角度 (FOVY)
    # 原理：确保相机能覆盖从基座到目标的整个操作区域
    horizontal_span = np.linalg.norm(target_pos - base_pos) * cfg_cam.base_to_target_dist_scale
    # 三角函数计算垂直视野
    fovy = 2 * np.degrees(np.arctan2(horizontal_span / 2, cfg_cam.cam_height))

    # 添加俯视相机
    worldbody.add_camera(
        name="downward_cam",
        pos=[mid_point[0], mid_point[1], cfg_cam.cam_height],
        quat=[1, 0, 0, 0],  # 使用四元数或 xyaxes 定义朝向，此处简化示意
        fovy=fovy,
    )

    # ----- 4. 目标物体添加 -----
    # 创建可抓取的立方体
    cube = worldbody.add_body(name="target_cube", pos=target_pos)
    
    # 几何形状：红色立方体
    cube.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=cfg_obj.size,
        rgba=cfg_obj.color,
        mass=cfg_obj.mass
    )
    
    # 自由关节：允许物体在被抓取后自由移动（6自由度）
    cube.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)

    print("[EnvBuilder] 模型构建完成，正在编译...")
    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


# ====================== 轨迹数据处理 ======================

def load_and_process_hand_trajectory(csv_path: str, hand_range_raw: List[float]) -> np.ndarray:
    """加载并预处理手部控制轨迹数据.

    处理流程：
    1. CSV 读取：跳过表头，加载数值数据。
    2. 归一化：将原始传感器数据映射到 [0, 0.01] 的关节角度空间。
    3. 降维广播：将多通道数据简化为单通道控制，并广播到所有手指。

    Args:
        csv_path: CSV 文件路径。
        hand_range_raw: 原始数据各通道的量程范围（用于归一化分母）。

    Returns:
        np.ndarray: 处理后的手部目标序列，形状为 (N, 6)。

    Raises:
        FileNotFoundError: 如果指定的 CSV 文件不存在。
        ValueError: 如果数据解析失败。
    """
    try:
        raw_data = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                raw_data.append([float(x) for x in row])
        
        raw_array = np.array(raw_data)  # (N, 7) 假设第一列为时间戳
        
        # ----- 数据归一化 -----
        # 将原始数据范围映射到 [0, 0.01] 弧度（或归一化位置）
        # 公式：normalized = (raw / range) * target_scale
        normalized_part = raw_array[:, 1:] / np.array(hand_range_raw) * 0.01

        # ----- 模式简化（仅用于演示） -----
        # 提取第一列作为控制信号，并广播到所有 6 个自由度
        # 这通常用于同步控制多指，或在数据维度高于实际硬件时进行降维
        single_channel = normalized_part[:, 0:1]  # (N, 1)
        hand_target_sequence = np.tile(single_channel, (1, 6))  # (N, 6)

        print(f"Loaded hand target sequence with shape: {hand_target_sequence.shape}")
        return hand_target_sequence

    except Exception as e:
        print(f"[数据加载错误] 无法处理文件 {csv_path}: {e}")
        raise


# ====================== 主程序 ======================

def main():
    """抓取环境演示主循环.

    流程：
    1. 初始化：构建环境，加载控制器，预处理轨迹数据。
    2. 控制循环：基于仿真时间索引轨迹数据，执行 PD 控制。
    3. 渲染：同步 Viewer 显示。
    4. 异常处理：捕获并打印致命错误，确保资源释放。
    """
    model: Optional[mujoco.MjModel] = None
    data: Optional[mujoco.MjData] = None
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    CSV_PATH = PROJECT_ROOT / "data" / "position_log.csv"
    
    try:
        # ===== 1. 环境与硬件抽象层初始化 =====
        model, data = build_custom_grasp_environment()
        
        # 底层硬件抽象接口
        hardware_interface = HandArmController(model)
        
        # 上层运动控制器 (OSC)
        pos_controller = OSC_PositionController(
            base=hardware_interface, 
            model=model
        )

        # ===== 2. 轨迹数据加载 =====
        HAND_RANGE_RAW = [1600, 1600, 1400, 1800, 1200, 2000]
        hand_target_sequence = load_and_process_hand_trajectory(
            CSV_PATH, 
            HAND_RANGE_RAW
        )

        # ===== 3. 预定义目标配置 =====
        # 机械臂参考构型（单位：度），需转换为弧度
        ARM_POSE_DEG = np.array([
            9.25, 82.21, -18.44, 133.08, 
            7.34, -125.17, 113.68
        ])
        arm_target = np.radians(ARM_POSE_DEG)  # 转换为弧度制

        # ===== 4. Viewer 启动与主循环 =====
        # 使用上下文管理器确保 Viewer 正确关闭
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("=== [Simulation] 开始运行，按 [ESC] 退出 ===")
            
            while viewer.is_running():
                sim_time = data.time  # 使用仿真内部时间更准确

                # ----- 控制逻辑计算 -----
                # 索引手部轨迹（循环播放）
                # 频率假设：0.01秒/步
                seq_idx = int(sim_time / 0.01) % len(hand_target_sequence)
                hand_target = hand_target_sequence[seq_idx]

                # 执行关节空间 PD 控制
                # 注意：此处使用 set_target 退化为纯关节 PD 模式
                pos_controller.set_target(
                    data=data,
                    arm_target=arm_target,
                    hand_target=hand_target
                )

                # ----- 物理步进 -----
                mujoco.mj_step(model, data)

                # ----- 渲染同步 -----
                viewer.sync()

                # ----- 状态监控 (可选) -----
                # print(f"\r[Time: {sim_time:.2f}s] Hand State: {hand_target}", end="")

    except Exception as e:
        print(f"\n[致命错误] 仿真异常终止: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n=== [Cleanup] 仿真结束，资源已释放 ===")


if __name__ == "__main__":
    main()