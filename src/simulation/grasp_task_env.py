"""
自定义抓取环境构建与演示模块.
"""

import csv
from pathlib import Path
import mujoco
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass, field

from src.robot.robot_arm_system import get_combined_spec
from src.controllers.position_controller import OSC_PositionController
from src.controllers.hand_arm_controller import HandArmController
from src.sensors.tactile_sensor import TactileReader, DISPLAY_ORDER, FINGER_PHALANX_ORDER  # ← 导入 TactileReader


# ====================== 配置数据类 ======================

@dataclass
class CameraConfig:
    width: int = 320
    height: int = 240
    cam_height: float = 3.0
    base_to_target_dist_scale: float = 2.0


@dataclass
class ObjectConfig:
    pos: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.0, 0.025]))
    size: np.ndarray = field(default_factory=lambda: np.array([0.025, 0.025, 0.025]))
    mass: float = 0.1
    color: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 1.0]))


# ====================== 环境构建 ======================

def build_custom_grasp_environment(
    tactile_backend: str = "physics",
) -> Tuple[mujoco.MjModel, mujoco.MjData, TactileReader]:
    """
    构建自定义抓取环境，返回编译好的模型和已绑定的触觉读取器.
    
    Args:
        tactile_backend: "physics"（弹性taxel，推荐）或 "simple"（轻量site）
    
    Returns:
        (model, data, reader): 已编译模型、仿真数据、已绑定的 TactileReader
    """
    print("=== [EnvBuilder] 开始构建自定义抓取环境 ===")
    cfg_cam = CameraConfig()
    cfg_obj = ObjectConfig()

    # 1. 获取 spec 和 reader（此时 reader 已 build，未 bind）
    spec, reader = get_combined_spec(  # ← 返回 (spec, reader)
        rot_xyz_deg=(-90, 0, 0),
        attach_point_name="right_hand",
        tactile_backend=tactile_backend,
    )
    worldbody = spec.worldbody

    # 2. 添加环境元素（光照、相机、物体）
    worldbody.add_light(
        name="top_light",
        pos=[0.0, 0.0, 2.0],
        dir=[0.0, 0.0, -1.0],
        diffuse=[0.8, 0.8, 0.8],
        ambient=[0.3, 0.3, 0.3],
    )

    base_pos = np.array([0.0, 0.0, 0.0])
    target_pos = cfg_obj.pos
    mid_point = (base_pos + target_pos) / 2.0
    horizontal_span = np.linalg.norm(target_pos - base_pos) * cfg_cam.base_to_target_dist_scale
    fovy = 2 * np.degrees(np.arctan2(horizontal_span / 2, cfg_cam.cam_height))

    worldbody.add_camera(
        name="downward_cam",
        pos=[mid_point[0], mid_point[1], cfg_cam.cam_height],
        quat=[1, 0, 0, 0],
        fovy=fovy,
    )

    cube = worldbody.add_body(name="target_cube", pos=target_pos)
    cube.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=cfg_obj.size,
        rgba=cfg_obj.color,
        mass=cfg_obj.mass,
    )
    cube.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)

    # 3. 编译模型
    print("[EnvBuilder] 模型构建完成，正在编译...")
    model = spec.compile()
    data = mujoco.MjData(model)

    # 4. 【关键】绑定 reader（必须在 compile 之后）
    reader.bind(model)
    print(f"[EnvBuilder] 触觉读取器已绑定: {reader}")

    return model, data, reader


# ====================== 轨迹数据处理 ======================

def load_and_process_hand_trajectory(
    csv_path: str, hand_range_raw: List[float]
) -> np.ndarray:
    try:
        raw_data = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                raw_data.append([float(x) for x in row])

        raw_array = np.array(raw_data)
        normalized_part = raw_array[:, 1:] / np.array(hand_range_raw) * 0.01
        single_channel = normalized_part[:, 0:1]
        hand_target_sequence = np.tile(single_channel, (1, 6))

        print(f"Loaded hand target sequence with shape: {hand_target_sequence.shape}")
        return hand_target_sequence

    except Exception as e:
        print(f"[数据加载错误] 无法处理文件 {csv_path}: {e}")
        raise


# ====================== 主程序 ======================

import cv2


def main():
    """抓取环境演示主循环：集成实时触觉图像显示."""
    model: Optional[mujoco.MjModel] = None
    data: Optional[mujoco.MjData] = None
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    CSV_PATH = PROJECT_ROOT / "data" / "position_log.csv"

    try:
        # ===== 1. 环境与触觉系统初始化 =====
        # 返回的 reader 已经是绑定好的，无需再调用 bind
        model, data, reader = build_custom_grasp_environment(tactile_backend="simple_avg")

        hardware_interface = HandArmController(model)
        pos_controller = OSC_PositionController(base=hardware_interface, model=model)

        # ===== 2. 轨迹数据准备 =====
        HAND_RANGE_RAW = [1600, 1600, 1400, 1800, 1200, 2000]
        hand_target_sequence = load_and_process_hand_trajectory(CSV_PATH, HAND_RANGE_RAW)
        ARM_POSE_DEG = np.array([9.25, 82.21, -18.44, 133.08, 7.34, -125.17, 113.68])
        arm_target = np.radians(ARM_POSE_DEG)

        SUB_H, SUB_W = 160, 120

        # ===== 3. 主循环 =====
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("=== [Simulation] 运行中，按 Q 关闭触觉窗口 ===")

            while viewer.is_running():
                sim_time = data.time

                # ----- 运动控制 -----
                seq_idx = int(sim_time / 0.01) % len(hand_target_sequence)
                pos_controller.set_target(
                    data=data,
                    arm_target=arm_target,
                    hand_target=hand_target_sequence[seq_idx],
                )

                # ----- 物理步进 -----
                mujoco.mj_step(model, data)

                # ----- 触觉图像读取（reader 已绑定，直接用）-----
                tactile_images = reader.read_image(data) 

                # ----- 按固定顺序生成每个指节的热力图帧 -----
                frames: dict = {}
                for name in DISPLAY_ORDER:
                    if name not in tactile_images:
                        frames[name] = np.zeros((SUB_H, SUB_W, 3), dtype=np.uint8)
                        continue

                    img = tactile_images[name]
                    enhanced = np.clip(img.astype(np.float32) * 5.0, 0, 255).astype(np.uint8)
                    resized = cv2.resize(enhanced, (SUB_W, SUB_H), interpolation=cv2.INTER_NEAREST)
                    heatmap = cv2.applyColorMap(resized, cv2.COLORMAP_JET)

                    # 标题文字
                    parts = name.split('_')
                    if parts[0] == "thumb":
                        short_name = f"T_{parts[1][:3].capitalize()}"
                    else:
                        short_name = f"F{parts[1]}_{parts[2][:3].capitalize()}"

                    cv2.rectangle(heatmap, (0, 0), (SUB_W, 25), (0, 0, 0), -1)
                    cv2.putText(heatmap, short_name, (5, 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    frames[name] = heatmap

                # ----- 按"指尖/中节/指根 × 5根手指"网格拼图 -----
                finger_keys = ["finger_0", "finger_1", "finger_2", "finger_3", "thumb"]
                phalanx_levels = ["top", "middle", "bottom"]

                grid_rows = []
                for level in phalanx_levels:
                    row_frames = []
                    for finger in finger_keys:
                        phalanx_name = FINGER_PHALANX_ORDER[finger][
                            {"top": 2, "middle": 1, "bottom": 0}[level]
                        ]
                        row_frames.append(frames[phalanx_name])
                    grid_rows.append(np.hstack(row_frames))

                combined_heatmap = np.vstack(grid_rows)

                cv2.imshow("Tactile Heatmap (Top / Mid / Bot)", combined_heatmap)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                viewer.sync()

    except Exception as e:
        print(f"\n[致命错误] 仿真异常终止: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print("\n=== [Cleanup] 仿真与触觉窗口已关闭 ===")


if __name__ == "__main__":
    main()