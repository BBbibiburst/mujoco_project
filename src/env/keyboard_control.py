"""
键盘控制机械臂与灵巧手演示脚本
支持所有继承自 RobotArmEnvBase 的任务环境

运行方式：
  python -m src.env.keyboard_control_demo --task pick_place
  python -m src.env.keyboard_control_demo --task stack --action-mode osc_pose

===============================================================
键盘映射
===============================================================
── 末端执行器位移（OSC 模式）──────────────────────────────────
  W / S       : Y 轴 +/-  （前进 / 后退）
  A / D       : X 轴 -/+  （左移 / 右移）
  Q / E       : Z 轴 +/-  （上升 / 下降）

── 末端执行器姿态（仅 osc_pose 模式）──────────────────────────
  I / K       : 绕 X 轴旋转 +/-  (Pitch)
  J / L       : 绕 Z 轴旋转 +/-  (Yaw)
  U / O       : 绕 Y 轴旋转 +/-  (Roll)

── 关节控制（joint_pd 模式）────────────────────────────────────
  1-7         : 选中关节 1-7
  +/-（= / -）: 增大 / 减小选中关节角度

── 灵巧手控制 ──────────────────────────────────────────────────
  F           : 抓握（渐进式闭合）
  G           : 张开（渐进式打开）
  R           : 手指完全复位

── 环境控制 ────────────────────────────────────────────────────
  SPACE       : 重置当前回合
  TAB         : 切换到下一个任务
  H           : 显示 / 隐藏帮助面板
  ESC / X     : 退出程序
===============================================================
"""

import sys
import time
import threading
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np
import cv2
from src.env.base_env import RobotArmEnvBase, RobotConfig

# ───────────────────── 任务注册表（复用 demo.py 中的结构）──────────────────────
TASK_REGISTRY: dict = {
    "pick_place": {
        "module": "src.env.pick_place_env",
        "env_class": "PickPlaceEnv",
        "cfg_class": "PickPlaceConfig",
        "display_name": "Pick and Place",
        "default_cfg_kwargs": {
            "r_step_penalty": -0.005,
            "r_place_bonus": 100.0,
            "r_grasp_bonus": 10.0,
        },
        "info_display": {
            "phase": "Phase",
            "dist_obj_target": "Obj-Target Dist(m)",
            "is_grasped": "Grasped",
        },
    },
    "stack": {
        "module": "src.env.stack_env",
        "env_class": "StackEnv",
        "cfg_class": "StackConfig",
        "display_name": "Stack Blocks",
        "default_cfg_kwargs": {},
        "info_display": {
            "phase": "Phase",
            "is_grasped": "Grasped",
            "is_stacked": "Stacked",
        },
    },
    "insert": {
        "module": "src.env.insert_env",
        "env_class": "InsertEnv",
        "cfg_class": "InsertConfig",
        "display_name": "Insert Peg",
        "default_cfg_kwargs": {},
        "info_display": {
            "phase": "Phase",
            "xy_dist_to_hole": "XY Dist to Hole(m)",
            "is_grasped": "Grasped",
            "is_inserted": "Inserted",
        },
    },
    "reorient": {
        "module": "src.env.reorient_env",
        "env_class": "ReorientEnv",
        "cfg_class": "ReorientConfig",
        "display_name": "Reorient Object",
        "default_cfg_kwargs": {},
        "info_display": {
            "phase": "Phase",
            "orient_error_rad": "Orient Error(rad)",
            "is_grasped": "Grasped",
            "is_success": "Success",
        },
    },
    "push": {
        "module": "src.env.push_env",
        "env_class": "PushEnv",
        "cfg_class": "PushConfig",
        "display_name": "Push Object",
        "default_cfg_kwargs": {},
        "info_display": {
            "phase": "Phase",
            "dist_obj_target": "Obj-Target Dist(m)",
            "tactile_max": "Tactile Max(Norm)",
            "is_success": "Success",
        },
    },
}

TASK_NAMES = list(TASK_REGISTRY.keys())


def _load_task(task_name: str, robot_cfg: RobotConfig) -> RobotArmEnvBase:
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"未知任务: '{task_name}'，可用: {TASK_NAMES}")
    reg = TASK_REGISTRY[task_name]
    import importlib
    mod = importlib.import_module(reg["module"])
    EnvClass = getattr(mod, reg["env_class"])
    CfgClass = getattr(mod, reg["cfg_class"])
    task_cfg = CfgClass(**reg["default_cfg_kwargs"])
    return EnvClass(robot_config=robot_cfg, task_config=task_cfg)


# ───────────────────────── 触觉热力图（来自 demo.py）─────────────────────────
def render_tactile_heatmap(obs: dict, sub_h: int = 100, sub_w: int = 120) -> np.ndarray:
    try:
        from src.sensors.tactile_sensor import FINGER_PHALANX_ORDER
    except ImportError:
        return np.zeros((sub_h * 3, sub_w * 5, 3), dtype=np.uint8)

    finger_keys = ["finger_0", "finger_1", "finger_2", "finger_3", "thumb"]
    level_order  = ["top", "middle", "bottom"]
    level_to_key = {"top": "tactile_top", "middle": "tactile_middle", "bottom": "tactile_bottom"}
    level_to_phalanx_idx = {"top": 2, "middle": 1, "bottom": 0}

    grid_rows = []
    for level in level_order:
        tac_key = level_to_key[level]
        if tac_key not in obs:
            continue
        imgs = obs[tac_key]
        if imgs.ndim == 4:
            imgs = imgs[..., 0]
        row_frames = []
        for finger_idx, finger in enumerate(finger_keys):
            img = imgs[finger_idx]
            enhanced = np.clip(img.astype(np.float32) * 5.0, 0, 255).astype(np.uint8)
            resized   = cv2.resize(enhanced, (sub_w, sub_h), interpolation=cv2.INTER_NEAREST)
            heatmap   = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
            phalanx_name = FINGER_PHALANX_ORDER[finger][level_to_phalanx_idx[level]]
            parts = phalanx_name.split('_')
            short_name = (f"T_{parts[1][:3].capitalize()}" if parts[0] == "thumb"
                          else f"F{parts[1]}_{parts[2][:3].capitalize()}")
            cv2.rectangle(heatmap, (0, 0), (sub_w, 18), (0, 0, 0), -1)
            cv2.putText(heatmap, short_name, (3, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            row_frames.append(heatmap)
        grid_rows.append(np.hstack(row_frames))
    if not grid_rows:
        return np.zeros((sub_h * 3, sub_w * 5, 3), dtype=np.uint8)
    return np.vstack(grid_rows)


# ───────────────────────── 帮助面板渲染 ──────────────────────────────────────
HELP_LINES = [
    ("=== 末端位移 ===",     None),
    ("W/S",                 "Y 轴 前进/后退"),
    ("A/D",                 "X 轴 左/右"),
    ("Q/E",                 "Z 轴 上升/下降"),
    ("",                    ""),
    ("=== 末端姿态 ===",     None),
    ("I/K",                 "Pitch +/-"),
    ("J/L",                 "Yaw  +/-"),
    ("U/O",                 "Roll +/-"),
    ("",                    ""),
    ("=== 关节（joint_pd）===", None),
    ("1~7",                 "选中关节"),
    ("= / -",               "角度 +/-"),
    ("",                    ""),
    ("=== 灵巧手 ===",       None),
    ("F",                   "抓握（闭合）"),
    ("G",                   "张开"),
    ("R",                   "完全复位"),
    ("",                    ""),
    ("=== 环境 ===",         None),
    ("SPACE",               "重置回合"),
    ("TAB",                 "下一任务"),
    ("H",                   "隐藏帮助"),
    ("ESC / X",             "退出"),
]

def draw_help_panel(frame: np.ndarray) -> np.ndarray:
    """在画面右侧叠加半透明帮助面板."""
    panel_w, line_h, pad = 280, 16, 6
    panel_h = line_h * len(HELP_LINES) + pad * 2
    x0 = frame.shape[1] - panel_w - 4
    y0 = 4

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    for i, (key, desc) in enumerate(HELP_LINES):
        y = y0 + pad + i * line_h + line_h - 4
        if desc is None:  # 标题行
            cv2.putText(frame, key, (x0 + 6, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 220, 255), 1, cv2.LINE_AA)
        elif key:
            cv2.putText(frame, key, (x0 + 6, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 210, 80), 1, cv2.LINE_AA)
            cv2.putText(frame, desc, (x0 + 80, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


# ───────────────────────── 键盘控制器 ────────────────────────────────────────
class KeyboardController:
    """
    将 OpenCV waitKey 扫描码转化为结构化动作向量。

    动作向量布局（以 osc_pose / 7+16=23 维为例）：
      [0:3]   末端位移   (dx, dy, dz)
      [3:6]   末端旋转   (drx, dry, drz)  ← 仅 osc_pose 时有效
      [6]     关节 placeholder / 其他
      [7:23]  手指关节角度增量（Shadow Hand 16 自由度）

    注意：实际维度由 env.action_space 决定，本类仅填充「语义槽」，
          维度不足时忽略姿态 / 手指部分，不会越界。
    """

    # 末端位移步长（单位与 action_scale 保持一致，归一化到 [-1, 1]）
    TRANS_STEP = 1.0    # 满幅，由 env 内部 action_scale 缩放
    ROT_STEP   = 1.0    # 旋转满幅
    GRIP_DELTA = 0.08   # 每帧手指角度变化量（rad-like）

    N_FINGER_DOF = 6

    # joint_pd 步长
    JOINT_STEP = 0.05   # rad

    def __init__(self, action_dim: int, action_mode: str = "osc_pose"):
        self.action_dim  = action_dim
        self.action_mode = action_mode

        # 手指目标位置（0=完全张开, 1=完全闭合，归一化）
        self._finger_target = 0.0   # [0.0, 1.0]

        # joint_pd：选中的关节索引（0-based）
        self.selected_joint = 0

        # 持续按键集合（由上层维护）
        self._pressed: set = set()

        # 统计
        self.ep_reward = 0.0
        self.ep_steps  = 0

    # ── 公开接口 ────────────────────────────────────────────────────────────
    def key_down(self, k: int):
        self._pressed.add(k)

    def key_up(self, k: int):
        self._pressed.discard(k)

    def compute_action(self) -> np.ndarray:
        """根据当前按键集合计算动作向量."""
        act = np.zeros(self.action_dim, dtype=np.float32)
        dim = self.action_dim

        # ── 末端位移（前3维）──────────────────────────────────────────────
        if dim > 0: act[1] += self._val(ord('w')) - self._val(ord('s'))   # Y
        if dim > 0: act[0] += self._val(ord('d')) - self._val(ord('a'))   # X
        if dim > 2: act[2] += self._val(ord('q')) - self._val(ord('e'))   # Z

        # ── 末端旋转（维度 3-5，仅 osc_pose）────────────────────────────
        if self.action_mode == "osc_pose" and dim > 5:
            act[3] += (self._val(ord('i')) - self._val(ord('k'))) * self.ROT_STEP  # Pitch
            act[4] += (self._val(ord('u')) - self._val(ord('o'))) * self.ROT_STEP  # Roll
            act[5] += (self._val(ord('j')) - self._val(ord('l'))) * self.ROT_STEP  # Yaw

        # ── joint_pd：单关节控制 ─────────────────────────────────────────
        if self.action_mode == "joint_pd":
            j = self.selected_joint
            if dim > j:
                act[j] += (self._val(ord('=')) - self._val(ord('-'))) * self.JOINT_STEP

        # ── 灵巧手（末尾 N_FINGER_DOF 维）────────────────────────────────
        # 设计原则：无按键时输出 0（保持当前位置），只在按键时输出 ±1 增量。
        # 环境内部应将此增量叠加到手指位置，而不是直接作为目标位置。
        finger_start = dim - self.N_FINGER_DOF
        if finger_start >= 0:
            grip_delta = 0.0
            if ord('f') in self._pressed:
                grip_delta = +1.0   # 闭合
            elif ord('g') in self._pressed:
                grip_delta = -1.0   # 张开
            elif ord('r') in self._pressed:
                # 复位：发送强烈负信号让手张开
                grip_delta = -1.0
                self._finger_target = 0.0

            # 同步更新 _finger_target 用于 HUD 显示
            self._finger_target = np.clip(
                self._finger_target + grip_delta * self.GRIP_DELTA, 0.0, 1.0
            )

            # 输出：有按键时发增量信号，无按键时输出 0（不打扰仿真）
            act[finger_start:] = grip_delta

        return act

    def handle_special_key(self, k: int) -> Optional[str]:
        """
        处理非连续按键（只触发一次的命令）。
        返回 "reset" / "quit" / "next_task" / "toggle_help" / None
        """
        k_lower = k if k < 128 else k  # cv2 已是小写
        if k_lower == 27 or k_lower == ord('x'):   # ESC 或 x
            return "quit"
        if k_lower == ord(' '):
            return "reset"
        if k_lower == 9:                             # TAB
            return "next_task"
        if k_lower == ord('h'):
            return "toggle_help"
        # 关节选择 1-7
        if ord('1') <= k_lower <= ord('7'):
            self.selected_joint = k_lower - ord('1')
            return None
        return None

    # ── 私有 ─────────────────────────────────────────────────────────────
    def _val(self, key: int) -> float:
        return self.TRANS_STEP if key in self._pressed else 0.0


# ───────────────────────── 主控循环 ──────────────────────────────────────────
def run_keyboard_control(
    task_name: str = "pick_place",
    action_mode: str = "osc_pose",
    controller_type: str = "osc",
    seed: int = 42,
):
    task_idx = TASK_NAMES.index(task_name) if task_name in TASK_NAMES else 0

    def make_env(name: str) -> RobotArmEnvBase:
        robot_cfg = RobotConfig(
            action_mode=action_mode,
            controller_type=controller_type,
            max_episode_steps=500,
            action_scale=0.03,
            action_scale_rot=0.05,
            control_freq=20.0,
            tactile_backend="simple_avg",
        )
        return _load_task(name, robot_cfg)

    # 初始化环境
    print(f"[KeyCtrl] 加载任务: {TASK_REGISTRY[task_name]['display_name']} ...")
    env  = make_env(task_name)
    obs, info = env.reset(seed=seed)
    action_dim = env.action_space.shape[0]
    print(f"[KeyCtrl] action_dim={action_dim}, action_mode={action_mode}")

    ctrl   = KeyboardController(action_dim, action_mode)
    ctrl.ep_reward = 0.0
    ctrl.ep_steps  = 0
    show_help      = True
    episode        = 0
    status_msg     = ""          # 一次性状态消息
    status_ttl     = 0           # 剩余显示帧数

    # ── 连续按键跟踪（OpenCV 无 key-up 事件，用帧间差分模拟）────────────
    # 策略：每帧 waitKey(16)，若有按键则加入 pressed，
    #       同时维护一个"上帧仍活跃"的集合。
    # 简化方案：对位移/旋转类按键每帧直接采样（无需 key_up），
    #            对特殊按键用节流（cooldown）防重触发。
    cooldown: dict[int, int] = {}  # key -> remaining frames

    def set_status(msg: str, ttl: int = 60):
        nonlocal status_msg, status_ttl
        status_msg = msg
        status_ttl = ttl

    def draw_hud(frame: np.ndarray) -> np.ndarray:
        """叠加 HUD 信息."""
        reg       = TASK_REGISTRY[task_name]
        info_line = " | ".join(
            f"{label}={'✓' if isinstance(info.get(k), bool) and info[k] else '✗' if isinstance(info.get(k), bool) else f'{info.get(k, 0):.3f}' if isinstance(info.get(k), float) else str(info.get(k, 'N/A'))}"
            for k, label in reg["info_display"].items()
        )
        lines = [
            f"Task: {reg['display_name']}  Ep#{episode+1}  Step:{ctrl.ep_steps}  Reward:{ctrl.ep_reward:.2f}",
            f"Grip:{ctrl._finger_target*100:.0f}%  Joint:{ctrl.selected_joint+1}  Mode:{action_mode}",
            info_line,
        ]
        for i, ln in enumerate(lines):
            y = 20 + i * 18
            cv2.putText(frame, ln, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, ln, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 120), 1, cv2.LINE_AA)

        if status_ttl > 0:
            sv = min(status_ttl / 30.0, 1.0)
            col = (int(80 * sv), int(200 * sv), int(255 * sv))
            cv2.putText(frame, status_msg, (6, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, status_msg, (6, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
        return frame

    print("\n[KeyCtrl] 仿真已启动。按 H 查看帮助。\n")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            # ── 1. 读取键盘（16ms 轮询） ──────────────────────────────
            raw_k = cv2.waitKey(16)
            # waitKey 无按键时返回 -1；& 0xFF 会把 -1 变成 255（0xFF），
            # 必须在 mask 前先判断是否有按键。
            key_pressed = raw_k != -1
            raw_k = raw_k & 0xFF if key_pressed else 255   # 255 = 无效哨兵值
            pressed_this_frame: set = set()

            # 连续型按键：有按键时加入集合，无按键时集合为空（= 输出零动作）
            HOLD_KEYS = set(map(ord, 'wasdqeikjluofgr=-'))
            if key_pressed and raw_k in HOLD_KEYS:
                pressed_this_frame.add(raw_k)

            ctrl._pressed = pressed_this_frame   # 每帧覆盖（模拟 hold / release）

            # 冷却计时器递减
            expired = [k for k, c in cooldown.items() if c <= 0]
            for k in expired:
                del cooldown[k]
            for k in cooldown:
                cooldown[k] -= 1

            # 特殊键（单次触发，需冷却）——只在真正有按键时处理
            if key_pressed and raw_k not in HOLD_KEYS:
                if raw_k not in cooldown:
                    cmd = ctrl.handle_special_key(raw_k)
                    cooldown[raw_k] = 20   # 20帧冷却 ≈ 320ms
                    if cmd == "quit":
                        print("[KeyCtrl] 退出。")
                        break
                    elif cmd == "reset":
                        obs, info = env.reset()
                        ctrl.ep_reward = 0.0
                        ctrl.ep_steps  = 0
                        set_status("⟳ 回合已重置")
                    elif cmd == "next_task":
                        task_idx = (task_idx + 1) % len(TASK_NAMES)
                        task_name = TASK_NAMES[task_idx]
                        env.close()
                        env = make_env(task_name)
                        obs, info = env.reset(seed=seed)
                        action_dim = env.action_space.shape[0]
                        ctrl = KeyboardController(action_dim, action_mode)
                        episode = 0
                        set_status(f"→ 切换任务: {TASK_REGISTRY[task_name]['display_name']}")
                    elif cmd == "toggle_help":
                        show_help = not show_help

            # 关节选择（1-7，节流）
            if key_pressed and ord('1') <= raw_k <= ord('7') and raw_k not in cooldown:
                ctrl.selected_joint = raw_k - ord('1')
                cooldown[raw_k] = 15
                set_status(f"关节 {ctrl.selected_joint+1} 已选中", 30)

            # ── 2. 计算并执行动作 ─────────────────────────────────────
            action = ctrl.compute_action()
            obs, reward, terminated, truncated, info = env.step(action)
            ctrl.ep_reward += reward
            ctrl.ep_steps  += 1
            if status_ttl > 0:
                status_ttl -= 1

            # ── 3. 回合结束处理 ───────────────────────────────────────
            if terminated or truncated:
                status = "✓ 任务完成！" if terminated else "✗ 超时"
                set_status(f"{status}  Reward={ctrl.ep_reward:.2f}", 90)
                print(
                    f"[Ep {episode+1}] {status}  steps={ctrl.ep_steps}  "
                    f"reward={ctrl.ep_reward:.2f}"
                )
                episode += 1
                obs, info = env.reset()
                ctrl.ep_reward = 0.0
                ctrl.ep_steps  = 0

            # ── 4. 渲染 ───────────────────────────────────────────────
            # 仿真视窗
            viewer.sync()

            # 相机 + HUD
            cam_bgr = cv2.cvtColor(obs["camera_rgb"], cv2.COLOR_RGB2BGR)
            cam_bgr = draw_hud(cam_bgr)
            if show_help:
                cam_bgr = draw_help_panel(cam_bgr)

            cv2.namedWindow("Camera | Keyboard Control", cv2.WINDOW_NORMAL)
            cv2.imshow("Camera | Keyboard Control", cam_bgr)
            cv2.resizeWindow("Camera | Keyboard Control", 700, 520)

            # 触觉热力图
            heatmap = render_tactile_heatmap(obs)
            cv2.namedWindow("Tactile", cv2.WINDOW_NORMAL)
            cv2.imshow("Tactile", heatmap)
            cv2.resizeWindow("Tactile", 600, 300)

    cv2.destroyAllWindows()
    env.close()
    print("[KeyCtrl] 已关闭环境。")


# ───────────────────────── 入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="键盘控制机械臂与灵巧手",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=TASK_NAMES,
        default="pick_place",
        help="初始任务（运行中可按 TAB 切换）:\n"
             + "\n".join(f"  {k}: {v['display_name']}" for k, v in TASK_REGISTRY.items()),
    )
    parser.add_argument(
        "--action-mode",
        choices=["osc_pose", "osc_pos", "joint_pd"],
        default="osc_pose",
        help="动作模式",
    )
    parser.add_argument(
        "--controller",
        choices=["osc", "ik"],
        default="osc",
        help="控制器类型",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    print(__doc__)
    run_keyboard_control(
        task_name=args.task,
        action_mode=args.action_mode,
        controller_type=args.controller,
        seed=args.seed,
    )