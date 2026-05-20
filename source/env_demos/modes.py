"""
演示模式实现.

每个函数对应一种演示模式，通过 __main__.py 调用。
"""

import json
import multiprocessing as _mp
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import mujoco
import numpy as np

from source.env.base_env import RobotArmEnvBase
from source.env.env_config import RobotConfig, SimConfig, ActionConfig
from source.env_demos.pipeline_overlay import PipelineStateOverlay
from source.env_demos.registry import TASK_REGISTRY, load_task

from .heatmap import render_tactile_heatmap
from .visualizers import (
    EETrajectoryVisualizer,
    TrajectoryVisualStyle,
    FingertipMidpointVisualizer,
)
from .keyboard_panel import KeyboardControlPanel
from .strategies import create_strategy

# ====================== 共用辅助 ======================


def _make_robot_cfg(**overrides) -> RobotConfig:
    """构造 RobotConfig，支持扁平关键字覆盖（向后兼容旧调用方式）."""
    sim_keys = {"control_freq", "sim_freq", "max_episode_steps"}
    action_keys = {
        "action_mode",
        "controller_type",
        "action_scale",
        "action_scale_rot",
        "action_scale_hand",
    }

    sim_kw = {k: v for k, v in overrides.items() if k in sim_keys}
    action_kw = {k: v for k, v in overrides.items() if k in action_keys}

    return RobotConfig(
        sim=SimConfig(**sim_kw),
        action=ActionConfig(**action_kw),
    )


def _quat_to_euler_deg(quat: np.ndarray) -> np.ndarray:
    """四元数 [w,x,y,z] → 欧拉角 ZYX (roll, pitch, yaw)，单位度."""
    w, x, y, z = quat
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.degrees(np.array([roll, pitch, yaw]))


# ====================== 日志记录辅助 ======================

# 模块顶部，缓存日志目录
_LOG_DIR: Optional[Path] = None


def _get_log_dir() -> Path:
    """
    获取日志根目录（按进程启动时间命名，同一会话内固定不变）.

    主进程首次调用时创建带时间戳的目录，并将路径写入环境变量
    DEMO_LOG_DIR，以便 spawn 子进程通过 args["log_dir"] 恢复后
    读取同一目录，保证整个运行只产生一个 log 文件夹。
    """
    global _LOG_DIR
    if _LOG_DIR is not None:
        return _LOG_DIR

    # spawn 子进程：从环境变量恢复主进程已创建的目录
    env_val = os.environ.get("DEMO_LOG_DIR")
    if env_val:
        _LOG_DIR = Path(env_val)
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        return _LOG_DIR

    # 主进程：首次调用，创建带时间戳目录并写入环境变量
    current_dir = Path(__file__).parent.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOG_DIR = current_dir / f"log_{timestamp}"
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["DEMO_LOG_DIR"] = str(_LOG_DIR)  # 供 spawn 子进程读取
    return _LOG_DIR


# 结果分类目录名映射
_OUTCOME_DIRS = {
    "success": "success",
    "timeout": "timeout",
    "fail":    "fail",
}


class InfoLogger:
    """
    每 episode 一个 JSONL 文件，记录每一步的 info。

    文件在 episode 进行期间写入临时目录（log/pending/），
    episode 结束后调用 close(outcome) 将文件移动到对应分类子目录：

        log/
          success/   ← terminated=True  且 success=True
          timeout/   ← truncated=True
          fail/      ← terminated=True  且 success=False
          pending/   ← 未调用 close(outcome) 时的临时位置

    参数
    ----
    outcome : str | None
        传给 close() 的结局，取值 "success" / "timeout" / "fail"。
        传 None 时文件留在 pending/ 不移动（异常退出等情况）。
    """

    # 子目录名集合，方便外部查询
    OUTCOME_SUCCESS = "success"
    OUTCOME_TIMEOUT = "timeout"
    OUTCOME_FAIL    = "fail"

    def __init__(self, task_name: str, mode_name: str, episode: int):
        self.task_name  = task_name
        self.mode_name  = mode_name
        self.episode    = episode
        self._root      = _get_log_dir()

        # 写入期间放在 pending/，避免结果未定时出现在分类目录
        pending_dir = self._root / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._filename = f"{task_name}_{mode_name}_ep{episode:04d}_{timestamp}.jsonl"
        self.file_path = pending_dir / self._filename

        self._file      = open(self.file_path, "w", encoding="utf-8")
        self.step_count = 0

    def log_step(
        self,
        step: int,
        info: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录单步 info."""
        record: Dict[str, Any] = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "info": info,
        }
        if extra:
            record["extra"] = extra
        self._file.write(
            json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"
        )
        self._file.flush()
        self.step_count += 1

    def close(self, outcome: Optional[str] = None) -> Path:
        """
        关闭文件并将其移动到对应分类子目录。

        参数
        ----
        outcome : "success" | "timeout" | "fail" | None
            结局分类；None 表示不移动，文件留在 pending/。

        返回
        ----
        最终文件路径。
        """
        if self._file:
            self._file.close()
            self._file = None

        if outcome is None or outcome not in _OUTCOME_DIRS:
            # 未知结局或异常退出，保留在 pending/
            return self.file_path

        dest_dir = self._root / _OUTCOME_DIRS[outcome]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / self._filename

        try:
            self.file_path.rename(dest_path)
            self.file_path = dest_path
        except OSError:
            # 跨设备移动时 rename 失败，退化为复制后删除
            import shutil
            shutil.move(str(self.file_path), str(dest_path))
            self.file_path = dest_path

        return self.file_path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 异常退出时 outcome=None，文件留在 pending/
        self.close(outcome=None)


def _json_default(obj: Any) -> Any:
    """JSON 序列化默认值处理（numpy 类型等）."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ====================== 模式 1：随机策略 ======================


def demo_random_policy(
    task_name: str = "pick_and_place",
    n_episodes: int = 3,
    render: bool = True,
    action_mode: str = "joint",
    controller_type: str = "osc",
    show_ee_traj: bool = True,
    show_fingertip_midpoint: bool = True,
) -> None:
    """随机策略演示：仿真窗口 + 触觉热力图 + 相机画面 + 末端轨迹可视化."""
    reg = TASK_REGISTRY[task_name]
    print(f"\n{'='*65}\n [Demo] 随机策略 | 任务={reg['display_name']}")
    print(f" action_mode={action_mode}, controller={controller_type}")
    print(f"{'='*65}")

    robot_cfg = _make_robot_cfg(
        action_mode=action_mode,
        controller_type=controller_type,
        max_episode_steps=200,
        action_scale=0.03,
        action_scale_rot=0.06,
        control_freq=20.0,
    )
    env = load_task(task_name, robot_cfg)

    if not render:
        _run_no_render(env, n_episodes)
        return

    obs, info = env.reset(seed=42)
    print(f"\n[初始化] obs keys: {list(obs.keys())}")
    for k, v in obs.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    traj_vis = (
        EETrajectoryVisualizer(TrajectoryVisualStyle(), max_history=30)
        if show_ee_traj
        else None
    )
    ft_vis = FingertipMidpointVisualizer() if show_fingertip_midpoint else None

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        episode, step = 0, 0
        while viewer.is_running() and episode < n_episodes:
            action = env.action_space.sample()
            obs, reward, terminated, success, truncated, info = env.step(action)
            step += 1

            viewer.user_scn.ngeom = 0
            if traj_vis:
                actual = env.get_ee_pose()[0]
                traj_vis.update(actual)
                traj_vis.draw(viewer)
            if ft_vis:
                ft_vis.update(env)
                ft_vis.draw(viewer)

            _show_cv_windows(obs)
            viewer.sync()

            if terminated or truncated:
                status = "✓ 终止" if terminated else "✗ 超时"
                print(f"[Episode {episode+1}] {status} {success} | steps={step}")
                episode += 1
                step = 0
                if traj_vis:
                    traj_vis.reset()
                if ft_vis:
                    ft_vis.reset()
                if episode < n_episodes:
                    obs, info = env.reset()

    cv2.destroyAllWindows()
    env.close()


def _run_no_render(env: RobotArmEnvBase, n_episodes: int) -> None:
    total_steps, successes = [], 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        ep_steps = 0
        while not done:
            _, _, terminated, success, truncated, _ = env.step(
                env.action_space.sample()
            )
            ep_steps += 1
            done = terminated or truncated
        total_steps.append(ep_steps)
        if success:
            successes += 1
        print(
            f"  Ep {ep+1:3d}: steps={ep_steps:4d}, "
            f"{'TERMINATED' if terminated else 'timeout'}, success={success}"
        )
    print(f"\n  平均步数: {np.mean(total_steps):.1f}")
    print(f"  成功率:   {successes / n_episodes * 100:.1f}%")
    env.close()


def _show_cv_windows(obs: dict) -> None:
    heatmap = render_tactile_heatmap(obs)
    cv2.namedWindow("Tactile (Top/Mid/Bot)", cv2.WINDOW_NORMAL)
    cv2.imshow("Tactile (Top/Mid/Bot)", heatmap)
    cv2.resizeWindow("Tactile (Top/Mid/Bot)", 1000, 480)

    if "camera_rgb" in obs:
        cam_bgr = cv2.cvtColor(obs["camera_rgb"], cv2.COLOR_RGB2BGR)
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.imshow("Camera", cam_bgr)
        cv2.resizeWindow("Camera", 640, 480)

    cv2.waitKey(1)


# ====================== 模式 2：观测空间验证 ======================


def demo_verify_observation_space(task_name: str = "pick_and_place") -> None:
    """验证所有观测分量的形状与数值范围."""
    reg = TASK_REGISTRY[task_name]
    print(
        f"\n{'='*65}\n [Demo] 观测空间验证 | 任务={reg['display_name']}\n{'='*65}"
    )

    env = load_task(
        task_name,
        _make_robot_cfg(
            action_mode="joint", controller_type="osc", max_episode_steps=100
        ),
    )
    obs, _ = env.reset(seed=0)

    print("\n--- 观测空间结构 ---")
    for key, val in obs.items():
        print(
            f"  {key}: shape={val.shape}, dtype={val.dtype}, "
            f"min={val.min():.2f}, max={val.max():.2f}"
        )
    print(f"\n--- 动作空间 ---")
    print(
        f"  shape={env.action_space.shape}, "
        f"low={env.action_space.low[0]:.1f}, high={env.action_space.high[0]:.1f}"
    )

    if "camera_rgb" in obs:
        cv2.imshow("Camera RGB", cv2.cvtColor(obs["camera_rgb"], cv2.COLOR_RGB2BGR))
    cv2.imshow("Tactile Heatmap", render_tactile_heatmap(obs))
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    env.close()


# ====================== 模式 3：基准测试 ======================


def demo_benchmark(
    task_name: str = "pick_and_place",
    n_episodes: int = 100,
    action_mode: str = "joint",
    controller_type: str = "osc",
) -> None:
    """无渲染高速基准测试."""
    reg = TASK_REGISTRY[task_name]
    print(f"[Benchmark] 任务={reg['display_name']}, n_episodes={n_episodes}")

    env = load_task(
        task_name,
        _make_robot_cfg(
            action_mode=action_mode,
            controller_type=controller_type,
            max_episode_steps=200,
            action_scale=0.03,
            action_scale_rot=0.06,
            control_freq=20.0,
        ),
    )

    t0 = time.time()
    total_steps = 0
    success_num = 0

    for ep in range(n_episodes):
        env.reset(seed=ep)
        done = False
        while not done:
            _, _, terminated, success, truncated, _ = env.step(
                env.action_space.sample()
            )
            total_steps += 1
            done = terminated or truncated
            if success and terminated:
                success_num += 1

    elapsed = time.time() - t0
    env.close()

    print(f"  总步数:  {total_steps} | 总时间: {elapsed:.1f}s")
    print(f"  步频:    {total_steps / elapsed:.0f} steps/s")
    print(f"  回合频:  {n_episodes / elapsed:.1f} eps/s")
    print(f"  成功率:  {success_num / n_episodes * 100:.1f}%")


# ====================== 模式 4：键盘控制 ======================

_HAND_MAX = 0.0095
_HAND_MIN = 0.0
_GRIPPER_OPEN = np.array(
    [_HAND_MAX, _HAND_MAX, _HAND_MIN, _HAND_MIN, _HAND_MIN, _HAND_MAX]
)


def demo_keyboard_control(
    task_name: str = "pick_and_place",
    n_episodes: int = 999_999,
    action_mode: str = "joint",
    controller_type: str = "osc",
    arm_step: float = 0.05,
    hand_step: float = 0.0005,
    pos_step: float = 0.01,
    rot_step: float = 0.05,
    show_fingertip_midpoint: bool = True,
    seed: Optional[int] = 42,
    log_info: bool = False,
) -> None:
    """
    键盘控制模式（joint / ee 双模式，禁用超时，仅手动 R 重置）.

    ←/→ 切换关节，↑/↓ 调整，R 重置，O/C 张/握手，G 夹爪，Q 退出
    """
    reg = TASK_REGISTRY[task_name]
    is_ee = action_mode == "ee"

    print(
        f"\n{'='*65}\n [Demo] 键盘控制 | 任务={reg['display_name']}  模式={action_mode}"
    )
    print(f" controller={controller_type}  超时=禁用（手动R重置）\n{'='*65}")
    if log_info:
        print(f" [Info 记录] 已启用，日志目录: {_get_log_dir()}")

    env = load_task(
        task_name,
        _make_robot_cfg(
            action_mode=action_mode,
            controller_type=controller_type,
            max_episode_steps=999_999,
            action_scale=1.0,
            action_scale_rot=1.0,
            action_scale_hand=1.0,
            control_freq=20.0,
        ),
    )

    cmd_q = queue.Queue()
    panel = KeyboardControlPanel(
        cmd_q,
        arm_dof=env.ARM_DOF,
        hand_dof=env.HAND_DOF,
        action_mode=action_mode,
        arm_step=arm_step,
        hand_step=hand_step,
        pos_step=pos_step,
        rot_step=rot_step,
    )
    threading.Thread(target=panel.run, daemon=True).start()
    panel.wait_ready(timeout=10.0)

    traj_vis = EETrajectoryVisualizer(TrajectoryVisualStyle(), max_history=40)
    ft_vis = FingertipMidpointVisualizer() if show_fingertip_midpoint else None

    obs, info = env.reset(seed=seed)

    ee_target_pos = ee_target_quat = joint_target = hand_target = None

    def _init_targets():
        nonlocal ee_target_pos, ee_target_quat, joint_target, hand_target
        if is_ee:
            ee_target_pos, ee_target_quat = env.get_ee_pose()
            ee_target_pos = ee_target_pos.copy()
            ee_target_quat = ee_target_quat.copy()
            hand_target = env.get_hand_qpos().copy()
        else:
            joint_target = np.concatenate([env.get_arm_qpos(), env.get_hand_qpos()])

    _init_targets()

    sel_idx = 0
    episode = 1
    step = 0
    ep_reward = 0.0
    reward = 0.0
    terminated = False
    running = True

    info_logger: Optional[InfoLogger] = None
    if log_info:
        info_logger = InfoLogger(task_name, "keyboard", episode)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 1.8
        viewer.cam.elevation = -25
        viewer.cam.azimuth = 135

        while viewer.is_running() and running and episode <= n_episodes:

            pending_reset = False
            while True:
                try:
                    cmd, val = cmd_q.get_nowait()
                except queue.Empty:
                    break

                if cmd == "quit":
                    running = False
                    break
                elif cmd == "reset":
                    pending_reset = True
                elif cmd == "sel":
                    n_ctrl = (env.ARM_DOF if not is_ee else 6) + 7
                    sel_idx = (sel_idx + val) % n_ctrl
                elif cmd == "open_hand":
                    if is_ee:
                        hand_target[:] = _HAND_MIN
                    else:
                        joint_target[env.ARM_DOF : env.ARM_DOF + env.HAND_DOF] = (
                            _HAND_MIN
                        )
                elif cmd == "close_hand":
                    if is_ee:
                        hand_target[:] = _HAND_MAX
                    else:
                        joint_target[env.ARM_DOF : env.ARM_DOF + env.HAND_DOF] = (
                            _HAND_MAX
                        )
                elif cmd == "gripper_open":
                    if is_ee:
                        hand_target[:] = _GRIPPER_OPEN
                    else:
                        joint_target[env.ARM_DOF : env.ARM_DOF + env.HAND_DOF] = (
                            _GRIPPER_OPEN
                        )
                elif cmd == "delta":
                    if not terminated:
                        _apply_delta(
                            val,
                            sel_idx,
                            is_ee,
                            env,
                            ee_target_pos,
                            ee_target_quat,
                            hand_target,
                            joint_target,
                            panel,
                        )

            if not running:
                break

            if pending_reset:
                label = "✅ 任务成功！" if terminated else "🔄 手动重置"
                print(
                    f"[回合 {episode}] {label}  累积奖励={ep_reward:.4f}  步数={step}"
                )
                if info_logger:
                    # keyboard 无超时：terminated+success → success，其余 → fail
                    _kb_outcome = (
                        InfoLogger.OUTCOME_SUCCESS if (terminated and success)
                        else InfoLogger.OUTCOME_FAIL
                    )
                    info_logger.close(outcome=_kb_outcome)

                obs, info = env.reset()
                traj_vis.reset()
                if ft_vis:
                    ft_vis.reset()
                _init_targets()
                episode += 1
                step = 0
                ep_reward = 0.0
                reward = 0.0
                terminated = False

                if log_info and episode <= n_episodes:
                    info_logger = InfoLogger(task_name, "keyboard", episode)
                continue

            if not terminated:
                action = _build_action(
                    is_ee, env, ee_target_pos, ee_target_quat, hand_target, joint_target
                )
                obs, reward, terminated, success, _, info = env.step(action)
                step += 1
                ep_reward += reward

                if info_logger:
                    info_logger.log_step(
                        step,
                        info,
                        extra={
                            "reward": float(reward),
                            "ep_reward": float(ep_reward),
                            "terminated": bool(terminated),
                            "success": bool(success),
                            "action_mode": action_mode,
                        },
                    )

                if terminated:
                    print(
                        f"[回合 {episode}] 终止，success={success}  步数={step}"
                        f"  累积奖励={ep_reward:.4f}  → 按 R 重置"
                    )

            viewer.user_scn.ngeom = 0
            actual = env.get_ee_pose()[0]
            traj_vis.update(
                actual, target_pos=ee_target_pos, target_quat=ee_target_quat
            )
            traj_vis.draw(viewer)
            if ft_vis:
                ft_vis.update(env)
                ft_vis.draw(viewer)

            if is_ee:
                rpy = _quat_to_euler_deg(ee_target_quat)
                sync = np.array(
                    [(hand_target[2] + hand_target[3] + hand_target[4]) / 3.0]
                )
                display_vals = np.concatenate([ee_target_pos, rpy, hand_target, sync])
            else:
                hand_part = joint_target[env.ARM_DOF : env.ARM_DOF + env.HAND_DOF]
                sync = np.array([(hand_part[2] + hand_part[3] + hand_part[4]) / 3.0])
                display_vals = np.concatenate(
                    [joint_target[: env.ARM_DOF + env.HAND_DOF], sync]
                )

            panel.update_state(
                sel_idx=sel_idx,
                display_vals=display_vals,
                reward=reward,
                ep_reward=ep_reward,
                step=step,
                terminated=terminated,
                episode=episode,
                info_extra={},
            )

            _show_cv_windows(obs)
            _overlay_camera(
                obs, action_mode, reward, ep_reward, step, episode, terminated
            )
            viewer.sync()

    if info_logger:
        # Q 退出时 episode 尚未通过 R 重置关闭，outcome 未定，留在 pending/
        info_logger.close(outcome=None)

    cv2.destroyAllWindows()
    env.close()
    print("\n[键盘控制] 已退出。")


def _apply_delta(
    val,
    sel_idx,
    is_ee,
    env,
    ee_target_pos,
    ee_target_quat,
    hand_target,
    joint_target,
    panel,
) -> None:
    """将方向键增量应用到控制目标."""
    if is_ee:
        if sel_idx < 3:
            ee_target_pos[sel_idx] += val * panel._pos_step
        elif sel_idx < 6:
            axis = np.zeros(3)
            axis[sel_idx - 3] = 1.0
            dq = np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, axis, val * panel._rot_step)
            new_q = np.zeros(4)
            mujoco.mju_mulQuat(new_q, ee_target_quat, dq)
            norm = np.linalg.norm(new_q)
            if norm > 1e-8:
                ee_target_quat[:] = new_q / norm
        else:
            hi = sel_idx - 6
            if hi < 6:
                hand_target[hi] = np.clip(
                    hand_target[hi] + val * panel._hand_step, _HAND_MIN, _HAND_MAX
                )
            else:
                for si in [2, 3, 4]:
                    hand_target[si] = np.clip(
                        hand_target[si] + val * panel._hand_step, _HAND_MIN, _HAND_MAX
                    )
    else:
        if sel_idx < env.ARM_DOF:
            joint_target[sel_idx] += val * panel._arm_step
        elif sel_idx < env.ARM_DOF + env.HAND_DOF:
            joint_target[sel_idx] = np.clip(
                joint_target[sel_idx] + val * panel._hand_step, _HAND_MIN, _HAND_MAX
            )
        else:
            base = env.ARM_DOF
            for si in [base + 2, base + 3, base + 4]:
                joint_target[si] = np.clip(
                    joint_target[si] + val * panel._hand_step, _HAND_MIN, _HAND_MAX
                )


def _build_action(
    is_ee, env, ee_target_pos, ee_target_quat, hand_target, joint_target
):
    """根据控制目标构造动作向量."""
    if is_ee:
        cur_pos, cur_quat = env.get_ee_pose()
        cur_inv = np.zeros(4)
        mujoco.mju_negQuat(cur_inv, cur_quat)
        dq = np.zeros(4)
        mujoco.mju_mulQuat(dq, ee_target_quat, cur_inv)
        rot_delta = np.zeros(3)
        mujoco.mju_quat2Vel(rot_delta, dq, 1.0)
        return np.concatenate(
            [ee_target_pos - cur_pos, rot_delta, hand_target - env.get_hand_qpos()]
        ).astype(np.float32)
    else:
        current = np.concatenate([env.get_arm_qpos(), env.get_hand_qpos()])
        return (joint_target - current).astype(np.float32)


def _overlay_camera(obs, action_mode, reward, ep_reward, step, episode, terminated):
    """在相机画面上叠加文字并显示."""
    if "camera_rgb" not in obs:
        return
    cam_bgr = cv2.cvtColor(obs["camera_rgb"], cv2.COLOR_RGB2BGR)
    texts = [
        f"[{action_mode.upper()}] Reward: {reward:+.4f}  Cum: {ep_reward:+.4f}",
        f"Step: {step}  Ep: {episode}  (R=reset, no timeout)",
    ]
    overlay = cam_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (640, 20 * len(texts) + 10), (0, 0, 0), -1)
    cam_bgr = cv2.addWeighted(overlay, 0.4, cam_bgr, 0.6, 0)
    for i, txt in enumerate(texts):
        y = 18 + i * 18
        cv2.putText(
            cam_bgr, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            cam_bgr, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )
    if terminated:
        cv2.putText(
            cam_bgr, "TASK SUCCESS!", (30, 130), cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0, 0, 0), 4, cv2.LINE_AA,
        )
        cv2.putText(
            cam_bgr, "TASK SUCCESS!", (30, 130), cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0, 255, 80), 2, cv2.LINE_AA,
        )
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.imshow("Camera", cam_bgr)
    cv2.resizeWindow("Camera", 640, 480)


# ====================== 模式 5：流程化任务执行（并行 Worker）======================

# ---------- Worker（必须定义在模块顶层才能被 multiprocessing pickle）----------

def _pipeline_worker_stream(args: dict, result_q: _mp.Queue) -> None:
    """
    子进程入口：独立运行若干 episode，每完成一个立即将结果放入队列。

    每个 worker 拥有独立的环境实例和策略实例，互不干扰。
    日志文件以 worker_id 区分，避免写入冲突。

    结果消息格式
    ------------
    普通结果::

        {
            "worker_id": int,
            "ep": int,           # 全局 episode 编号（0-based）
            "steps": int,
            "success": bool,
            "truncated": bool,
            "status": "SUCCESS" | "TIMEOUT" | "FAIL",
        }

    哨兵（worker 全部 episode 跑完后发送一次）::

        {"worker_id": int, "done": True}
    """
    # spawn 模式下子进程是全新解释器，os.environ 的运行时修改不会自动继承，
    # 需要从 args 显式恢复，使 _get_log_dir() 复用主进程已创建的目录，
    # 保证整次运行只产生一个 log_<timestamp> 文件夹。
    if args.get("log_dir"):
        os.environ["DEMO_LOG_DIR"] = args["log_dir"]

    task_name       = args["task_name"]
    ep_indices      = args["ep_indices"]   # 本 worker 负责的全局 episode 编号列表
    action_mode     = args["action_mode"]
    controller_type = args["controller_type"]
    seed_base       = args["seed_base"]
    log_info        = args["log_info"]
    worker_id       = args["worker_id"]

    # 子进程内延迟 import，避免主进程 fork 时携带 OpenGL context
    from source.env_demos.registry import load_task, load_strategy

    robot_cfg = _make_robot_cfg(
        action_mode=action_mode,
        controller_type=controller_type,
        max_episode_steps=2000,
        action_scale=0.05,
        action_scale_rot=0.1,
        action_scale_hand=0.005,
        control_freq=20.0,
    )
    env = load_task(task_name, robot_cfg)
    strategy = load_strategy(task_name)

    for ep in ep_indices:
        ep_seed = None if seed_base is None else (ep + 1) * seed_base
        obs, info = env.reset(seed=ep_seed)
        strategy.reset()
        step = 0
        done = False

        info_logger: Optional[InfoLogger] = (
            InfoLogger(task_name, f"pipeline_w{worker_id:02d}", ep + 1)
            if log_info
            else None
        )

        while not done:
            action, _ = strategy.tick(obs, info, step, env)
            obs, _, terminated, success, truncated, info = env.step(action)
            step += 1
            done = terminated or truncated

            if info_logger:
                info_logger.log_step(
                    step,
                    info,
                    extra={
                        "terminated": bool(terminated),
                        "success": bool(success),
                        "truncated": bool(truncated),
                        "phase_idx": getattr(strategy, "phase_idx", None),
                    },
                )

        if info_logger:
            _w_outcome = (
                InfoLogger.OUTCOME_SUCCESS if success
                else (InfoLogger.OUTCOME_TIMEOUT if truncated else InfoLogger.OUTCOME_FAIL)
            )
            info_logger.close(outcome=_w_outcome)

        # 每 episode 完成立即上报，不等待整个 worker 批次结束
        result_q.put({
            "worker_id": worker_id,
            "ep": ep,
            "steps": step,
            "success": bool(success),
            "truncated": bool(truncated),
            "status": (
                "SUCCESS" if success else ("TIMEOUT" if truncated else "FAIL")
            ),
        })

    env.close()
    # 哨兵：通知主进程该 worker 已全部完成
    result_q.put({"worker_id": worker_id, "done": True})


# ---------- 主函数 ----------

def demo_pipeline(
    task_name: str = "block_lifting",
    n_episodes: int = 3,
    render: bool = True,
    action_mode: str = "ee",
    controller_type: str = "osc",
    show_ee_traj: bool = True,
    show_fingertip_midpoint: bool = True,
    summary_interval: int = 50,
    seed: Optional[int] = 42,
    log_info: bool = False,
    n_workers: int = 0,
) -> None:
    """
    流程化任务执行：按任务类型自动编排行为序列完成目标。

    三种运行模式：
      - render=True        : 可视化演示，单进程实时渲染 + 统计成功率
      - render=False, n_workers<=1 : 无渲染单进程批量测试（保留原有行为）
      - render=False, n_workers>1  : 无渲染多进程并行批量测试（大幅提升吞吐）

    参数
    ----
    n_workers : int
        并行 worker 数量（仅 render=False 时生效）。
        0  → 自动取 os.cpu_count()；
        1  → 单进程模式（与原行为完全一致）；
        N  → N 个子进程并行，每个进程独占一个环境实例。
        summary_interval 以实际完成的 episode 数计，与 worker 批次无关。
    """
    from source.env_demos.registry import load_strategy

    reg = TASK_REGISTRY[task_name]
    print(f"\n{'='*65}\n [Demo] 流程化执行 | 任务={reg['display_name']}")
    print(f" action_mode={action_mode}, controller={controller_type}")
    print(f" 渲染模式: {'可视化' if render else '无渲染批量测试'}")
    if not render:
        _n_workers = n_workers if n_workers > 0 else (_mp.cpu_count() or 1)
        _n_workers = min(_n_workers, n_episodes)
        print(f" 并行 Worker 数: {_n_workers}")
    if log_info:
        print(f" [Info 记录] 已启用，日志目录: {_get_log_dir()}")
    print(f"{'='*65}")

    robot_cfg = _make_robot_cfg(
        action_mode=action_mode,
        controller_type=controller_type,
        max_episode_steps=2000,
        action_scale=0.05,
        action_scale_rot=0.1,
        action_scale_hand=0.005,
        control_freq=20.0,
    )

    total_steps: List[int] = []
    successes = 0
    timeouts = 0
    t0 = time.time()

    # ===================================================================
    # 无渲染模式
    # ===================================================================
    if not render:
        _n_workers = n_workers if n_workers > 0 else (_mp.cpu_count() or 1)
        _n_workers = min(_n_workers, n_episodes)

        # ---------- 单进程（n_workers == 1）：保留原有行为 ----------
        if _n_workers == 1:
            env = load_task(task_name, robot_cfg)
            strategy = load_strategy(task_name)

            for ep in range(n_episodes):
                ep_seed = None if seed is None else (ep + 1) * seed
                obs, info = env.reset(seed=ep_seed)
                strategy.reset()
                step = 0
                done = False

                info_logger = (
                    InfoLogger(task_name, "pipeline", ep + 1) if log_info else None
                )

                while not done:
                    action, _ = strategy.tick(obs, info, step, env)
                    obs, _, terminated, success, truncated, info = env.step(action)
                    step += 1
                    done = terminated or truncated

                    if info_logger:
                        info_logger.log_step(
                            step,
                            info,
                            extra={
                                "terminated": bool(terminated),
                                "success": bool(success),
                                "truncated": bool(truncated),
                                "phase_idx": getattr(strategy, "phase_idx", None),
                            },
                        )

                total_steps.append(step)
                if success:
                    successes += 1
                if truncated:
                    timeouts += 1

                status = (
                    "SUCCESS" if success else ("TIMEOUT" if truncated else "FAIL")
                )
                print(f"  Ep {ep+1:3d}/{n_episodes}: steps={step:4d} | {status}")

                if info_logger:
                    _sp_outcome = (
                        InfoLogger.OUTCOME_SUCCESS if success
                        else (InfoLogger.OUTCOME_TIMEOUT if truncated else InfoLogger.OUTCOME_FAIL)
                    )
                    info_logger.close(outcome=_sp_outcome)

                completed = ep + 1
                if completed % summary_interval == 0 and completed < n_episodes:
                    _print_pipeline_summary(
                        task_name, completed, total_steps, successes, timeouts,
                        elapsed=time.time() - t0, is_interim=True,
                    )

            _print_pipeline_summary(
                task_name, n_episodes, total_steps, successes, timeouts,
                elapsed=time.time() - t0, is_interim=False,
            )
            env.close()
            return

        # ---------- 多进程（n_workers > 1）：流式并行执行 ----------
        #
        # 将 n_episodes 个 episode 均匀分配给各 worker，余数轮流补给前几个 worker。
        # 示例：10 episodes, 3 workers → [0,3,6,9], [1,4,7], [2,5,8]
        #
        # log_dir 由主进程在构造 worker_args 前统一调用 _get_log_dir() 确定，
        # 并通过 args["log_dir"] 显式传入每个子进程，子进程启动时写入
        # os.environ["DEMO_LOG_DIR"]，使 _get_log_dir() 复用同一目录，
        # 保证整次运行只产生一个 log_<timestamp> 文件夹。
        log_dir_str = str(_get_log_dir()) if log_info else None

        ep_indices_all = list(range(n_episodes))
        chunks = [ep_indices_all[i::_n_workers] for i in range(_n_workers)]

        worker_args = [
            {
                "task_name": task_name,
                "ep_indices": chunk,
                "action_mode": action_mode,
                "controller_type": controller_type,
                "seed_base": seed if seed is not None else 42,
                "log_info": log_info,
                "worker_id": i,
                "log_dir": log_dir_str,  # 主进程确定的统一目录，子进程复用
            }
            for i, chunk in enumerate(chunks)
            if chunk  # 过滤空分片（n_workers > n_episodes 时出现）
        ]

        actual_workers = len(worker_args)
        print(
            f"  分配方案: {actual_workers} workers × ~{n_episodes // actual_workers} episodes/worker\n"
        )

        # 使用 Manager Queue 实现跨进程流式传递
        # （普通 multiprocessing.Queue 在某些平台 fork 后也可用，
        #   但 Manager Queue 更安全，无需担心 fork 时的 fd 继承问题）
        ctx = _mp.get_context("spawn")  # spawn 避免 fork + OpenGL context 冲突
        manager = ctx.Manager()
        result_q = manager.Queue()

        processes = [
            ctx.Process(
                target=_pipeline_worker_stream,
                args=(wa, result_q),
                daemon=True,
            )
            for wa in worker_args
        ]
        for p in processes:
            p.start()

        finished_workers = 0
        last_summary_at = 0   # 上次打 summary 时已完成的 episode 数

        while finished_workers < actual_workers:
            try:
                r = result_q.get(timeout=300)  # 5 分钟超时保护
            except Exception:
                print("[警告] 等待结果超时，可能有 worker 异常退出，强制结束收集。")
                break

            if r.get("done"):
                # 哨兵：该 worker 全部完成
                finished_workers += 1
                continue

            # 普通结果：立即累计并打印
            wid = r["worker_id"]
            total_steps.append(r["steps"])
            if r["success"]:
                successes += 1
            if r["truncated"]:
                timeouts += 1

            completed = len(total_steps)

            # 精确按 summary_interval 触发：每新增满 interval 个就打一次
            if (
                completed - last_summary_at >= summary_interval
                and completed < n_episodes
            ):
                _print_pipeline_summary(
                    task_name, completed, total_steps, successes, timeouts,
                    elapsed=time.time() - t0, is_interim=True,
                )
                last_summary_at = completed

        for p in processes:
            p.join()
        manager.shutdown()

        _print_pipeline_summary(
            task_name, n_episodes, total_steps, successes, timeouts,
            elapsed=time.time() - t0, is_interim=False,
        )
        return

    # ===================================================================
    # 渲染模式（单进程，与原逻辑完全一致）
    # ===================================================================
    env = load_task(task_name, robot_cfg)
    strategy = load_strategy(task_name)

    obs, info = env.reset(seed=seed)
    strategy.reset()

    overlay = PipelineStateOverlay(strategy)
    traj_vis = (
        EETrajectoryVisualizer(TrajectoryVisualStyle(), max_history=100)
        if show_ee_traj
        else None
    )
    ft_vis = FingertipMidpointVisualizer() if show_fingertip_midpoint else None

    viewer = mujoco.viewer.launch_passive(env.model, env.data)

    info_logger: Optional[InfoLogger] = (
        InfoLogger(task_name, "pipeline", 1) if log_info else None
    )

    try:
        viewer.cam.distance = 1.5
        viewer.cam.elevation = -30
        viewer.cam.azimuth = 120

        episode = 0
        while viewer.is_running() and episode < n_episodes:
            if episode > 0:
                ep_seed = (episode + 1) * seed
                obs, info = env.reset(seed=ep_seed)
                strategy.reset()
                overlay.reset()
                if traj_vis:
                    traj_vis.reset()
                if ft_vis:
                    ft_vis.reset()

                if log_info:
                    if info_logger:
                        _prev_outcome = (
                            InfoLogger.OUTCOME_SUCCESS if success
                            else (InfoLogger.OUTCOME_TIMEOUT if truncated else InfoLogger.OUTCOME_FAIL)
                        )
                        info_logger.close(outcome=_prev_outcome)
                    info_logger = InfoLogger(task_name, "pipeline", episode + 1)

            step = 0
            done = False

            while viewer.is_running() and not done:
                action, action_context = strategy.tick(obs, info, step, env)
                ee_target_pos = action_context.ee_target_pos
                ee_target_quat = action_context.ee_target_quat
                ee_delta_pos = action_context.ee_delta_pos
                ee_delta_rot = action_context.ee_delta_rot

                if ee_target_pos is None and ee_target_quat is None:
                    if ee_delta_pos is not None and ee_delta_rot is not None:
                        ee_pos, ee_quat = env.get_ee_pose()
                        ee_target_pos = ee_pos + ee_delta_pos
                        delta_quat = np.zeros(4)
                        mujoco.mju_euler2Quat(delta_quat, ee_delta_rot, "xyz")
                        ee_target_quat = np.zeros(4)
                        mujoco.mju_mulQuat(ee_target_quat, delta_quat, ee_quat)

                obs, reward, terminated, success, truncated, info = env.step(action)
                step += 1
                done = terminated or truncated

                if info_logger:
                    info_logger.log_step(
                        step,
                        info,
                        extra={
                            "reward": float(reward),
                            "terminated": bool(terminated),
                            "success": bool(success),
                            "truncated": bool(truncated),
                            "phase_idx": getattr(strategy, "phase_idx", None),
                        },
                    )

                switch_msg = overlay.update(step)
                if switch_msg:
                    print(switch_msg)

                viewer.user_scn.ngeom = 0
                if traj_vis:
                    actual = env.get_ee_pose()[0]
                    traj_vis.update(
                        actual, target_pos=ee_target_pos, target_quat=ee_target_quat
                    )
                    traj_vis.draw(viewer)
                if ft_vis:
                    ft_vis.update(env)
                    ft_vis.draw(viewer)

                overlay.draw_viewer_indicator(viewer)
                _show_cv_windows_pipeline(obs, overlay, reward, env)
                viewer.sync()

            total_steps.append(step)
            if success:
                successes += 1
            if truncated:
                timeouts += 1

            status = "✓ 成功" if success else "✗ 失败/超时"
            print(f"\n  [回合 {episode+1} 结束] {status}")
            print(
                f"   总步数: {step} | 最终阶段: "
                f"{overlay._get_phase_name(strategy.phase_idx)}"
            )

            status_dict = strategy.get_status_dict()
            print("   策略内部状态：")
            for k, v in status_dict.items():
                print(f"   {k}: {v}")

            completed = episode + 1
            if completed % summary_interval == 0 and completed < n_episodes:
                _print_pipeline_summary(
                    task_name, completed, total_steps, successes, timeouts,
                    elapsed=time.time() - t0, is_interim=True,
                )

            print(f"  {'='*40}")
            episode += 1

        elapsed = time.time() - t0
        _print_pipeline_summary(
            task_name, n_episodes, total_steps, successes, timeouts,
            elapsed=elapsed, is_interim=False,
        )

    finally:
        # 严格保证析构顺序：viewer → cv2 → env，顺序错误会导致 segfault
        if info_logger:
            # 最后一个 episode：正常结束时 success/truncated 已赋值；
            # 若 viewer 被强制关闭则 episode 可能未完成，留 pending。
            try:
                _fin_outcome = (
                    InfoLogger.OUTCOME_SUCCESS if success
                    else (InfoLogger.OUTCOME_TIMEOUT if truncated else InfoLogger.OUTCOME_FAIL)
                )
            except NameError:
                _fin_outcome = None  # episode 从未执行过
            info_logger.close(outcome=_fin_outcome)
        viewer.close()
        cv2.destroyAllWindows()
        env.close()


def _print_pipeline_summary(
    task_name: str,
    n_completed: int,
    total_steps: list,
    successes: int,
    timeouts: int,
    elapsed: float,
    is_interim: bool = False,
) -> None:
    """输出 pipeline 统计总结（阶段性或最终）."""
    label = "阶段性总结" if is_interim else "最终总结"
    header = f"\n{'='*50}\n  Pipeline {label} ({task_name})\n{'='*50}"
    print(header)
    print(f"  已完成回合: {n_completed}")
    print(f"  成功数:     {successes}")
    print(f"  超时数:     {timeouts}")
    print(f"  失败数:     {n_completed - successes - timeouts}")
    print(f"  通关率:     {successes / n_completed * 100:.1f}%")
    print(f"  平均步数:   {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
    print(f"  总耗时:     {elapsed:.1f}s")
    print(f"  回合/秒:    {n_completed / elapsed:.2f}")
    print(f"  步/秒:      {sum(total_steps) / elapsed:.0f}")
    print(f"{'='*50}")


def _show_cv_windows_pipeline(
    obs: dict, overlay: PipelineStateOverlay, reward: float, env
) -> None:
    """通用 OpenCV 显示，不依赖具体任务."""
    from .heatmap import render_tactile_heatmap

    heatmap = render_tactile_heatmap(obs)
    cv2.namedWindow("Tactile (Top/Mid/Bot)", cv2.WINDOW_NORMAL)
    cv2.imshow("Tactile (Top/Mid/Bot)", heatmap)
    cv2.resizeWindow("Tactile (Top/Mid/Bot)", 1000, 480)

    if "camera_rgb" in obs:
        cam_bgr = cv2.cvtColor(obs["camera_rgb"], cv2.COLOR_RGB2BGR)
        cam_bgr = overlay.draw_camera_overlay(cam_bgr, reward)
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.imshow("Camera", cam_bgr)
        cv2.resizeWindow("Camera", 640, 480)

    cv2.waitKey(1)