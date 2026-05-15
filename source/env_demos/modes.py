"""
演示模式实现.

每个函数对应一种演示模式，通过 __main__.py 调用。
"""

import queue
import threading
import time
from typing import Optional

import cv2
import mujoco
import numpy as np

from source.env.base_env import RobotArmEnvBase
from source.env.env_config import RobotConfig, SimConfig, ActionConfig
from source.env_demos.pipeline_overlay import PipelineStateOverlay
from source.env_demos.registry import TASK_REGISTRY, load_task

from .heatmap import render_tactile_heatmap
from .visualizers import (
    EETrajectoryVisualizer, TrajectoryVisualStyle,
    FingertipMidpointVisualizer,
)
from .keyboard_panel import KeyboardControlPanel
from .strategies import create_strategy


# ====================== 共用辅助 ======================

def _make_robot_cfg(**overrides) -> RobotConfig:
    """构造 RobotConfig，支持扁平关键字覆盖（向后兼容旧调用方式）."""
    sim_keys    = {"control_freq", "sim_freq", "max_episode_steps"}
    action_keys = {"action_mode", "controller_type", "action_scale",
                   "action_scale_rot", "action_scale_hand"}

    sim_kw    = {k: v for k, v in overrides.items() if k in sim_keys}
    action_kw = {k: v for k, v in overrides.items() if k in action_keys}

    return RobotConfig(
        sim=SimConfig(**sim_kw),
        action=ActionConfig(**action_kw),
    )


def _quat_to_euler_deg(quat: np.ndarray) -> np.ndarray:
    """四元数 [w,x,y,z] → 欧拉角 ZYX (roll, pitch, yaw)，单位度."""
    w, x, y, z = quat
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.degrees(np.array([roll, pitch, yaw]))


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
        action_mode=action_mode, controller_type=controller_type,
        max_episode_steps=200, action_scale=0.03, action_scale_rot=0.06,
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

    traj_vis = EETrajectoryVisualizer(TrajectoryVisualStyle(), max_history=30) if show_ee_traj else None
    ft_vis   = FingertipMidpointVisualizer() if show_fingertip_midpoint else None

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
                if traj_vis: traj_vis.reset()
                if ft_vis:   ft_vis.reset()
                if episode < n_episodes:
                    obs, info = env.reset()

    cv2.destroyAllWindows()
    env.close()


def _run_no_render(env: RobotArmEnvBase, n_episodes: int) -> None:
    total_steps, successes = [], 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done   = False
        ep_steps = 0
        while not done:
            _, _, terminated, success, truncated, _ = env.step(env.action_space.sample())
            ep_steps += 1
            done = terminated or truncated
        total_steps.append(ep_steps)
        if success:
            successes += 1
        print(f"  Ep {ep+1:3d}: steps={ep_steps:4d}, {'TERMINATED' if terminated else 'timeout'}, success={success}")
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
    print(f"\n{'='*65}\n [Demo] 观测空间验证 | 任务={reg['display_name']}\n{'='*65}")

    env = load_task(task_name, _make_robot_cfg(action_mode="joint", controller_type="osc",
                                               max_episode_steps=100))
    obs, _ = env.reset(seed=0)

    print("\n--- 观测空间结构 ---")
    for key, val in obs.items():
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}, "
              f"min={val.min():.2f}, max={val.max():.2f}")
    print(f"\n--- 动作空间 ---")
    print(f"  shape={env.action_space.shape}, "
          f"low={env.action_space.low[0]:.1f}, high={env.action_space.high[0]:.1f}")

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

    env = load_task(task_name, _make_robot_cfg(
        action_mode=action_mode, controller_type=controller_type,
        max_episode_steps=200, action_scale=0.03, action_scale_rot=0.06,
        control_freq=20.0,
    ))

    t0          = time.time()
    total_steps = 0
    success_num = 0

    for ep in range(n_episodes):
        env.reset(seed=ep)
        done = False
        while not done:
            _, _, terminated, success, truncated, _ = env.step(env.action_space.sample())
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
_GRIPPER_OPEN = np.array([_HAND_MAX, _HAND_MAX, _HAND_MIN, _HAND_MIN, _HAND_MIN, _HAND_MAX])


def demo_keyboard_control(
    task_name: str = "pick_and_place",
    action_mode: str = "joint",
    controller_type: str = "osc",
    arm_step: float = 0.05,
    hand_step: float = 0.0005,
    pos_step: float = 0.01,
    rot_step: float = 0.05,
    show_fingertip_midpoint: bool = True,
) -> None:
    """
    键盘控制模式（joint / ee 双模式，禁用超时，仅手动 R 重置）.

    ←/→ 切换关节，↑/↓ 调整，R 重置，O/C 张/握手，G 夹爪，Q 退出
    """
    reg    = TASK_REGISTRY[task_name]
    is_ee  = action_mode == "ee"

    print(f"\n{'='*65}\n [Demo] 键盘控制 | 任务={reg['display_name']}  模式={action_mode}")
    print(f" controller={controller_type}  超时=禁用（手动R重置）\n{'='*65}")

    env = load_task(task_name, _make_robot_cfg(
        action_mode=action_mode, controller_type=controller_type,
        max_episode_steps=999_999,
        action_scale=1.0, action_scale_rot=1.0, action_scale_hand=1.0,
        control_freq=20.0,
    ))

    cmd_q = queue.Queue()
    panel = KeyboardControlPanel(
        cmd_q, arm_dof=env.ARM_DOF, hand_dof=env.HAND_DOF,
        action_mode=action_mode,
        arm_step=arm_step, hand_step=hand_step,
        pos_step=pos_step, rot_step=rot_step,
    )
    threading.Thread(target=panel.run, daemon=True).start()
    panel.wait_ready(timeout=10.0)

    traj_vis = EETrajectoryVisualizer(TrajectoryVisualStyle(), max_history=40)
    ft_vis   = FingertipMidpointVisualizer() if show_fingertip_midpoint else None

    obs, _ = env.reset(seed=42)

    # 控制目标
    ee_target_pos = ee_target_quat = joint_target = hand_target = None

    def _init_targets():
        nonlocal ee_target_pos, ee_target_quat, joint_target, hand_target
        if is_ee:
            ee_target_pos, ee_target_quat = env.get_ee_pose()
            ee_target_pos  = ee_target_pos.copy()
            ee_target_quat = ee_target_quat.copy()
            hand_target    = env.get_hand_qpos().copy()
        else:
            joint_target = np.concatenate([env.get_arm_qpos(), env.get_hand_qpos()])

    _init_targets()

    sel_idx    = 0
    episode    = 1
    step       = 0
    ep_reward  = 0.0
    reward     = 0.0
    terminated = False
    running    = True

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance  = 1.8
        viewer.cam.elevation = -25
        viewer.cam.azimuth   = 135

        while viewer.is_running() and running:

            # ---- 消费命令队列 ----
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
                    if is_ee: hand_target[:] = _HAND_MIN
                    else: joint_target[env.ARM_DOF:env.ARM_DOF + env.HAND_DOF] = _HAND_MIN
                elif cmd == "close_hand":
                    if is_ee: hand_target[:] = _HAND_MAX
                    else: joint_target[env.ARM_DOF:env.ARM_DOF + env.HAND_DOF] = _HAND_MAX
                elif cmd == "gripper_open":
                    if is_ee: hand_target[:] = _GRIPPER_OPEN
                    else: joint_target[env.ARM_DOF:env.ARM_DOF + env.HAND_DOF] = _GRIPPER_OPEN
                elif cmd == "delta":
                    if not terminated:
                        _apply_delta(
                            val, sel_idx, is_ee, env,
                            ee_target_pos, ee_target_quat, hand_target, joint_target, panel,
                        )

            if not running:
                break

            # ---- 重置 ----
            if pending_reset:
                label = "✅ 任务成功！" if terminated else "🔄 手动重置"
                print(f"[回合 {episode}] {label}  累积奖励={ep_reward:.4f}  步数={step}")
                obs, _   = env.reset()
                traj_vis.reset()
                if ft_vis: ft_vis.reset()
                _init_targets()
                episode   += 1
                step       = 0
                ep_reward  = 0.0
                reward     = 0.0
                terminated = False

            # ---- 仿真步进 ----
            if not terminated:
                action = _build_action(
                    is_ee, env, ee_target_pos, ee_target_quat, hand_target, joint_target
                )
                obs, reward, terminated, success, _, _ = env.step(action)
                step     += 1
                ep_reward += reward
                if terminated:
                    print(f"[回合 {episode}] 终止，success={success}  步数={step}  累积奖励={ep_reward:.4f}  → 按 R 重置")

            # ---- 可视化 ----
            viewer.user_scn.ngeom = 0
            actual = env.get_ee_pose()[0]
            traj_vis.update(actual,
                            target_pos=ee_target_pos,
                            target_quat=ee_target_quat)
            traj_vis.draw(viewer)
            if ft_vis:
                ft_vis.update(env)
                ft_vis.draw(viewer)

            # 面板数据
            if is_ee:
                rpy  = _quat_to_euler_deg(ee_target_quat)
                sync = np.array([(hand_target[2] + hand_target[3] + hand_target[4]) / 3.0])
                display_vals = np.concatenate([ee_target_pos, rpy, hand_target, sync])
            else:
                hand_part = joint_target[env.ARM_DOF:env.ARM_DOF + env.HAND_DOF]
                sync = np.array([(hand_part[2] + hand_part[3] + hand_part[4]) / 3.0])
                display_vals = np.concatenate([joint_target[:env.ARM_DOF + env.HAND_DOF], sync])

            panel.update_state(
                sel_idx=sel_idx, display_vals=display_vals,
                reward=reward, ep_reward=ep_reward,
                step=step, terminated=terminated, episode=episode,
                info_extra={},
            )

            # CV 窗口
            _show_cv_windows(obs)
            _overlay_camera(obs, action_mode, reward, ep_reward, step, episode, terminated)
            viewer.sync()

    cv2.destroyAllWindows()
    env.close()
    print("\n[键盘控制] 已退出。")


def _apply_delta(val, sel_idx, is_ee, env, ee_target_pos, ee_target_quat,
                 hand_target, joint_target, panel) -> None:
    """将方向键增量应用到控制目标."""
    if is_ee:
        if sel_idx < 3:
            ee_target_pos[sel_idx] += val * panel._pos_step
        elif sel_idx < 6:
            axis  = np.zeros(3)
            axis[sel_idx - 3] = 1.0
            dq    = np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, axis, val * panel._rot_step)
            new_q = np.zeros(4)
            mujoco.mju_mulQuat(new_q, ee_target_quat, dq)
            norm  = np.linalg.norm(new_q)
            if norm > 1e-8:
                ee_target_quat[:] = new_q / norm
        else:
            hi = sel_idx - 6
            if hi < 6:
                hand_target[hi] = np.clip(hand_target[hi] + val * panel._hand_step, _HAND_MIN, _HAND_MAX)
            else:
                for si in [2, 3, 4]:
                    hand_target[si] = np.clip(hand_target[si] + val * panel._hand_step, _HAND_MIN, _HAND_MAX)
    else:
        if sel_idx < env.ARM_DOF:
            joint_target[sel_idx] += val * panel._arm_step
        elif sel_idx < env.ARM_DOF + env.HAND_DOF:
            joint_target[sel_idx] = np.clip(
                joint_target[sel_idx] + val * panel._hand_step, _HAND_MIN, _HAND_MAX
            )
        else:
            base = env.ARM_DOF
            for si in [base+2, base+3, base+4]:
                joint_target[si] = np.clip(joint_target[si] + val * panel._hand_step, _HAND_MIN, _HAND_MAX)


def _build_action(is_ee, env, ee_target_pos, ee_target_quat, hand_target, joint_target):
    """根据控制目标构造动作向量."""
    if is_ee:
        cur_pos, cur_quat = env.get_ee_pose()
        cur_inv   = np.zeros(4)
        mujoco.mju_negQuat(cur_inv, cur_quat)
        dq        = np.zeros(4)
        mujoco.mju_mulQuat(dq, ee_target_quat, cur_inv)
        rot_delta = np.zeros(3)
        mujoco.mju_quat2Vel(rot_delta, dq, 1.0)
        return np.concatenate([ee_target_pos - cur_pos, rot_delta,
                                hand_target - env.get_hand_qpos()]).astype(np.float32)
    else:
        current = np.concatenate([env.get_arm_qpos(), env.get_hand_qpos()])
        return (joint_target - current).astype(np.float32)


def _overlay_camera(obs, action_mode, reward, ep_reward, step, episode, terminated):
    """在相机画面上叠加文字并显示."""
    if "camera_rgb" not in obs:
        return
    cam_bgr = cv2.cvtColor(obs["camera_rgb"], cv2.COLOR_RGB2BGR)
    texts   = [
        f"[{action_mode.upper()}] Reward: {reward:+.4f}  Cum: {ep_reward:+.4f}",
        f"Step: {step}  Ep: {episode}  (R=reset, no timeout)",
    ]
    overlay = cam_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (640, 20 * len(texts) + 10), (0, 0, 0), -1)
    cam_bgr = cv2.addWeighted(overlay, 0.4, cam_bgr, 0.6, 0)
    for i, txt in enumerate(texts):
        y = 18 + i * 18
        cv2.putText(cam_bgr, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(cam_bgr, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    if terminated:
        cv2.putText(cam_bgr, "TASK SUCCESS!", (30, 130), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(cam_bgr, "TASK SUCCESS!", (30, 130), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0,255,80), 2, cv2.LINE_AA)
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.imshow("Camera", cam_bgr)
    cv2.resizeWindow("Camera", 640, 480)
    
    
# ====================== 模式 5：流程化任务执行 ======================

def demo_pipeline(
    task_name: str = "block_lifting",
    n_episodes: int = 3,
    render: bool = True,
    action_mode: str = "ee",
    controller_type: str = "osc",
    show_ee_traj: bool = True,
    show_fingertip_midpoint: bool = True,
) -> None:
    """
    流程化任务执行：按任务类型自动编排行为序列完成目标。
    
    两种运行模式：
      - render=True  : 可视化单回合演示（原有体验）
      - render=False : 无渲染批量测试，统计通关率
    """
    from .registry import load_strategy
    from .pipeline_overlay import PipelineStateOverlay

    reg = TASK_REGISTRY[task_name]
    print(f"\n{'='*65}\n [Demo] 流程化执行 | 任务={reg['display_name']}")
    print(f" action_mode={action_mode}, controller={controller_type}")
    print(f" 渲染模式: {'可视化' if render else '无渲染批量测试'}")
    print(f"{'='*65}")

    robot_cfg = _make_robot_cfg(
        action_mode=action_mode, controller_type=controller_type,
        max_episode_steps=1500,
        action_scale=0.05,
        action_scale_rot=0.1,
        action_scale_hand=0.005,
        control_freq=20.0,
    )
    env = load_task(task_name, robot_cfg)
    strategy = load_strategy(task_name)

    if not render:
        _run_pipeline_benchmark(env, strategy, task_name, n_episodes)
        return

    # ===== 渲染模式：单回合可视化演示 =====
    obs, info = env.reset(seed=42)
    strategy.reset()

    overlay = PipelineStateOverlay(strategy)
    traj_vis = EETrajectoryVisualizer(TrajectoryVisualStyle(), max_history=100) if show_ee_traj else None
    ft_vis = FingertipMidpointVisualizer() if show_fingertip_midpoint else None

    viewer = mujoco.viewer.launch_passive(env.model, env.data)
    try:
        viewer.cam.distance = 1.5
        viewer.cam.elevation = -30
        viewer.cam.azimuth = 120

        episode, step = 0, 0

        while viewer.is_running() and episode < n_episodes:
            action, action_context = strategy.tick(obs, info, step, env)
            ee_target_pos = action_context.ee_target_pos
            ee_target_quat = action_context.ee_target_quat
            ee_delta_pos = action_context.ee_delta_pos
            ee_delta_rot = action_context.ee_delta_rot

            # 反推绝对目标（仅用于可视化）
            if ee_target_pos is None and ee_target_quat is None:
                if ee_delta_pos is not None and ee_delta_rot is not None:
                    ee_pos, ee_quat = env.get_ee_pose()
                    ee_target_pos = ee_pos + ee_delta_pos
                    delta_quat = np.zeros(4)
                    mujoco.mju_euler2Quat(delta_quat, ee_delta_rot, 'xyz')
                    ee_target_quat = np.zeros(4)
                    mujoco.mju_mulQuat(ee_target_quat, delta_quat, ee_quat)

            obs, reward, terminated, success, truncated, info = env.step(action)
            step += 1

            switch_msg = overlay.update(step)
            if switch_msg:
                print(switch_msg)

            # ---- 可视化 ----
            viewer.user_scn.ngeom = 0
            if traj_vis:
                actual = env.get_ee_pose()[0]
                traj_vis.update(actual, target_pos=ee_target_pos, target_quat=ee_target_quat)
                traj_vis.draw(viewer)
            if ft_vis:
                ft_vis.update(env)
                ft_vis.draw(viewer)

            overlay.draw_viewer_indicator(viewer)
            _show_cv_windows_pipeline(obs, overlay, reward, env)
            viewer.sync()

            if terminated or truncated:
                status = "✓ 成功" if success else "✗ 失败/超时"
                print(f"\n  [回合 {episode+1} 结束] {status}")
                print(f"   总步数: {step} | 最终阶段: {overlay._get_phase_name(strategy.phase_idx)}")

                status_dict = strategy.get_status_dict()
                print("   策略内部状态：")
                for k, v in status_dict.items():
                    print(f"   {k}: {v}")

                print(f"  {'='*40}")

                episode += 1
                step = 0
                if traj_vis:
                    traj_vis.reset()
                if ft_vis:
                    ft_vis.reset()
                overlay.reset()
                if episode < n_episodes:
                    obs, info = env.reset()
                    strategy.reset()

    finally:
        # 严格保证析构顺序：viewer 先关闭释放 OpenGL/渲染资源，
        # cv2 窗口次之，env 最后清理 MuJoCo 数据——顺序错误会导致 segfault。
        viewer.close()
        cv2.destroyAllWindows()
        env.close()


def _run_pipeline_benchmark(
    env: RobotArmEnvBase,
    strategy,
    task_name: str,
    n_episodes: int,
) -> None:
    """
    无渲染批量基准测试：统计通关率、平均步数、耗时。
    """
    print(f"\n[Pipeline Benchmark] 开始 {n_episodes} 回合无渲染测试...\n")

    t0 = time.time()
    total_steps = []
    successes = 0
    timeouts = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        strategy.reset()
        step = 0
        done = False

        while not done:
            action, action_context = strategy.tick(obs, info, step, env)
            obs, _, terminated, success, truncated, info = env.step(action)
            step += 1
            done = terminated or truncated

        total_steps.append(step)
        if success:
            successes += 1
        if truncated:
            timeouts += 1

        status = "SUCCESS" if success else ("TIMEOUT" if truncated else "FAIL")
        print(f"  Ep {ep+1:3d}/{n_episodes}: steps={step:4d} | {status}")

    elapsed = time.time() - t0

    print(f"\n{'='*50}")
    print(f"  Pipeline Benchmark 结果 ({task_name})")
    print(f"{'='*50}")
    print(f"  总回合数:   {n_episodes}")
    print(f"  成功数:     {successes}")
    print(f"  超时数:     {timeouts}")
    print(f"  失败数:     {n_episodes - successes - timeouts}")
    print(f"  通关率:     {successes / n_episodes * 100:.1f}%")
    print(f"  平均步数:   {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
    print(f"  总耗时:     {elapsed:.1f}s")
    print(f"  回合/秒:    {n_episodes / elapsed:.2f}")
    print(f"  步/秒:      {sum(total_steps) / elapsed:.0f}")
    print(f"{'='*50}")

    env.close()


def _show_cv_windows_pipeline(obs: dict, overlay: PipelineStateOverlay,
                              reward: float, env) -> None:
    """
    通用 OpenCV 显示，不依赖具体任务.
    """
    # 触觉热力图
    from .heatmap import render_tactile_heatmap
    heatmap = render_tactile_heatmap(obs)
    cv2.namedWindow("Tactile (Top/Mid/Bot)", cv2.WINDOW_NORMAL)
    cv2.imshow("Tactile (Top/Mid/Bot)", heatmap)
    cv2.resizeWindow("Tactile (Top/Mid/Bot)", 1000, 480)

    # 相机画面 + 通用状态叠加
    if "camera_rgb" in obs:
        cam_bgr = cv2.cvtColor(obs["camera_rgb"], cv2.COLOR_RGB2BGR)
        cam_bgr = overlay.draw_camera_overlay(cam_bgr, reward)

        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.imshow("Camera", cam_bgr)
        cv2.resizeWindow("Camera", 640, 480)

    cv2.waitKey(1)