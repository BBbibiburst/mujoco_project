"""
通用任务环境演示脚本（视觉-触觉-本体感觉版本）.

支持所有继承自 RobotArmEnvBase 的任务环境，通过 --task 参数切换任务。

运行方式：
    # 从项目根目录执行
    python -m src.env.task_demo --task pick_place
    python -m src.env.task_demo --task stack
    python -m src.env.task_demo --task insert
    python -m src.env.task_demo --task reorient
    python -m src.env.task_demo --task push

完整参数示例：
    python -m src.env.task_demo \\
        --task stack \\
        --mode random \\
        --episodes 5 \\
        --action-mode osc_pose \\
        --controller osc \\
        --no-render

功能：
    1. random   : 随机策略回合演示（仿真窗口 + 触觉热力图 + 相机画面）
    2. scripted : 脚本化策略演示（仅 pick_place / stack 支持，其余降级为随机）
    3. verify   : 观测空间形状与数值范围验证
    4. benchmark: 无渲染高速基准测试（N 回合）
    5. compare  : 对比当前任务的不同 action_mode × controller 组合
"""

import sys
import time
from pathlib import Path
from typing import Optional, Type

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np
import cv2

from src.env.base_env import RobotArmEnvBase, RobotConfig


# ====================== 任务注册表 ======================
# 延迟导入，避免初始化时加载所有环境（加载 MuJoCo 模型耗时）

# ====================== 任务注册表 ======================
# 延迟导入，避免初始化时加载所有环境（加载 MuJoCo 模型耗时）
TASK_REGISTRY: dict = {
    "pick_place": {
        "module": "src.env.pick_place_env",
        "env_class": "PickPlaceEnv",
        "cfg_class": "PickPlaceConfig",
        "display_name": "Pick and Place",  # 修改为英文
        "default_cfg_kwargs": {
            "r_step_penalty": -0.005,
            "r_place_bonus": 100.0,
            "r_grasp_bonus": 10.0,
        },
        # 展示哪些 info 字段（key: 英文标签）
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
        "display_name": "Stack Blocks",  # 修改为英文
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
        "display_name": "Insert Peg",  # 修改为英文
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
        "display_name": "Reorient Object",  # 修改为英文
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
        "display_name": "Push Object",  # 修改为英文
        "default_cfg_kwargs": {},
        "info_display": {
            "phase": "Phase",
            "dist_obj_target": "Obj-Target Dist(m)",
            "tactile_max": "Tactile Max(Norm)",
            "is_success": "Success",
        },
    },
}


def _load_task(task_name: str, robot_cfg: RobotConfig) -> RobotArmEnvBase:
    """动态加载任务环境，避免顶层全量导入."""
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"未知任务: '{task_name}'。"
            f"可用任务: {list(TASK_REGISTRY.keys())}"
        )
    reg = TASK_REGISTRY[task_name]

    import importlib
    mod = importlib.import_module(reg["module"])
    EnvClass = getattr(mod, reg["env_class"])
    CfgClass = getattr(mod, reg["cfg_class"])

    task_cfg = CfgClass(**reg["default_cfg_kwargs"])
    return EnvClass(robot_config=robot_cfg, task_config=task_cfg)


# ====================== 通用可视化工具 ======================

def render_tactile_heatmap(obs: dict, sub_h: int = 160, sub_w: int = 200) -> np.ndarray:
    """
    将扁平化触觉图像渲染为热力图网格。
    支持任意包含 tactile_bottom / tactile_middle / tactile_top 键的观测字典。

    布局：行=指节层（top/middle/bottom），列=手指（5根）
    返回 shape: (3*sub_h, 5*sub_w, 3)
    """
    from src.sensors.tactile_sensor import FINGER_PHALANX_ORDER

    finger_keys = ["finger_0", "finger_1", "finger_2", "finger_3", "thumb"]
    level_order = ["top", "middle", "bottom"]
    level_to_key = {
        "top": "tactile_top",
        "middle": "tactile_middle",
        "bottom": "tactile_bottom",
    }
    level_to_phalanx_idx = {"top": 2, "middle": 1, "bottom": 0}

    grid_rows = []
    for level in level_order:
        tac_key = level_to_key[level]
        if tac_key not in obs:
            continue

        imgs = obs[tac_key]  # (5, H, W) 或 (5, H, W, 1) for TactileShapeWrapper
        if imgs.ndim == 4:
            imgs = imgs[..., 0]   # squeeze 最后维度

        row_frames = []
        for finger_idx, finger in enumerate(finger_keys):
            img = imgs[finger_idx]  # (H, W)

            # 增强对比度
            enhanced = np.clip(img.astype(np.float32) * 5.0, 0, 255).astype(np.uint8)
            resized = cv2.resize(enhanced, (sub_w, sub_h), interpolation=cv2.INTER_NEAREST)
            heatmap = cv2.applyColorMap(resized, cv2.COLORMAP_JET)

            # 标题
            phalanx_name = FINGER_PHALANX_ORDER[finger][level_to_phalanx_idx[level]]
            parts = phalanx_name.split('_')
            if parts[0] == "thumb":
                short_name = f"T_{parts[1][:3].capitalize()}"
            else:
                short_name = f"F{parts[1]}_{parts[2][:3].capitalize()}"

            cv2.rectangle(heatmap, (0, 0), (sub_w, 22), (0, 0, 0), -1)
            cv2.putText(heatmap, short_name, (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            row_frames.append(heatmap)

        grid_rows.append(np.hstack(row_frames))

    if not grid_rows:
        return np.zeros((sub_h * 3, sub_w * 5, 3), dtype=np.uint8)
    return np.vstack(grid_rows)


def _format_info_line(info: dict, display_keys: dict) -> str:
    """将 info 字典的指定字段格式化为单行字符串."""
    parts = []
    for k, label in display_keys.items():
        v = info.get(k, "N/A")
        if isinstance(v, float):
            parts.append(f"{label}={v:.3f}")
        elif isinstance(v, bool):
            parts.append(f"{label}={'✓' if v else '✗'}")
        else:
            parts.append(f"{label}={v}")
    return " | ".join(parts)


# ====================== 演示模式1：随机策略 ======================

def demo_random_policy(
    task_name: str = "pick_place",
    n_episodes: int = 3,
    render: bool = True,
    action_mode: str = "osc_pose",
    controller_type: str = "osc",
):
    """随机策略演示：仿真窗口 + 触觉热力图 + 任务状态信息."""
    reg = TASK_REGISTRY[task_name]
    print("=" * 65)
    print(f"  [Demo] 随机策略 | 任务={reg['display_name']}")
    print(f"         action_mode={action_mode}, controller={controller_type}")
    print("=" * 65)

    robot_cfg = RobotConfig(
        action_mode=action_mode,
        controller_type=controller_type,
        max_episode_steps=200,
        action_scale=0.03,
        action_scale_rot=0.06,
        control_freq=20.0,
        tactile_backend="simple_avg",
    )
    env = _load_task(task_name, robot_cfg)
    info_display = reg["info_display"]

    if render:
        obs, info = env.reset(seed=42)
        print(f"\n[初始化] obs keys: {list(obs.keys())}")
        for k, v in obs.items():
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        print(f"  action_dim: {env.action_space.shape[0]}")

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            episode = 0
            step = 0
            ep_reward = 0.0

            while viewer.is_running() and episode < n_episodes:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                step += 1

                # 触觉热力图
                heatmap = render_tactile_heatmap(obs)
                cv2.namedWindow("Tactile (Top/Mid/Bot)", cv2.WINDOW_NORMAL)
                cv2.imshow("Tactile (Top/Mid/Bot)", heatmap)
                cv2.resizeWindow("Tactile (Top/Mid/Bot)", 1000, 480)

                # 相机图像
                cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
                cam_bgr = cv2.cvtColor(obs["camera_rgb"], cv2.COLOR_RGB2BGR)
                # 叠加任务信息文字
                info_str = _format_info_line(info, info_display)
                cv2.putText(cam_bgr, info_str, (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow("Camera", cam_bgr)
                cv2.resizeWindow("Camera", 640, 480)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                viewer.sync()

                if terminated or truncated:
                    status = "✓ 成功" if terminated else "✗ 超时"
                    info_line = _format_info_line(info, info_display)
                    print(
                        f"[Episode {episode+1}] {status} | "
                        f"steps={step}, reward={ep_reward:.2f} | {info_line}"
                    )
                    episode += 1
                    step = 0
                    ep_reward = 0.0
                    if episode < n_episodes:
                        obs, info = env.reset()
    else:
        total_rewards, total_steps, successes = [], [], 0
        for ep in range(n_episodes):
            obs, info = env.reset(seed=ep)
            ep_reward = 0.0
            ep_steps = 0
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_steps += 1
                done = terminated or truncated
            total_rewards.append(ep_reward)
            total_steps.append(ep_steps)
            if terminated:
                successes += 1
            info_line = _format_info_line(info, info_display)
            print(
                f"  Ep {ep+1:3d}: reward={ep_reward:7.2f}, "
                f"steps={ep_steps:4d}, {'SUCCESS' if terminated else 'timeout'} | {info_line}"
            )
        print(f"\n  平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"  平均步数: {np.mean(total_steps):.1f}")
        print(f"  成功率:   {successes/n_episodes*100:.1f}%")

    cv2.destroyAllWindows()
    env.close()


# ====================== 演示模式2：脚本化策略 ======================

def demo_scripted_policy(
    task_name: str = "pick_place",
    render: bool = True,
    action_mode: str = "osc_pose",
    controller_type: str = "osc",
):
    """
    脚本化策略演示。

    pick_place / stack：执行完整的抓取+运输脚本
    insert / reorient  ：执行接近+抓取脚本（对准/重定向留给随机策略）
    push               ：执行接近+推动脚本
    其余未知任务        ：降级为随机策略
    """
    reg = TASK_REGISTRY.get(task_name)
    print("=" * 65)
    print(f"  [Demo] 脚本化策略 | 任务={reg['display_name'] if reg else task_name}")
    print("=" * 65)

    robot_cfg = RobotConfig(
        action_mode=action_mode,
        controller_type=controller_type,
        max_episode_steps=500,
        action_scale=0.02,
        action_scale_rot=0.04,
        control_freq=20.0,
        tactile_backend="simple_avg",
    )
    env = _load_task(task_name, robot_cfg)
    obs, info = env.reset(seed=0)

    print(f"  动作维度: {env.action_space.shape[0]}")
    if reg:
        print(f"  初始状态: {_format_info_line(info, reg['info_display'])}")

    # ---- 通用辅助函数 ----
    def get_ee_pos():
        pos, _ = env.get_ee_pose()
        return pos

    def move_to(target_pos, hand_target, viewer, tol=0.03, max_steps=150):
        """控制末端移动到目标位置（与任务无关的通用实现）."""
        for _ in range(max_steps):
            ee_pos = get_ee_pos()
            delta = target_pos - ee_pos
            dist = np.linalg.norm(delta)
            if dist < tol:
                return True
            direction = delta / (dist + 1e-6)
            arm_action = np.clip(direction * 2.0, -1, 1)
            hand_act = np.full(6, hand_target)

            if env.cfg.action_mode == "osc_pose":
                action = np.concatenate([arm_action, np.zeros(3), hand_act])
            elif env.cfg.action_mode == "osc_pos":
                action = np.concatenate([arm_action, hand_act])
            else:
                raise NotImplementedError("joint_pd 不支持脚本化策略")

            env.step(action)
            if viewer is not None and hasattr(viewer, "sync"):
                viewer.sync()
        return False

    def close_fingers(n_steps: int, viewer, hand_val: float = 1.0):
        for _ in range(n_steps):
            if env.cfg.action_mode == "osc_pose":
                action = np.concatenate([np.zeros(6), np.full(6, hand_val)])
            else:
                action = np.concatenate([np.zeros(3), np.full(6, hand_val)])
            env.step(action)
            if viewer is not None and hasattr(viewer, "sync"):
                viewer.sync()

    # ---- 选择脚本 ----
    def run(viewer):
        if task_name in ("pick_place", "stack"):
            _script_pick_and_transport(env, task_name, get_ee_pos, move_to, close_fingers, viewer)
        elif task_name == "insert":
            _script_insert(env, get_ee_pos, move_to, close_fingers, viewer)
        elif task_name == "reorient":
            _script_reorient(env, get_ee_pos, move_to, close_fingers, viewer)
        elif task_name == "push":
            _script_push(env, get_ee_pos, move_to, viewer)
        else:
            print("  [提示] 未知任务，执行随机策略 100 步...")
            for _ in range(100):
                env.step(env.action_space.sample())
                if viewer:
                    viewer.sync()

    class FakeViewer:
        def sync(self): pass
        def is_running(self): return True

    if render:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            run(viewer)
    else:
        run(FakeViewer())

    cv2.destroyAllWindows()
    env.close()


def _script_pick_and_transport(env, task_name, get_ee_pos, move_to, close_fingers, viewer):
    """通用抓取+运输脚本（pick_place 和 stack 共用）."""
    # 获取物体位置
    if task_name == "pick_place":
        obj_pos = env._get_obj_pos()
    else:  # stack
        obj_pos = env._get_block_a_pos()

    print(f"\n  阶段1: 移到物体正上方 {obj_pos[:2]}")
    ok = move_to(obj_pos + [0, 0, 0.15], hand_target=0.0, viewer=viewer)
    print(f"    {'到达' if ok else '超时'}")

    print("  阶段2: 下降接近物体")
    if task_name == "pick_place":
        obj_pos = env._get_obj_pos()
    else:
        obj_pos = env._get_block_a_pos()
    ok = move_to(obj_pos + [0, 0, 0.02], hand_target=0.0, viewer=viewer, tol=0.03)
    print(f"    {'到达' if ok else '超时'}")

    print("  阶段3: 闭合手指")
    close_fingers(60, viewer, hand_val=1.0)

    print("  阶段4: 抬起")
    ee = get_ee_pos()
    ok = move_to(ee + [0, 0, 0.15], hand_target=1.0, viewer=viewer)
    if task_name == "pick_place":
        obj_final = env._get_obj_pos()
    else:
        obj_final = env._get_block_a_pos()
    print(f"    {'成功' if ok else '超时'} | 物体高度: {obj_final[2]:.3f}m")

    # stack：移动到 block_B 上方
    if task_name == "stack":
        target_xy = env._block_b_pos[:2]
        stack_target = np.array([target_xy[0], target_xy[1], 0.15])
        print(f"  阶段5: 运输到 block_B 上方 {target_xy}")
        ok = move_to(stack_target, hand_target=1.0, viewer=viewer)
        print(f"    {'到达' if ok else '超时'}")

        print("  阶段6: 缓慢下降放置")
        place_target = np.array([target_xy[0], target_xy[1], env.task_cfg.obj_size * 2.5])
        ok = move_to(place_target, hand_target=1.0, viewer=viewer, tol=0.02)
        close_fingers(30, viewer, hand_val=0.0)   # 松开
        print(f"    {'成功' if ok else '超时'} | 堆叠: {env._is_stacked_success()}")

    print(f"\n  最终 info: {env._get_info()}")


def _script_insert(env, get_ee_pos, move_to, close_fingers, viewer):
    """插孔任务脚本."""
    peg_pos = env._get_peg_pos()
    print(f"\n  阶段1: 移到 peg 上方")
    move_to(peg_pos + [0, 0, 0.15], hand_target=0.0, viewer=viewer)

    print("  阶段2: 下降抓取 peg")
    peg_pos = env._get_peg_pos()
    move_to(peg_pos + [0, 0, 0.04], hand_target=0.0, viewer=viewer, tol=0.03)

    print("  阶段3: 闭合手指")
    close_fingers(60, viewer)

    print("  阶段4: 抬起 peg")
    ee = get_ee_pos()
    move_to(ee + [0, 0, 0.12], hand_target=1.0, viewer=viewer)

    print("  阶段5: 水平对准 hole")
    hole_xy = env._hole_pos[:2]
    align_target = np.array([hole_xy[0], hole_xy[1], get_ee_pos()[2]])
    ok = move_to(align_target, hand_target=1.0, viewer=viewer, tol=0.02)
    print(f"    {'对准' if ok else '超时'} | XY误差: {np.linalg.norm(env._get_peg_pos()[:2] - hole_xy):.3f}m")

    print("  阶段6: 尝试垂直插入（有限步数）")
    insert_target = np.array([hole_xy[0], hole_xy[1], 0.01])
    move_to(insert_target, hand_target=1.0, viewer=viewer, tol=0.01, max_steps=80)
    print(f"  最终 info: {env._get_info()}")


def _script_reorient(env, get_ee_pos, move_to, close_fingers, viewer):
    """重定向任务脚本（只完成抓取，旋转用随机动作模拟）."""
    obj_pos = env._get_obj_pos()
    print(f"\n  阶段1: 接近物体")
    move_to(obj_pos + [0, 0, 0.12], hand_target=0.0, viewer=viewer)

    obj_pos = env._get_obj_pos()
    move_to(obj_pos + [0, 0, 0.02], hand_target=0.0, viewer=viewer, tol=0.03)

    print("  阶段2: 抓取")
    close_fingers(60, viewer)

    print("  阶段3: 随机旋转 200 步（演示触觉反馈）")
    import numpy as np
    for _ in range(200):
        if env.cfg.action_mode == "osc_pose":
            action = np.concatenate([
                np.zeros(3),
                np.random.uniform(-0.5, 0.5, 3),   # 随机旋转分量
                np.ones(6),                          # 保持闭合
            ])
        else:
            action = np.concatenate([np.zeros(3), np.ones(6)])
        env.step(action)
        if viewer and hasattr(viewer, "sync"):
            viewer.sync()

    print(f"  最终 info: {env._get_info()}")


def _script_push(env, get_ee_pos, move_to, viewer):
    """推动任务脚本."""
    obj_pos = env._get_obj_pos()
    target_xy = env._target_pos[:2]

    # 计算推动方向：从物体背面接近
    push_dir = target_xy - obj_pos[:2]
    norm = np.linalg.norm(push_dir)
    if norm > 1e-6:
        push_dir = push_dir / norm

    approach_pt = obj_pos[:2] - push_dir * 0.08
    approach_3d = np.array([approach_pt[0], approach_pt[1], obj_pos[2] + 0.01])

    print(f"\n  阶段1: 从背面接近物体")
    ok = move_to(approach_3d, hand_target=0.5, viewer=viewer, tol=0.04)
    print(f"    {'到达' if ok else '超时'}")

    print("  阶段2: 向目标推动 200 步")
    for _ in range(200):
        ee_pos = get_ee_pos()
        obj_pos = env._get_obj_pos()
        delta = target_xy - obj_pos[:2]
        dist = np.linalg.norm(delta)
        if dist < 0.04:
            print("    物体已接近目标，停止推动")
            break
        direction = np.array([delta[0], delta[1], 0.0]) / (dist + 1e-6)
        arm_action = np.clip(direction * 1.5, -1, 1)
        if env.cfg.action_mode == "osc_pose":
            action = np.concatenate([arm_action, np.zeros(3), np.full(6, 0.5)])
        else:
            action = np.concatenate([arm_action, np.full(6, 0.5)])
        env.step(action)
        if viewer and hasattr(viewer, "sync"):
            viewer.sync()

    print(f"  最终 info: {env._get_info()}")


# ====================== 演示模式3：观测空间验证 ======================

def demo_verify_observation_space(task_name: str = "pick_place"):
    """验证所有观测分量的形状与数值范围."""
    reg = TASK_REGISTRY[task_name]
    print("=" * 65)
    print(f"  [Demo] 观测空间验证 | 任务={reg['display_name']}")
    print("=" * 65)

    robot_cfg = RobotConfig(
        action_mode="osc_pose",
        controller_type="osc",
        max_episode_steps=100,
        tactile_backend="simple_avg",
    )
    env = _load_task(task_name, robot_cfg)
    obs, info = env.reset(seed=0)

    print("\n--- 观测空间结构 ---")
    for key, val in obs.items():
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}, "
              f"min={val.min():.2f}, max={val.max():.2f}")

    print("\n--- 动作空间 ---")
    print(f"  shape={env.action_space.shape}, "
          f"low={env.action_space.low[0]:.1f}, high={env.action_space.high[0]:.1f}")

    print(f"\n--- 初始任务状态 ---")
    for k, label in reg["info_display"].items():
        print(f"  {label}: {info.get(k, 'N/A')}")

    # 可视化
    cam_bgr = cv2.cvtColor(obs["camera_rgb"], cv2.COLOR_RGB2BGR)
    cv2.namedWindow("Camera RGB", cv2.WINDOW_NORMAL)
    cv2.imshow("Camera RGB", cam_bgr)
    cv2.resizeWindow("Camera RGB", 640, 480)

    heatmap = render_tactile_heatmap(obs)
    cv2.namedWindow("Tactile Heatmap", cv2.WINDOW_NORMAL)
    cv2.imshow("Tactile Heatmap", heatmap)
    cv2.resizeWindow("Tactile Heatmap", 1000, 480)

    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    env.close()


# ====================== 演示模式4：基准测试 ======================

def demo_benchmark(
    task_name: str = "pick_place",
    n_episodes: int = 100,
    action_mode: str = "osc_pose",
    controller_type: str = "osc",
):
    """无渲染高速基准测试."""
    reg = TASK_REGISTRY[task_name]
    print(f"[Benchmark] 任务={reg['display_name']}, n_episodes={n_episodes}")

    robot_cfg = RobotConfig(
        action_mode=action_mode,
        controller_type=controller_type,
        max_episode_steps=200,
        action_scale=0.03,
        control_freq=20.0,
        tactile_backend="simple_avg",
    )
    env = _load_task(task_name, robot_cfg)

    t0 = time.time()
    total_steps = 0
    successes = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            total_steps += 1
            done = terminated or truncated
        if terminated:
            successes += 1

    elapsed = time.time() - t0
    env.close()

    print(f"  总步数: {total_steps} | 总时间: {elapsed:.1f}s")
    print(f"  步频: {total_steps/elapsed:.0f} steps/s")
    print(f"  回合频: {n_episodes/elapsed:.1f} eps/s")
    print(f"  成功率: {successes/n_episodes*100:.1f}%")


# ====================== 演示模式5：多配置对比 ======================

def demo_compare_configs(task_name: str = "pick_place"):
    """对比同任务下不同 action_mode × controller_type 组合."""
    reg = TASK_REGISTRY[task_name]
    print(f"\n[Compare] 任务={reg['display_name']}")

    configs = [
        ("osc_pose", "osc", "OSC 6D位姿"),
        ("osc_pos",  "osc", "OSC 3D位置"),
        ("osc_pos",  "ik",  "IK 3D位置"),
        ("joint_pd", "osc", "OSC 关节PD"),
    ]
    for action_mode, controller_type, desc in configs:
        print(f"\n  --- {desc} (action={action_mode}, ctrl={controller_type}) ---")
        try:
            demo_random_policy(
                task_name=task_name,
                n_episodes=2,
                render=False,
                action_mode=action_mode,
                controller_type=controller_type,
            )
        except Exception as e:
            print(f"  [错误] {e}")


# ====================== 入口 ======================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="通用任务环境演示（支持5个灵巧手任务）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=list(TASK_REGISTRY.keys()),
        default="pick_place",
        help="要演示的任务名称：\n"
             + "\n".join(f"  {k}: {v['display_name']}" for k, v in TASK_REGISTRY.items()),
    )
    parser.add_argument(
        "--mode",
        choices=["random", "scripted", "verify", "benchmark", "compare"],
        default="random",
        help="演示模式",
    )
    parser.add_argument("--no-render", action="store_true", help="禁用可视化（加速运行）")
    parser.add_argument("--episodes", type=int, default=3, help="演示回合数")
    parser.add_argument(
        "--action-mode",
        choices=["osc_pose", "osc_pos", "joint_pd"],
        default="osc_pose",
    )
    parser.add_argument(
        "--controller",
        choices=["osc", "ik"],
        default="osc",
    )
    args = parser.parse_args()

    render = not args.no_render

    print(f"\n{'='*65}")
    print(f"  任务: {TASK_REGISTRY[args.task]['display_name']}")
    print(f"  模式: {args.mode}")
    print(f"  渲染: {'是' if render else '否'}")
    print(f"{'='*65}\n")

    if args.mode == "random":
        demo_random_policy(
            task_name=args.task,
            n_episodes=args.episodes,
            render=render,
            action_mode=args.action_mode,
            controller_type=args.controller,
        )
    elif args.mode == "scripted":
        demo_scripted_policy(
            task_name=args.task,
            render=render,
            action_mode=args.action_mode,
            controller_type=args.controller,
        )
    elif args.mode == "verify":
        demo_verify_observation_space(task_name=args.task)
    elif args.mode == "benchmark":
        demo_benchmark(
            task_name=args.task,
            n_episodes=args.episodes if args.episodes != 3 else 100,
            action_mode=args.action_mode,
            controller_type=args.controller,
        )
    elif args.mode == "compare":
        demo_compare_configs(task_name=args.task)
