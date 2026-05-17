"""
通用任务环境演示脚本

基于模块化架构的演示入口，支持所有继承自 RobotArmEnvBase 的任务环境。
通过 --task 参数切换任务，通过 --mode 参数切换演示模式。

运行方式：
    python -m source.env_demos.demo --task block_lifting --mode random
    python -m source.env_demos.demo --task block_lifting --mode keyboard --action-mode ee
    # 可视化模式测试 pipeline（默认，跑3回合）
    python -m source.env_demos.demo --task block_lifting --mode pipeline
    # 无渲染批量测试 pipeline 100次
    python -m source.env_demos.demo --task block_lifting --mode pipeline --no-render --episodes 100

模式说明：
    random    : 随机策略回合演示（仿真窗口 + 触觉热力图 + 相机画面 + 可视化）
    verify    : 观测空间形状与数值范围验证
    benchmark : 无渲染高速基准测试（N 回合）
    keyboard  : 键盘逐关节控制（Tkinter 控制面板 + 实时状态显示）
    pipeline  : 基于流程的任务完成演示和测试
"""

import argparse

from source.env_demos.registry import TASK_REGISTRY
from source.env_demos.modes import (
    demo_random_policy,
    demo_verify_observation_space,
    demo_benchmark,
    demo_keyboard_control,
    demo_pipeline,
)



def main():
    parser = argparse.ArgumentParser(
        description="通用任务环境演示（重构版）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=list(TASK_REGISTRY.keys()),
        default="block_lifting",
        help="任务名称",
    )
    parser.add_argument(
        "--mode",
        choices=["random", "verify", "benchmark", "keyboard", "pipeline"],
        default="random",
        help=(
            "演示模式：\n"
            "  random    随机策略 + 可视化\n"
            "  verify    观测空间验证\n"
            "  benchmark 无渲染基准测试\n"
            "  keyboard  键盘控制\n"
            "  pipeline  流程化任务执行（自动策略）\n"
        ),
    )
    parser.add_argument("--no-render", action="store_true", help="禁用可视化（random模式）")
    parser.add_argument("--no-traj", action="store_true", help="禁用末端轨迹可视化")
    parser.add_argument("--no-ft-mid", action="store_true", help="禁用指尖中点可视化")
    parser.add_argument("--episodes", type=int, default=3, help="回合数")
    parser.add_argument("--action-mode", choices=["joint", "ee"], default="joint")
    parser.add_argument("--controller", choices=["osc", "ik"], default="osc")
    parser.add_argument("--arm-step", type=float, default=0.05, help="键盘模式臂步长(rad)")
    parser.add_argument("--hand-step", type=float, default=0.0005, help="键盘模式手步长(m)")
    parser.add_argument("--pos-step", type=float, default=0.01, help="键盘模式位置步长(m)")
    parser.add_argument("--rot-step", type=float, default=0.05, help="键盘模式旋转步长(rad)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--log_info", action="store_true", help="记录回合信息")
    parser.add_argument("--n_workers", type=int, default=32, help="无渲染模式并行环境数")

    args = parser.parse_args()

    render = not args.no_render
    show_traj = not args.no_traj
    show_ft_mid = not args.no_ft_mid

    print(f"\n{'='*65}")
    print(f"  任务:       {TASK_REGISTRY[args.task]['display_name']}")
    print(f"  模式:       {args.mode}")
    if args.mode == "random":
        print(f"  渲染:       {'是' if render else '否'}")
        print(f"  轨迹可视化: {'是' if show_traj else '否'}")
        print(f"  指尖中点:   {'是' if show_ft_mid else '否'}")
    elif args.mode == "keyboard":
        print(f"  臂步长:     {args.arm_step} rad")
        print(f"  手步长:     {args.hand_step} m")
        print(f"  指尖中点:   {'是' if show_ft_mid else '否'}")
    print(f"{'='*65}\n")

    if args.mode == "random":
        demo_random_policy(
            task_name=args.task,
            n_episodes=args.episodes,
            render=render,
            action_mode=args.action_mode,
            controller_type=args.controller,
            show_ee_traj=show_traj,
            show_fingertip_midpoint=show_ft_mid,
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
    elif args.mode == "keyboard":
        demo_keyboard_control(
            task_name=args.task,
            action_mode=args.action_mode,
            controller_type=args.controller,
            arm_step=args.arm_step,
            hand_step=args.hand_step,
            pos_step=args.pos_step,
            rot_step=args.rot_step,
            show_fingertip_midpoint=show_ft_mid,
            seed=args.seed,
            log_info=args.log_info,
        )
    elif args.mode == "pipeline":
        demo_pipeline(
            task_name=args.task,
            n_episodes=args.episodes,
            render=render,
            action_mode=args.action_mode,
            controller_type=args.controller,
            show_ee_traj=show_traj,
            show_fingertip_midpoint=show_ft_mid,
            seed=args.seed,
            log_info=args.log_info,
        )


if __name__ == "__main__":
    main()