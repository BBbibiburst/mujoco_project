#!/usr/bin/env python3
"""
Multi-Stage Diffusion Policy - 主入口
整合所有模块的训练和推理

用法:
  训练: python -m source.train.main --mode train
  演示: python -m source.train.main --mode demo
  推理: python -m source.train.main --mode infer --model_path <model_path> --stats_path <stats_path>
"""

import os
import sys
import argparse
from pathlib import Path

# 项目路径配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "block_lifting"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "models" / "block_lifting"
DEFAULT_CACHE_PATH = PROJECT_ROOT / "cache" / "block_lifting"

# 将项目根目录加入Python路径
sys.path.insert(0, str(PROJECT_ROOT))

# 导入本地模块
from dataset import LazyMultiModalGraspDataset, demo_dataset
from encoders import demo_encoders
from unet import demo_unet
from model import MultiStageDiffusionPolicy, train, demo_model
from inference import MultiStageDiffusionPolicyInference, demo_inference


def main():
    parser = argparse.ArgumentParser(description='Multi-Stage Diffusion Policy')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'demo', 'infer', 'dataset', 'encoders', 'unet'],
                       help='运行模式')

    # 路径参数 (使用项目相对路径)
    parser.add_argument('--data_dir', type=str, default=str(DEFAULT_DATA_PATH),
                       help='数据目录 (默认: PROJECT_ROOT/data/block_lifting)')
    parser.add_argument('--output_dir', type=str, default=str(DEFAULT_OUTPUT_PATH),
                       help='输出目录 (默认: PROJECT_ROOT/models/block_lifting)')
    parser.add_argument('--cache_dir', type=str, default=str(DEFAULT_CACHE_PATH),
                       help='缓存目录 (默认: PROJECT_ROOT/cache/block_lifting)')
    parser.add_argument('--model_path', type=str,
                       default=str(DEFAULT_OUTPUT_PATH / "best_model.pt"),
                       help='模型路径')
    parser.add_argument('--stats_path', type=str,
                       default=str(DEFAULT_OUTPUT_PATH / "stats.pkl"),
                       help='统计量路径')

    # 模型参数
    parser.add_argument('--switch_strategy', type=str, default='progressive',
                       choices=['hard', 'progressive'])
    parser.add_argument('--switch_timestep', type=int, default=50)
    parser.add_argument('--num_diffusion_steps', type=int, default=100)
    parser.add_argument('--pred_horizon', type=int, default=16)
    parser.add_argument('--obs_horizon', type=int, default=2)
    parser.add_argument('--action_horizon', type=int, default=8)

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader worker数量')
    parser.add_argument('--max_episode_cache', type=int, default=4,
                       help='内存中最大缓存episode数')
    parser.add_argument('--enable_disk_cache', type=bool, default=True,
                       help='是否启用磁盘缓存')

    args = parser.parse_args()

    # 打印项目路径信息
    print(f"[Main] PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"[Main] Data path: {args.data_dir}")
    print(f"[Main] Output path: {args.output_dir}")
    print(f"[Main] Cache path: {args.cache_dir}")

    if args.mode == 'train':
        print("=" * 60)
        print("开始训练 Multi-Stage Diffusion Policy")
        print("=" * 60)
        train(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            switch_strategy=args.switch_strategy,
            switch_timestep=args.switch_timestep,
            num_diffusion_steps=args.num_diffusion_steps,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            pred_horizon=args.pred_horizon,
            obs_horizon=args.obs_horizon,
            action_horizon=args.action_horizon,
            max_episode_cache=args.max_episode_cache,
            num_workers=args.num_workers,
        )

    elif args.mode == 'demo':
        print("=" * 60)
        print("运行完整演示")
        print("=" * 60)

        print("\n[1/5] Dataset 演示")
        if os.path.exists(args.data_dir):
            demo_dataset(args.data_dir)
        else:
            print(f"数据目录 {args.data_dir} 不存在，跳过数据集演示")

        print("\n[2/5] Encoders 演示")
        demo_encoders()

        print("\n[3/5] U-Net 演示")
        demo_unet()

        print("\n[4/5] Model 演示")
        demo_model()

        print("\n[5/5] Inference 演示")
        demo_inference()

    elif args.mode == 'infer':
        print("=" * 60)
        print("推理模式")
        print("=" * 60)

        if not os.path.exists(args.model_path):
            print(f"模型文件 {args.model_path} 不存在")
            return

        inferencer = MultiStageDiffusionPolicyInference(
            model_path=args.model_path,
            stats_path=args.stats_path,
            obs_horizon=args.obs_horizon,
            action_horizon=args.action_horizon,
        )

        print(f"推理器加载完成")
        print(f"  模型: {args.model_path}")
        print(f"  策略: {inferencer.model.switch_strategy}")
        print(f"  切换时间步: {inferencer.model.switch_timestep}")
        print(f"\n在你的仿真循环中使用:")
        print(f"  inferencer.update_obs(step_info)")
        print(f"  action = inferencer.get_action()")

    elif args.mode == 'dataset':
        if os.path.exists(args.data_dir):
            demo_dataset(args.data_dir)
        else:
            print(f"数据目录 {args.data_dir} 不存在")

    elif args.mode == 'encoders':
        demo_encoders()

    elif args.mode == 'unet':
        demo_unet()


if __name__ == "__main__":
    main()