#!/usr/bin/env python3
"""
Encoders - 多模态编码器模块
包含: VisionEncoder, TactileEncoder, ProprioceptionEncoder

独立运行: python -m source.train.encoders --demo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple
import argparse


class VisionEncoder(nn.Module):
    """视觉编码器 -> 全局场景特征"""
    def __init__(self, obs_shape, feature_dim=512):
        super().__init__()
        C, H, W = obs_shape

        # ResNet-like backbone
        self.conv_net = nn.Sequential(
            nn.Conv2d(C, 64, 7, stride=2, padding=3),  # /2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),       # /4

            # Residual block 1
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Downsample
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # /8
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Residual block 2
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Downsample
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # /16
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        # x: [B, C, H, W]
        h = self.conv_net(x)
        h = h.reshape(h.shape[0], -1)
        return self.fc(h)


class TactileEncoder(nn.Module):
    """触觉编码器 -> 精细接触特征"""
    def __init__(self, tactile_dim=700, feature_dim=256):
        super().__init__()

        # 将700维触觉数据编码为紧凑特征
        self.mlp = nn.Sequential(
            nn.Linear(tactile_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)


class ProprioceptionEncoder(nn.Module):
    """本体感知编码器 (arm + hand)"""
    def __init__(self, feature_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def demo_encoders():
    """编码器演示 - 独立运行查看模型结构"""
    print("=" * 60)
    print("Encoder 模型结构演示")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 1. VisionEncoder
    print("" + "-" * 60)
    print("[1] VisionEncoder")
    print("-" * 60)
    vision_encoder = VisionEncoder(obs_shape=(3, 96, 96), feature_dim=512).to(device)
    print(f"输入形状: (B, 3, 96, 96)")
    print(f"输出形状: (B, 512)")
    print(f"参数量: {count_parameters(vision_encoder):,}")

    # 前向测试
    dummy_input = torch.randn(2, 3, 96, 96).to(device)
    output = vision_encoder(dummy_input)
    print(f"测试通过: {dummy_input.shape} -> {output.shape}")

    # 2. TactileEncoder
    print("" + "-" * 60)
    print("[2] TactileEncoder")
    print("-" * 60)
    tactile_encoder = TactileEncoder(tactile_dim=700, feature_dim=256).to(device)
    print(f"输入形状: (B, 700)")
    print(f"输出形状: (B, 256)")
    print(f"参数量: {count_parameters(tactile_encoder):,}")

    dummy_input = torch.randn(2, 700).to(device)
    output = tactile_encoder(dummy_input)
    print(f"测试通过: {dummy_input.shape} -> {output.shape}")

    # 3. ProprioceptionEncoder
    print("" + "-" * 60)
    print("[3] ProprioceptionEncoder")
    print("-" * 60)
    proprio_encoder = ProprioceptionEncoder(feature_dim=64).to(device)
    print(f"输入形状: (B, 13)")
    print(f"输出形状: (B, 64)")
    print(f"参数量: {count_parameters(proprio_encoder):,}")

    dummy_input = torch.randn(2, 13).to(device)
    output = proprio_encoder(dummy_input)
    print(f"测试通过: {dummy_input.shape} -> {output.shape}")

    # 4. 多帧融合测试
    print("" + "-" * 60)
    print("[4] 多帧视觉融合测试")
    print("-" * 60)
    obs_horizon = 2
    visual_fusion = nn.Sequential(
        nn.Linear(512 * obs_horizon, 512),
        nn.ReLU(),
    ).to(device)

    # 模拟2帧视觉特征
    features = torch.randn(2, obs_horizon, 512).to(device)
    features_flat = features.reshape(2, -1)
    fused = visual_fusion(features_flat)
    print(f"输入: {features.shape} -> 展平: {features_flat.shape} -> 融合: {fused.shape}")

    # 5. 触觉时序融合测试
    print("" + "-" * 60)
    print("[5] 多帧触觉融合测试")
    print("-" * 60)
    tactile_fusion = nn.Sequential(
        nn.Linear(64 + 256, 256),
        nn.ReLU(),
    ).to(device)
    tactile_temporal_fusion = nn.Sequential(
        nn.Linear(256 * obs_horizon, 256),
        nn.ReLU(),
    ).to(device)

    proprio_feat = torch.randn(2, 64).to(device)
    tactile_feat = torch.randn(2, 256).to(device)
    fused_tactile = tactile_fusion(torch.cat([proprio_feat, tactile_feat], dim=-1))
    print(f"单帧融合: proprio(64) + tactile(256) -> {fused_tactile.shape}")

    tactile_seq = torch.randn(2, obs_horizon, 256).to(device)
    tactile_flat = tactile_seq.reshape(2, -1)
    temporal_fused = tactile_temporal_fusion(tactile_flat)
    print(f"时序融合: {tactile_seq.shape} -> 展平: {tactile_flat.shape} -> {temporal_fused.shape}")

    print("" + "=" * 60)
    print("Encoder 演示完成")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='运行模型结构演示')
    args = parser.parse_args()

    if args.demo:
        demo_encoders()
    else:
        print("用法: python -m source.train.encoders --demo")