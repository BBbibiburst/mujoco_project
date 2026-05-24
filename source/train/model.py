#!/usr/bin/env python3
"""
MultiStageDiffusionPolicy - 多阶段条件扩散策略主模型
整合: encoders + unet + diffusion schedule

独立运行: 
  训练: python -m source.train.model --mode train --data_dir ./grasp_data
  演示: python -m source.train.model --demo
"""

import os
import sys
import math
import pickle
import argparse
import numpy as np
from collections import deque
from typing import Dict, Tuple, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange
from PIL import Image
from torchvision import transforms

# 导入本地模块
from source.train.dataset import LazyMultiModalGraspDataset, decode_tactile
from source.train.encoders import VisionEncoder, TactileEncoder, ProprioceptionEncoder
from source.train.unet import MultiStageDiffusionUNet1D


class MultiStageDiffusionPolicy(nn.Module):
    """
    多阶段条件扩散策略

    阶段1 (t >= switch_timestep): 视觉主导 -> 全局理解
    阶段2 (t < switch_timestep): 触觉主导 -> 精细调整

    两种策略:
      - progressive: 渐进式混合，通过门控网络平滑过渡
      - hard: 硬切换，在指定时间步直接切换条件
    """
    def __init__(
        self,
        action_dim: int = 13,
        pred_horizon: int = 16,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        num_diffusion_steps: int = 100,
        switch_timestep: int = 50,
        switch_strategy: str = "progressive",
        image_size: Tuple[int, int] = (96, 96),
        vision_feature_dim: int = 512,
        tactile_dim: int = 700,
        tactile_feature_dim: int = 256,
        diffusion_step_embed_dim: int = 256,
        down_dims = [256, 512, 1024],
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.num_diffusion_steps = num_diffusion_steps
        self.switch_timestep = switch_timestep
        self.switch_strategy = switch_strategy

        # 视觉编码器
        self.vision_encoder = VisionEncoder(
            obs_shape=(3, image_size[0], image_size[1]),
            feature_dim=vision_feature_dim
        )
        self.visual_fusion = nn.Sequential(
            nn.Linear(vision_feature_dim * obs_horizon, vision_feature_dim),
            nn.ReLU(),
        )

        # 触觉编码器
        self.proprio_encoder = ProprioceptionEncoder(feature_dim=64)
        self.tactile_encoder = TactileEncoder(tactile_dim=tactile_dim, feature_dim=tactile_feature_dim)
        self.tactile_fusion = nn.Sequential(
            nn.Linear(64 + tactile_feature_dim, tactile_feature_dim),
            nn.ReLU(),
        )
        self.tactile_temporal_fusion = nn.Sequential(
            nn.Linear(tactile_feature_dim * obs_horizon, tactile_feature_dim),
            nn.ReLU(),
        )

        # 噪声预测网络
        self.noise_pred_net = MultiStageDiffusionUNet1D(
            input_dim=action_dim,
            visual_cond_dim=vision_feature_dim,
            tactile_cond_dim=tactile_feature_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            switch_strategy=switch_strategy,
            switch_timestep=switch_timestep,
        )

        self._setup_diffusion_schedule()

    def _setup_diffusion_schedule(self):
        num_steps = self.num_diffusion_steps

        self.betas = torch.linspace(1e-4, 0.02, num_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device):
        super().to(device)
        for key in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                    'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
                    'posterior_variance', 'posterior_log_variance_clipped',
                    'posterior_mean_coef1', 'posterior_mean_coef2']:
            setattr(self, key, getattr(self, key).to(device))
        return self

    def encode_visual(self, visual_obs):
        B, T = visual_obs.shape[:2]
        visual_obs = rearrange(visual_obs, 'b t c h w -> (b t) c h w')
        features = self.vision_encoder(visual_obs)
        features = rearrange(features, '(b t) d -> b (t d)', b=B, t=T)
        visual_cond = self.visual_fusion(features)
        return visual_cond

    def encode_tactile(self, tactile_obs):
        B, T = tactile_obs.shape[:2]

        proprio = tactile_obs[:, :, :13]
        tactile = tactile_obs[:, :, 13:]

        tactile_features = []
        for t in range(T):
            p = self.proprio_encoder(proprio[:, t])
            ta = self.tactile_encoder(tactile[:, t])
            fused = self.tactile_fusion(torch.cat([p, ta], dim=-1))
            tactile_features.append(fused)

        tactile_seq = torch.stack(tactile_features, dim=1)
        tactile_seq = rearrange(tactile_seq, 'b t d -> b (t d)')
        tactile_cond = self.tactile_temporal_fusion(tactile_seq)
        return tactile_cond

    def predict_noise(self, noisy_actions, timesteps, visual_obs, tactile_obs):
        visual_cond = self.encode_visual(visual_obs)
        tactile_cond = self.encode_tactile(tactile_obs)

        noise_pred = self.noise_pred_net(
            noisy_actions, timesteps, visual_cond, tactile_cond
        )
        return noise_pred

    def compute_loss(self, batch):
        visual_obs = batch['visual_obs']
        tactile_obs = batch['tactile_obs']
        actions = batch['action']

        B = actions.shape[0]
        device = actions.device

        actions = actions.permute(0, 2, 1)

        timesteps = torch.randint(0, self.num_diffusion_steps, (B,), device=device).long()

        noise = torch.randn_like(actions)

        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[timesteps].reshape(B, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].reshape(B, 1, 1)

        noisy_actions = sqrt_alpha_cumprod_t * actions + sqrt_one_minus_alpha_cumprod_t * noise

        noise_pred = self.predict_noise(noisy_actions, timesteps, visual_obs, tactile_obs)

        loss = F.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, visual_obs, tactile_obs, num_samples=1):
        B = visual_obs.shape[0]
        device = visual_obs.device

        noisy_actions = torch.randn(B, self.action_dim, self.pred_horizon, device=device)

        for t in reversed(range(self.num_diffusion_steps)):
            timesteps = torch.full((B,), t, device=device, dtype=torch.long)

            noise_pred = self.predict_noise(noisy_actions, timesteps, visual_obs, tactile_obs)

            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]

            pred_x0 = (noisy_actions - torch.sqrt(1 - alpha_cumprod) * noise_pred) / torch.sqrt(alpha_cumprod)
            pred_x0_clipped = torch.clamp(pred_x0, -3.0, 3.0)

            model_mean = (
                self.posterior_mean_coef1[t] * pred_x0_clipped +
                self.posterior_mean_coef2[t] * noisy_actions
            )

            if t > 0:
                noise = torch.randn_like(noisy_actions)
                variance = torch.exp(0.5 * self.posterior_log_variance_clipped[t])
                noisy_actions = model_mean + variance * noise
            else:
                noisy_actions = model_mean

        actions = noisy_actions.permute(0, 2, 1)
        return actions


def _grad_norm(model: nn.Module) -> float:
    """计算当前梯度的 L2 范数"""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total ** 0.5


def train(data_dir: str, output_dir: str, **kwargs):
    """训练函数"""
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)

    epochs     = kwargs.get('epochs', 100)
    batch_size = kwargs.get('batch_size', 32)
    lr         = kwargs.get('lr', 1e-4)

    # ── 数据集（HDF5 版本）──────────────────────────────────────────────────
    h5_path    = kwargs.get('h5_path',    data_dir)   # data_dir 传入 h5 文件路径
    stats_path = kwargs.get('stats_path', os.path.join(output_dir, 'stats.pkl'))

    dataset = LazyMultiModalGraspDataset(
        h5_path=h5_path,
        stats_path=stats_path,
        pred_horizon=kwargs.get('pred_horizon', 16),
        obs_horizon=kwargs.get('obs_horizon', 2),
        action_horizon=kwargs.get('action_horizon', 8),
        normalize=True,
    )

    num_workers = kwargs.get('num_workers', 12)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )

    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # ── 模型 ────────────────────────────────────────────────────────────────
    model = MultiStageDiffusionPolicy(
        action_dim=13,
        pred_horizon=kwargs.get('pred_horizon', 16),
        obs_horizon=kwargs.get('obs_horizon', 2),
        action_horizon=kwargs.get('action_horizon', 8),
        num_diffusion_steps=kwargs.get('num_diffusion_steps', 100),
        switch_timestep=kwargs.get('switch_timestep', 50),
        switch_strategy=kwargs.get('switch_strategy', 'progressive'),
    ).to(device)

    # torch.compile：首个 batch 会有一次编译耗时（约 30~60s），之后约提速 10~30%
    if hasattr(torch, 'compile'):
        print("[Train] torch.compile 编译模型中（首 batch 稍慢属正常）...")
        model = torch.compile(model)


    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── 训练头部摘要 ────────────────────────────────────────────────────────
    sep = "=" * 70
    print(sep)
    print("  Multi-Stage Diffusion Policy — Training")
    print(sep)
    print(f"  Device          : {device}")
    print(f"  Dataset size    : {len(dataset):,} samples")
    print(f"  Batch size      : {batch_size}  |  Steps/epoch: {len(dataloader)}")
    print(f"  Epochs          : {epochs}")
    print(f"  LR (initial)    : {lr:.2e}  (CosineAnnealing)")
    print(f"  Trainable params: {total_params:,}")
    print(f"  Strategy        : {kwargs.get('switch_strategy', 'progressive')}  "
          f"switch_t={kwargs.get('switch_timestep', 50)}")
    print(f"  num_workers     : {num_workers}  prefetch_factor=4  persistent_workers=True")
    print(f"  num_workers     : {num_workers}  prefetch=4  persistent=True")
    print(f"  HDF5            : {h5_path}")
    print(f"  Output dir      : {output_dir}")
    print(sep)

    best_loss   = float('inf')
    global_step = 0

    config_dict = {
        'switch_strategy'    : kwargs.get('switch_strategy', 'progressive'),
        'switch_timestep'    : kwargs.get('switch_timestep', 50),
        'num_diffusion_steps': kwargs.get('num_diffusion_steps', 100),
        'pred_horizon'       : kwargs.get('pred_horizon', 16),
        'obs_horizon'        : kwargs.get('obs_horizon', 2),
        'action_horizon'     : kwargs.get('action_horizon', 8),
    }

    # ── Epoch 进度条 ─────────────────────────────────────────────────────────
    epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch")

    try:
        for epoch in epoch_bar:
            model.train()
            epoch_losses = []
            epoch_gnorms = []

            # ── Batch 进度条 ─────────────────────────────────────────────────
            batch_bar = tqdm(
                dataloader,
                desc=f"Epoch {epoch+1:03d}/{epochs}",
                unit="batch",
                leave=False,
            )

            for batch in batch_bar:
                batch = {k: v.to(device) for k, v in batch.items()}

                loss = model.compute_loss(batch)

                optimizer.zero_grad()
                loss.backward()
                gnorm = _grad_norm(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                loss_val = loss.item()
                epoch_losses.append(loss_val)
                epoch_gnorms.append(gnorm)
                global_step += 1

                # 实时更新 batch 进度条后缀
                cur_lr = optimizer.param_groups[0]['lr']
                batch_bar.set_postfix(
                    loss=f"{loss_val:.4f}",
                    gnorm=f"{gnorm:.3f}",
                    lr=f"{cur_lr:.1e}",
                    step=global_step,
                )

            batch_bar.close()

            # ── Epoch 汇总 ───────────────────────────────────────────────────
            avg_loss  = float(np.mean(epoch_losses))
            avg_gnorm = float(np.mean(epoch_gnorms))
            min_loss  = float(np.min(epoch_losses))
            max_loss  = float(np.max(epoch_losses))
            cur_lr    = optimizer.param_groups[0]['lr']
            is_best   = avg_loss < best_loss

            scheduler.step()

            # 更新外层 epoch 进度条后缀
            epoch_bar.set_postfix(
                avg_loss=f"{avg_loss:.4f}",
                best=f"{best_loss:.4f}",
                lr=f"{cur_lr:.1e}",
            )

            # epoch 汇总固定打印一行（不被进度条覆盖）
            mark = "★" if is_best else " "
            tqdm.write(
                f"  {mark} Epoch {epoch+1:03d}/{epochs}"
                f"  avg={avg_loss:.6f} [min={min_loss:.6f} max={max_loss:.6f}]"
                f"  gnorm={avg_gnorm:.3f}"
                f"  lr={cur_lr:.2e}"
            )

            # ── 保存最佳模型 ─────────────────────────────────────────────────
            if is_best:
                best_loss = avg_loss
                ckpt_path = os.path.join(output_dir, 'best_model.pt')
                torch.save({
                    'epoch'               : epoch,
                    'global_step'         : global_step,
                    'model_state_dict'    : model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss'                : avg_loss,
                    'stats'               : dataset.stats,
                    'config'              : config_dict,
                }, ckpt_path)
                tqdm.write(f"    → best model saved [{ckpt_path}]")

        # ── 训练结束 ─────────────────────────────────────────────────────────
        final_path = os.path.join(output_dir, 'final_model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'stats'           : dataset.stats,
            'config'          : config_dict,
        }, final_path)

        with open(os.path.join(output_dir, 'stats.pkl'), 'wb') as f:
            pickle.dump(dataset.stats, f)

        print(sep)
        print(f"  Training complete!  best_loss={best_loss:.6f}")
        print(f"  Final model : {final_path}")
        print(sep)

    finally:
        dataset.cleanup()

    return model, dataset.stats


def demo_model():
    """模型演示 - 独立运行查看完整模型结构和前向传播"""
    print("=" * 60)
    print("MultiStageDiffusionPolicy 完整模型演示")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 创建模型
    model = MultiStageDiffusionPolicy(
        action_dim=13,
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        num_diffusion_steps=100,
        switch_timestep=50,
        switch_strategy="progressive",
    ).to(device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")

    # 各组件参数量
    components = {
        'Vision Encoder': model.vision_encoder,
        'Visual Fusion': model.visual_fusion,
        'Proprio Encoder': model.proprio_encoder,
        'Tactile Encoder': model.tactile_encoder,
        'Tactile Fusion': model.tactile_fusion,
        'Tactile Temporal Fusion': model.tactile_temporal_fusion,
        'Noise Pred U-Net': model.noise_pred_net,
    }

    print(f"各组件参数量:")
    for name, module in components.items():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name:25s}: {params:10,} ({params/total_params*100:.1f}%)")

    # 前向测试
    print(f"前向传播测试:")
    batch_size = 2
    visual_obs = torch.randn(batch_size, 2, 3, 96, 96).to(device)
    tactile_obs = torch.randn(batch_size, 2, 713).to(device)
    actions = torch.randn(batch_size, 16, 13).to(device)

    batch = {
        'visual_obs': visual_obs,
        'tactile_obs': tactile_obs,
        'action': actions,
    }

    # 训练模式 (计算损失)
    model.train()
    loss = model.compute_loss(batch)
    print(f"  Loss: {loss.item():.6f}")

    # 推理模式 (采样)
    model.eval()
    with torch.no_grad():
        sampled_actions = model.sample(visual_obs, tactile_obs)
    print(f"  Sampled actions: {sampled_actions.shape}")

    # 阶段行为验证
    print(f"阶段行为验证:")
    print(f"  策略: {model.switch_strategy}")
    print(f"  切换时间步: {model.switch_timestep}")

    if model.switch_strategy == "progressive":
        print(f"门控值随时间步变化:")
        for t in [0, 25, 50, 75, 99]:
            ts = torch.full((batch_size,), t).to(device)
            gate = model.noise_pred_net.down_modules[0][0].timestep_gate(ts)
            print(f"    t={t:3d}: alpha={gate.mean().item():.4f} (0=视觉主导, 1=触觉主导)")

    print("" + "=" * 60)
    print("模型演示完成")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='demo', choices=['train', 'demo'])
    parser.add_argument('--data_dir', type=str, default='./grasp_data')
    parser.add_argument('--output_dir', type=str, default='./multistage_output')
    parser.add_argument('--switch_strategy', type=str, default='progressive', choices=['hard', 'progressive'])
    parser.add_argument('--switch_timestep', type=int, default=50)
    parser.add_argument('--num_diffusion_steps', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--pred_horizon', type=int, default=16)
    parser.add_argument('--obs_horizon', type=int, default=2)
    parser.add_argument('--action_horizon', type=int, default=8)
    parser.add_argument('--max_episode_cache', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    if args.mode == 'demo':
        demo_model()
    elif args.mode == 'train':
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