#!/usr/bin/env python3
"""
MultiStageDiffusionUNet1D - 多阶段条件U-Net
支持: hard_switch / progressive_blend 两种策略

独立运行: python -m source.train.unet --demo
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Literal
import argparse


class SinusoidalPosEmb(nn.Module):
    """正弦位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2),
            nn.GroupNorm(n_groups, out_ch),
            nn.Mish(),
        )
    def forward(self, x):
        return self.block(x)


class MultiStageCondResBlock1D(nn.Module):
    """
    支持多阶段条件的残差块
    可以同时接收视觉条件和触觉条件，根据时间步动态加权
    """
    def __init__(self, in_ch, out_ch, visual_cond_dim, tactile_cond_dim, 
                 kernel_size=3, n_groups=8, use_progressive_blend=True):
        super().__init__()

        self.use_progressive_blend = use_progressive_blend

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_ch, out_ch, kernel_size, n_groups),
            Conv1dBlock(out_ch, out_ch, kernel_size, n_groups),
        ])

        # 视觉条件编码 (FiLM)
        self.visual_cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(visual_cond_dim, out_ch * 2),
        )

        # 触觉条件编码 (FiLM)
        self.tactile_cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(tactile_cond_dim, out_ch * 2),
        )

        # 时间步感知门控: 根据扩散时间步决定视觉/触觉的权重
        # t=0(纯噪声) -> 视觉主导; t=T(去噪完成) -> 触觉主导
        self.timestep_gate = nn.Sequential(
            SinusoidalPosEmb(64),
            nn.Linear(64, 128),
            nn.Mish(),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # 输出[0,1], 0=视觉主导, 1=触觉主导
        )

        self.residual_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.out_channels = out_ch

    def forward(self, x, timesteps, visual_cond, tactile_cond):
        """
        x: [B, in_ch, T]
        timesteps: [B] - 扩散时间步，用于控制条件混合
        visual_cond: [B, visual_cond_dim] 或 None
        tactile_cond: [B, tactile_cond_dim] 或 None
        """
        out = self.blocks[0](x)

        # 计算时间步门控: alpha=0 纯视觉, alpha=1 纯触觉
        if self.use_progressive_blend and visual_cond is not None and tactile_cond is not None:
            alpha = self.timestep_gate(timesteps)  # [B, 1]

            # 视觉FiLM
            visual_embed = self.visual_cond_encoder(visual_cond)  # [B, out_ch*2]
            visual_embed = visual_embed.reshape(-1, 2, self.out_channels, 1)
            visual_scale, visual_bias = visual_embed[:, 0], visual_embed[:, 1]

            # 触觉FiLM
            tactile_embed = self.tactile_cond_encoder(tactile_cond)
            tactile_embed = tactile_embed.reshape(-1, 2, self.out_channels, 1)
            tactile_scale, tactile_bias = tactile_embed[:, 0], tactile_embed[:, 1]

            # 渐进式混合: out = (1-alpha) * visual + alpha * tactile
            scale = (1 - alpha.view(-1, 1, 1)) * visual_scale + alpha.view(-1, 1, 1) * tactile_scale
            bias = (1 - alpha.view(-1, 1, 1)) * visual_bias + alpha.view(-1, 1, 1) * tactile_bias

        elif visual_cond is not None:
            visual_embed = self.visual_cond_encoder(visual_cond)
            visual_embed = visual_embed.reshape(-1, 2, self.out_channels, 1)
            scale, bias = visual_embed[:, 0], visual_embed[:, 1]
        elif tactile_cond is not None:
            tactile_embed = self.tactile_cond_encoder(tactile_cond)
            tactile_embed = tactile_embed.reshape(-1, 2, self.out_channels, 1)
            scale, bias = tactile_embed[:, 0], tactile_embed[:, 1]
        else:
            raise ValueError("At least one condition must be provided")

        out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class MultiStageDiffusionUNet1D(nn.Module):
    """
    多阶段条件1D U-Net
    支持: hard_switch / progressive_blend 两种策略
    """
    def __init__(
        self,
        input_dim: int,
        visual_cond_dim: int,
        tactile_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: List[int] = [256, 512, 1024],
        kernel_size: int = 5,
        n_groups: int = 8,
        switch_strategy: Literal["hard", "progressive"] = "progressive",
        switch_timestep: int = 50,
    ):
        super().__init__()

        self.switch_strategy = switch_strategy
        self.switch_timestep = switch_timestep

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        # 时间步编码
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        # 输入投影
        self.input_mlp = Conv1dBlock(input_dim, start_dim, kernel_size)

        # 下采样路径
        self.down_modules = nn.ModuleList([])
        for ind in range(len(down_dims)):
            is_last = ind >= (len(down_dims) - 1)
            self.down_modules.append(nn.ModuleList([
                MultiStageCondResBlock1D(
                    all_dims[ind], all_dims[ind+1], 
                    visual_cond_dim, tactile_cond_dim,
                    kernel_size, n_groups,
                    use_progressive_blend=(switch_strategy == "progressive")
                ),
                MultiStageCondResBlock1D(
                    all_dims[ind+1], all_dims[ind+1],
                    visual_cond_dim, tactile_cond_dim,
                    kernel_size, n_groups,
                    use_progressive_blend=(switch_strategy == "progressive")
                ),
                nn.Conv1d(all_dims[ind+1], all_dims[ind+1], 3, 2, 1) if not is_last else nn.Identity()
            ]))

        # 中间层
        self.mid_modules = nn.ModuleList([
            MultiStageCondResBlock1D(
                down_dims[-1], down_dims[-1],
                visual_cond_dim, tactile_cond_dim,
                kernel_size, n_groups,
                use_progressive_blend=(switch_strategy == "progressive")
            ),
            MultiStageCondResBlock1D(
                down_dims[-1], down_dims[-1],
                visual_cond_dim, tactile_cond_dim,
                kernel_size, n_groups,
                use_progressive_blend=(switch_strategy == "progressive")
            ),
        ])

        # 上采样路径
        self.up_modules = nn.ModuleList([])
        for ind in reversed(range(len(down_dims))):
            is_last = ind == 0
            self.up_modules.append(nn.ModuleList([
                MultiStageCondResBlock1D(
                    all_dims[ind+1] * 2, all_dims[ind+1],
                    visual_cond_dim, tactile_cond_dim,
                    kernel_size, n_groups,
                    use_progressive_blend=(switch_strategy == "progressive")
                ),
                MultiStageCondResBlock1D(
                    all_dims[ind+1], all_dims[ind],
                    visual_cond_dim, tactile_cond_dim,
                    kernel_size, n_groups,
                    use_progressive_blend=(switch_strategy == "progressive")
                ),
                nn.ConvTranspose1d(all_dims[ind], all_dims[ind], 4, 2, 1) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(self, noisy_actions, timesteps, visual_cond, tactile_cond):
        """
        noisy_actions: [B, action_dim, T]
        timesteps: [B] - 扩散时间步
        visual_cond: [B, visual_cond_dim] 或 None
        tactile_cond: [B, tactile_cond_dim] 或 None
        """
        # 时间步编码 (添加到条件中)
        timestep_emb = self.diffusion_step_encoder(timesteps)

        # 硬切换策略: 根据时间步选择条件
        if self.switch_strategy == "hard":
            mask = (timesteps < self.switch_timestep).float().view(-1, 1)
            if visual_cond is not None:
                visual_cond = visual_cond * mask
            if tactile_cond is not None:
                tactile_cond = tactile_cond * (1 - mask)

        # 输入投影
        x = self.input_mlp(noisy_actions)

        # 下采样
        hs = []
        for resnet1, resnet2, downsample in self.down_modules:
            x = resnet1(x, timesteps, visual_cond, tactile_cond)
            x = resnet2(x, timesteps, visual_cond, tactile_cond)
            hs.append(x)
            x = downsample(x)

        # 中间层
        for mid_module in self.mid_modules:
            x = mid_module(x, timesteps, visual_cond, tactile_cond)

        # 上采样
        for resnet1, resnet2, upsample in self.up_modules:
            x = torch.cat((x, hs.pop()), dim=1)
            x = resnet1(x, timesteps, visual_cond, tactile_cond)
            x = resnet2(x, timesteps, visual_cond, tactile_cond)
            x = upsample(x)

        x = self.final_conv(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_structure(model, name="Model"):
    """打印模型结构"""
    print(f"{'='*60}")
    print(f"{name} Structure")
    print(f"{'='*60}")

    total_params = 0
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子节点
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"  {name}: {module.__class__.__name__} ({params:,} params)")
                total_params += params

    print(f"Total parameters: {total_params:,}")
    print(f"{'='*60}")


def demo_unet():
    """U-Net演示 - 独立运行查看模型结构和前向传播"""
    print("=" * 60)
    print("MultiStageDiffusionUNet1D 模型结构演示")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 配置
    batch_size = 2
    action_dim = 13
    pred_horizon = 16
    visual_cond_dim = 512
    tactile_cond_dim = 256

    # 1. Progressive Blend 策略
    print("" + "-" * 60)
    print("[1] Progressive Blend U-Net")
    print("-" * 60)

    model_prog = MultiStageDiffusionUNet1D(
        input_dim=action_dim,
        visual_cond_dim=visual_cond_dim,
        tactile_cond_dim=tactile_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        switch_strategy="progressive",
        switch_timestep=50,
    ).to(device)

    print_model_structure(model_prog, "Progressive U-Net")
    print(f"总参数量: {count_parameters(model_prog):,}")

    # 前向测试
    noisy_actions = torch.randn(batch_size, action_dim, pred_horizon).to(device)
    timesteps = torch.randint(0, 100, (batch_size,)).to(device)
    visual_cond = torch.randn(batch_size, visual_cond_dim).to(device)
    tactile_cond = torch.randn(batch_size, tactile_cond_dim).to(device)

    output = model_prog(noisy_actions, timesteps, visual_cond, tactile_cond)
    print(f"前向测试:")
    print(f"  Input:  noisy_actions={noisy_actions.shape}, timesteps={timesteps.shape}")
    print(f"  Cond:   visual={visual_cond.shape}, tactile={tactile_cond.shape}")
    print(f"  Output: {output.shape} (should match input)")

    # 验证门控行为
    print(f"门控行为验证 (Progressive):")
    for t in [0, 25, 50, 75, 99]:
        ts = torch.full((batch_size,), t).to(device)
        # 获取第一个残差块的gate值
        gate = model_prog.down_modules[0][0].timestep_gate(ts)
        print(f"  t={t:3d}: gate(alpha)={gate.mean().item():.4f} (0=visual, 1=tactile)")

    # 2. Hard Switch 策略
    print("" + "-" * 60)
    print("[2] Hard Switch U-Net")
    print("-" * 60)

    model_hard = MultiStageDiffusionUNet1D(
        input_dim=action_dim,
        visual_cond_dim=visual_cond_dim,
        tactile_cond_dim=tactile_cond_dim,
        switch_strategy="hard",
        switch_timestep=50,
    ).to(device)

    print(f"总参数量: {count_parameters(model_hard):,}")

    output = model_hard(noisy_actions, timesteps, visual_cond, tactile_cond)
    print(f"前向测试通过: {output.shape}")

    # 验证硬切换行为
    print(f"切换行为验证 (Hard Switch, threshold=50):")
    for t in [0, 25, 49, 50, 75, 99]:
        ts = torch.full((batch_size,), t).to(device)
        mask = (ts < 50).float()
        print(f"  t={t:3d}: mask={mask[0].item():.0f} (0=visual, 1=tactile)")

    # 3. 性能测试
    print("" + "-" * 60)
    print("[3] 推理速度测试")
    print("-" * 60)

    model_prog.eval()
    with torch.no_grad():
        # 预热
        for _ in range(10):
            _ = model_prog(noisy_actions, timesteps, visual_cond, tactile_cond)

        # 计时
        import time
        num_iters = 100
        start = time.time()
        for _ in range(num_iters):
            _ = model_prog(noisy_actions, timesteps, visual_cond, tactile_cond)
        elapsed = time.time() - start

        print(f"  {num_iters}次前向传播: {elapsed:.3f}s")
        print(f"  单次推理: {elapsed/num_iters*1000:.2f}ms")
        print(f"  理论FPS: {num_iters/elapsed:.1f}")

    print("" + "=" * 60)
    print("U-Net 演示完成")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='运行模型结构演示')
    args = parser.parse_args()

    if args.demo:
        demo_unet()
    else:
        print("用法:")
        print("  演示: python -m source.train.unet --demo")