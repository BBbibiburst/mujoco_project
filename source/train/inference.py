#!/usr/bin/env python3
"""
MultiStageDiffusionPolicyInference - 推理器
用于仿真环境集成

独立运行: python -m source.train.inference --demo
"""

import base64
import io
import os
from pathlib import Path
import sys
import pickle
import argparse
import numpy as np
from collections import deque
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# 导入本地模块
from source.train.model import MultiStageDiffusionPolicy
from source.train.dataset import decode_tactile


class MultiStageDiffusionPolicyInference:
    """多阶段扩散策略推理器"""

    def __init__(
        self,
        model_path: str,
        stats_path: str,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        # 加载统计量
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)

        # 加载模型配置和权重
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint.get('config', {})

        self.model = MultiStageDiffusionPolicy(
            action_dim=13,
            pred_horizon=config.get('pred_horizon', 16),
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            num_diffusion_steps=config.get('num_diffusion_steps', 100),
            switch_timestep=config.get('switch_timestep', 50),
            switch_strategy=config.get('switch_strategy', 'progressive'),
        ).to(device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # 观测队列
        self.visual_deque = deque(maxlen=obs_horizon)
        self.tactile_deque = deque(maxlen=obs_horizon)
        self.action_queue = deque(maxlen=action_horizon)

        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def _normalize_tactile(self, obs):
        mean = self.stats['tactile_obs_mean']
        std = self.stats['tactile_obs_std']
        return (obs - mean) / std

    def _denormalize_action(self, action):
        mean = self.stats['action_mean']
        std = self.stats['action_std']
        return action * std + mean

    def reset(self):
        """重置所有队列"""
        self.visual_deque.clear()
        self.tactile_deque.clear()
        self.action_queue.clear()

    def update_obs(self, step_info: Dict):
        """
        从环境step_info更新观测

        Args:
            step_info: 环境返回的info字典，包含:
                - images: {camera_rgb: base64图像}
                - arm_qpos: [7]
                - hand_qpos: [6]
                - tactile: {finger_0_bottom: {...}, ...}
        """
        # 视觉观测
        img_b64 = step_info['images']['camera_rgb']
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        visual_obs = self.image_transform(img)
        self.visual_deque.append(visual_obs)

        # 触觉观测
        arm = np.array(step_info['arm_qpos'], dtype=np.float32)
        hand = np.array(step_info['hand_qpos'], dtype=np.float32)
        tactile = decode_tactile(step_info['tactile'])
        tactile_obs = np.concatenate([arm, hand, tactile])
        tactile_obs = self._normalize_tactile(tactile_obs)
        self.tactile_deque.append(tactile_obs)

    def get_action(self) -> Optional[np.ndarray]:
        """
        获取动作

        Returns:
            [13] 动作向量 (arm_qpos[7], hand_qpos[6])，或None如果观测不足
        """
        # 如果动作队列中还有动作，直接返回
        if len(self.action_queue) > 0:
            return self.action_queue.popleft()

        # 检查观测是否足够
        if len(self.visual_deque) < self.obs_horizon or len(self.tactile_deque) < self.obs_horizon:
            return None

        # 堆叠观测
        visual_seq = torch.stack(list(self.visual_deque)).unsqueeze(0).to(self.device)
        tactile_seq = torch.from_numpy(np.stack(list(self.tactile_deque))).float().unsqueeze(0).to(self.device)

        # 采样动作序列
        with torch.no_grad():
            action_seq = self.model.sample(visual_seq, tactile_seq)

        action_seq = action_seq.cpu().numpy()[0]
        action_seq = self._denormalize_action(action_seq)

        # 将动作序列加入队列
        for i in range(min(self.action_horizon, len(action_seq))):
            self.action_queue.append(action_seq[i])

        return self.action_queue.popleft() if len(self.action_queue) > 0 else None


def demo_inference():
    """推理器演示 - 独立运行测试推理流程"""
    print("=" * 60)
    print("MultiStageDiffusionPolicyInference 推理演示")
    print("=" * 60)

    # 注意：这里使用随机初始化的模型进行演示
    # 实际使用时需要加载训练好的模型

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 创建临时模型和统计量用于演示
    model = MultiStageDiffusionPolicy(
        action_dim=13,
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        num_diffusion_steps=100,
        switch_timestep=50,
        switch_strategy="progressive",
    ).to(device)

    # 模拟统计量
    stats = {
        'action_mean': np.zeros(13, dtype=np.float32),
        'action_std': np.ones(13, dtype=np.float32),
        'tactile_obs_mean': np.zeros(713, dtype=np.float32),
        'tactile_obs_std': np.ones(713, dtype=np.float32),
    }

    # 保存临时模型
    os.makedirs('./temp_output', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'stats': stats,
        'config': {
            'switch_strategy': 'progressive',
            'switch_timestep': 50,
            'num_diffusion_steps': 100,
            'pred_horizon': 16,
            'obs_horizon': 2,
            'action_horizon': 8,
        }
    }, './temp_output/temp_model.pt')

    with open('./temp_output/temp_stats.pkl', 'wb') as f:
        pickle.dump(stats, f)

    # 加载推理器
    inferencer = MultiStageDiffusionPolicyInference(
        model_path='./temp_output/temp_model.pt',
        stats_path='./temp_output/temp_stats.pkl',
        obs_horizon=2,
        action_horizon=8,
        device=device,
    )

    print(f"推理器初始化完成")
    print(f"  观测队列长度: {inferencer.obs_horizon}")
    print(f"  动作队列长度: {inferencer.action_horizon}")
    print(f"  设备: {inferencer.device}")

    # 模拟环境step_info
    print(f"模拟推理流程:")

    # 创建模拟的step_info
    def create_mock_step_info(step_idx):
        import base64
        from io import BytesIO

        # 创建模拟图像
        img = Image.new('RGB', (640, 480), color=(step_idx*10 % 255, 100, 150))
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        img_b64 = base64.b64encode(buffer.getvalue()).decode()

        # 创建模拟触觉数据 (全零表示无接触)
        tactile_data = "data:application/octet-stream;base64," + base64.b64encode(
            np.zeros(700, dtype=np.uint8).tobytes()
        ).decode()

        return {
            'arm_qpos': np.random.randn(7).astype(np.float32) * 0.1,
            'hand_qpos': np.random.rand(6).astype(np.float32) * 0.01,
            'tactile': {
                'finger_0_bottom': {'data': tactile_data, 'shape': [10, 7], 'dtype': 'uint8'},
                'finger_0_middle': {'data': tactile_data, 'shape': [8, 5], 'dtype': 'uint8'},
                'finger_0_top': {'data': tactile_data, 'shape': [6, 5], 'dtype': 'uint8'},
                'finger_1_bottom': {'data': tactile_data, 'shape': [10, 7], 'dtype': 'uint8'},
                'finger_1_middle': {'data': tactile_data, 'shape': [8, 5], 'dtype': 'uint8'},
                'finger_1_top': {'data': tactile_data, 'shape': [6, 5], 'dtype': 'uint8'},
                'finger_2_bottom': {'data': tactile_data, 'shape': [10, 7], 'dtype': 'uint8'},
                'finger_2_middle': {'data': tactile_data, 'shape': [8, 5], 'dtype': 'uint8'},
                'finger_2_top': {'data': tactile_data, 'shape': [6, 5], 'dtype': 'uint8'},
                'finger_3_bottom': {'data': tactile_data, 'shape': [10, 7], 'dtype': 'uint8'},
                'finger_3_middle': {'data': tactile_data, 'shape': [8, 5], 'dtype': 'uint8'},
                'finger_3_top': {'data': tactile_data, 'shape': [6, 5], 'dtype': 'uint8'},
                'thumb_bottom': {'data': tactile_data, 'shape': [10, 7], 'dtype': 'uint8'},
                'thumb_middle': {'data': tactile_data, 'shape': [8, 5], 'dtype': 'uint8'},
                'thumb_top': {'data': tactile_data, 'shape': [6, 5], 'dtype': 'uint8'},
            },
            'images': {'camera_rgb': f'data:image/jpeg;base64,{img_b64}'},
        }

    # 填充初始观测
    print(f"填充初始观测队列...")
    for i in range(inferencer.obs_horizon):
        step_info = create_mock_step_info(i)
        inferencer.update_obs(step_info)
        print(f"    Step {i}: visual_deque={len(inferencer.visual_deque)}, tactile_deque={len(inferencer.tactile_deque)}")

    # 获取动作
    print(f"获取动作...")
    action = inferencer.get_action()
    if action is not None:
        print(f"    Action shape: {action.shape}")
        print(f"    Arm action: {action[:7]}")
        print(f"    Hand action: {action[7:]}")

    # 连续推理测试
    print(f"连续推理测试 (10 steps):")
    inferencer.reset()

    # 重新填充
    for i in range(inferencer.obs_horizon):
        inferencer.update_obs(create_mock_step_info(i))

    for step in range(10):
        action = inferencer.get_action()
        if action is None:
            # 需要更多观测
            inferencer.update_obs(create_mock_step_info(step + inferencer.obs_horizon))
            action = inferencer.get_action()

        if action is not None:
            print(f"    Step {step}: action_queue={len(inferencer.action_queue)}, action_mean={action.mean():.4f}")

    # 清理临时文件
    import shutil
    shutil.rmtree('./temp_output')

    print("" + "=" * 60)
    print("推理演示完成")
    print("=" * 60)

# 项目路径配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "block_lifting"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "models" / "block_lifting"
DEFAULT_CACHE_PATH = PROJECT_ROOT / "cache" / "block_lifting"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=DEFAULT_OUTPUT_PATH / "best_model.pt")
    parser.add_argument('--stats_path', type=str, default=DEFAULT_OUTPUT_PATH / "stats.pkl")
    parser.add_argument('--obs_horizon', type=int, default=2)
    parser.add_argument('--action_horizon', type=int, default=8)
    parser.add_argument('--demo', action='store_true', help='运行推理演示')
    args = parser.parse_args()

    if args.demo:
        demo_inference()
    else:
        print("用法:")
        print("  演示: python -m source.train.inference --demo")
        print("  实际使用: python -m source.train.inference --model_path <model_path> --stats_path <stats_path>")