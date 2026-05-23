#!/usr/bin/env python3
"""
LazyMultiModalGraspDataset - 惰性加载多模态抓取数据集
解决80GB大数据集内存问题

特性:
  - 惰性加载: 不预加载所有episode，按需读取
  - 分片缓存: 只缓存最近N个episode在内存
  - 索引预构建: 启动时只构建索引，不加载数据
  - 支持多worker: DataLoader多进程读取
  - 进度条显示: 使用tqdm显示构建和统计量计算进度

独立运行: python -m source.train.dataset --demo
"""

import os
import json
import base64
import io
import glob
import argparse
import pickle
import hashlib
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 尝试导入tqdm，如果没有则提供降级方案
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("[警告] tqdm未安装，进度条功能不可用。安装: pip install tqdm")


# ==================== 配置 ====================

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "block_lifting"
DEFAULT_CACHE_PATH = PROJECT_ROOT / "cache" / "block_lifting"

# 缓存控制
MAX_EPISODE_CACHE = 1      # 内存中最多缓存的episode数
ENABLE_DISK_CACHE = True   # 是否启用磁盘缓存预处理后的样本


def decode_tactile(tactile_dict: Dict) -> np.ndarray:
    """解码触觉传感器为700维向量"""
    tactile_flat = []
    sensor_order = [
        'finger_0_bottom', 'finger_0_middle', 'finger_0_top',
        'finger_1_bottom', 'finger_1_middle', 'finger_1_top',
        'finger_2_bottom', 'finger_2_middle', 'finger_2_top',
        'finger_3_bottom', 'finger_3_middle', 'finger_3_top',
        'thumb_bottom', 'thumb_middle', 'thumb_top',
    ]
    for key in sensor_order:
        if key not in tactile_dict:
            shape = [10, 7] if 'bottom' in key else ([8, 5] if 'middle' in key else [6, 5])
            data = np.zeros(shape, dtype=np.float32)
        else:
            sensor = tactile_dict[key]
            data_b64 = sensor['data']
            if ',' in data_b64:
                data_b64 = data_b64.split(',')[1]
            raw_bytes = base64.b64decode(data_b64)
            shape = sensor['shape']
            dtype = np.dtype(sensor['dtype'])
            data = np.frombuffer(raw_bytes, dtype=dtype).reshape(shape).astype(np.float32)
            if dtype == np.uint8:
                data = data / 255.0
        tactile_flat.append(data.flatten())
    return np.concatenate(tactile_flat)


def get_tactile_dim() -> int:
    return 700


class LazyMultiModalGraspDataset(Dataset):
    """
    惰性加载数据集 - 解决大内存问题

    核心机制:
      1. 启动时只扫描文件，构建索引 (文件路径+偏移)
      2. 按需读取episode，解析JSONL
      3. LRU缓存最近使用的episode
      4. 可选磁盘缓存预处理后的样本
    """

    def __init__(
        self,
        data_dir: str,
        pred_horizon: int = 16,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        image_size: Tuple[int, int] = (96, 96),
        normalize: bool = True,
        max_episode_cache: int = MAX_EPISODE_CACHE,
        enable_disk_cache: bool = ENABLE_DISK_CACHE,
        cache_dir: Optional[str] = None,
        show_progress: bool = True,  # 新增: 是否显示进度条
    ):
        self.data_dir = Path(data_dir)
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.normalize = normalize
        self.image_size = image_size
        self.max_episode_cache = max_episode_cache
        self.enable_disk_cache = enable_disk_cache
        self.show_progress = show_progress and TQDM_AVAILABLE

        # 磁盘缓存目录
        if enable_disk_cache and cache_dir is None:
            cache_dir = DEFAULT_CACHE_PATH
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

        # 扫描文件并构建索引
        print("[LazyDataset] 扫描数据文件...")
        self.episode_files = self._scan_files()
        print(f"[LazyDataset] 找到 {len(self.episode_files)} 个episode文件")

        print("[LazyDataset] 构建样本索引...")
        self.indices = self._build_indices()

        # LRU缓存: OrderedDict实现
        self._episode_cache = OrderedDict()

        # 统计量 (延迟计算或从缓存加载)
        self.stats = None
        self._stats_path = self.cache_dir / "stats.pkl" if self.cache_dir else None
        if normalize:
            self.stats = self._load_or_compute_stats()

        print(f"[LazyDataset] 文件数: {len(self.episode_files)}, 样本数: {len(self.indices)}")
        print(f"[LazyDataset] 内存缓存: {max_episode_cache} episodes")
        print(f"[LazyDataset] 磁盘缓存: {enable_disk_cache} ({self.cache_dir})")

    def _scan_files(self) -> List[Path]:
        """扫描所有episode文件，只记录路径不加载"""
        files = sorted(self.data_dir.glob("*.jsonl"))
        if not files:
            files = sorted(self.data_dir.rglob("*.jsonl"))

        if not files:
            raise ValueError(f"在 {self.data_dir} 中未找到 .jsonl 文件")

        return files

    def _build_indices(self) -> List[Tuple[int, int]]:
        """
        构建样本索引: (episode_idx, start_step)
        通过快速扫描每文件的行数来确定长度，不解析内容
        """
        indices = []

        # 使用tqdm显示进度
        file_iter = self.episode_files
        if self.show_progress:
            file_iter = tqdm(self.episode_files, desc="[索引构建] 扫描文件", unit="file")

        for ep_idx, fp in enumerate(file_iter):
            # 快速计数行数
            line_count = 0
            with open(fp, 'r', encoding='utf-8') as f:
                for _ in f:
                    line_count += 1

            # 每个episode的有效起始位置
            num_valid = max(0, line_count - self.pred_horizon + 1)
            for start in range(num_valid):
                indices.append((ep_idx, start))

        return indices

    def _get_cache_key(self, episode_idx: int) -> str:
        """生成episode的缓存key"""
        fp = self.episode_files[episode_idx]
        # 使用文件路径+修改时间的hash作为key
        mtime = fp.stat().st_mtime
        return hashlib.md5(f"{fp}:{mtime}".encode()).hexdigest()

    def _load_episode(self, episode_idx: int) -> List[Dict]:
        """
        加载单个episode，带LRU缓存
        """
        # 检查内存缓存
        if episode_idx in self._episode_cache:
            # 移到末尾(最近使用)
            self._episode_cache.move_to_end(episode_idx)
            return self._episode_cache[episode_idx]

        # 检查磁盘缓存
        cache_key = self._get_cache_key(episode_idx)
        disk_cache_path = self.cache_dir / f"ep_{cache_key}.pkl" if self.cache_dir else None

        if disk_cache_path and disk_cache_path.exists():
            # 从磁盘加载预处理后的episode
            with open(disk_cache_path, 'rb') as f:
                episode = pickle.load(f)
            self._add_to_cache(episode_idx, episode)
            return episode

        # 从原始文件解析
        fp = self.episode_files[episode_idx]
        episode = []
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    episode.append(json.loads(line)['info'])
                except:
                    continue

        # 保存到磁盘缓存
        if disk_cache_path:
            with open(disk_cache_path, 'wb') as f:
                pickle.dump(episode, f)

        self._add_to_cache(episode_idx, episode)
        return episode

    def _add_to_cache(self, episode_idx: int, episode: List[Dict]):
        """添加到LRU缓存，超出限制时移除最旧的"""
        self._episode_cache[episode_idx] = episode
        self._episode_cache.move_to_end(episode_idx)

        # 清理旧缓存
        while len(self._episode_cache) > self.max_episode_cache:
            self._episode_cache.popitem(last=False)

    def _compute_stats(self) -> Dict[str, np.ndarray]:
        """计算统计量 - 流式处理，不保存所有数据"""
        print("[LazyDataset] 计算统计量 (流式)...")

        action_sum = np.zeros(13, dtype=np.float64)
        action_sq_sum = np.zeros(13, dtype=np.float64)
        tactile_sum = np.zeros(713, dtype=np.float64)
        tactile_sq_sum = np.zeros(713, dtype=np.float64)
        count = 0

        # 使用tqdm显示进度
        ep_range = range(len(self.episode_files))
        if self.show_progress:
            ep_range = tqdm(ep_range, desc="[统计量] 处理episodes", unit="ep")

        # 流式处理每个episode
        for ep_idx in ep_range:
            episode = self._load_episode(ep_idx)

            for step in episode:
                # Action
                action = np.concatenate([
                    np.array(step['arm_qpos'], dtype=np.float32),
                    np.array(step['hand_qpos'], dtype=np.float32)
                ])
                action_sum += action
                action_sq_sum += action ** 2

                # Tactile obs
                arm = np.array(step['arm_qpos'], dtype=np.float32)
                hand = np.array(step['hand_qpos'], dtype=np.float32)
                tactile = decode_tactile(step['tactile'])
                tactile_obs = np.concatenate([arm, hand, tactile])
                tactile_sum += tactile_obs
                tactile_sq_sum += tactile_obs ** 2

                count += 1

            # 清理缓存控制内存
            if ep_idx in self._episode_cache:
                del self._episode_cache[ep_idx]

        # 计算mean和std
        action_mean = action_sum / count
        action_std = np.sqrt(action_sq_sum / count - action_mean ** 2) + 1e-8

        tactile_mean = tactile_sum / count
        tactile_std = np.sqrt(tactile_sq_sum / count - tactile_mean ** 2) + 1e-8

        stats = {
            'action_mean': action_mean.astype(np.float32),
            'action_std': action_std.astype(np.float32),
            'tactile_obs_mean': tactile_mean.astype(np.float32),
            'tactile_obs_std': tactile_std.astype(np.float32),
        }

        # 保存统计量缓存
        if self._stats_path:
            with open(self._stats_path, 'wb') as f:
                pickle.dump(stats, f)

        print(f"[LazyDataset] 统计量计算完成 (处理{count}步)")
        return stats

    def _load_or_compute_stats(self) -> Dict[str, np.ndarray]:
        """加载或计算统计量"""
        if self._stats_path and self._stats_path.exists():
            print(f"[LazyDataset] 从缓存加载统计量")
            with open(self._stats_path, 'rb') as f:
                return pickle.load(f)
        return self._compute_stats()

    def _decode_image(self, step: Dict) -> torch.Tensor:
        """解码base64图像"""
        img_b64 = step['images']['camera_rgb']
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        return self.image_transform(img)

    def _build_tactile_obs(self, step: Dict) -> np.ndarray:
        """构建触觉观测"""
        arm = np.array(step['arm_qpos'], dtype=np.float32)
        hand = np.array(step['hand_qpos'], dtype=np.float32)
        tactile = decode_tactile(step['tactile'])
        return np.concatenate([arm, hand, tactile])  # [713]

    def _normalize(self, data: np.ndarray, key: str) -> np.ndarray:
        if self.stats is None:
            return data
        return (data - self.stats[f'{key}_mean']) / self.stats[f'{key}_std']

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, start = self.indices[idx]

        # 按需加载episode
        episode = self._load_episode(ep_idx)

        # 边界检查
        if start + self.pred_horizon > len(episode):
            # 回退到有效范围
            start = max(0, len(episode) - self.pred_horizon)

        # 构建视觉观测序列
        visual_list = []
        for t in range(start, min(start + self.obs_horizon, len(episode))):
            visual_list.append(self._decode_image(episode[t]))
        while len(visual_list) < self.obs_horizon:
            visual_list.append(visual_list[-1])
        visual_seq = torch.stack(visual_list)

        # 构建触觉观测序列
        tactile_list = []
        for t in range(start, min(start + self.obs_horizon, len(episode))):
            tactile_obs = self._build_tactile_obs(episode[t])
            tactile_obs = self._normalize(tactile_obs, 'tactile_obs')
            tactile_list.append(tactile_obs)
        while len(tactile_list) < self.obs_horizon:
            tactile_list.append(tactile_list[-1])
        tactile_seq = torch.from_numpy(np.stack(tactile_list)).float()

        # 构建动作序列
        action_list = []
        end_idx = min(start + self.pred_horizon, len(episode))
        for t in range(start, end_idx):
            action = np.concatenate([
                np.array(episode[t]['arm_qpos'], dtype=np.float32),
                np.array(episode[t]['hand_qpos'], dtype=np.float32)
            ])
            action = self._normalize(action, 'action')
            action_list.append(action)

        # 如果不足pred_horizon，用最后一个动作填充
        while len(action_list) < self.pred_horizon:
            action_list.append(action_list[-1] if action_list else np.zeros(13, dtype=np.float32))

        action_seq = torch.from_numpy(np.stack(action_list)).float()

        return {
            'visual_obs': visual_seq,      # [T, C, H, W]
            'tactile_obs': tactile_seq,    # [T, 713]
            'action': action_seq,          # [T, 13]
        }


def demo_dataset(data_dir: str):
    """数据集演示 - 展示惰性加载和内存控制"""
    import psutil
    import time

    print("=" * 60)
    print("LazyMultiModalGraspDataset 惰性加载演示")
    print("=" * 60)

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024
    print(f"初始内存: {mem_before:.1f} MB")

    # 创建数据集
    start_time = time.time()
    dataset = LazyMultiModalGraspDataset(
        data_dir=data_dir,
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        image_size=(96, 96),
        normalize=True,
        max_episode_cache=2,      # 只缓存2个episode
        enable_disk_cache=True,
        show_progress=True,       # 启用进度条
    )
    init_time = time.time() - start_time
    mem_after_init = process.memory_info().rss / 1024 / 1024

    print(f"\n初始化时间: {init_time:.2f}s")
    print(f"初始化后内存: {mem_after_init:.1f} MB (增加 {mem_after_init-mem_before:.1f} MB)")
    print(f"数据集大小: {len(dataset)} 样本")
    print(f"Episode文件数: {len(dataset.episode_files)}")

    # 随机访问测试
    print(f"\n随机访问测试 (访问10个样本):")
    np.random.seed(42)
    sample_indices = np.random.choice(len(dataset), min(10, len(dataset)), replace=False)

    access_times = []
    # 使用tqdm显示访问进度
    iter_range = sample_indices
    if TQDM_AVAILABLE:
        iter_range = tqdm(sample_indices, desc="[访问测试] 读取样本", unit="sample")

    for i, idx in enumerate(iter_range):
        t0 = time.time()
        sample = dataset[idx]
        t1 = time.time()
        access_times.append(t1 - t0)

        if i < 3:  # 只打印前3个
            print(f"  Sample {idx}: visual={sample['visual_obs'].shape}, "
                  f"tactile={sample['tactile_obs'].shape}, action={sample['action'].shape}, "
                  f"time={t1-t0:.3f}s")

    mem_after_access = process.memory_info().rss / 1024 / 1024
    print(f"\n访问后内存: {mem_after_access:.1f} MB (增加 {mem_after_access-mem_after_init:.1f} MB)")
    print(f"平均访问时间: {np.mean(access_times)*1000:.1f}ms")
    print(f"缓存命中率: 观察访问时间变化判断")

    # DataLoader测试 (多worker)
    print(f"\nDataLoader测试 (num_workers=2, batch_size=4):")
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=2,      # 多进程加载
        pin_memory=True,
    )

    t0 = time.time()
    batch = next(iter(dataloader))
    t1 = time.time()

    for key, value in batch.items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    print(f"  首batch加载时间: {t1-t0:.2f}s")

    # 统计信息
    print(f"\n统计信息:")
    print(f"  Action mean: {dataset.stats['action_mean']}")
    print(f"  Action std:  {dataset.stats['action_std']}")

    print("\n" + "=" * 60)
    print("惰性加载演示完成")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument('--demo', action='store_true', help='运行数据演示')
    args = parser.parse_args()

    if args.demo:
        demo_dataset(args.data_dir)
    else:
        print("用法: python -m source.train.dataset --demo")