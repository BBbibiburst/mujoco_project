#!/usr/bin/env python3
"""
HDF5MultiModalGraspDataset - 基于 HDF5 的高速数据集
需要先用 preprocess.py 生成 HDF5 文件。

独立运行: python -m source.train.dataset --demo --h5 data/block_lifting.h5
"""

import os
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import h5py
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# ── 项目路径 ────────────────────────────────────────────────────────────────
PROJECT_ROOT      = Path(__file__).resolve().parent.parent.parent
DEFAULT_H5_PATH   = PROJECT_ROOT / "data"   / "block_lifting.h5"
DEFAULT_STATS_PATH= PROJECT_ROOT / "models" / "block_lifting" / "stats.pkl"


# ── 保留原版 decode_tactile，供 inference.py 继续使用 ───────────────────────
import base64

SENSOR_ORDER = [
    'finger_0_bottom', 'finger_0_middle', 'finger_0_top',
    'finger_1_bottom', 'finger_1_middle', 'finger_1_top',
    'finger_2_bottom', 'finger_2_middle', 'finger_2_top',
    'finger_3_bottom', 'finger_3_middle', 'finger_3_top',
    'thumb_bottom',    'thumb_middle',    'thumb_top',
]

def decode_tactile(tactile_dict: Dict) -> np.ndarray:
    """解码触觉传感器为 700 维向量（供推理器使用，与原版一致）"""
    parts = []
    for key in SENSOR_ORDER:
        if key not in tactile_dict:
            shape = [10, 7] if 'bottom' in key else ([8, 5] if 'middle' in key else [6, 5])
            parts.append(np.zeros(shape[0] * shape[1], dtype=np.float32))
        else:
            sensor = tactile_dict[key]
            b64    = sensor['data']
            if ',' in b64:
                b64 = b64.split(',')[1]
            raw   = base64.b64decode(b64)
            shape = sensor['shape']
            dtype = np.dtype(sensor['dtype'])
            data  = np.frombuffer(raw, dtype=dtype).reshape(shape).astype(np.float32)
            if dtype == np.uint8:
                data = data / 255.0
            parts.append(data.flatten())
    return np.concatenate(parts)


def get_tactile_dim() -> int:
    return 700


# ── 主数据集类 ───────────────────────────────────────────────────────────────

class HDF5MultiModalGraspDataset(Dataset):
    """
    基于 HDF5 的高速多模态数据集。

    HDF5 文件结构（由 preprocess.py 生成）:
      /episode_00001/
        images    float32 [T, C, H, W]  已归一化
        tactiles  float32 [T, 713]
        actions   float32 [T, 13]
      /episode_00002/
        ...

    __getitem__ 只做切片 + normalize，无任何解码，速度极快。
    """

    def __init__(
        self,
        h5_path: str,
        stats_path: str,
        pred_horizon:   int  = 16,
        obs_horizon:    int  = 2,
        action_horizon: int  = 8,
        normalize:      bool = True,
    ):
        self.h5_path       = str(h5_path)
        self.pred_horizon  = pred_horizon
        self.obs_horizon   = obs_horizon
        self.action_horizon= action_horizon
        self.normalize     = normalize

        # ── 加载统计量 ──────────────────────────────────────────────────────
        if normalize:
            with open(stats_path, 'rb') as f:
                self.stats = pickle.load(f)
        else:
            self.stats = None

        # ── 扫描 HDF5，构建索引 (episode_name, start_step) ─────────────────
        # 只读 episode 长度，不加载数据
        print(f"[HDF5Dataset] 扫描 {self.h5_path} ...")
        self._indices: List[Tuple[str, int]] = []
        self._ep_lengths: Dict[str, int]     = {}

        with h5py.File(self.h5_path, 'r') as f:
            ep_names = sorted(f.keys())
            for name in ep_names:
                T = f[name]['actions'].shape[0]
                self._ep_lengths[name] = T
                num_valid = max(0, T - pred_horizon + 1)
                for start in range(num_valid):
                    self._indices.append((name, start))

        print(f"[HDF5Dataset] episodes: {len(self._ep_lengths)},  samples: {len(self._indices)}")

        # ── 每个 worker 独立持有一个文件句柄（懒加载）──────────────────────
        # 不在 __init__ 里打开文件，避免多进程 fork 时句柄冲突
        self._h5file: Optional[h5py.File] = None

    # ── 文件句柄懒加载（多进程安全）────────────────────────────────────────
    def _get_h5(self) -> h5py.File:
        if self._h5file is None:
            self._h5file = h5py.File(self.h5_path, 'r', swmr=True)
        return self._h5file

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_name, start = self._indices[idx]
        h5 = self._get_h5()
        grp = h5[ep_name]

        T = self._ep_lengths[ep_name]

        # ── 视觉观测 [obs_horizon, C, H, W] ────────────────────────────────
        vis_end  = min(start + self.obs_horizon, T)
        vis_data = grp['images'][start:vis_end]             # float32, 已归一化
        # 不足 obs_horizon 时用最后一帧填充
        if vis_data.shape[0] < self.obs_horizon:
            pad = np.repeat(vis_data[[-1]], self.obs_horizon - vis_data.shape[0], axis=0)
            vis_data = np.concatenate([vis_data, pad], axis=0)
        visual_seq = torch.from_numpy(vis_data)             # [T, C, H, W]

        # ── 触觉观测 [obs_horizon, 713] ─────────────────────────────────────
        tac_end  = min(start + self.obs_horizon, T)
        tac_data = grp['tactiles'][start:tac_end].copy()    # float32 [<=obs_horizon, 713]
        if tac_data.shape[0] < self.obs_horizon:
            pad = np.repeat(tac_data[[-1]], self.obs_horizon - tac_data.shape[0], axis=0)
            tac_data = np.concatenate([tac_data, pad], axis=0)
        if self.normalize and self.stats is not None:
            tac_data = (tac_data - self.stats['tactile_obs_mean']) / self.stats['tactile_obs_std']
        tactile_seq = torch.from_numpy(tac_data).float()    # [T, 713]

        # ── 动作序列 [pred_horizon, 13] ─────────────────────────────────────
        act_end  = min(start + self.pred_horizon, T)
        act_data = grp['actions'][start:act_end].copy()     # float32 [<=pred_horizon, 13]
        if act_data.shape[0] < self.pred_horizon:
            pad = np.repeat(act_data[[-1]], self.pred_horizon - act_data.shape[0], axis=0)
            act_data = np.concatenate([act_data, pad], axis=0)
        if self.normalize and self.stats is not None:
            act_data = (act_data - self.stats['action_mean']) / self.stats['action_std']
        action_seq = torch.from_numpy(act_data).float()     # [T, 13]

        return {
            'visual_obs':  visual_seq,   # [obs_horizon, C, H, W]
            'tactile_obs': tactile_seq,  # [obs_horizon, 713]
            'action':      action_seq,   # [pred_horizon, 13]
        }

    def cleanup(self):
        if self._h5file is not None:
            try:
                self._h5file.close()
            except Exception:
                pass
            self._h5file = None


# ── 向后兼容别名（model.py 里用的是 LazyMultiModalGraspDataset）──────────────
LazyMultiModalGraspDataset = HDF5MultiModalGraspDataset


# ── demo ─────────────────────────────────────────────────────────────────────

def demo_dataset(h5_path: str, stats_path: str):
    import time
    print("=" * 60)
    print("HDF5MultiModalGraspDataset 演示")
    print("=" * 60)

    dataset = HDF5MultiModalGraspDataset(
        h5_path=h5_path,
        stats_path=stats_path,
        pred_horizon=16,
        obs_horizon=2,
        normalize=True,
    )
    print(f"数据集大小: {len(dataset)} 样本")

    # 随机访问测试
    indices = np.random.choice(len(dataset), min(10, len(dataset)), replace=False)
    times = []
    for i, idx in enumerate(indices):
        t0 = time.time()
        sample = dataset[int(idx)]
        times.append(time.time() - t0)
        if i < 3:
            print(f"  Sample {idx}: visual={sample['visual_obs'].shape}, "
                  f"tactile={sample['tactile_obs'].shape}, action={sample['action'].shape}, "
                  f"time={times[-1]*1000:.1f}ms")

    print(f"平均访问时间: {np.mean(times)*1000:.1f}ms")

    # DataLoader 测试
    loader = DataLoader(dataset, batch_size=32, num_workers=4,
                        pin_memory=True, prefetch_factor=4,
                        persistent_workers=True)
    t0 = time.time()
    batch = next(iter(loader))
    print(f"首 batch 加载时间: {time.time()-t0:.2f}s")
    for k, v in batch.items():
        print(f"  {k}: {v.shape}")

    dataset.cleanup()
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5',    type=str, default=str(DEFAULT_H5_PATH))
    parser.add_argument('--stats', type=str, default=str(DEFAULT_STATS_PATH))
    parser.add_argument('--demo',  action='store_true')
    args = parser.parse_args()

    if args.demo:
        demo_dataset(args.h5, args.stats)
    else:
        print("用法: python -m source.train.dataset --demo --h5 data/block_lifting.h5 --stats models/block_lifting/stats.pkl")