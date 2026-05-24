#!/usr/bin/env python3
"""
preprocess.py - 离线预处理脚本（修复内存泄漏版）
将原始 JSONL 数据集转换为 HDF5 格式，只需运行一次。

用法:
  python preprocess.py --data_dir data/block_lifting --output data/block_lifting.h5
  python preprocess.py --data_dir data/block_lifting --output data/block_lifting.h5 --workers 2
"""

import io
import os
import json
import base64
import pickle
import argparse
import tempfile
import shutil
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ── 与 dataset.py 保持一致的图像预处理 ─────────────────────────────────────
IMAGE_SIZE   = (96, 96)
IMAGE_MEAN   = [0.485, 0.456, 0.406]
IMAGE_STD    = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
])

SENSOR_ORDER = [
    'finger_0_bottom', 'finger_0_middle', 'finger_0_top',
    'finger_1_bottom', 'finger_1_middle', 'finger_1_top',
    'finger_2_bottom', 'finger_2_middle', 'finger_2_top',
    'finger_3_bottom', 'finger_3_middle', 'finger_3_top',
    'thumb_bottom',    'thumb_middle',    'thumb_top',
]


def decode_tactile(tactile_dict) -> np.ndarray:
    """解码触觉传感器为 700 维向量（与 dataset.py 完全一致）"""
    parts = []
    for key in SENSOR_ORDER:
        if key not in tactile_dict:
            shape = [10, 7] if 'bottom' in key else ([8, 5] if 'middle' in key else [6, 5])
            parts.append(np.zeros(shape[0] * shape[1], dtype=np.float32))
        else:
            sensor  = tactile_dict[key]
            b64     = sensor['data']
            if ',' in b64:
                b64 = b64.split(',')[1]
            raw   = base64.b64decode(b64)
            shape = sensor['shape']
            dtype = np.dtype(sensor['dtype'])
            data  = np.frombuffer(raw, dtype=dtype).reshape(shape).astype(np.float32)
            if dtype == np.uint8:
                data = data / 255.0
            parts.append(data.flatten())
    return np.concatenate(parts)  # [700]


def process_one_episode(jsonl_path: Path):
    """
    处理单个 episode 文件，返回预处理后的 numpy 数组。
    这个函数在子进程中运行，不依赖任何共享状态。

    Returns:
        images:   float32 [T, C, H, W]  已归一化
        tactiles: float32 [T, 713]       arm+hand+tactile
        actions:  float32 [T, 13]        arm+hand
        ep_name:  str
    """
    steps = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                steps.append(json.loads(line)['info'])
            except Exception:
                continue

    if not steps:
        return None

    images, tactiles, actions = [], [], []

    for step in steps:
        # 图像
        b64 = step['images']['camera_rgb']
        if ',' in b64:
            b64 = b64.split(',')[1]
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')
        img_array = _transform(img).numpy()  # float32 [C,H,W]
        images.append(img_array)
        img.close()  # 显式关闭 PIL Image，释放解码缓存
        del img, img_array

        # 本体感知 + 触觉
        arm  = np.array(step['arm_qpos'],  dtype=np.float32)   # [7]
        hand = np.array(step['hand_qpos'], dtype=np.float32)   # [6]
        tact = decode_tactile(step['tactile'])                  # [700]
        tactiles.append(np.concatenate([arm, hand, tact]))     # [713]
        actions.append(np.concatenate([arm, hand]))            # [13]
        del arm, hand, tact

    result = (
        np.stack(images),    # [T, C, H, W]
        np.stack(tactiles),  # [T, 713]
        np.stack(actions),   # [T, 13]
        jsonl_path.stem,     # episode 名称
    )
    
    # 清理中间列表，减少子进程内存占用
    del images, tactiles, actions, steps
    return result


def process_one_episode_to_temp(jsonl_path: Path, temp_dir: str):
    """
    处理单个 episode，直接写入临时 HDF5 文件。
    子进程内完成所有 heavy lifting，主进程只接收轻量路径。
    
    Returns:
        temp_file_path: str  or  None（处理失败）
    """
    result = process_one_episode(jsonl_path)
    if result is None:
        return None
    
    images, tactiles, actions, ep_name = result
    temp_file = Path(temp_dir) / f"{ep_name}.h5"
    
    try:
        with h5py.File(temp_file, 'w') as f:
            f.create_dataset('images', data=images, compression='lzf', chunks=(1, *images.shape[1:]))
            f.create_dataset('tactiles', data=tactiles, compression='lzf', chunks=(min(16, len(tactiles)), 713))
            f.create_dataset('actions', data=actions, compression='lzf', chunks=(min(16, len(actions)), 13))
    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()
        raise e
    
    # 子进程内立即释放大数组
    del images, tactiles, actions, result
    return str(temp_file)


def compute_stats_from_hdf5(h5_path: str) -> dict:
    """
    从已生成的 HDF5 文件快速计算统计量（随机采样 500 个 episode）。
    比从原始 JSONL 计算快很多，因为 HDF5 列式读取很高效。
    """
    print("[Stats] 从 HDF5 采样计算统计量...")
    rng = np.random.default_rng(seed=42)

    with h5py.File(h5_path, 'r') as f:
        ep_names = list(f.keys())
        sample_names = rng.choice(ep_names, size=min(500, len(ep_names)), replace=False)

        actions_list, tactile_list = [], []
        for name in tqdm(sample_names, desc="[Stats] 采样"):
            grp = f[name]
            # 每个 episode 只取前 10 步
            actions_list.append(grp['actions'][:10])    # [<=10, 13]
            tactile_list.append(grp['tactiles'][:10])   # [<=10, 713]

    actions_arr = np.concatenate(actions_list, axis=0)   # [N, 13]
    tactile_arr = np.concatenate(tactile_list, axis=0)   # [N, 713]

    stats = {
        'action_mean':      actions_arr.mean(0).astype(np.float32),
        'action_std':       (actions_arr.std(0) + 1e-8).astype(np.float32),
        'tactile_obs_mean': tactile_arr.mean(0).astype(np.float32),
        'tactile_obs_std':  (tactile_arr.std(0) + 1e-8).astype(np.float32),
    }
    print(f"[Stats] 完成，采样 {len(actions_arr)} 步")
    return stats


def preprocess(data_dir: str, output_h5: str, num_workers: int = 2,
               stats_output: str = None):
    data_dir  = Path(data_dir)
    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(data_dir.glob('*.jsonl'))
    if not jsonl_files:
        jsonl_files = sorted(data_dir.rglob('*.jsonl'))
    if not jsonl_files:
        raise ValueError(f"在 {data_dir} 中未找到 .jsonl 文件")

    print(f"[Preprocess] 找到 {len(jsonl_files)} 个 episode 文件")
    print(f"[Preprocess] 输出: {output_h5}")
    print(f"[Preprocess] 并行 workers: {num_workers}")

    # ── 创建临时目录存放子进程的中间文件 ───────────────────────────────────
    temp_dir = tempfile.mkdtemp(prefix="preprocess_")
    print(f"[Preprocess] 临时目录: {temp_dir}")

    try:
        # ── 阶段 1：多进程并行处理，每个 episode 写入独立临时 HDF5 ─────────
        temp_files = []
        success = 0
        failed  = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_one_episode_to_temp, fp, temp_dir): fp
                for fp in jsonl_files
            }

            pbar = tqdm(as_completed(futures), total=len(futures),
                        desc="[Preprocess] 转换进度", unit="ep")

            for future in pbar:
                fp = futures[future]
                try:
                    temp_path = future.result()
                    if temp_path is None:
                        failed += 1
                        continue

                    temp_files.append(temp_path)
                    success += 1
                    pbar.set_postfix(ok=success, fail=failed)

                except Exception as e:
                    failed += 1
                    tqdm.write(f"[警告] {fp.name} 处理失败: {e}")

        print(f"\n[Preprocess] 阶段 1 完成: {success} 成功, {failed} 失败")

        if not temp_files:
            raise RuntimeError("没有成功处理任何 episode")

        # ── 阶段 2：主进程合并所有临时文件到最终 HDF5 ──────────────────────
        print(f"[Preprocess] 阶段 2: 合并 {len(temp_files)} 个临时文件...")
        
        with h5py.File(output_h5, 'w') as h5f:
            # 保存元信息
            h5f.attrs['image_size']  = IMAGE_SIZE
            h5f.attrs['image_mean']  = IMAGE_MEAN
            h5f.attrs['image_std']   = IMAGE_STD
            h5f.attrs['num_episodes'] = len(temp_files)

            for temp_path in tqdm(temp_files, desc="[Preprocess] 合并", unit="ep"):
                ep_name = Path(temp_path).stem
                
                with h5py.File(temp_path, 'r') as tmp:
                    grp = h5f.create_group(ep_name)
                    # 逐数据集拷贝，内存友好
                    h5f.copy(tmp['images'], grp, name='images')
                    h5f.copy(tmp['tactiles'], grp, name='tactiles')
                    h5f.copy(tmp['actions'], grp, name='actions')
                
                # 合并后立即删除临时文件，释放磁盘和内存
                Path(temp_path).unlink()
                del grp

        print(f"[Preprocess] 合并完成，已清理临时文件")

    finally:
        # 确保临时目录被清理（即使出错）
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"[Preprocess] 临时目录已清理")

    print(f"\n[Preprocess] 完成: {success} 成功, {failed} 失败")
    print(f"[Preprocess] HDF5 文件: {output_h5}  ({output_h5.stat().st_size/1e9:.2f} GB)")

    # ── 计算并保存统计量 ───────────────────────────────────────────────────
    stats_path = stats_output or str(output_h5.parent / 'stats.pkl')
    stats = compute_stats_from_hdf5(str(output_h5))
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f"[Preprocess] 统计量已保存: {stats_path}")

    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='离线预处理：JSONL → HDF5')
    parser.add_argument('--data_dir',  type=str, required=True,  help='原始 JSONL 数据目录')
    parser.add_argument('--output',    type=str, required=True,  help='输出 HDF5 文件路径')
    parser.add_argument('--workers',   type=int, default=2,      help='并行进程数（默认 2，内存敏感建议 1-2）')
    parser.add_argument('--stats_out', type=str, default=None,   help='统计量输出路径（默认与 HDF5 同目录）')
    args = parser.parse_args()

    preprocess(
        data_dir=args.data_dir,
        output_h5=args.output,
        num_workers=args.workers,
        stats_output=args.stats_out,
    )