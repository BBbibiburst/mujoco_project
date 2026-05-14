"""
触觉热力图渲染.
"""

from typing import Dict
import cv2
import numpy as np

from source.sensors.tactile_sensor import FINGER_PHALANX_ORDER


_FINGER_KEYS  = ["finger_0", "finger_1", "finger_2", "finger_3", "thumb"]
_LEVEL_ORDER  = ["top", "middle", "bottom"]
_LEVEL_TO_KEY = {"top": "tactile_top", "middle": "tactile_middle", "bottom": "tactile_bottom"}
_LEVEL_TO_IDX = {"top": 2, "middle": 1, "bottom": 0}


def render_tactile_heatmap(obs: dict, sub_h: int = 160, sub_w: int = 200) -> np.ndarray:
    """
    将触觉图像渲染为热力图网格.

    布局：行=指节层（top/middle/bottom），列=手指（5根）
    返回 shape: (3*sub_h, 5*sub_w, 3), dtype uint8, BGR 格式
    """
    grid_rows = []
    for level in _LEVEL_ORDER:
        key = _LEVEL_TO_KEY[level]
        if key not in obs:
            continue

        imgs = obs[key]  # (5, H, W) 或 (5, H, W, 1)
        if imgs.ndim == 4:
            imgs = imgs[..., 0]

        row_frames = []
        phalanx_idx = _LEVEL_TO_IDX[level]
        for fi, finger in enumerate(_FINGER_KEYS):
            img = imgs[fi]
            enhanced = np.clip(img.astype(np.float32) * 5.0, 0, 255).astype(np.uint8)
            resized  = cv2.resize(enhanced, (sub_w, sub_h), interpolation=cv2.INTER_NEAREST)
            heatmap  = cv2.applyColorMap(resized, cv2.COLORMAP_JET)

            # 指节标签
            phalanx = FINGER_PHALANX_ORDER[finger][phalanx_idx]
            parts   = phalanx.split("_")
            short   = (f"T_{parts[1][:3].capitalize()}" if parts[0] == "thumb"
                       else f"F{parts[1]}_{parts[2][:3].capitalize()}")
            cv2.rectangle(heatmap, (0, 0), (sub_w, 22), (0, 0, 0), -1)
            cv2.putText(heatmap, short, (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            row_frames.append(heatmap)

        grid_rows.append(np.hstack(row_frames))

    if not grid_rows:
        return np.zeros((sub_h * 3, sub_w * 5, 3), dtype=np.uint8)
    return np.vstack(grid_rows)