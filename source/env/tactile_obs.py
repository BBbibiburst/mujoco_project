"""
触觉观测辅助类.

封装所有触觉传感器相关逻辑，供各 env 复用，消除重复代码。
"""

from typing import Dict, Optional, Tuple
import numpy as np
from gymnasium import spaces

from source.sensors.tactile_sensor import TactileReader, FINGER_PHALANX_ORDER, DISPLAY_ORDER


# ====================== 常量 ======================

FINGER_NAMES: Tuple[str, ...] = ("finger_0", "finger_1", "finger_2", "finger_3", "thumb")
N_FINGERS: int = len(FINGER_NAMES)

# 每个指节层的传感器分辨率 (rows, cols)
TACTILE_LEVELS: Dict[str, Tuple[int, int]] = {
    "bottom": (10, 7),
    "middle": (8, 5),
    "top":    (6, 5),
}

# 指节名称 → 指节层（调试用）
_PHALANX_TO_LEVEL: Dict[str, str] = {}
_LEVEL_IDX = {0: "bottom", 1: "middle", 2: "top"}
for _finger, _phalanges in FINGER_PHALANX_ORDER.items():
    for _idx, _name in enumerate(_phalanges):
        _PHALANX_TO_LEVEL[_name] = _LEVEL_IDX[_idx]


class TactileObsHelper:
    """
    触觉传感器观测助手.

    用法::

        # 在 env 的 __init__ 或 _init_simulation 后：
        self._tactile = TactileObsHelper(self.reader)

        # 在 _get_obs 中：
        tactile = self._tactile.get_grouped(self.data)
        obs["tactile_bottom"] = tactile["bottom"]

        # 检查是否有触觉激活：
        active = self._tactile.is_active(self.data, threshold=50.0)
    """

    def __init__(self, reader: Optional[TactileReader] = None):
        self._reader = reader

    def bind(self, reader: TactileReader) -> None:
        """绑定传感器（在 model 编译后、reader.bind(model) 之后调用）."""
        self._reader = reader

    # ====================== 观测空间 ======================

    @staticmethod
    def observation_spaces() -> Dict[str, spaces.Box]:
        """返回触觉相关的 observation_space 条目，供 env 合并到 spaces.Dict."""
        return {
            "tactile_bottom": spaces.Box(0, 255, (N_FINGERS, 10, 7), dtype=np.uint8),
            "tactile_middle": spaces.Box(0, 255, (N_FINGERS, 8, 5),  dtype=np.uint8),
            "tactile_top":    spaces.Box(0, 255, (N_FINGERS, 6, 5),  dtype=np.uint8),
        }

    # ====================== 观测获取 ======================

    def get_grouped(self, data) -> Dict[str, np.ndarray]:
        """
        获取触觉图像，按指节层分组.

        Returns:
            dict with keys "bottom" / "middle" / "top",
            each shape (N_FINGERS, H, W), dtype uint8.
        """
        if self._reader is None:
            return self.empty()

        try:
            raw = self._reader.read_image(data)
            if not raw:
                return self.empty()
            return self._parse_raw(raw)
        except Exception:
            return self.empty()

    def is_active(self, data, threshold: float = 50.0) -> bool:
        """任意指节层最大值超过阈值则返回 True."""
        tactile = self.get_grouped(data)
        return any(v.max() > threshold for v in tactile.values())

    @staticmethod
    def empty() -> Dict[str, np.ndarray]:
        """返回全零的分组触觉图像."""
        return {
            "bottom": np.zeros((N_FINGERS, 10, 7), dtype=np.uint8),
            "middle": np.zeros((N_FINGERS, 8, 5),  dtype=np.uint8),
            "top":    np.zeros((N_FINGERS, 6, 5),  dtype=np.uint8),
        }

    # ====================== 调试 ======================

    def verify_shapes(self, data) -> None:
        """打印传感器实际分辨率（调试用）."""
        if self._reader is None:
            print("[Tactile] reader is None")
            return
        raw = self._reader.read_image(data)
        if not raw:
            print("[Tactile] no images returned")
            return

        print("=== 触觉传感器实际分辨率 ===")
        for name in DISPLAY_ORDER:
            if name in raw:
                img = raw[name]
                level = _PHALANX_TO_LEVEL.get(name, "?")
                print(f"  {name} ({level}): shape={img.shape}, max={img.max():.1f}")
            else:
                print(f"  {name}: MISSING")

        print("\n=== 预期分辨率 (rows, cols) ===")
        for level, (h, w) in TACTILE_LEVELS.items():
            print(f"  {level}: ({h}, {w})")

    # ====================== 私有方法 ======================

    def _parse_raw(self, raw: dict) -> Dict[str, np.ndarray]:
        result = {}
        for level, level_idx in [("bottom", 0), ("middle", 1), ("top", 2)]:
            expected_h, expected_w = TACTILE_LEVELS[level]
            imgs = []
            for finger in FINGER_NAMES:
                phalanx = FINGER_PHALANX_ORDER[finger][level_idx]
                if phalanx in raw:
                    img = raw[phalanx]
                    img = self._ensure_shape(img, expected_h, expected_w)
                else:
                    img = np.zeros((expected_h, expected_w), dtype=np.uint8)
                imgs.append(img)
            result[level] = np.stack(imgs, axis=0)
        return result

    @staticmethod
    def _ensure_shape(img: np.ndarray, h: int, w: int) -> np.ndarray:
        """保证图像形状为 (h, w)，必要时转置或 resize."""
        if img.shape == (w, h):
            return img.T.astype(np.uint8)
        if img.shape == (h, w):
            return img.astype(np.uint8)
        import cv2
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)