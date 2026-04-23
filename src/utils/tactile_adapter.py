"""
触觉传感器统一适配层 (Tactile Adapter)
================================================================================

本模块为两套触觉传感器实现提供统一接口，使上层任务代码无需关心底层实现。

两套底层实现：
    - SimpleBackend  ← touch_sensor_builder.py
                       直接将 site 挂到 body 上，无弹性形变，输出原始法向力 [N]
    - PhysicsBackend ← touch_sensor_builder_physic_based.py
                       每个 taxel 有独立弹性滑动关节，力值更平滑，输出可归一化

统一接口 (TactileReader)：
    ┌──────────────────────────────────────────────────────┐
    │  build_sensor(spec, hand_path, prefix, ...)          │  构建阶段
    │  bind(model)                                         │  编译后绑定
    │  read_raw(data)  → Dict[str, ndarray(rows,cols) f32] │  读取原始力 [N]
    │  read_image(data)→ Dict[str, ndarray(rows,cols) u8]  │  读取归一化图像
    │  read_flat(data) → Dict[str, ndarray(N,) f32]        │  读取展平力向量
    │  metadata        → Dict[str, PhalanxMeta]            │  阵列元信息
    └──────────────────────────────────────────────────────┘

快速使用示例：
    # 1. 选择后端（改这一行即可切换实现）
    reader = TactileReaderFactory.create("physics", spec, hand_path, prefix)
    # reader = TactileReaderFactory.create("simple", spec, hand_path, prefix)

    # 2. 编译模型后绑定
    model = spec.compile()
    reader.bind(model)

    # 3. 仿真循环中读取（接口完全一致，与后端无关）
    tactile = reader.read_raw(data)      # Dict[str -> (rows, cols) float32, N]
    images  = reader.read_image(data)    # Dict[str -> (rows, cols) uint8, 0-255]
    flat    = reader.read_flat(data)     # Dict[str -> (N,) float32, N]

命名键对齐规则：
    Simple  版键名来自 SkinConfig.mesh_name,  如 "skin_0_0_p"
    Physics 版键名来自 PhalanxConfig.phalanx_name, 如 "finger_0_bottom"
    适配层在初始化时建立 skin_name ↔ phalanx_name 的双向映射，
    统一对外暴露 phalanx_name 作为标准键。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np


# ===========================================================================
# 公共数据结构
# ===========================================================================

# Simple版 skin_name → Physics版 phalanx_name 的标准映射
# 两套配置覆盖相同的15块皮肤，顺序与 SKIN_CONFIGS / PHALANX_CONFIGS 一致
_SKIN_TO_PHALANX: Dict[str, str] = {
    "skin_0_0_p": "finger_0_bottom",
    "skin_0_1_p": "finger_0_middle",
    "skin_0_2_p": "finger_0_top",
    "skin_1_0_p": "finger_1_bottom",
    "skin_1_1_p": "finger_1_middle",
    "skin_1_2_p": "finger_1_top",
    "skin_2_0_p": "finger_2_bottom",
    "skin_2_1_p": "finger_2_middle",
    "skin_2_2_p": "finger_2_top",
    "skin_3_0_p": "finger_3_bottom",
    "skin_3_1_p": "finger_3_middle",
    "skin_3_2_p": "finger_3_top",
    "skin_4_0_p": "thumb_bottom",
    "skin_4_1_p": "thumb_middle",
    "skin_4_2_p": "thumb_top",
}
_PHALANX_TO_SKIN: Dict[str, str] = {v: k for k, v in _SKIN_TO_PHALANX.items()}

# 标准展示顺序（15个指节，从食指底部到拇指顶部）
DISPLAY_ORDER: List[str] = [
    "finger_0_bottom", "finger_0_middle", "finger_0_top",
    "finger_1_bottom", "finger_1_middle", "finger_1_top",
    "finger_2_bottom", "finger_2_middle", "finger_2_top",
    "finger_3_bottom", "finger_3_middle", "finger_3_top",
    "thumb_bottom",    "thumb_middle",    "thumb_top",
]

# 各手指的指节顺序（index 0=bottom, 1=middle, 2=top）
# 与 physics_based 版原始定义保持一致，供可视化拼图使用
FINGER_PHALANX_ORDER: Dict[str, List[str]] = {
    "finger_0": ["finger_0_bottom", "finger_0_middle", "finger_0_top"],
    "finger_1": ["finger_1_bottom", "finger_1_middle", "finger_1_top"],
    "finger_2": ["finger_2_bottom", "finger_2_middle", "finger_2_top"],
    "finger_3": ["finger_3_bottom", "finger_3_middle", "finger_3_top"],
    "thumb":    ["thumb_bottom",    "thumb_middle",    "thumb_top"],
}

# 各指节形状（phalanx_name → (rows, cols)），两版本一致
_PHALANX_SHAPE: Dict[str, Tuple[int, int]] = {
    "finger_0_bottom": (10, 7), "finger_0_middle": (8, 5), "finger_0_top": (6, 5),
    "finger_1_bottom": (10, 7), "finger_1_middle": (8, 5), "finger_1_top": (6, 5),
    "finger_2_bottom": (10, 7), "finger_2_middle": (8, 5), "finger_2_top": (6, 5),
    "finger_3_bottom": (10, 7), "finger_3_middle": (8, 5), "finger_3_top": (6, 5),
    "thumb_bottom":    (10, 7), "thumb_middle":    (8, 5), "thumb_top":    (6, 5),
}


@dataclass(frozen=True)
class PhalanxMeta:
    """单个指节的元信息，与后端无关。"""
    phalanx_name: str          # 统一键名，如 "finger_0_bottom"
    skin_name: str             # Simple版键名，如 "skin_0_0_p"
    rows: int                  # taxel 行数
    cols: int                  # taxel 列数
    total_taxels: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "total_taxels", self.rows * self.cols)


# ===========================================================================
# 抽象基类
# ===========================================================================

class TactileReader(ABC):
    """
    触觉传感器统一读取接口（抽象基类）。

    所有方法均以 phalanx_name 为键（如 "finger_0_bottom"），
    与底层 Simple/Physics 实现无关。

    生命周期：
        1. __init__  → 存储构建参数（此时 spec 尚未修改）
        2. build_sensor(spec, ...) → 向 spec 添加传感器（编译前调用）
        3. bind(model)  → 将传感器名称绑定到编译后模型的 ID（编译后调用）
        4. read_*(data) → 仿真循环中读取数据
    """

    # 归一化饱和力阈值 [N]，子类可覆盖
    FORCE_MAX_NEWTON: float = 5.0

    def __init__(self):
        self._bound = False
        self._meta: Dict[str, PhalanxMeta] = {
            name: PhalanxMeta(
                phalanx_name=name,
                skin_name=_PHALANX_TO_SKIN[name],
                rows=_PHALANX_SHAPE[name][0],
                cols=_PHALANX_SHAPE[name][1],
            )
            for name in DISPLAY_ORDER
        }

    # ── 构建与绑定 ──────────────────────────────────────────────────────────

    @abstractmethod
    def build_sensor(
        self,
        spec: mujoco.MjSpec,
        hand_path: Path,
        prefix: str = "inspirehand_",
        **kwargs,
    ) -> None:
        """
        向 MjSpec 添加传感器（编译前调用）。

        Args:
            spec:      未编译的 MjSpec 对象，函数将直接修改它。
            hand_path: 灵巧手 XML 路径，用于定位 meshes/ 目录。
            prefix:    body/sensor 名称前缀，需与 attach_body 时一致。
            **kwargs:  后端专属参数（可选）。
        """

    @abstractmethod
    def bind(self, model: mujoco.MjModel) -> None:
        """
        将传感器名称绑定到编译后模型中的整数 ID（编译后、仿真前调用一次）。

        Args:
            model: 已编译的 MjModel 对象。

        Raises:
            RuntimeError: 若 build_sensor 尚未调用。
            ValueError:   若传感器名称在模型中找不到。
        """

    # ── 数据读取 ────────────────────────────────────────────────────────────

    @abstractmethod
    def _read_raw_impl(self, data: mujoco.MjData) -> Dict[str, np.ndarray]:
        """
        后端实现：返回 phalanx_name → (rows, cols) float32 [N]。
        由子类实现，不直接对外暴露。
        """

    def read_raw(self, data: mujoco.MjData) -> Dict[str, np.ndarray]:
        """
        读取各指节触觉数据，原始法向力 [N]。

        Returns:
            Dict[phalanx_name, ndarray(rows, cols, float32)]
            值域约为 [0, FORCE_MAX_NEWTON]，未裁剪。

        Raises:
            RuntimeError: 若 bind() 尚未调用。
        """
        self._check_bound()
        return self._read_raw_impl(data)

    def read_image(
        self,
        data: mujoco.MjData,
        force_max: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        读取各指节触觉图像，归一化为 uint8 [0, 255]。

        归一化公式：image = clip(raw, 0, force_max) / force_max * 255

        Args:
            data:      MjData 对象。
            force_max: 饱和力 [N]，None 时使用 self.FORCE_MAX_NEWTON。

        Returns:
            Dict[phalanx_name, ndarray(rows, cols, uint8)]
        """
        fmax = force_max if force_max is not None else self.FORCE_MAX_NEWTON
        raw  = self.read_raw(data)
        return {
            name: (np.clip(arr, 0.0, fmax) / fmax * 255.0).astype(np.uint8)
            for name, arr in raw.items()
        }

    def read_flat(self, data: mujoco.MjData) -> Dict[str, np.ndarray]:
        """
        读取各指节触觉数据，展平为一维向量 [N]。

        Returns:
            Dict[phalanx_name, ndarray(rows*cols, float32)]
        """
        return {name: arr.ravel() for name, arr in self.read_raw(data).items()}

    def read_concat(self, data: mujoco.MjData) -> np.ndarray:
        """
        将所有指节数据按 DISPLAY_ORDER 顺序拼接为单一向量 [N]。

        Returns:
            ndarray(total_taxels,) float32，顺序固定，可直接送入神经网络。
        """
        raw = self.read_raw(data)
        return np.concatenate([raw[name].ravel() for name in DISPLAY_ORDER])

    def read_with_metadata(self, data: mujoco.MjData) -> Dict[str, dict]:
        """
        读取带完整元信息的触觉帧，格式与真实传感器协议对齐。

        Returns:
            Dict[phalanx_name, {
                "timestamp":    float,           # Unix 时间戳 [s]
                "phalanx_name": str,
                "rows":         int,
                "cols":         int,
                "frame":        ndarray(rows, cols, float32),  # 原始力 [N]
                "image":        ndarray(rows, cols, uint8),    # 归一化图像
                "force_max_N":  float,
                "contact_mask": ndarray(rows, cols, bool),     # 接触检测
                "contact_area": int,                           # 接触 taxel 数
                "centroid":     ndarray(2,) float | None,      # 接触重心 (row, col)
                "total_force_N":float,                         # 总接触力 [N]
            }]
        """
        ts  = time.time()
        raw = self.read_raw(data)
        out = {}
        for name, frame in raw.items():
            meta  = self._meta[name]
            image = (np.clip(frame, 0.0, self.FORCE_MAX_NEWTON)
                     / self.FORCE_MAX_NEWTON * 255.0).astype(np.uint8)
            mask  = frame > 0.01  # 接触阈值 0.01 N

            # 接触重心（行列坐标，加权平均）
            centroid: Optional[np.ndarray] = None
            if mask.any():
                rows_idx, cols_idx = np.indices(frame.shape)
                total = frame[mask].sum() + 1e-12
                cr = (frame * rows_idx).sum() / total
                cc = (frame * cols_idx).sum() / total
                centroid = np.array([cr, cc], dtype=np.float32)

            out[name] = {
                "timestamp":     ts,
                "phalanx_name":  name,
                "rows":          meta.rows,
                "cols":          meta.cols,
                "frame":         frame.astype(np.float32),
                "image":         image,
                "force_max_N":   self.FORCE_MAX_NEWTON,
                "contact_mask":  mask,
                "contact_area":  int(mask.sum()),
                "centroid":      centroid,
                "total_force_N": float(frame.sum()),
            }
        return out

    # ── 元信息查询 ──────────────────────────────────────────────────────────

    @property
    def metadata(self) -> Dict[str, PhalanxMeta]:
        """返回所有指节的元信息字典（phalanx_name → PhalanxMeta）。"""
        return self._meta

    @property
    def total_taxels(self) -> int:
        """所有指节 taxel 总数。"""
        return sum(m.total_taxels for m in self._meta.values())

    @property
    def backend_name(self) -> str:
        """后端名称字符串，用于日志和调试。"""
        return self.__class__.__name__

    def __repr__(self) -> str:
        bound_str = "bound" if self._bound else "not bound"
        return (f"<{self.backend_name} | {len(self._meta)} phalanges | "
                f"{self.total_taxels} taxels | {bound_str}>")

    # ── 内部工具 ────────────────────────────────────────────────────────────

    def _check_bound(self):
        if not self._bound:
            raise RuntimeError(
                f"{self.backend_name}: 请先调用 bind(model) 再读取数据。"
            )


# ===========================================================================
# Simple 后端适配器
# ===========================================================================

class SimpleTactileReader(TactileReader):
    """
    Simple 后端适配器，封装 touch_sensor_builder.py。

    底层使用轻量 site（3mm 半径），直接挂在 body 上，无弹性形变。
    适合快速原型验证，仿真计算开销小。
    """

    FORCE_MAX_NEWTON = 5.0

    def __init__(self):
        super().__init__()
        # phalanx_name → sensor_id_array (rows, cols)，bind 后填充
        self._sensor_ids: Dict[str, np.ndarray] = {}
        # phalanx_name → List[sensor_name]，build 后填充
        self._sensor_names: Dict[str, List[str]] = {}
        self._built = False

    def build_sensor(
        self,
        spec: mujoco.MjSpec,
        hand_path: Path,
        prefix: str = "inspirehand_",
        site_group: int = 4,
        site_rgba: Tuple[float, ...] = (1.0, 0.2, 0.2, 0.6),
        **kwargs,
    ) -> None:
        """
        调用 touch_sensor_builder.add_touch_sensors_to_spec，
        并将返回的 skin_name 键转换为统一的 phalanx_name 键。
        """
        from src.utils.touch_sensor_builder import add_touch_sensors_to_spec

        raw_map = add_touch_sensors_to_spec(
            spec=spec,
            hand_path=hand_path,
            prefix=prefix,
            site_group=site_group,
            site_rgba=site_rgba,
        )

        # skin_name → phalanx_name 键转换
        for skin_name, sensor_names in raw_map.items():
            phalanx_name = _SKIN_TO_PHALANX.get(skin_name)
            if phalanx_name is None:
                # 未知 skin，按原键保留（兼容自定义配置）
                phalanx_name = skin_name
            self._sensor_names[phalanx_name] = sensor_names

        self._built = True
        print(f"[SimpleTactileReader] build 完成，共 {len(self._sensor_names)} 个指节")

    def bind(self, model: mujoco.MjModel) -> None:
        if not self._built:
            raise RuntimeError("请先调用 build_sensor() 再调用 bind()。")

        for phalanx_name, names in self._sensor_names.items():
            meta = self._meta.get(phalanx_name)
            rows, cols = (meta.rows, meta.cols) if meta else (len(names), 1)

            ids = np.zeros((rows, cols), dtype=np.int32)
            for idx, sname in enumerate(names):
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sname)
                if sid < 0:
                    raise ValueError(
                        f"[SimpleTactileReader] 传感器 '{sname}' 在模型中不存在，"
                        f"请确认 build_sensor() 与当前模型一致。"
                    )
                r, c = divmod(idx, cols)
                ids[r, c] = sid

            self._sensor_ids[phalanx_name] = ids

        self._bound = True
        print(f"[SimpleTactileReader] bind 完成，共绑定 {len(self._sensor_ids)} 个指节")

    def _read_raw_impl(self, data: mujoco.MjData) -> Dict[str, np.ndarray]:
        return {
            name: data.sensordata[ids.ravel()].reshape(ids.shape).astype(np.float32)
            for name, ids in self._sensor_ids.items()
        }


# ===========================================================================
# Physics 后端适配器
# ===========================================================================

class PhysicsTactileReader(TactileReader):
    """
    Physics 后端适配器，封装 touch_sensor_builder_physic_based.py。

    每个 taxel 有独立弹性滑动关节（stiffness=200 N/m，行程 2mm），
    力值更平滑，更接近真实触觉传感器（如 BioTac / XELA）的响应特性。
    注意：该后端会向模型添加大量 joint/geom，仿真计算开销略高。
    """

    FORCE_MAX_NEWTON = 5.0

    def __init__(self):
        super().__init__()
        # phalanx_name → PhalanxSensorArray，build 后填充
        self._arrays: Dict = {}
        self._built = False

    def build_sensor(
        self,
        spec: mujoco.MjSpec,
        hand_path: Path,
        prefix: str = "inspirehand_",
        **kwargs,
    ) -> None:
        """
        调用 touch_sensor_builder_physic_based.add_elastic_taxel_arrays，
        返回的 phalanx_name 键已与统一标准一致，无需转换。
        """
        from src.utils.touch_sensor_builder_physic_based import add_elastic_taxel_arrays

        self._arrays = add_elastic_taxel_arrays(
            spec=spec,
            hand_path=hand_path,
            prefix=prefix,
        )
        self._built = True
        print(f"[PhysicsTactileReader] build 完成，共 {len(self._arrays)} 个指节")

    def bind(self, model: mujoco.MjModel) -> None:
        if not self._built:
            raise RuntimeError("请先调用 build_sensor() 再调用 bind()。")

        from src.utils.touch_sensor_builder_physic_based import bind_all
        bind_all(self._arrays, model)
        self._bound = True
        print(f"[PhysicsTactileReader] bind 完成，共绑定 {len(self._arrays)} 个指节")

    def _read_raw_impl(self, data: mujoco.MjData) -> Dict[str, np.ndarray]:
        return {
            name: arr.read_raw(data).astype(np.float32)
            for name, arr in self._arrays.items()
        }


# ===========================================================================
# 工厂函数
# ===========================================================================

class TactileReaderFactory:
    """
    触觉读取器工厂，通过字符串选择后端，上层代码无需 import 具体类。

    支持的 backend 字符串：
        "simple"   → SimpleTactileReader
        "physics"  → PhysicsTactileReader

    示例：
        reader = TactileReaderFactory.create("physics")
        reader.build_sensor(spec, hand_path, prefix="inspirehand_")
        model = spec.compile()
        reader.bind(model)
    """

    _REGISTRY = {
        "simple":  SimpleTactileReader,
        "physics": PhysicsTactileReader,
    }

    @classmethod
    def create(cls, backend: str) -> TactileReader:
        """
        创建触觉读取器实例。

        Args:
            backend: 后端名称，"simple" 或 "physics"。

        Returns:
            TactileReader 实例（未 build，未 bind）。

        Raises:
            ValueError: 若 backend 字符串不合法。
        """
        backend = backend.lower().strip()
        if backend not in cls._REGISTRY:
            raise ValueError(
                f"未知后端 '{backend}'，支持的选项：{list(cls._REGISTRY.keys())}"
            )
        instance = cls._REGISTRY[backend]()
        print(f"[TactileReaderFactory] 创建 {instance.backend_name}")
        return instance

    @classmethod
    def available_backends(cls) -> List[str]:
        """返回所有已注册的后端名称。"""
        return list(cls._REGISTRY.keys())


# ===========================================================================
# 便捷函数：一步完成构建 + 绑定
# ===========================================================================

def build_and_bind_tactile_reader(
    backend: str,
    spec: mujoco.MjSpec,
    hand_path: Path,
    model: mujoco.MjModel,
    prefix: str = "inspirehand_",
    **kwargs,
) -> TactileReader:
    """
    一步完成创建、build_sensor、bind 的便捷函数。

    ⚠️  注意：spec.compile() 必须在 build_sensor 之后、bind 之前调用。
        因此本函数要求传入已编译的 model。若 spec 尚未编译，请手动分步调用。

    Args:
        backend:   "simple" 或 "physics"。
        spec:      已通过 build_sensor 修改并编译的 MjSpec（函数内部仅 build）。
        hand_path: 灵巧手 XML 路径。
        model:     已编译的 MjModel（由 spec.compile() 得到）。
        prefix:    body/sensor 名称前缀。
        **kwargs:  透传给 build_sensor 的额外参数。

    Returns:
        已绑定的 TactileReader 实例，可直接调用 read_* 方法。

    示例（推荐用法）：
        spec = get_combined_spec(...)          # 获取未编译的 spec

        reader = TactileReaderFactory.create("physics")
        reader.build_sensor(spec, hand_path, prefix)  # 修改 spec

        model = spec.compile()                 # 编译
        data  = mujoco.MjData(model)

        reader.bind(model)                     # 绑定 ID
    """
    reader = TactileReaderFactory.create(backend)
    reader.build_sensor(spec, hand_path, prefix=prefix, **kwargs)
    reader.bind(model)
    return reader


# ===========================================================================
# 公开符号列表
# ===========================================================================

__all__ = [
    # 核心类
    "TactileReader",
    "SimpleTactileReader",
    "PhysicsTactileReader",
    "TactileReaderFactory",
    # 数据结构
    "PhalanxMeta",
    # 常量
    "DISPLAY_ORDER",
    "FINGER_PHALANX_ORDER",
    # 便捷函数
    "build_and_bind_tactile_reader",
]