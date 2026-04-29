"""
触觉传感器统一适配层 (Tactile Adapter)

对外只暴露 TactileReader 接口，内部封装四种后端：
- simple:      轻量 site 方案，单点采样
- physics:     弹性 taxel 方案，单点采样
- simple_avg:  Simple 后端 + 3x3 子采样平均
- physics_avg: Physics 后端 + 3x3 子采样平均
"""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import mujoco
import numpy as np


# ===========================================================================
# 常量与配置
# ===========================================================================

_SKIN_TO_PHALANX = {
    "skin_0_0_p": "finger_0_bottom", "skin_0_1_p": "finger_0_middle", "skin_0_2_p": "finger_0_top",
    "skin_1_0_p": "finger_1_bottom", "skin_1_1_p": "finger_1_middle", "skin_1_2_p": "finger_1_top",
    "skin_2_0_p": "finger_2_bottom", "skin_2_1_p": "finger_2_middle", "skin_2_2_p": "finger_2_top",
    "skin_3_0_p": "finger_3_bottom", "skin_3_1_p": "finger_3_middle", "skin_3_2_p": "finger_3_top",
    "skin_4_0_p": "thumb_bottom",    "skin_4_1_p": "thumb_middle",    "skin_4_2_p": "thumb_top",
}

DISPLAY_ORDER: List[str] = [
    "finger_0_bottom", "finger_0_middle", "finger_0_top",
    "finger_1_bottom", "finger_1_middle", "finger_1_top",
    "finger_2_bottom", "finger_2_middle", "finger_2_top",
    "finger_3_bottom", "finger_3_middle", "finger_3_top",
    "thumb_bottom",    "thumb_middle",    "thumb_top",
]

_PHALANX_SHAPE: Dict[str, Tuple[int, int]] = {
    "finger_0_bottom": (10, 7), "finger_0_middle": (8, 5), "finger_0_top": (6, 5),
    "finger_1_bottom": (10, 7), "finger_1_middle": (8, 5), "finger_1_top": (6, 5),
    "finger_2_bottom": (10, 7), "finger_2_middle": (8, 5), "finger_2_top": (6, 5),
    "finger_3_bottom": (10, 7), "finger_3_middle": (8, 5), "finger_3_top": (6, 5),
    "thumb_bottom":    (10, 7), "thumb_middle":    (8, 5), "thumb_top":    (6, 5),
}

FINGER_PHALANX_ORDER: Dict[str, List[str]] = {
    "finger_0": ["finger_0_bottom", "finger_0_middle", "finger_0_top"],
    "finger_1": ["finger_1_bottom", "finger_1_middle", "finger_1_top"],
    "finger_2": ["finger_2_bottom", "finger_2_middle", "finger_2_top"],
    "finger_3": ["finger_3_bottom", "finger_3_middle", "finger_3_top"],
    "thumb":    ["thumb_bottom",    "thumb_middle",    "thumb_top"],
}

_PHALANX_CONFIGS: List[Tuple[str, str, str, int, int]] = [
    ("finger_0_bottom", "finger_first_0_p",  "skin_0_0_p.STL", 10, 7),
    ("finger_0_middle", "finger_second_0_p", "skin_0_1_p.STL",  8, 5),
    ("finger_0_top",    "finger_second_0_p", "skin_0_2_p.STL",  6, 5),
    ("finger_1_bottom", "finger_first_1_p",  "skin_1_0_p.STL", 10, 7),
    ("finger_1_middle", "finger_second_1_p", "skin_1_1_p.STL",  8, 5),
    ("finger_1_top",    "finger_second_1_p", "skin_1_2_p.STL",  6, 5),
    ("finger_2_bottom", "finger_first_2_p",  "skin_2_0_p.STL", 10, 7),
    ("finger_2_middle", "finger_second_2_p", "skin_2_1_p.STL",  8, 5),
    ("finger_2_top",    "finger_second_2_p", "skin_2_2_p.STL",  6, 5),
    ("finger_3_bottom", "finger_first_3_p",  "skin_3_0_p.STL", 10, 7),
    ("finger_3_middle", "finger_second_3_p", "skin_3_1_p.STL",  8, 5),
    ("finger_3_top",    "finger_second_3_p", "skin_3_2_p.STL",  6, 5),
    ("thumb_bottom",    "thumb_first_p",     "skin_4_0_p.STL", 10, 7),
    ("thumb_middle",    "thumb_second_p",    "skin_4_1_p.STL",  8, 5),
    ("thumb_top",       "thumb_third_p",     "skin_4_2_p.STL",  6, 5),
]

# ===========================================================================
# 数据结构
# ===========================================================================

@dataclass(frozen=True)
class PhalanxMeta:
    phalanx_name: str
    skin_name: str
    rows: int
    cols: int
    total_taxels: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "total_taxels", self.rows * self.cols)


# ===========================================================================
# 抽象基类（对外接口）
# ===========================================================================

class TactileReader(ABC):
    """
    统一读取接口。所有方法以 phalanx_name 为键（如 "finger_0_bottom"）。
    
    使用流程：
        reader = TactileReader.create("physics_avg")
        reader.build(spec, hand_path, prefix)
        model = spec.compile()
        reader.bind(model)
        
        raw = reader.read_raw(data)      # Dict[str, ndarray(rows,cols)]
        img = reader.read_image(data)    # Dict[str, ndarray(rows,cols, uint8)]
        vec = reader.read_concat(data)   # ndarray(total_taxels,) 直接送网络
    """

    FORCE_MAX_NEWTON: float = 5.0
    SUBGRID: int = 1

    def __init__(self):
        self._bound = False
        self._meta: Dict[str, PhalanxMeta] = {
            name: PhalanxMeta(name, _SKIN_TO_PHALANX.get(name, name), *_PHALANX_SHAPE[name])
            for name in DISPLAY_ORDER
        }

    # ── 构建与绑定 ──────────────────────────────────────────────────────────

    @abstractmethod
    def build(self, spec: mujoco.MjSpec, hand_path: Path, prefix: str = "inspirehand_", **kwargs) -> None:
        """向 spec 添加传感器（编译前调用）。"""

    @abstractmethod
    def bind(self, model: mujoco.MjModel) -> None:
        """绑定到编译后模型（编译后调用一次）。"""

    # ── 数据读取（通用实现，子类无需覆盖）────────────────────────────────────

    def read_raw(self, data: mujoco.MjData) -> Dict[str, np.ndarray]:
        self._check_bound()
        return self._read_raw_impl(data)

    def read_image(self, data: mujoco.MjData, force_max: Optional[float] = None) -> Dict[str, np.ndarray]:
        fmax = force_max or self.FORCE_MAX_NEWTON
        return {k: (np.clip(v, 0, fmax) / fmax * 255).astype(np.uint8)
                for k, v in self.read_raw(data).items()}

    def read_flat(self, data: mujoco.MjData) -> Dict[str, np.ndarray]:
        return {k: v.ravel() for k, v in self.read_raw(data).items()}

    def read_concat(self, data: mujoco.MjData) -> np.ndarray:
        """按 DISPLAY_ORDER 拼接为单一向量，可直接输入神经网络。"""
        raw = self.read_raw(data)
        return np.concatenate([raw[name].ravel() for name in DISPLAY_ORDER])

    def read_with_metadata(self, data: mujoco.MjData) -> Dict[str, dict]:
        """读取带完整元信息的触觉帧（与真实传感器协议对齐）。"""
        ts, raw = time.time(), self.read_raw(data)
        out = {}
        for name, frame in raw.items():
            meta = self._meta[name]
            mask = frame > 0.01
            centroid = None
            if mask.any():
                rows_idx, cols_idx = np.indices(frame.shape)
                total = frame[mask].sum() + 1e-12
                centroid = np.array([
                    (frame * rows_idx).sum() / total,
                    (frame * cols_idx).sum() / total
                ], dtype=np.float32)
            out[name] = {
                "timestamp": ts, "phalanx_name": name,
                "rows": meta.rows, "cols": meta.cols,
                "frame": frame.astype(np.float32),
                "image": (np.clip(frame, 0, self.FORCE_MAX_NEWTON) / self.FORCE_MAX_NEWTON * 255).astype(np.uint8),
                "force_max_N": self.FORCE_MAX_NEWTON,
                "contact_mask": mask, "contact_area": int(mask.sum()),
                "centroid": centroid, "total_force_N": float(frame.sum()),
            }
        return out

    # ── 属性 ────────────────────────────────────────────────────────────────

    @property
    def metadata(self) -> Dict[str, PhalanxMeta]: return self._meta
    @property
    def total_taxels(self) -> int: return sum(m.total_taxels for m in self._meta.values())
    @property
    def backend_name(self) -> str: return self.__class__.__name__

    def __repr__(self) -> str:
        return f"<{self.backend_name} | {len(self._meta)} phalanges | {self.total_taxels} taxels | {'bound' if self._bound else 'not bound'}>"

    # ── 内部 ────────────────────────────────────────────────────────────────

    @abstractmethod
    def _read_raw_impl(self, data: mujoco.MjData) -> Dict[str, np.ndarray]: ...

    def _check_bound(self):
        if not self._bound:
            raise RuntimeError(f"{self.backend_name}: 请先调用 bind(model)")

    # ── 工厂方法 ────────────────────────────────────────────────────────────

    @classmethod
    def create(cls, backend: str) -> TactileReader:
        """工厂方法：backend='simple' | 'physics' | 'simple_avg' | 'physics_avg'"""
        mapping = {
            "simple": SimpleReader,
            "physics": PhysicsReader,
            "simple_avg": SimpleAvgReader,
            "physics_avg": PhysicsAvgReader,
        }
        backend = backend.lower().strip()
        if backend not in mapping:
            raise ValueError(f"未知后端 '{backend}'，支持: {', '.join(mapping.keys())}")
        return mapping[backend]()


# ===========================================================================
# 3x3 采样 Mixin
# ===========================================================================

class MultiPointAvgMixin:
    """
    提供 3x3 子采样位置计算。
    
    关键假设：generate_surface_mesh_points_from_stl 按 (u,v) 参数空间的
    行优先规则网格输出点，其中 u,v ∈ [0,1] 是参数坐标。
    """
    
    SUBGRID: int = 3

    def _get_expanded_points(self, stl_path: Path, rows: int, cols: int) -> np.ndarray:
        """
        生成 3x3 子采样点阵。
        
        Returns:
            pts: shape (rows, cols, 3, 3, 3)，每个 taxel 对应一个 (3,3,3) 局部点阵
        """
        from src.sensors.stl_mesh_sampler import generate_surface_mesh_points_from_stl
        
        fine_rows, fine_cols = rows * self.SUBGRID, cols * self.SUBGRID
        pts_fine = generate_surface_mesh_points_from_stl(stl_path, fine_rows, fine_cols)
        
        expected = fine_rows * fine_cols
        if len(pts_fine) != expected:
            raise ValueError(
                f"采样器输出点数 {len(pts_fine)} 与期望 {expected} 不符，"
                f"可能使用了非规则网格采样策略"
            )
        
        pts = pts_fine.reshape(rows, self.SUBGRID, cols, self.SUBGRID, 3)
        pts = pts.transpose(0, 2, 1, 3, 4)
        return pts


# ===========================================================================
# Simple 后端 — 单点采样
# ===========================================================================

class SimpleReader(TactileReader):
    """轻量 site 方案，直接挂在 body 上，无弹性，计算开销小。"""

    def build(self, spec, hand_path, prefix="inspirehand_", site_group=4,
              site_rgba=(1.0, 0.2, 0.2, 0.6), **kwargs):
        from src.sensors.stl_mesh_sampler import generate_surface_mesh_points_from_stl

        configs = [
            ("skin_0_0_p", "finger_first_0_p",  "skin_0_0_p.STL", 10, 7),
            ("skin_0_1_p", "finger_second_0_p", "skin_0_1_p.STL",  8, 5),
            ("skin_0_2_p", "finger_second_0_p", "skin_0_2_p.STL",  6, 5),
            ("skin_1_0_p", "finger_first_1_p",  "skin_1_0_p.STL", 10, 7),
            ("skin_1_1_p", "finger_second_1_p", "skin_1_1_p.STL",  8, 5),
            ("skin_1_2_p", "finger_second_1_p", "skin_1_2_p.STL",  6, 5),
            ("skin_2_0_p", "finger_first_2_p",  "skin_2_0_p.STL", 10, 7),
            ("skin_2_1_p", "finger_second_2_p", "skin_2_1_p.STL",  8, 5),
            ("skin_2_2_p", "finger_second_2_p", "skin_2_2_p.STL",  6, 5),
            ("skin_3_0_p", "finger_first_3_p",  "skin_3_0_p.STL", 10, 7),
            ("skin_3_1_p", "finger_second_3_p", "skin_3_1_p.STL",  8, 5),
            ("skin_3_2_p", "finger_second_3_p", "skin_3_2_p.STL",  6, 5),
            ("skin_4_0_p", "thumb_first_p",     "skin_4_0_p.STL", 10, 7),
            ("skin_4_1_p", "thumb_second_p",    "skin_4_1_p.STL",  8, 5),
            ("skin_4_2_p", "thumb_third_p",     "skin_4_2_p.STL",  6, 5),
        ]

        meshes_dir = Path(hand_path).parent / "meshes"
        self._sensor_names: Dict[str, List[str]] = {}
        sensor_radius = 0.003

        for skin_name, body_name, stl_file, rows, cols in configs:
            stl_path = meshes_dir / stl_file
            pts_mesh = generate_surface_mesh_points_from_stl(stl_path, rows, cols)
            
            geom_name = skin_name
            try:
                geom = spec.geom(prefix + geom_name)
            except KeyError:
                geom = spec.geom(geom_name)
            
            pos, quat = np.array(geom.pos), np.array(geom.quat)
            R = _quat_to_rot(quat)
            pts_body = pts_mesh @ R.T + pos

            target_body = spec.body(prefix + body_name)
            phalanx_name = _SKIN_TO_PHALANX[skin_name]
            names = []

            for idx, pt in enumerate(pts_body):
                site_name = f"touch_site_{skin_name}_{idx}"
                sensor_name = f"touch_{skin_name}_{idx}"

                site = target_body.add_site()
                site.name, site.type = site_name, mujoco.mjtGeom.mjGEOM_SPHERE
                site.size, site.pos, site.group, site.rgba = [sensor_radius, 0, 0], pt.tolist(), site_group, list(site_rgba)

                sensor = spec.add_sensor()
                sensor.name, sensor.type, sensor.objtype, sensor.objname = sensor_name, mujoco.mjtSensor.mjSENS_TOUCH, mujoco.mjtObj.mjOBJ_SITE, site_name
                sensor.cutoff, sensor.noise = 0.0, 0.001
                names.append(sensor_name)

            self._sensor_names[phalanx_name] = names

    def bind(self, model):
        self._sensor_ids = {}
        for phalanx_name, names in self._sensor_names.items():
            meta = self._meta[phalanx_name]
            ids = np.zeros((meta.rows, meta.cols), dtype=np.int32)
            for i, name in enumerate(names):
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
                if sid < 0: raise ValueError(f"Sensor '{name}' not found")
                ids[i // meta.cols, i % meta.cols] = sid
            self._sensor_ids[phalanx_name] = ids
        self._bound = True

    def _read_raw_impl(self, data):
        return {name: data.sensordata[ids.ravel()].reshape(ids.shape).astype(np.float32)
                for name, ids in self._sensor_ids.items()}


# ===========================================================================
# Physics 后端 — 单点采样
# ===========================================================================

class PhysicsReader(TactileReader):
    """
    弹性 taxel 方案，每个点有独立滑动关节，力值更平滑。
    
    物理参数可在创建时配置：
        reader = PhysicsReader(stiffness=300, damping=5, taxel_radius=0.002)
    """
    
    def __init__(self, stiffness=200.0, damping=2.0, elastic_range=0.002,
                 taxel_radius=0.001, site_group=4, site_rgba=(0.95, 0.45, 0.05, 0.6)):
        super().__init__()
        self.stiffness = stiffness
        self.damping = damping
        self.elastic_range = elastic_range
        self.taxel_radius = taxel_radius
        self.site_group = site_group
        self.site_rgba = site_rgba

    def build(self, spec, hand_path, prefix="inspirehand_", **kwargs):
        from src.sensors.stl_mesh_sampler import generate_surface_mesh_points_from_stl

        configs = [
            ("finger_0_bottom", "finger_first_0_p",  "skin_0_0_p.STL", 10, 7),
            ("finger_0_middle", "finger_second_0_p", "skin_0_1_p.STL",  8, 5),
            ("finger_0_top",    "finger_second_0_p", "skin_0_2_p.STL",  6, 5),
            ("finger_1_bottom", "finger_first_1_p",  "skin_1_0_p.STL", 10, 7),
            ("finger_1_middle", "finger_second_1_p", "skin_1_1_p.STL",  8, 5),
            ("finger_1_top",    "finger_second_1_p", "skin_1_2_p.STL",  6, 5),
            ("finger_2_bottom", "finger_first_2_p",  "skin_2_0_p.STL", 10, 7),
            ("finger_2_middle", "finger_second_2_p", "skin_2_1_p.STL",  8, 5),
            ("finger_2_top",    "finger_second_2_p", "skin_2_2_p.STL",  6, 5),
            ("finger_3_bottom", "finger_first_3_p",  "skin_3_0_p.STL", 10, 7),
            ("finger_3_middle", "finger_second_3_p", "skin_3_1_p.STL",  8, 5),
            ("finger_3_top",    "finger_second_3_p", "skin_3_2_p.STL",  6, 5),
            ("thumb_bottom",    "thumb_first_p",     "skin_4_0_p.STL", 10, 7),
            ("thumb_middle",    "thumb_second_p",    "skin_4_1_p.STL",  8, 5),
            ("thumb_top",       "thumb_third_p",     "skin_4_2_p.STL",  6, 5),
        ]

        meshes_dir = Path(hand_path).parent / "meshes"
        self._arrays: Dict[str, _PhalanxArray] = {}

        for phalanx_name, body_name, stl_file, rows, cols in configs:
            stl_path = meshes_dir / stl_file
            pts_mesh = generate_surface_mesh_points_from_stl(stl_path, rows, cols)

            mesh_name = Path(stl_file).stem
            try:
                geom = spec.geom(prefix + mesh_name)
            except KeyError:
                geom = spec.geom(mesh_name)
            pos, quat = np.array(geom.pos), np.array(geom.quat)
            R = _quat_to_rot(quat)
            pts_local = pts_mesh @ R.T + pos

            interior_ref = np.zeros(3)
            normals = _outward_normals(pts_local, interior_ref)

            target_body = spec.body(prefix + body_name)
            sensor_names = []

            for r in range(rows):
                row_names = []
                for c in range(cols):
                    idx = r * cols + c
                    pt, nvec = pts_local[idx], normals[idx]
                    tag = f"{phalanx_name}_r{r:02d}_c{c:02d}"

                    tb = target_body.add_body()
                    tb.name = f"{prefix}taxel_body_{tag}"
                    tb.pos = pt.tolist()

                    jt = tb.add_joint()
                    jt.name = f"{prefix}taxel_j_{tag}"
                    jt.type = mujoco.mjtJoint.mjJNT_SLIDE
                    jt.axis = nvec.tolist()
                    jt.stiffness, jt.damping = self.stiffness, self.damping
                    jt.range, jt.limited = [-self.elastic_range, 0.0], True

                    gm = tb.add_geom()
                    gm.name = f"{prefix}taxel_geom_{tag}"
                    gm.type, gm.size = mujoco.mjtGeom.mjGEOM_SPHERE, [self.taxel_radius, 0, 0]
                    gm.condim, gm.group, gm.rgba = 1, self.site_group, list(self.site_rgba)

                    st = tb.add_site()
                    st.name = f"{prefix}site_{tag}"
                    st.type, st.size = mujoco.mjtGeom.mjGEOM_SPHERE, [self.taxel_radius*1.5]*3
                    st.group, st.rgba = self.site_group, list(self.site_rgba)

                    sn = spec.add_sensor()
                    sn.name = f"{prefix}taxel_sens_{tag}"
                    sn.type, sn.objtype, sn.objname = mujoco.mjtSensor.mjSENS_TOUCH, mujoco.mjtObj.mjOBJ_SITE, st.name
                    sn.cutoff, sn.noise = 0.0, 0.0
                    row_names.append(sn.name)
                sensor_names.append(row_names)

            self._arrays[phalanx_name] = _PhalanxArray(phalanx_name, rows, cols, sensor_names)

    def bind(self, model):
        for arr in self._arrays.values():
            arr.bind(model)
        self._bound = True

    def _read_raw_impl(self, data):
        return {name: arr.read(data) for name, arr in self._arrays.items()}


# ===========================================================================
# SimpleAvg 后端 — Simple + 3x3 子采样平均
# ===========================================================================

class SimpleAvgReader(TactileReader, MultiPointAvgMixin):
    """
    在 Body 上直接生成 rows*cols*9 个 site。
    读取时将每 9 个传感器的值取平均值，作为一个 Taxel 的输出。
    
    相比 SimpleReader，信号更平滑，适合强化学习训练。
    """
    
    SUBGRID: int = 3

    def __init__(self):
        super().__init__()
        self._sensor_names: Dict[str, List[List[List[str]]]] = {}

    def build(self, spec, hand_path, prefix="inspirehand_", site_group=4,
              site_rgba=(1.0, 0.2, 0.2, 0.6), **kwargs):
        from src.sensors.stl_mesh_sampler import generate_surface_mesh_points_from_stl

        configs = [
            ("skin_0_0_p", "finger_first_0_p",  "skin_0_0_p.STL", 10, 7),
            ("skin_0_1_p", "finger_second_0_p", "skin_0_1_p.STL",  8, 5),
            ("skin_0_2_p", "finger_second_0_p", "skin_0_2_p.STL",  6, 5),
            ("skin_1_0_p", "finger_first_1_p",  "skin_1_0_p.STL", 10, 7),
            ("skin_1_1_p", "finger_second_1_p", "skin_1_1_p.STL",  8, 5),
            ("skin_1_2_p", "finger_second_1_p", "skin_1_2_p.STL",  6, 5),
            ("skin_2_0_p", "finger_first_2_p",  "skin_2_0_p.STL", 10, 7),
            ("skin_2_1_p", "finger_second_2_p", "skin_2_1_p.STL",  8, 5),
            ("skin_2_2_p", "finger_second_2_p", "skin_2_2_p.STL",  6, 5),
            ("skin_3_0_p", "finger_first_3_p",  "skin_3_0_p.STL", 10, 7),
            ("skin_3_1_p", "finger_second_3_p", "skin_3_1_p.STL",  8, 5),
            ("skin_3_2_p", "finger_second_3_p", "skin_3_2_p.STL",  6, 5),
            ("skin_4_0_p", "thumb_first_p",     "skin_4_0_p.STL", 10, 7),
            ("skin_4_1_p", "thumb_second_p",    "skin_4_1_p.STL",  8, 5),
            ("skin_4_2_p", "thumb_third_p",     "skin_4_2_p.STL",  6, 5),
        ]

        meshes_dir = Path(hand_path).parent / "meshes"
        self._sensor_names = {}
        sensor_radius = 0.0015

        for skin_name, body_name, stl_file, rows, cols in configs:
            stl_path = meshes_dir / stl_file
            
            pts_grid = self._get_expanded_points(stl_path, rows, cols)
            
            geom_name = skin_name
            try:
                geom = spec.geom(prefix + geom_name)
            except KeyError:
                geom = spec.geom(geom_name)
            pos, quat = np.array(geom.pos), np.array(geom.quat)
            R = _quat_to_rot(quat)
            
            pts_flat = pts_grid.reshape(-1, 3)
            pts_body_flat = pts_flat @ R.T + pos
            pts_body = pts_body_flat.reshape(rows, cols, 3, 3, 3)
            
            target_body = spec.body(prefix + body_name)
            phalanx_name = _SKIN_TO_PHALANX[skin_name]
            taxel_sensors: List[List[List[str]]] = []

            for r in range(rows):
                row_sensors = []
                for c in range(cols):
                    sub_sensors = []
                    for sr in range(3):
                        for sc in range(3):
                            idx = sr * 3 + sc
                            pt = pts_body[r, c, sr, sc]
                            
                            site_name = f"touch_site_{skin_name}_r{r}_c{c}_s{idx}"
                            sensor_name = f"touch_{skin_name}_r{r}_c{c}_s{idx}"
                            
                            site = target_body.add_site()
                            site.name = site_name
                            site.type = mujoco.mjtGeom.mjGEOM_SPHERE
                            site.size = [sensor_radius, 0, 0]
                            site.pos = pt.tolist()
                            site.group = site_group
                            site.rgba = list(site_rgba)
                            
                            sensor = spec.add_sensor()
                            sensor.name = sensor_name
                            sensor.type = mujoco.mjtSensor.mjSENS_TOUCH
                            sensor.objtype = mujoco.mjtObj.mjOBJ_SITE
                            sensor.objname = site_name
                            sensor.cutoff = 0.0
                            sensor.noise = 0.001
                            
                            sub_sensors.append(sensor_name)
                    row_sensors.append(sub_sensors)
                taxel_sensors.append(row_sensors)
            
            self._sensor_names[phalanx_name] = taxel_sensors

    def bind(self, model):
        self._sensor_ids = {}
        for phalanx_name, taxel_grid in self._sensor_names.items():
            meta = self._meta[phalanx_name]
            ids = np.zeros((meta.rows, meta.cols, 9), dtype=np.int32)
            for r in range(meta.rows):
                for c in range(meta.cols):
                    for s, name in enumerate(taxel_grid[r][c]):
                        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
                        if sid < 0:
                            raise ValueError(f"Sensor '{name}' not found")
                        ids[r, c, s] = sid
            self._sensor_ids[phalanx_name] = ids
        self._bound = True

    def _read_raw_impl(self, data):
        return {
            name: data.sensordata[ids.reshape(-1)].reshape(
                ids.shape[0], ids.shape[1], 9
            ).mean(axis=-1).astype(np.float32)
            for name, ids in self._sensor_ids.items()
        }


# ===========================================================================
# PhysicsAvg 后端 — Physics + 3x3 子采样平均
# ===========================================================================

class PhysicsAvgReader(PhysicsReader, MultiPointAvgMixin):
    """
    每个 Taxel 拥有 1 个物理滑块 Body（共享关节），
    该 Body 上附着 9 个采样 site。
    
    这模拟了"平头"触觉单元：整个单元作为一个刚体沿法向移动，
    但接触检测在 9 个空间分散的点上进行。
    """
    
    SUBGRID: int = 3

    def __init__(self, stiffness=200.0, damping=2.0, elastic_range=0.002,
                 taxel_radius=0.001, site_group=4, site_rgba=(0.95, 0.45, 0.05, 0.6)):
        super().__init__(stiffness, damping, elastic_range, taxel_radius, site_group, site_rgba)
        self._arrays: Dict[str, _MultiPhalanxArray] = {}

    def build(self, spec, hand_path, prefix="inspirehand_", **kwargs):
        from src.sensors.stl_mesh_sampler import generate_surface_mesh_points_from_stl

        configs = [
            ("finger_0_bottom", "finger_first_0_p",  "skin_0_0_p.STL", 10, 7),
            ("finger_0_middle", "finger_second_0_p", "skin_0_1_p.STL",  8, 5),
            ("finger_0_top",    "finger_second_0_p", "skin_0_2_p.STL",  6, 5),
            ("finger_1_bottom", "finger_first_1_p",  "skin_1_0_p.STL", 10, 7),
            ("finger_1_middle", "finger_second_1_p", "skin_1_1_p.STL",  8, 5),
            ("finger_1_top",    "finger_second_1_p", "skin_1_2_p.STL",  6, 5),
            ("finger_2_bottom", "finger_first_2_p",  "skin_2_0_p.STL", 10, 7),
            ("finger_2_middle", "finger_second_2_p", "skin_2_1_p.STL",  8, 5),
            ("finger_2_top",    "finger_second_2_p", "skin_2_2_p.STL",  6, 5),
            ("finger_3_bottom", "finger_first_3_p",  "skin_3_0_p.STL", 10, 7),
            ("finger_3_middle", "finger_second_3_p", "skin_3_1_p.STL",  8, 5),
            ("finger_3_top",    "finger_second_3_p", "skin_3_2_p.STL",  6, 5),
            ("thumb_bottom",    "thumb_first_p",     "skin_4_0_p.STL", 10, 7),
            ("thumb_middle",    "thumb_second_p",    "skin_4_1_p.STL",  8, 5),
            ("thumb_top",       "thumb_third_p",     "skin_4_2_p.STL",  6, 5),
        ]

        meshes_dir = Path(hand_path).parent / "meshes"
        self._arrays = {}

        for phalanx_name, body_name, stl_file, rows, cols in configs:
            stl_path = meshes_dir / stl_file
            pts_grid = self._get_expanded_points(stl_path, rows, cols)

            mesh_name = Path(stl_file).stem
            try:
                geom = spec.geom(prefix + mesh_name)
            except KeyError:
                geom = spec.geom(mesh_name)
            pos, quat = np.array(geom.pos), np.array(geom.quat)
            R = _quat_to_rot(quat)
            
            pts_flat = pts_grid.reshape(-1, 3)
            pts_local = pts_flat @ R.T + pos
            pts_local = pts_local.reshape(rows, cols, 3, 3, 3)

            centers = pts_local.mean(axis=(2, 3))
            interior_ref = np.zeros(3)
            normals = _outward_normals(
                centers.reshape(-1, 3), interior_ref
            ).reshape(rows, cols, 3)

            target_body = spec.body(prefix + body_name)
            sensor_names: List[List[List[str]]] = []

            for r in range(rows):
                row_names = []
                for c in range(cols):
                    pt_center = centers[r, c]
                    nvec = normals[r, c]
                    tag = f"{phalanx_name}_r{r:02d}_c{c:02d}"

                    tb = target_body.add_body()
                    tb.name = f"{prefix}taxel_body_{tag}"
                    tb.pos = pt_center.tolist()

                    jt = tb.add_joint()
                    jt.name = f"{prefix}taxel_j_{tag}"
                    jt.type = mujoco.mjtJoint.mjJNT_SLIDE
                    jt.axis = nvec.tolist()
                    jt.stiffness = self.stiffness
                    jt.damping = self.damping
                    jt.range = [-self.elastic_range, 0.0]
                    jt.limited = True

                    gm = tb.add_geom()
                    gm.name = f"{prefix}taxel_geom_{tag}"
                    gm.type = mujoco.mjtGeom.mjGEOM_SPHERE
                    gm.size = [self.taxel_radius, 0, 0]
                    gm.condim = 1
                    gm.group = self.site_group
                    gm.rgba = list(self.site_rgba)

                    sub_sensors = []
                    for sr in range(3):
                        for sc in range(3):
                            idx = sr * 3 + sc
                            pt_local = pts_local[r, c, sr, sc]
                            offset = pt_local - pt_center
                            
                            st = tb.add_site()
                            st.name = f"{prefix}site_{tag}_s{idx}"
                            st.type = mujoco.mjtGeom.mjGEOM_SPHERE
                            st.size = [self.taxel_radius * 0.8] * 3
                            st.pos = offset.tolist()
                            st.group = self.site_group
                            st.rgba = list(self.site_rgba)

                            sn = spec.add_sensor()
                            sn.name = f"{prefix}taxel_sens_{tag}_s{idx}"
                            sn.type = mujoco.mjtSensor.mjSENS_TOUCH
                            sn.objtype = mujoco.mjtObj.mjOBJ_SITE
                            sn.objname = st.name
                            sn.cutoff = 0.0
                            sn.noise = 0.0
                            sub_sensors.append(sn.name)
                    row_names.append(sub_sensors)
                sensor_names.append(row_names)

            self._arrays[phalanx_name] = _MultiPhalanxArray(
                phalanx_name, rows, cols, sensor_names
            )

    def bind(self, model):
        for arr in self._arrays.values():
            arr.bind(model)
        self._bound = True

    def _read_raw_impl(self, data):
        return {
            name: arr.read(data) 
            for name, arr in self._arrays.items()
        }


# ===========================================================================
# 内部辅助类与函数
# ===========================================================================

class _PhalanxArray:
    """Physics 后端内部使用的传感器阵列描述符。"""

    def __init__(self, name: str, rows: int, cols: int, names: List[List[str]]):
        self.name, self.rows, self.cols, self.names = name, rows, cols, names
        self.ids: Optional[np.ndarray] = None

    def bind(self, model: mujoco.MjModel):
        self.ids = np.zeros((self.rows, self.cols), dtype=np.int32)
        for r in range(self.rows):
            for c in range(self.cols):
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, self.names[r][c])
                if sid < 0: raise ValueError(f"Sensor '{self.names[r][c]}' not found")
                self.ids[r, c] = sid

    def read(self, data: mujoco.MjData) -> np.ndarray:
        if self.ids is None: raise RuntimeError("Not bound")
        return data.sensordata[self.ids.ravel()].reshape(self.rows, self.cols).astype(np.float32)


class _MultiPhalanxArray:
    """支持 3x3 子采样的传感器阵列描述符"""

    def __init__(self, name: str, rows: int, cols: int, 
                 names: List[List[List[str]]]):
        self.name, self.rows, self.cols = name, rows, cols
        self.names = names
        self.ids: Optional[np.ndarray] = None

    def bind(self, model: mujoco.MjModel):
        self.ids = np.zeros((self.rows, self.cols, 9), dtype=np.int32)
        for r in range(self.rows):
            for c in range(self.cols):
                for s, name in enumerate(self.names[r][c]):
                    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
                    if sid < 0:
                        raise ValueError(f"Sensor '{name}' not found")
                    self.ids[r, c, s] = sid

    def read(self, data: mujoco.MjData) -> np.ndarray:
        if self.ids is None:
            raise RuntimeError("Not bound")
        raw = data.sensordata[self.ids.reshape(-1)].reshape(
            self.rows, self.cols, 9
        )
        return raw.mean(axis=-1).astype(np.float32)


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """四元数 [w,x,y,z] 转旋转矩阵。"""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)  ],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)  ],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])


def _outward_normals(pts_local: np.ndarray, interior_ref: np.ndarray) -> np.ndarray:
    """
    计算点云朝外法向量。
    
    Args:
        pts_local: 变换到 body 坐标系后的点云，shape (N, 3)。
        interior_ref: body 坐标系下的内部参考点（如 body 原点 [0,0,0]），
                      法向量方向将与 (pt → interior_ref) 相反，即真正朝外。
    """
    centered = pts_local - pts_local.mean(0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    thin_axis = vh[-1]
    radial = centered - (centered @ thin_axis)[:, None] * thin_axis
    norms = np.linalg.norm(radial, axis=1, keepdims=True) + 1e-12
    normals = radial / norms

    inward_vecs = interior_ref[None, :] - pts_local
    dots = np.einsum('ij,ij->i', normals, inward_vecs)
    flip_mask = dots > 0
    normals[flip_mask] *= -1

    return normals


# ===========================================================================
# 便捷函数
# ===========================================================================

def build_tactile_reader(backend: str, spec: mujoco.MjSpec, hand_path: Path,
                         prefix: str = "inspirehand_", **kwargs) -> TactileReader:
    """
    创建并 build 的便捷函数（注意：bind 需在 compile 后手动调用）。
    
    用法：
        reader = build_tactile_reader("physics_avg", spec, hand_path)
        model = spec.compile()
        reader.bind(model)
    """
    reader = TactileReader.create(backend)
    reader.build(spec, hand_path, prefix, **kwargs)
    return reader


__all__ = [
    "TactileReader", 
    "PhalanxMeta", 
    "DISPLAY_ORDER", 
    "FINGER_PHALANX_ORDER",  
    "build_tactile_reader",
    "SimpleReader",
    "PhysicsReader",
    "SimpleAvgReader",
    "PhysicsAvgReader",
    "MultiPointAvgMixin",
]