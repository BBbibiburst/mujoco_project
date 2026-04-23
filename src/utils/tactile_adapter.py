"""
触觉传感器统一适配层 (Tactile Adapter) — 精简修正版

对外只暴露 TactileReader 接口，内部封装 Simple / Physics 两种后端。
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

# 15 块皮肤的标准映射
_SKIN_TO_PHALANX = {
    "skin_0_0_p": "finger_0_bottom", "skin_0_1_p": "finger_0_middle", "skin_0_2_p": "finger_0_top",
    "skin_1_0_p": "finger_1_bottom", "skin_1_1_p": "finger_1_middle", "skin_1_2_p": "finger_1_top",
    "skin_2_0_p": "finger_2_bottom", "skin_2_1_p": "finger_2_middle", "skin_2_2_p": "finger_2_top",
    "skin_3_0_p": "finger_3_bottom", "skin_3_1_p": "finger_3_middle", "skin_3_2_p": "finger_3_top",
    "skin_4_0_p": "thumb_bottom",    "skin_4_1_p": "thumb_middle",    "skin_4_2_p": "thumb_top",
}

# 展示顺序（固定，神经网络输入用）
DISPLAY_ORDER: List[str] = [
    "finger_0_bottom", "finger_0_middle", "finger_0_top",
    "finger_1_bottom", "finger_1_middle", "finger_1_top",
    "finger_2_bottom", "finger_2_middle", "finger_2_top",
    "finger_3_bottom", "finger_3_middle", "finger_3_top",
    "thumb_bottom",    "thumb_middle",    "thumb_top",
]

# 各指节形状
_PHALANX_SHAPE: Dict[str, Tuple[int, int]] = {
    "finger_0_bottom": (10, 7), "finger_0_middle": (8, 5), "finger_0_top": (6, 5),
    "finger_1_bottom": (10, 7), "finger_1_middle": (8, 5), "finger_1_top": (6, 5),
    "finger_2_bottom": (10, 7), "finger_2_middle": (8, 5), "finger_2_top": (6, 5),
    "finger_3_bottom": (10, 7), "finger_3_middle": (8, 5), "finger_3_top": (6, 5),
    "thumb_bottom":    (10, 7), "thumb_middle":    (8, 5), "thumb_top":    (6, 5),
}

# 各手指的指节顺序（index 0=bottom, 1=middle, 2=top）
FINGER_PHALANX_ORDER: Dict[str, List[str]] = {
    "finger_0": ["finger_0_bottom", "finger_0_middle", "finger_0_top"],
    "finger_1": ["finger_1_bottom", "finger_1_middle", "finger_1_top"],
    "finger_2": ["finger_2_bottom", "finger_2_middle", "finger_2_top"],
    "finger_3": ["finger_3_bottom", "finger_3_middle", "finger_3_top"],
    "thumb":    ["thumb_bottom",    "thumb_middle",    "thumb_top"],
}

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
        reader = TactileReader.create("physics")  # 或 "simple"
        reader.build(spec, hand_path, prefix)
        model = spec.compile()
        reader.bind(model)
        
        # 仿真循环
        raw = reader.read_raw(data)      # Dict[str, ndarray(rows,cols)]
        img = reader.read_image(data)    # Dict[str, ndarray(rows,cols, uint8)]
        vec = reader.read_concat(data)   # ndarray(total_taxels,) 直接送网络
    """

    FORCE_MAX_NEWTON: float = 5.0

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
        """工厂方法：backend='simple' | 'physics'"""
        backend = backend.lower().strip()
        if backend == "simple": return SimpleReader()
        if backend == "physics": return PhysicsReader()
        raise ValueError(f"未知后端 '{backend}'，支持: simple, physics")


# ===========================================================================
# Simple 后端
# ===========================================================================

class SimpleReader(TactileReader):
    """轻量 site 方案，直接挂在 body 上，无弹性，计算开销小。"""

    def build(self, spec, hand_path, prefix="inspirehand_", site_group=4,
              site_rgba=(1.0, 0.2, 0.2, 0.6), **kwargs):
        from src.utils.stl_mesh_sampler import generate_surface_mesh_points_from_stl

        configs = [
            # (skin_name, body_name, stl_file, rows, cols)
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
            
            # 获取 geom 在 body 中的位姿
            geom_name = skin_name
            try:
                geom = spec.geom(prefix + geom_name)
            except KeyError:
                geom = spec.geom(geom_name)
            
            # 变换到 body 坐标系
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
# Physics 后端
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
        from src.utils.stl_mesh_sampler import generate_surface_mesh_points_from_stl

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

            # 获取 geom 位姿
            mesh_name = Path(stl_file).stem
            try:
                geom = spec.geom(prefix + mesh_name)
            except KeyError:
                geom = spec.geom(mesh_name)
            pos, quat = np.array(geom.pos), np.array(geom.quat)
            R = _quat_to_rot(quat)
            pts_local = pts_mesh @ R.T + pos

            # 【关键修正】使用 body 原点作为 interior_ref 校正法向量
            # body 原点在手指内部，确保法向量真正朝外
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

                    # 子 body + 弹性关节
                    tb = target_body.add_body()
                    tb.name = f"{prefix}taxel_body_{tag}"
                    tb.pos = pt.tolist()

                    jt = tb.add_joint()
                    jt.name = f"{prefix}taxel_j_{tag}"
                    jt.type = mujoco.mjtJoint.mjJNT_SLIDE
                    jt.axis = nvec.tolist()  # 朝外为正
                    jt.stiffness, jt.damping = self.stiffness, self.damping
                    jt.range, jt.limited = [-self.elastic_range, 0.0], True

                    # 接触球
                    gm = tb.add_geom()
                    gm.type, gm.size = mujoco.mjtGeom.mjGEOM_SPHERE, [self.taxel_radius, 0, 0]
                    gm.condim, gm.group, gm.rgba = 1, self.site_group, list(self.site_rgba)

                    # Site + Sensor
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
    计算点云朝外法向量（关键修正：使用 interior_ref 确保方向正确）。
    
    Args:
        pts_local: 变换到 body 坐标系后的点云，shape (N, 3)。
        interior_ref: body 坐标系下的内部参考点（如 body 原点 [0,0,0]），
                      法向量方向将与 (pt → interior_ref) 相反，即真正朝外。
    """
    centered = pts_local - pts_local.mean(0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    # 最小奇异值对应的向量 ≈ 点云薄壁方向
    thin_axis = vh[-1]
    # 去除薄壁分量，得到径向方向
    radial = centered - (centered @ thin_axis)[:, None] * thin_axis
    norms = np.linalg.norm(radial, axis=1, keepdims=True) + 1e-12
    normals = radial / norms

    # 【关键修正】用 interior_ref 校正方向
    # interior_ref 在手指内部，pt → interior_ref 是朝内的
    # 法向量应与其方向相反（dot < 0），若同向则翻转
    inward_vecs = interior_ref[None, :] - pts_local  # (N,3)，朝内
    dots = np.einsum('ij,ij->i', normals, inward_vecs)
    flip_mask = dots > 0
    normals[flip_mask] *= -1

    return normals


# ===========================================================================
# 便捷函数（已修正逻辑）
# ===========================================================================

def build_tactile_reader(backend: str, spec: mujoco.MjSpec, hand_path: Path,
                         prefix: str = "inspirehand_", **kwargs) -> TactileReader:
    """
    创建并 build 的便捷函数（注意：bind 需在 compile 后手动调用）。
    
    正确用法：
        reader = build_tactile_reader("physics", spec, hand_path)
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
]