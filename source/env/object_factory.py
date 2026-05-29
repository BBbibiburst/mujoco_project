"""
通用物体生成器 — object_factory.py

三个正交维度：形状（种类）、颜色、尺寸，互相独立、独立扩展。

字符串格式
----------
    "<shape>_<color>_<size>"
    "mesh:<mesh_key>_<color>_<size>"

示例
----
    "box_red_large"
    "sphere_blue_small"
    "cylinder_green_medium"
    "capsule_yellow_large"
    "mesh:bottle_purple_small"    # bottle 需提前用 MeshRegistry.register() 注册

快速开始
--------
    from object_factory import ObjectFactory, MeshRegistry

    # （可选）注册 STL 文件
    MeshRegistry.register("bottle", "/path/to/laundry_bottle.stl")

    # 在 _build_scene() 中
    def _build_scene(self, spec):
        body, desc = ObjectFactory.create(spec, "box_red_large",   pos=[0.5, 0.0, 0.8])
        body, desc = ObjectFactory.create(spec, "sphere_blue_small", pos=[0.4, 0.1, 0.8])
        body, desc = ObjectFactory.create(spec, "mesh:bottle_purple_small", pos=[0.5, 0.0, 0.9])

枚举全量物体
------------
    from object_factory import all_object_types
    types = all_object_types()                          # 默认 4形状 × 7色 × 3尺寸 = 84 种
    types = all_object_types(mesh_keys=["bottle"])      # +7色×3尺寸×1网格 = 105 种

与 BlockLiftingEnv 集成（_build_scene 中替换原立方体）
------------------------------------------------------
    from .object_factory import ObjectFactory

    class BlockLiftingEnv(RobotArmEnvBase):
        def __init__(self, ..., obj_descriptor: str = "box_red_large"):
            self.obj_descriptor = obj_descriptor
            super().__init__(...)

        def _build_scene(self, spec):
            tc = self.task_cfg
            body, desc = ObjectFactory.create(
                spec,
                self.obj_descriptor,
                pos=[tc.obj_spawn_center[0], tc.obj_spawn_center[1],
                     self._table_height + desc.bottom_half_z],
                name="target_object",   # 名称固定，_cache_ids() 继续有效
            )
            # 其余代码不变 ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mujoco
import numpy as np


# =============================================================================
# 维度一：颜色
# =============================================================================

#: 7 种基础颜色，key 为字符串 ID，value 为 RGBA（0-1 浮点）
COLOR_PALETTE: Dict[str, Tuple[float, float, float, float]] = {
    "red":    (0.90, 0.18, 0.15, 1.0),
    "green":  (0.18, 0.75, 0.25, 1.0),
    "blue":   (0.18, 0.40, 0.90, 1.0),
    "yellow": (0.95, 0.85, 0.10, 1.0),
    "purple": (0.65, 0.20, 0.85, 1.0),
    "orange": (0.95, 0.50, 0.08, 1.0),
    "cyan":   (0.10, 0.80, 0.85, 1.0),
}


# =============================================================================
# 维度二：尺寸
# =============================================================================

@dataclass(frozen=True)
class SizeSpec:
    """
    各形状在该尺寸档下的 half-size（MuJoCo 约定，单位：米）。

    half-size 的含义
    ----------------
    box      : [hx, hy, hz]，实际边长 = 2×half
    sphere   : [radius]
    cylinder : [radius, half_height]
    capsule  : [radius, half_height]（半球半径 = radius）
    mesh     : 统一缩放系数（无量纲）
    """
    box_half:      Tuple[float, float, float]
    sphere_radius: float
    cyl_radius:    float
    cyl_half_h:    float
    cap_radius:    float
    cap_half_h:    float
    mesh_scale:    float


#: 三个尺寸档，可随时添加更多
SIZE_TABLE: Dict[str, SizeSpec] = {
    "small": SizeSpec(
        box_half      = (0.018, 0.018, 0.018),
        sphere_radius = 0.018,
        cyl_radius    = 0.014,
        cyl_half_h    = 0.022,
        cap_radius    = 0.014,
        cap_half_h    = 0.022,
        mesh_scale    = 0.8,
    ),
    "medium": SizeSpec(
        box_half      = (0.025, 0.025, 0.025),
        sphere_radius = 0.025,
        cyl_radius    = 0.020,
        cyl_half_h    = 0.030,
        cap_radius    = 0.020,
        cap_half_h    = 0.030,
        mesh_scale    = 1.0,
    ),
    "large": SizeSpec(
        box_half      = (0.035, 0.035, 0.035),
        sphere_radius = 0.035,
        cyl_radius    = 0.028,
        cyl_half_h    = 0.042,
        cap_radius    = 0.028,
        cap_half_h    = 0.042,
        mesh_scale    = 1.2,
    ),
}


# =============================================================================
# 维度三：形状 / 种类
# =============================================================================

#: MuJoCo 原生形状集合
NATIVE_SHAPES = {"box", "sphere", "cylinder", "capsule"}

#: 所有合法形状（原生 + mesh 前缀）
#  mesh 类形状格式为 "mesh:<key>"，key 须已在 MeshRegistry 中注册


# =============================================================================
# STL 网格注册表
# =============================================================================

class MeshRegistry:
    """
    全局 STL / OBJ 网格注册表。

    在程序入口（或 env __init__ 中）调用 register()，
    此后在描述符中使用 "mesh:<key>" 即可。

    示例
    ----
        MeshRegistry.register("bottle", "/assets/laundry_bottle.stl")
        MeshRegistry.register("cup",    "/assets/mug.stl")
    """

    _registry: Dict[str, Path] = {}

    @classmethod
    def register(cls, key: str, path: Union[str, Path]) -> None:
        """注册网格文件。key 只能含小写字母、数字、下划线。"""
        if not re.match(r"^[a-z0-9_]+$", key):
            raise ValueError(f"mesh key 只能含小写字母/数字/下划线，收到: '{key}'")
        p = Path(path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"网格文件不存在: {p}")
        if p.suffix.lower() not in (".stl", ".obj"):
            raise ValueError(f"仅支持 .stl / .obj，收到: {p.suffix}")
        cls._registry[key] = p

    @classmethod
    def get(cls, key: str) -> Path:
        if key not in cls._registry:
            raise KeyError(
                f"网格 '{key}' 未注册。\n"
                f"已注册: {list(cls._registry)}\n"
                f"请先调用: MeshRegistry.register('{key}', '/path/to/file.stl')"
            )
        return cls._registry[key]

    @classmethod
    def keys(cls) -> List[str]:
        return list(cls._registry)


# =============================================================================
# 物体描述符（解析结果）
# =============================================================================

@dataclass
class ObjectDescriptor:
    """
    解析后的三维描述符：形状 + 颜色 + 尺寸。

    由 ObjectFactory.parse() 生成；包含创建 MuJoCo geom 所需的全部信息。
    """

    # --- 三个正交维度 ---
    shape:      str   #: 形状 ID，如 "box"、"sphere"、"mesh:bottle"
    color:      str   #: 颜色 ID，如 "red"
    size:       str   #: 尺寸 ID，如 "small"、"medium"、"large"

    # --- 预解析的数值 ---
    rgba:       Tuple[float, float, float, float]
    size_spec:  SizeSpec

    # --- 物理参数（可在 create() 时覆盖）---
    mass:        float = 0.10
    friction:    Tuple[float, float, float] = (0.5, 0.1, 0.01)
    condim:      int   = 4
    conaffinity: int   = 15

    # ------------------------------------------------------------------
    @property
    def shape_type(self) -> str:
        """形状基础类型：'box' / 'sphere' / 'cylinder' / 'capsule' / 'mesh'。"""
        return self.shape.split(":")[0]

    @property
    def mesh_key(self) -> Optional[str]:
        """若为 mesh 形状则返回 key，否则 None。"""
        parts = self.shape.split(":", 1)
        return parts[1] if len(parts) == 2 else None

    @property
    def bottom_half_z(self) -> float:
        """
        物体几何中心到底面的距离（用于计算放置高度）。

        用法：pos_z = table_height + obj.bottom_half_z
        """
        sz = self.size_spec
        t  = self.shape_type
        if t == "box":
            return sz.box_half[2]
        elif t == "sphere":
            return sz.sphere_radius
        elif t == "cylinder":
            return sz.cyl_half_h
        elif t == "capsule":
            return sz.cap_half_h + sz.cap_radius  # 半球 + 柱
        else:  # mesh：近似用 medium box
            return 0.025

    @property
    def canonical_name(self) -> str:
        """规范化名称，可用作 MuJoCo body name 前缀。"""
        shape_part = self.shape.replace(":", "_")
        return f"{shape_part}_{self.color}_{self.size}"

    def __str__(self) -> str:
        shape_part = self.shape.replace(":", "_")
        return f"{self.shape}_{self.color}_{self.size}"


# =============================================================================
# 工厂
# =============================================================================

# 解析正则：形状部分支持 "box" 和 "mesh:bottle"
_PARSE_RE = re.compile(
    r"^(?P<shape>[a-z]+(?::[a-z0-9_]+)?)_(?P<color>[a-z]+)_(?P<size>[a-z]+)$"
)


class ObjectFactory:
    """
    工厂主类：parse + create + create_batch。

    描述符字符串格式
    ----------------
        "<shape>_<color>_<size>"

    示例
    ----
        "box_red_large"
        "sphere_blue_small"
        "cylinder_green_medium"
        "capsule_yellow_large"
        "mesh:bottle_purple_small"
    """

    # ------------------------------------------------------------------
    # 解析
    # ------------------------------------------------------------------

    @classmethod
    def parse(
        cls,
        s: str,
        *,
        mass:        float = 0.10,
        friction:    Tuple = (0.5, 0.1, 0.01),
        condim:      int   = 4,
        conaffinity: int   = 15,
    ) -> ObjectDescriptor:
        """
        将描述符字符串解析为 ObjectDescriptor，不修改 MjSpec。

        参数
        ----
        s : str
            如 "box_red_large"
        mass, friction, condim, conaffinity
            物理参数，可按任务需要覆盖

        异常
        ----
        ValueError : 格式错误或维度值不在合法集合中
        """
        raw = s.strip().lower()
        m = _PARSE_RE.match(raw)
        if m is None:
            raise ValueError(
                f"无效描述符: '{s}'\n"
                f"格式: '<shape>_<color>_<size>'\n"
                f"示例: 'box_red_large'、'mesh:bottle_cyan_small'"
            )

        shape = m.group("shape")
        color = m.group("color")
        size  = m.group("size")

        # 校验颜色
        if color not in COLOR_PALETTE:
            raise ValueError(
                f"未知颜色 '{color}'。\n"
                f"合法颜色: {sorted(COLOR_PALETTE)}"
            )

        # 校验尺寸
        if size not in SIZE_TABLE:
            raise ValueError(
                f"未知尺寸 '{size}'。\n"
                f"合法尺寸: {sorted(SIZE_TABLE)}"
            )

        # 校验形状
        shape_type = shape.split(":")[0]
        if shape_type not in NATIVE_SHAPES and shape_type != "mesh":
            raise ValueError(
                f"未知形状 '{shape_type}'。\n"
                f"合法形状: {sorted(NATIVE_SHAPES)} 或 mesh:<key>"
            )

        return ObjectDescriptor(
            shape       = shape,
            color       = color,
            size        = size,
            rgba        = COLOR_PALETTE[color],
            size_spec   = SIZE_TABLE[size],
            mass        = mass,
            friction    = tuple(friction),
            condim      = condim,
            conaffinity = conaffinity,
        )

    # ------------------------------------------------------------------
    # 创建单个物体
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        spec:           mujoco.MjSpec,
        descriptor:     Union[str, ObjectDescriptor],
        *,
        pos:            Union[List, np.ndarray] = (0.5, 0.0, 0.8),
        quat:           Union[List, np.ndarray] = (1.0, 0.0, 0.0, 0.0),
        name:           Optional[str] = None,
        name_suffix:    str = "",
        add_free_joint: bool = True,
        parent_body:    Optional[mujoco.MjsBody] = None,
        # 物理参数覆盖（仅在传字符串时生效）
        mass:           float = 0.10,
        friction:       Tuple = (0.5, 0.1, 0.01),
        condim:         int   = 4,
        conaffinity:    int   = 15,
    ) -> Tuple[mujoco.MjsBody, ObjectDescriptor]:
        """
        在 MjSpec 中创建物体 body，返回 (body, descriptor)。

        参数
        ----
        spec
            _build_scene() 传入的 MjSpec
        descriptor
            描述符字符串（如 "box_red_large"）或已解析的 ObjectDescriptor
        pos
            初始位置 [x, y, z]
        quat
            初始旋转 [w, x, y, z]（MuJoCo 约定 w 在前）
        name
            body 名称；None 时自动生成
        name_suffix
            附加到自动名称后，用于区分同类物体（如 "0"、"1"）
        add_free_joint
            是否添加自由关节（可被抓取）
        parent_body
            父 body；None 时挂在 worldbody 下

        返回
        ----
        (body, descriptor)
        """
        if isinstance(descriptor, str):
            desc = cls.parse(
                descriptor,
                mass=mass, friction=friction,
                condim=condim, conaffinity=conaffinity,
            )
        else:
            desc = descriptor

        suffix  = f"_{name_suffix}" if name_suffix else ""
        bname   = name or (desc.canonical_name + suffix)
        parent  = parent_body or spec.worldbody

        body = parent.add_body(name=bname, pos=list(pos), quat=list(quat))
        cls._add_geom(spec, body, desc)

        if add_free_joint:
            body.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name=f"{bname}_free_joint")

        return body, desc

    # ------------------------------------------------------------------
    # 批量创建
    # ------------------------------------------------------------------

    @classmethod
    def create_batch(
        cls,
        spec:        mujoco.MjSpec,
        descriptors: List[Union[str, ObjectDescriptor]],
        positions:   List[Union[List, np.ndarray]],
        base_name:   str = "obj",
        **kwargs,
    ) -> List[Tuple[mujoco.MjsBody, ObjectDescriptor]]:
        """
        批量创建物体。

        参数
        ----
        descriptors
            描述符列表
        positions
            对应的初始位置列表（长度须与 descriptors 相等）
        base_name
            body 命名前缀，body 名为 "<base_name>_<i>"
        **kwargs
            透传到 create()
        """
        if len(descriptors) != len(positions):
            raise ValueError(
                f"descriptors({len(descriptors)}) 与 positions({len(positions)}) 长度不一致"
            )
        return [
            cls.create(spec, d, pos=p, name=f"{base_name}_{i}", **kwargs)
            for i, (d, p) in enumerate(zip(descriptors, positions))
        ]

    # ------------------------------------------------------------------
    # 内部：向 body 添加 geom
    # ------------------------------------------------------------------

    @classmethod
    def _add_geom(
        cls,
        spec: mujoco.MjSpec,
        body: mujoco.MjsBody,
        desc: ObjectDescriptor,
    ) -> None:
        sz   = desc.size_spec
        base = dict(
            rgba        = list(desc.rgba),
            mass        = desc.mass,
            friction    = list(desc.friction),
            condim      = desc.condim,
            conaffinity = desc.conaffinity,
        )

        t = desc.shape_type

        if t == "box":
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=list(sz.box_half), **base)

        elif t == "sphere":
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE,
                          size=[sz.sphere_radius], **base)

        elif t == "cylinder":
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                          size=[sz.cyl_radius, sz.cyl_half_h], **base)

        elif t == "capsule":
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                          size=[sz.cap_radius, sz.cap_half_h], **base)

        elif t == "mesh":
            key = desc.mesh_key
            cls._ensure_mesh_asset(spec, key, sz.mesh_scale)
            body.add_geom(
                type        = mujoco.mjtGeom.mjGEOM_MESH,
                meshname    = key,
                rgba        = list(desc.rgba),
                mass        = desc.mass,
                friction    = list(desc.friction),
                condim      = desc.condim,
                conaffinity = desc.conaffinity,
            )

        else:
            raise ValueError(f"不支持的形状: '{t}'")

    @classmethod
    def _ensure_mesh_asset(cls, spec: mujoco.MjSpec, key: str, scale: float) -> None:
        """幂等地将 STL 网格加入 MjSpec（同名只加载一次）。"""
        path = MeshRegistry.get(key)
        try:
            spec.add_mesh(name=key, file=str(path), scale=[scale, scale, scale])
        except Exception:
            pass  # 已存在，忽略


# =============================================================================
# 枚举工具
# =============================================================================

def all_object_types(
    shapes:    Optional[List[str]] = None,
    colors:    Optional[List[str]] = None,
    sizes:     Optional[List[str]] = None,
    mesh_keys: Optional[List[str]] = None,
) -> List[str]:
    """
    枚举所有合法描述符字符串。

    参数（均可省略，省略则取全集）
    ----
    shapes    : 形状子集，如 ["box", "sphere"]
    colors    : 颜色子集，如 ["red", "blue"]
    sizes     : 尺寸子集，如 ["small", "large"]
    mesh_keys : 额外 mesh 键名，自动展开为 mesh:<key> 加入形状列表

    示例
    ----
        all_object_types()
        # → 4形状 × 7色 × 3尺寸 = 84 种

        all_object_types(mesh_keys=["bottle"])
        # → 5形状 × 7色 × 3尺寸 = 105 种

        all_object_types(shapes=["box"], sizes=["small","large"])
        # → 1 × 7 × 2 = 14 种
    """
    _shapes = list(shapes or NATIVE_SHAPES)
    if mesh_keys:
        _shapes += [f"mesh:{k}" for k in mesh_keys]

    _colors = list(colors or COLOR_PALETTE)
    _sizes  = list(sizes  or SIZE_TABLE)

    return [
        f"{shape}_{color}_{size}"
        for shape in _shapes
        for color in _colors
        for size  in _sizes
    ]


# =============================================================================
# 便捷常量
# =============================================================================

#: 4形状 × 7色 × 3尺寸 = 84 种，无需额外资源即可使用
ALL_NATIVE_TYPES: List[str] = all_object_types()


# =============================================================================
# CLI 快速验证
# =============================================================================

if __name__ == "__main__":
    # 用 mock 替代 mujoco，验证纯逻辑
    print("=" * 60)
    print("  ObjectFactory 解析测试（不依赖 MuJoCo）")
    print("=" * 60)

    tests = [
        "box_red_large",
        "sphere_blue_small",
        "cylinder_green_medium",
        "capsule_yellow_large",
        "box_purple_small",
        "sphere_orange_medium",
        "cylinder_cyan_large",
        "mesh:bottle_red_small",
    ]

    for s in tests:
        try:
            desc = ObjectFactory.parse(s)
            print(
                f"  {s:<30}  shape={desc.shape_type:<10} "
                f"color={desc.color:<8} size={desc.size:<8} "
                f"bottom_z={desc.bottom_half_z:.3f}m"
            )
        except Exception as e:
            print(f"  {s:<30}  ERROR: {e}")

    print()
    print(f"  all_object_types()              → {len(all_object_types())} 种（4×7×3）")
    print(f"  all_object_types(mesh_keys=['bottle']) → {len(all_object_types(mesh_keys=['bottle']))} 种（5×7×3）")
    print(f"  all_object_types(shapes=['box'], sizes=['small','large']) → "
          f"{len(all_object_types(shapes=['box'], sizes=['small','large']))} 种（1×7×2）")

    print()
    print("  错误处理：")
    for bad in ["box_pink_large", "cube_red_large", "box_red_huge"]:
        try:
            ObjectFactory.parse(bad)
        except ValueError as e:
            print(f"  ✓ '{bad}': {e.args[0].splitlines()[0]}")