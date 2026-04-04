"""
触觉传感器布局模块 —— 弹性接触点仿真版本（三节指节配置）
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple
import mujoco
import numpy as np
from src.utils.stl_mesh_sampler import generate_surface_mesh_points_from_stl

# ====================== 物理参数常量 ======================
TAXEL_RADIUS = 0.001        # 接触球半径 [m]
ELASTIC_STIFFNESS = 200.0   # 弹性刚度 [N/m]
ELASTIC_DAMPING = 2.0       # 阻尼 [N·s/m]
ELASTIC_RANGE = 0.002       # 最大压缩量 [m]
FORCE_MAX_NEWTON = 5.0      # 饱和力阈值 [N]
SITE_GROUP = 4
SITE_RGBA = (0.95, 0.45, 0.05, 0.6)


# ====================== 指节配置 ======================
class PhalanxConfig(NamedTuple):
    phalanx_name: str   # 指节标识，如 "finger_0_bottom"
    body_name: str      # 指节 body 名称（不含 prefix）
    stl_file: str       # 对应 STL 文件名
    rows: int           # taxel 行数
    cols: int           # taxel 列数


PHALANX_CONFIGS: List[PhalanxConfig] = [
    # 手指 0
    PhalanxConfig("finger_0_bottom", "finger_first_0_p",  "skin_0_0_p.STL", 10, 7),
    PhalanxConfig("finger_0_middle", "finger_second_0_p", "skin_0_1_p.STL",  8, 5),
    PhalanxConfig("finger_0_top",    "finger_second_0_p", "skin_0_2_p.STL",  6, 5),
    # 手指 1
    PhalanxConfig("finger_1_bottom", "finger_first_1_p",  "skin_1_0_p.STL", 10, 7),
    PhalanxConfig("finger_1_middle", "finger_second_1_p", "skin_1_1_p.STL",  8, 5),
    PhalanxConfig("finger_1_top",    "finger_second_1_p", "skin_1_2_p.STL",  6, 5),
    # 手指 2
    PhalanxConfig("finger_2_bottom", "finger_first_2_p",  "skin_2_0_p.STL", 10, 7),
    PhalanxConfig("finger_2_middle", "finger_second_2_p", "skin_2_1_p.STL",  8, 5),
    PhalanxConfig("finger_2_top",    "finger_second_2_p", "skin_2_2_p.STL",  6, 5),
    # 手指 3
    PhalanxConfig("finger_3_bottom", "finger_first_3_p",  "skin_3_0_p.STL", 10, 7),
    PhalanxConfig("finger_3_middle", "finger_second_3_p", "skin_3_1_p.STL",  8, 5),
    PhalanxConfig("finger_3_top",    "finger_second_3_p", "skin_3_2_p.STL",  6, 5),
    # 拇指 4
    PhalanxConfig("thumb_bottom", "thumb_first_p",  "skin_4_0_p.STL", 10, 7),
    PhalanxConfig("thumb_middle", "thumb_second_p", "skin_4_1_p.STL",  8, 5),
    PhalanxConfig("thumb_top",    "thumb_third_p",  "skin_4_2_p.STL",  6, 5),
]

# 显示时各手指的指节顺序（用于 grasp_task_env 中的有序拼图）
FINGER_PHALANX_ORDER = {
    "finger_0": ["finger_0_bottom", "finger_0_middle", "finger_0_top"],
    "finger_1": ["finger_1_bottom", "finger_1_middle", "finger_1_top"],
    "finger_2": ["finger_2_bottom", "finger_2_middle", "finger_2_top"],
    "finger_3": ["finger_3_bottom", "finger_3_middle", "finger_3_top"],
    "thumb":    ["thumb_bottom",    "thumb_middle",    "thumb_top"],
}

# 显示用的有序 key 列表，供 grasp_task_env 按此顺序索引 vis_frames
DISPLAY_ORDER: List[str] = [
    name
    for finger in ["finger_0", "finger_1", "finger_2", "finger_3", "thumb"]
    for name in FINGER_PHALANX_ORDER[finger]
]


# ====================== 传感器阵列描述符 ======================
class PhalanxSensorArray:
    def __init__(self, phalanx_name: str, rows: int, cols: int,
                 sensor_names: List[List[str]]):
        self.phalanx_name = phalanx_name
        self.rows = rows
        self.cols = cols
        self.sensor_names = sensor_names
        self.sensor_ids: Optional[np.ndarray] = None

    def bind(self, model: mujoco.MjModel) -> None:
        ids = np.zeros((self.rows, self.cols), dtype=np.int32)
        for r in range(self.rows):
            for c in range(self.cols):
                name = self.sensor_names[r][c]
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
                if sid < 0:
                    raise ValueError(f"Sensor '{name}' not found in model.")
                ids[r, c] = sid
        self.sensor_ids = ids

    def read_raw(self, data: mujoco.MjData) -> np.ndarray:
        if self.sensor_ids is None:
            raise RuntimeError("Sensor not bound. Call bind() first.")
        return data.sensordata[self.sensor_ids.ravel()].reshape(self.rows, self.cols)

    def read_image(self, data: mujoco.MjData,
                   force_max: float = FORCE_MAX_NEWTON) -> np.ndarray:
        raw = self.read_raw(data)
        return (np.clip(raw, 0.0, force_max) / force_max * 255.0).astype(np.uint8)


# ====================== 坐标系工具 ======================
def _get_skin_geom_pose_in_body(
    spec: mujoco.MjSpec, geom_name: str, prefix: str
) -> Tuple[np.ndarray, np.ndarray]:
    """从 MjSpec 读取 skin geom 相对于其 parent body 的位姿 (pos, quat[wxyz])。"""
    full_geom_name = prefix + geom_name
    try:
        geom = spec.geom(full_geom_name)
    except KeyError:
        geom = spec.geom(geom_name)
    return np.array(geom.pos, dtype=float), np.array(geom.quat, dtype=float)


def _quat_to_rot(quat_wxyz: np.ndarray) -> np.ndarray:
    """四元数 [w,x,y,z] 转旋转矩阵。"""
    w, x, y, z = quat_wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)    ],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z),  2*(y*z - x*w)    ],
        [2*(x*z - y*w),     2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ], dtype=float)


def _apply_geom_transform(
    pts_mesh: np.ndarray, geom_pos: np.ndarray, geom_quat: np.ndarray
) -> np.ndarray:
    """将 mesh 局部坐标点变换到 parent body 局部坐标系：P_body = R·P_mesh + pos。"""
    R = _quat_to_rot(geom_quat)
    return pts_mesh @ R.T + geom_pos


def _outward_normals(pts_local: np.ndarray,
                     interior_ref: np.ndarray) -> np.ndarray:
    """
    计算点云各点的向外单位法向量。

    【修正】原版法向量基于点云质心估算"向外"，对开放弯曲薄片 (皮肤 STL) 不可靠：
    质心可能落在曲面凹侧，导致法向量整体翻转。

    修正策略：
    1. 用 SVD 求局部法向量（径向分量）。
    2. 以传入的 interior_ref（指节 body 原点，始终位于手指内部）为参考，
       确保法向量方向与 (pt → interior_ref) 方向相反，即真正朝外。

    Args:
        pts_local:    变换到 body 坐标系后的点云，shape (N, 3)。
        interior_ref: body 坐标系下的内部参考点，通常为 [0,0,0]（body 原点），shape (3,)。
    """
    centered = pts_local - pts_local.mean(0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    # 最小奇异值对应的向量 ≈ 点云法轴
    thin_axis = vh[-1]
    # 去除法轴分量，得到径向方向
    radial = centered - (centered @ thin_axis)[:, None] * thin_axis
    norms = np.linalg.norm(radial, axis=1, keepdims=True) + 1e-12
    normals = radial / norms  # shape (N, 3)，方向尚未确定朝内/外

    # ---- 【修正核心】用 interior_ref 校正方向 ----
    # interior_ref 在手指内部，pt → interior_ref 方向 = interior_ref - pt，是朝内的
    # 法向量应与其方向相反，即 dot(normal, interior_ref - pt) < 0 为朝外（正确）
    # 若 dot > 0 说明法向量朝内，需要翻转
    inward_vecs = interior_ref[None, :] - pts_local          # (N,3)，朝内
    dots = np.einsum('ij,ij->i', normals, inward_vecs)       # (N,)
    # 哪些点的法向量与朝内向量同向（即法向量朝内，需要翻转）
    flip_mask = dots > 0
    normals[flip_mask] *= -1

    return normals


# ====================== 核心构建函数 ======================
def _build_phalanx_array(
    spec: mujoco.MjSpec,
    cfg: PhalanxConfig,
    meshes_dir: Path,
    prefix: str,
) -> PhalanxSensorArray:
    """构建单节指节的弹性传感器阵列。"""
    stl_path = meshes_dir / cfg.stl_file
    if not stl_path.exists():
        raise FileNotFoundError(f"STL not found: {stl_path}")

    # --- 1. 从 STL 采样点云（Mesh 局部坐标系） ---
    pts_mesh = generate_surface_mesh_points_from_stl(stl_path, m=cfg.rows, n=cfg.cols)

    # --- 2. 读取 XML 中 Skin Geom 的 pos/quat ---
    mesh_name = Path(cfg.stl_file).stem   # 'skin_0_0_p.STL' -> 'skin_0_0_p'
    try:
        geom_pos, geom_quat = _get_skin_geom_pose_in_body(spec, mesh_name, prefix)
    except Exception:
        # 【修正 Bug1】原代码写的是 cfg.mesh_name，该字段不存在会 AttributeError
        # 正确应该用 cfg.stl_file 来拼出 mesh_name（上面已做），这里只需用 cfg.stl_file 打印
        print(f"[Warning] Geom '{mesh_name}' not found in spec, using identity transform. "
              f"(stl_file={cfg.stl_file})")
        geom_pos  = np.zeros(3)
        geom_quat = np.array([1.0, 0.0, 0.0, 0.0])

    # --- 3. 坐标变换：Mesh 局部 → Body 局部 ---
    pts_local = _apply_geom_transform(pts_mesh, geom_pos, geom_quat)

    # --- 4. 计算朝外法向量（在 body 坐标系下） ---
    # 【修正 Bug2】原代码先在 mesh 坐标系算法向量再旋转，且没有可靠的朝向校正。
    # 修正为：先把点云变换到 body 坐标系，再算法向量，同时用 body 原点（内部参考）校正朝向。
    R = _quat_to_rot(geom_quat)
    # body 原点在 body 坐标系下就是 [0,0,0]，始终位于手指内部
    interior_ref = np.zeros(3)
    normals_local = _outward_normals(pts_local, interior_ref)

    # --- 5. 获取挂载目标 Body ---
    full_body_name = prefix + cfg.body_name
    try:
        target_body = spec.body(full_body_name)
    except KeyError:
        raise KeyError(f"Target body '{full_body_name}' not found in spec.")

    # --- 6. 逐 taxel 创建子 body / 弹性关节 / 接触 geom / site / sensor ---
    sensor_names_2d: List[List[str]] = []
    for row in range(cfg.rows):
        row_names: List[str] = []
        for col in range(cfg.cols):
            idx  = row * cfg.cols + col
            pt   = pts_local[idx]
            nvec = normals_local[idx]   # 已确认朝外

            tag = f"{cfg.phalanx_name}_r{row:02d}_c{col:02d}"

            # -- 子 Body --
            tb      = target_body.add_body()
            tb.name = f"{prefix}taxel_body_{tag}"
            tb.pos  = pt.tolist()

            # -- 弹性滑动关节 --
            # 【修正 Bug3】关节轴方向与 range 的逻辑：
            # axis = +nvec（朝外），range = [-ELASTIC_RANGE, 0]
            # 含义：物体从外部压入 → 关节沿 -nvec 方向运动 → q 为负值
            # stiffness 产生恢复力把 taxel 推回 q=0（自然伸出位置）
            # 这与"弹性接触点被向后压"的物理描述完全吻合。
            #
            # 原代码写的是 axis=-nvec, range=[0, ELASTIC_RANGE]：
            # 轴反向后 q 正值代表向外伸出，range 下限为 0 无法产生负位移，
            # 实际上关节被限位在 0，物体一碰就顶死，弹性行程失效。
            jt           = tb.add_joint()
            jt.name      = f"{prefix}taxel_j_{tag}"
            jt.type      = mujoco.mjtJoint.mjJNT_SLIDE
            jt.axis      = nvec.tolist()            # 朝外为正方向
            jt.stiffness = ELASTIC_STIFFNESS
            jt.damping   = ELASTIC_DAMPING
            jt.range     = [-ELASTIC_RANGE, 0.0]   # 只允许向内压缩
            jt.limited   = True

            # -- 接触球体 --
            gm        = tb.add_geom()
            gm.type   = mujoco.mjtGeom.mjGEOM_SPHERE
            gm.size   = [TAXEL_RADIUS, 0.0, 0.0]
            gm.condim = 1
            gm.group  = SITE_GROUP
            gm.rgba   = list(SITE_RGBA)

            # -- Site（用于传感器绑定） --
            # 【修正 Bug4】st.size 原代码传 numpy array，部分 MuJoCo 版本会报类型错误，改为 list
            st       = tb.add_site()
            st.name  = f"{prefix}site_{tag}"
            st.type  = mujoco.mjtGeom.mjGEOM_SPHERE
            st.size  = [TAXEL_RADIUS * 1.5, TAXEL_RADIUS * 1.5, TAXEL_RADIUS * 1.5]
            st.group = SITE_GROUP
            st.rgba  = list(SITE_RGBA)

            # -- Touch Sensor --
            sn          = spec.add_sensor()
            sn.name     = f"{prefix}taxel_sens_{tag}"
            sn.type     = mujoco.mjtSensor.mjSENS_TOUCH
            sn.objtype  = mujoco.mjtObj.mjOBJ_SITE
            sn.objname  = st.name
            sn.cutoff   = 0.0
            sn.noise    = 0.0

            row_names.append(sn.name)
        sensor_names_2d.append(row_names)

    print(f"[SensorBuilder] Added {cfg.rows * cfg.cols} taxels on '{cfg.phalanx_name}'")
    return PhalanxSensorArray(cfg.phalanx_name, cfg.rows, cfg.cols, sensor_names_2d)


# ====================== 公开接口 ======================
def add_elastic_taxel_arrays(
    spec: mujoco.MjSpec,
    hand_path: Path,
    prefix: str = "inspirehand_",
) -> Dict[str, PhalanxSensorArray]:
    """添加所有指节的弹性传感器，返回按 phalanx_name 索引的描述符字典。"""
    meshes_dir = Path(hand_path).parent / "meshes"
    if not meshes_dir.exists():
        raise FileNotFoundError(f"Meshes dir not found: {meshes_dir}")

    arrays: Dict[str, PhalanxSensorArray] = {}
    for cfg in PHALANX_CONFIGS:
        arrays[cfg.phalanx_name] = _build_phalanx_array(spec, cfg, meshes_dir, prefix)

    total = sum(arr.rows * arr.cols for arr in arrays.values())
    print(f"[SensorBuilder] Done. Total elastic taxels: {total}")
    return arrays


def bind_all(arrays: Dict[str, PhalanxSensorArray], model: mujoco.MjModel) -> None:
    """将所有传感器名称绑定到编译后模型的 sensor ID。"""
    for arr in arrays.values():
        arr.bind(model)
    print(f"[SensorBuilder] Bound {len(arrays)} phalanx arrays.")


def read_all_tactile(
    arrays: Dict[str, PhalanxSensorArray], data: mujoco.MjData
) -> Dict[str, np.ndarray]:
    """读取所有指节的触觉图像（uint8，0~255）。"""
    return {name: arr.read_image(data) for name, arr in arrays.items()}