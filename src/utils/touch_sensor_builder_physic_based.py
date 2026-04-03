"""
触觉传感器布局模块 —— 弹性接触点仿真版本（三节指节配置）
修正版：移除了对不存在的 Skin Geom 的依赖，直接基于指节 Body 坐标系生成传感器。
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Any
import mujoco
import numpy as np
from src.utils.stl_mesh_sampler import generate_surface_mesh_points_from_stl

# ====================== 物理参数常量 ======================
TAXEL_RADIUS = 0.001  # 接触球半径 [m]
ELASTIC_STIFFNESS = 200.0  # 弹性刚度 [N/m]
ELASTIC_DAMPING = 2.0  # 阻尼 [N·s/m]
ELASTIC_RANGE = 0.002  # 最大压缩量 [m]
FORCE_MAX_NEWTON = 5.0  # 饱和力阈值 [N]
SITE_GROUP = 4
SITE_RGBA = (0.95, 0.45, 0.05, 0.6)  # 可视化颜色


# ====================== 指节配置 ======================
class PhalanxConfig(NamedTuple):
    phalanx_name: str  # 指节标识，如 "finger_0_bottom"
    body_name: str     # 指节 body 名称（不含 prefix）
    stl_file: str      # 对应 STL 文件名（仅用于读取网格形状，不用于查找 Geom）
    rows: int          # taxel 行数
    cols: int          # taxel 列数


PHALANX_CONFIGS: List[PhalanxConfig] = [
    # 手指 0
    PhalanxConfig("finger_0_bottom", "finger_first_0_p", "skin_0_0_p.STL", 10, 7),
    PhalanxConfig("finger_0_middle", "finger_second_0_p", "skin_0_1_p.STL", 8, 5),
    PhalanxConfig("finger_0_top", "finger_second_0_p", "skin_0_2_p.STL", 6, 5),
    # 手指 1
    PhalanxConfig("finger_1_bottom", "finger_first_1_p", "skin_1_0_p.STL", 10, 7),
    PhalanxConfig("finger_1_middle", "finger_second_1_p", "skin_1_1_p.STL", 8, 5),
    PhalanxConfig("finger_1_top", "finger_second_1_p", "skin_1_2_p.STL", 6, 5),
    # 手指 2
    PhalanxConfig("finger_2_bottom", "finger_first_2_p", "skin_2_0_p.STL", 10, 7),
    PhalanxConfig("finger_2_middle", "finger_second_2_p", "skin_2_1_p.STL", 8, 5),
    PhalanxConfig("finger_2_top", "finger_second_2_p", "skin_2_2_p.STL", 6, 5),
    # 手指 3
    PhalanxConfig("finger_3_bottom", "finger_first_3_p", "skin_3_0_p.STL", 10, 7),
    PhalanxConfig("finger_3_middle", "finger_second_3_p", "skin_3_1_p.STL", 8, 5),
    PhalanxConfig("finger_3_top", "finger_second_3_p", "skin_3_2_p.STL", 6, 5),
    # 拇指 4
    PhalanxConfig("thumb_bottom", "thumb_first_p", "skin_4_0_p.STL", 10, 7),
    PhalanxConfig("thumb_middle", "thumb_second_p", "skin_4_1_p.STL", 8, 5),
    PhalanxConfig("thumb_top", "thumb_third_p", "skin_4_2_p.STL", 6, 5),
]

FINGER_PHALANX_ORDER = {
    "finger_0": ["finger_0_bottom", "finger_0_middle", "finger_0_top"],
    "finger_1": ["finger_1_bottom", "finger_1_middle", "finger_1_top"],
    "finger_2": ["finger_2_bottom", "finger_2_middle", "finger_2_top"],
    "finger_3": ["finger_3_bottom", "finger_3_middle", "finger_3_top"],
    "thumb": ["thumb_bottom", "thumb_middle", "thumb_top"],
}


# ====================== 传感器阵列描述符 ======================
class PhalanxSensorArray:
    def __init__(self, phalanx_name: str, rows: int, cols: int, sensor_names: List[List[str]]):
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

    def read_image(self, data: mujoco.MjData, force_max: float = FORCE_MAX_NEWTON) -> np.ndarray:
        raw = self.read_raw(data)
        return (np.clip(raw, 0.0, force_max) / force_max * 255.0).astype(np.uint8)


# ====================== 坐标系工具 (修正核心) ======================
def _get_skin_geom_pose_in_body(spec: mujoco.MjSpec, geom_name: str, prefix: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 MjSpec 中读取 skin geom 相对于其 parent body 的位姿。
    这是复用 XML 中人类专家调好的参数的关键。
    """
    full_geom_name = prefix + geom_name
    try:
        geom = spec.geom(full_geom_name)
    except KeyError:
        # 兼容性处理：有时候 geom 名字不带 prefix
        geom = spec.geom(geom_name)
    
    pos = np.array(geom.pos, dtype=float)
    quat = np.array(geom.quat, dtype=float) # [w,x,y,z]
    return pos, quat

def _quat_to_rot(quat_wxyz: np.ndarray) -> np.ndarray:
    """四元数转旋转矩阵"""
    w, x, y, z = quat_wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=float)

def _apply_geom_transform(pts_mesh: np.ndarray, geom_pos: np.ndarray, geom_quat: np.ndarray) -> np.ndarray:
    """
    将 mesh 局部坐标点变换到 parent body 局部坐标系。
    公式: P_body = R * P_mesh + Pos
    """
    R = _quat_to_rot(geom_quat)
    # 批量矩阵乘法: (N,3) @ (3,3) -> (N,3)
    pts_rotated = pts_mesh @ R.T # 注意转置
    pts_body = pts_rotated + geom_pos
    return pts_body


def _outward_normals(pts_local: np.ndarray) -> np.ndarray:
    """计算向外法向量"""
    centered = pts_local - pts_local.mean(0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[-1]
    radial = centered - (centered @ axis)[:, None] * axis
    norms = np.linalg.norm(radial, axis=1, keepdims=True) + 1e-12
    return radial / norms


def _build_phalanx_array(spec: mujoco.MjSpec, cfg: PhalanxConfig, meshes_dir: Path, prefix: str) -> PhalanxSensorArray:
    """
    构建单节指节的传感器阵列。
    【核心修正】：不再假设 STL 原点即 Body 原点，而是读取 XML 中 geom 的 pos/quat 进行对齐。
    """
    stl_path = meshes_dir / cfg.stl_file
    if not stl_path.exists():
        raise FileNotFoundError(f"STL not found: {stl_path}")

    # --- 1. 读取 STL 生成点云 (Mesh Local) ---
    # 注意：此时的 pts_mesh 是相对于 STL 文件原点的坐标
    pts_mesh = generate_surface_mesh_points_from_stl(stl_path, m=cfg.rows, n=cfg.cols)

    # --- 2. 【关键】读取 XML 中 Skin Geom 的位姿 (Pos/Quat) ---
    # 这里的 cfg.mesh_name 需要对应 XML 中 <geom> 的名字
    # 注意：你的 PhalanxConfig 里可能没有 mesh_name，需要加一个，或者这里用 cfg.body_name 相关的逻辑
    # 假设你的 PhalanxConfig 里有一个属性叫 mesh_name，例如 "skin_0_0_p"
    # 如果你的配置里没有，你需要根据 cfg.phalanx_name 映射出来
    mesh_name = Path(cfg.stl_file).stem  # 'skin_0_0_p.STL' -> 'skin_0_0_p'
    try:
        # 调用工具函数读取该 Geom 在父 Body 中的偏移
        geom_pos, geom_quat = _get_skin_geom_pose_in_body(spec, mesh_name, prefix)
    except Exception as e:
        # 容错：如果找不到 Geom，打印警告并退化为原点（虽然会偏，但能跑）
        print(f"[Warning] Geom {cfg.mesh_name} not found, using origin. Sensor will be misaligned.")
        geom_pos, geom_quat = np.zeros(3), np.array([1, 0, 0, 0])

    # --- 3. 坐标变换：将 STL 点云从 Mesh 局部坐标 转换到 Body 局部坐标 ---
    # 这一步是魔法发生的地方：pts_mesh * R + Pos
    pts_local = _apply_geom_transform(pts_mesh, geom_pos, geom_quat)

    # --- 4. 计算法向量 (用于弹性关节方向) ---
    # 注意：法向量也需要随着坐标变换旋转
    normals = _outward_normals(pts_mesh) # 注意：这里如果法向量依赖于坐标，最好也变换一下
    # 简单做法：假设法向量在 Mesh 坐标系是向外的，变换矩阵 R 也会作用于法向量
    R = _quat_to_rot(geom_quat)
    normals_local = (R @ normals.T).T # 将法向量旋转到 Body 坐标系

    # --- 5. 获取挂载目标 Body ---
    full_body_name = prefix + cfg.body_name
    try:
        target_body = spec.body(full_body_name)
    except:
        raise KeyError(f"Target body '{full_body_name}' not found.")

    # --- 6. 创建传感器 (保持你的弹性关节逻辑不变) ---
    sensor_names_2d = []
    for row in range(cfg.rows):
        row_names = []
        for col in range(cfg.cols):
            idx = row * cfg.cols + col
            pt = pts_local[idx]       # 【已修正】使用变换后的坐标
            nvec = normals_local[idx] # 【已修正】使用变换后的法向量
            
            tag = f"{cfg.phalanx_name}_r{row:02d}_c{col:002d}"

            # --- 子 Body (Taxel) ---
            tb = target_body.add_body()
            tb.name = f"{prefix}taxel_body_{tag}"
            tb.pos = pt.tolist() # 现在它就在手指肚上了！

            # --- 弹性关节 (滑动关节) ---
            jt = tb.add_joint()
            jt.name = f"{prefix}taxel_j_{tag}"
            jt.type = mujoco.mjtJoint.mjJNT_SLIDE
            jt.axis = (-nvec).tolist()  # 压入方向
            jt.stiffness = ELASTIC_STIFFNESS
            jt.damping = ELASTIC_DAMPING
            jt.range = [0.0, ELASTIC_RANGE]
            jt.limited = True

            # --- 接触几何体 (球体) ---
            gm = tb.add_geom()
            gm.type = mujoco.mjtGeom.mjGEOM_SPHERE
            gm.size = [TAXEL_RADIUS, 0, 0] # Sphere 只需要第一个 size
            gm.condim = 1
            gm.group = SITE_GROUP
            gm.rgba = list(SITE_RGBA)

            # --- Site & Sensor (略) ---
            st = tb.add_site()
            st.name = f"{prefix}site_{tag}"
            st.type = mujoco.mjtGeom.mjGEOM_SPHERE
            st.size = np.array([TAXEL_RADIUS * 1.5, TAXEL_RADIUS * 1.5, TAXEL_RADIUS * 1.5]) # 强制 3D 数组
            st.group = SITE_GROUP
            st.rgba = SITE_RGBA
            sn = spec.add_sensor()
            sn.name = f"{prefix}taxel_sens_{tag}"
            sn.type = mujoco.mjtSensor.mjSENS_TOUCH
            sn.objtype = mujoco.mjtObj.mjOBJ_SITE
            sn.objname = st.name  # 绑定到刚才创建的 Site
            sn.cutoff = 0.0
            sn.noise = 0.0

            row_names.append(sn.name)
        sensor_names_2d.append(row_names)

    print(f"[SensorBuilder] Added {len(pts_local)} taxels on {cfg.phalanx_name}")
    return PhalanxSensorArray(cfg.phalanx_name, cfg.rows, cfg.cols, sensor_names_2d)

# ====================== 公开接口 ======================
def add_elastic_taxel_arrays(
    spec: mujoco.MjSpec,
    hand_path: Path,
    prefix: str = "inspirehand_"
) -> Dict[str, PhalanxSensorArray]:
    """
    添加所有指节的弹性传感器。
    """
    meshes_dir = Path(hand_path).parent / "meshes"
    if not meshes_dir.exists():
        raise FileNotFoundError(f"Meshes dir not found: {meshes_dir}")

    arrays = {}
    for cfg in PHALANX_CONFIGS:
        arrays[cfg.phalanx_name] = _build_phalanx_array(spec, cfg, meshes_dir, prefix)

    total = sum(arr.rows * arr.cols for arr in arrays.values())
    print(f"[SensorBuilder] Success! Added {total} elastic taxels.")
    return arrays


def bind_all(arrays: Dict[str, PhalanxSensorArray], model: mujoco.MjModel):
    """绑定所有传感器 ID"""
    for arr in arrays.values():
        arr.bind(model)
    print(f"[SensorBuilder] Bound {len(arrays)} phalanx arrays.")


def read_all_tactile(arrays: Dict[str, PhalanxSensorArray], data: mujoco.MjData) -> Dict[str, np.ndarray]:
    """读取所有触觉图像"""
    return {name: arr.read_image(data) for name, arr in arrays.items()}