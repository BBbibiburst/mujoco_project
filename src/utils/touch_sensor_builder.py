"""
触觉传感器布局模块 (Touch Sensor Layout Module).

在灵巧手 skin mesh 的曲面上均匀布置 MuJoCo touch sensor。
每个 sensor 通过在对应 body 下添加 site 来实现，坐标系从 STL 世界坐标
逆变换回对应 body 的局部坐标系。

核心功能：
    1. 批量传感器生成：为15块手指皮肤（4指+拇指）自动布局触觉传感器
    2. 分层采样密度：底部指节70个(10×7)、中部40个(8×5)、顶部30个(6×5)
    3. 坐标系自动转换：STL世界坐标 → 皮肤几何体局部坐标 → 父级body局部坐标
    4. 可视化配置：支持自定义site颜色、分组，便于仿真调试

使用方式：
    在 robot_arm_system.py 的 get_combined_spec() 末尾、return 之前调用：

        from touch_sensor_builder import add_touch_sensors_to_spec
        touch_sensor_map = add_touch_sensors_to_spec(
            spec=arm_spec,
            hand_path=hand_path,
            prefix="inspirehand_"
        )

依赖：
    - generate_surface_mesh_points_from_stl (来自 stl_mesh_sampler 模块)
    - mujoco (MjSpec API)
    - numpy
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import mujoco
import numpy as np
from ..utils.stl_mesh_sampler import generate_surface_mesh_points_from_stl


# ====================== 传感器布局配置 ======================

class SkinConfig(NamedTuple):
    """
    单块 skin mesh 的传感器配置.

    Attributes:
        mesh_name: skin 的 mesh/geom 名称（不含 prefix）。
        body_name: 该 skin 所挂载的 parent body 名称（不含 prefix）。
        stl_file: 对应的 STL 文件名。
        m: 圆周方向采样数（周向网格数）。
        n: 轴向采样数（高度方向网格数）。
        sensor_size: touch sensor site 半径 [m]。
    """
    mesh_name: str       # skin 的 mesh/geom 名称（不含 prefix）
    body_name: str       # 该 skin 所挂载的 parent body 名称（不含 prefix）
    stl_file: str        # 对应的 STL 文件名
    m: int               # 圆周方向采样数
    n: int               # 轴向采样数
    sensor_size: float   # touch sensor site 半径 [m]


# 指节类型 → (m, n) 映射
_BOTTOM_MN = (10, 7)   # 底部指节 skin_X_0_p  → 70 个传感器
_MIDDLE_MN = (8,  5)   # 中部指节 skin_X_1_p  → 40 个传感器
_TOP_MN    = (6,  5)   # 顶部指节 skin_X_2_p  → 30 个传感器

# 默认 sensor site 半径（约 3 mm，可按需调整）
_DEFAULT_SENSOR_RADIUS = 0.003

# 15 块 skin 的完整配置
# body_name 对应 XML 中 <body name="..."> 的名字（不含 inspirehand_ prefix）
SKIN_CONFIGS: List[SkinConfig] = [
    # ── 手指 0 ──────────────────────────────────────────────
    SkinConfig("skin_0_0_p", "finger_first_0_p",  "skin_0_0_p.STL", *_BOTTOM_MN, _DEFAULT_SENSOR_RADIUS),
    SkinConfig("skin_0_1_p", "finger_second_0_p", "skin_0_1_p.STL", *_MIDDLE_MN, _DEFAULT_SENSOR_RADIUS),
    SkinConfig("skin_0_2_p", "finger_second_0_p", "skin_0_2_p.STL", *_TOP_MN,    _DEFAULT_SENSOR_RADIUS),
    # ── 手指 1 ──────────────────────────────────────────────
    SkinConfig("skin_1_0_p", "finger_first_1_p",  "skin_1_0_p.STL", *_BOTTOM_MN, _DEFAULT_SENSOR_RADIUS),
    SkinConfig("skin_1_1_p", "finger_second_1_p", "skin_1_1_p.STL", *_MIDDLE_MN, _DEFAULT_SENSOR_RADIUS),
    SkinConfig("skin_1_2_p", "finger_second_1_p", "skin_1_2_p.STL", *_TOP_MN,    _DEFAULT_SENSOR_RADIUS),
    # ── 手指 2 ──────────────────────────────────────────────
    SkinConfig("skin_2_0_p", "finger_first_2_p",  "skin_2_0_p.STL", *_BOTTOM_MN, _DEFAULT_SENSOR_RADIUS),
    SkinConfig("skin_2_1_p", "finger_second_2_p", "skin_2_1_p.STL", *_MIDDLE_MN, _DEFAULT_SENSOR_RADIUS),
    SkinConfig("skin_2_2_p", "finger_second_2_p", "skin_2_2_p.STL", *_TOP_MN,    _DEFAULT_SENSOR_RADIUS),
    # ── 手指 3 ──────────────────────────────────────────────
    SkinConfig("skin_3_0_p", "finger_first_3_p",  "skin_3_0_p.STL", *_BOTTOM_MN, _DEFAULT_SENSOR_RADIUS),
    SkinConfig("skin_3_1_p", "finger_second_3_p", "skin_3_1_p.STL", *_MIDDLE_MN, _DEFAULT_SENSOR_RADIUS),
    SkinConfig("skin_3_2_p", "finger_second_3_p", "skin_3_2_p.STL", *_TOP_MN,    _DEFAULT_SENSOR_RADIUS),
    # ── 拇指 4 ──────────────────────────────────────────────
    SkinConfig("skin_4_0_p", "thumb_first_p",  "skin_4_0_p.STL", *_BOTTOM_MN, _DEFAULT_SENSOR_RADIUS),
    SkinConfig("skin_4_1_p", "thumb_second_p", "skin_4_1_p.STL", *_MIDDLE_MN, _DEFAULT_SENSOR_RADIUS),
    SkinConfig("skin_4_2_p", "thumb_third_p",  "skin_4_2_p.STL", *_TOP_MN,    _DEFAULT_SENSOR_RADIUS),
]


# ====================== 坐标系工具 ======================

def _body_world_transform(spec: mujoco.MjSpec, body_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    通过临时编译模型获取 body 在世界坐标系中的位姿.

    实现方式：
        临时编译 spec → 创建 MjData → 执行前向运动学 → 读取 xpos/xmat。

    Args:
        spec: 未编译的 MjSpec 对象。
        body_name: body 的完整名称（含 prefix）。

    Returns:
        Tuple[np.ndarray, np.ndarray]: (pos_w, rot_w)
            - pos_w: (3,) body 原点在世界系中的位置。
            - rot_w: (3,3) body 姿态旋转矩阵（行向量为 body 坐标轴在世界系的投影）。

    Raises:
        ValueError: body_name 不存在于模型中。

    Note:
        每次调用都会临时编译 spec，适合离线预处理。
        若性能敏感（如批量处理大量点），可缓存结果或改用 mj_kinematics。
    """
    # 临时编译以读取 xpos / xmat
    tmp_model = spec.compile()
    tmp_data  = mujoco.MjData(tmp_model)
    mujoco.mj_kinematics(tmp_model, tmp_data)   # 前向运动学，填充 xpos/xmat

    body_id = mujoco.mj_name2id(tmp_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' 不存在于已编译模型中")

    pos_w = tmp_data.xpos[body_id].copy()               # (3,)
    rot_w = tmp_data.xmat[body_id].reshape(3, 3).copy() # (3,3), 行主序 → row=world axis
    return pos_w, rot_w


def _world_to_body_local(
    pts_world: np.ndarray,
    body_pos_w: np.ndarray,
    body_rot_w: np.ndarray,
) -> np.ndarray:
    """
    将世界坐标点批量转换到 body 局部坐标系.

    MuJoCo xmat 存储的是旋转矩阵 R，满足：
        p_world = R @ p_local + pos_world
    因此逆变换为：
        p_local = R^T @ (p_world - pos_world)

    Args:
        pts_world: (N, 3) 世界坐标点。
        body_pos_w: (3,) body 原点世界位置。
        body_rot_w: (3, 3) body 旋转矩阵（行主序）。

    Returns:
        np.ndarray: (N, 3) body 局部坐标点。

    Note:
        使用矩阵乘法 @ 实现批量变换，比循环更高效。
    """
    delta = pts_world - body_pos_w[None, :]          # (N, 3)
    pts_local = delta @ body_rot_w                   # R^T @ delta，因为 xmat 行=world→body
    return pts_local


# ====================== 主接口 ======================

def add_touch_sensors_to_spec(
    spec: mujoco.MjSpec,
    hand_path: Path,
    prefix: str = "inspirehand_",
    sensor_configs: Optional[List[SkinConfig]] = None,
    site_group: int = 4,
    site_rgba: Tuple[float, ...] = (1.0, 0.2, 0.2, 0.6),
) -> Dict[str, List[str]]:
    """
    为灵巧手 skin mesh 在曲面上批量添加 MuJoCo touch sensor.

    完整处理流程：
        1. 遍历每块 skin 配置，加载对应 STL 文件。
        2. 调用 generate_surface_mesh_points_from_stl 生成世界坐标采样点。
        3. 获取 skin geom 在其 parent body 中的局部位姿。
        4. 将采样点从 mesh 局部坐标变换到 body 局部坐标。
        5. 在 target body 下逐点添加 site（sphere）。
        6. 为每个 site 添加 touch sensor，绑定到该 site。
        7. 收集所有 sensor 名称，按 skin 分组返回。

    Args:
        spec: 已合并（未编译）的 MjSpec 对象。
        hand_path: 灵巧手模型目录（含 meshes/ 子目录的 Path）。
        prefix: 灵巧手 body/sensor 名称前缀，默认 "inspirehand_"。
            必须与 attach_body 时的 prefix 一致。
        sensor_configs: 自定义 SkinConfig 列表，None → 使用默认 SKIN_CONFIGS。
        site_group: site 的 MuJoCo group（用于可视化分层，默认 4）。
            可通过 MuJoCo 可视化选项按 group 开关显示。
        site_rgba: site 颜色 RGBA，默认半透明红色 (1,0.2,0.2,0.6)，便于可视化调试。

    Returns:
        Dict[str, List[str]]: 每块 skin 对应的 sensor 名称列表。
            键为 skin 名称（如 "skin_0_0_p"），值为该 skin 上所有 sensor 名称的列表。

    Raises:
        FileNotFoundError: STL 文件不存在。
        ValueError: body 未找到或 geom 未找到。

    Examples:
        >>> # 基础用法
        >>> sensor_map = add_touch_sensors_to_spec(spec, hand_path, prefix="inspirehand_")
        >>> print(f"共添加 {sum(len(v) for v in sensor_map.values())} 个传感器")
        
        >>> # 自定义配置（仅给拇指添加传感器）
        >>> thumb_configs = [c for c in SKIN_CONFIGS if c.mesh_name.startswith("skin_4")]
        >>> sensor_map = add_touch_sensors_to_spec(
        ...     spec, hand_path, sensor_configs=thumb_configs, site_rgba=(0,1,0,0.5)
        ... )

    Note:
        此函数直接修改输入的 spec 对象（添加 site 和 sensor）。
        必须在 spec.compile() 之前调用。
        传感器输出为法向接触力，单位牛顿（N）。
    """
    configs = sensor_configs or SKIN_CONFIGS
    meshes_dir = Path(hand_path).parent / "meshes"
    sensor_map: Dict[str, List[str]] = {}
    total_sensors = 0                       
    
    print(f"[TouchSensor] 开始添加 touch sensor，目标 skin 数量: {len(configs)}")
    
    for cfg in configs:
        # ----- 1. 加载 STL 并生成采样点 -----
        stl_path = meshes_dir / cfg.stl_file
        if not stl_path.exists():
            raise FileNotFoundError(f"STL 文件不存在: {stl_path}")

        # 生成世界坐标系下的采样点（来自 stl_mesh_sampler）
        pts_mesh = generate_surface_mesh_points_from_stl(stl_path, cfg.m, cfg.n)

        # ----- 2. 获取 skin geom 在 body 中的位姿 -----
        geom_pos_local, geom_quat_local = _get_skin_geom_pose_in_body(
            spec, cfg.mesh_name, prefix
        )
        
        # ----- 3. 坐标变换：mesh 局部 → body 局部 -----
        pts_body = _apply_geom_transform(pts_mesh, geom_pos_local, geom_quat_local)

        # ----- 4. 获取目标 body 并添加传感器 -----
        full_body_name = prefix + cfg.body_name
        target_body = spec.body(full_body_name)

        skin_sensors: List[str] = []
        
        for idx, pt in enumerate(pts_body):
            site_name   = f"touch_site_{cfg.mesh_name}_{idx}"
            sensor_name = f"touch_{cfg.mesh_name}_{idx}"

            # 添加 site（球形，半径=sensor_size）
            site = target_body.add_site()
            site.name    = site_name
            site.type    = mujoco.mjtGeom.mjGEOM_SPHERE
            site.size    = [cfg.sensor_size, 0.0, 0.0]  # sphere 只需第一个元素
            site.pos     = pt.tolist()
            site.group   = site_group
            site.rgba    = list(site_rgba)

            # 添加 touch sensor，绑定到 site
            sensor = spec.add_sensor()
            sensor.name    = sensor_name
            sensor.type    = mujoco.mjtSensor.mjSENS_TOUCH
            sensor.objtype = mujoco.mjtObj.mjOBJ_SITE
            sensor.objname = site_name
            sensor.cutoff  = 0.0     # 不截断（输出原始法向力，单位 N）
            sensor.noise   = 0.001   # 微小噪声，模拟真实传感器特性

            skin_sensors.append(sensor_name)

        sensor_map[cfg.mesh_name] = skin_sensors
        total_sensors += len(skin_sensors)
        print(f"[TouchSensor]   → 在 '{cfg.mesh_name}' 上添加 {cfg.m} * {cfg.n} = {len(skin_sensors)} 个 sensor")

    print(f"[TouchSensor] 完成！共添加 {total_sensors} 个 touch sensor，"
          f"覆盖 {len(configs)} 块 skin mesh。")
    return sensor_map


# ====================== 内部工具函数 ======================

def _get_skin_geom_pose_in_body(
    spec: mujoco.MjSpec,
    geom_name: str,
    prefix: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 MjSpec 中读取 skin geom 相对于其 parent body 的位姿.

    Args:
        spec: MjSpec 对象。
        geom_name: skin geom 名称（可能含或不含 prefix）。
        prefix: 模型前缀。

    Returns:
        Tuple[np.ndarray, np.ndarray]: (pos, quat)
            - pos: (3,) geom 在 body 局部坐标系中的位置。
            - quat: (4,) geom 在 body 局部坐标系中的四元数 [w,x,y,z]。

    Note:
        兼容 geom 名称带或不带 prefix 的情况。
        先尝试带 prefix 查找，失败则尝试原名。
    """
    full_geom_name = prefix + geom_name if not geom_name.startswith(prefix) else geom_name
    
    # MjSpec 中的 geom 名称在 attach 时已添加 prefix
    try:
        geom = spec.geom(full_geom_name)
    except KeyError:
        # 部分 skin geom 在 XML 中名称不带 prefix，直接用原名查找
        geom = spec.geom(geom_name)

    pos  = np.array(geom.pos,  dtype=float)
    quat = np.array(geom.quat, dtype=float)  # [w,x,y,z]
    return pos, quat


def _quat_to_rot(quat_wxyz: np.ndarray) -> np.ndarray:
    """
    四元数 [w,x,y,z] → 3×3 旋转矩阵（MuJoCo 约定）.

    使用标准四元数到旋转矩阵的转换公式：
        R = [[1-2(y²+z²), 2(xy-zw), 2(xz+yw)],
             [2(xy+zw), 1-2(x²+z²), 2(yz-xw)],
             [2(xz-yw), 2(yz+xw), 1-2(x²+y²)]]

    Args:
        quat_wxyz: (4,) 四元数 [w, x, y, z]。

    Returns:
        np.ndarray: (3, 3) 旋转矩阵。
    """
    w, x, y, z = quat_wxyz
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ], dtype=float)


def _apply_geom_transform(
    pts_mesh: np.ndarray,
    geom_pos: np.ndarray,
    geom_quat: np.ndarray,
) -> np.ndarray:
    """
    将 mesh 局部坐标点变换到 parent body 局部坐标系.

    变换关系：
        p_body = R_geom @ p_mesh + pos_geom

    其中 R_geom 由 geom 的 quat 决定（MuJoCo 中 geom pose = body 局部系下的偏移）。

    Args:
        pts_mesh: (N, 3) STL mesh 局部坐标系下的点。
        geom_pos: (3,) geom 在 body 局部系中的位置。
        geom_quat: (4,) geom 在 body 局部系中的四元数 [w,x,y,z]。

    Returns:
        np.ndarray: (N, 3) body 局部坐标系下的点。

    Note:
        使用矩阵乘法实现批量变换，避免 Python 循环。
    """
    R = _quat_to_rot(geom_quat)                      # (3,3)
    pts_body = (R @ pts_mesh.T).T + geom_pos[None, :] # (N,3)
    return pts_body


# ====================== 与 robot_arm_system 的集成示例 ======================

def patch_get_combined_spec_example():
    """
    展示如何在 get_combined_spec() 中集成 touch sensor。
    
    在 robot_arm_system.py 的 get_combined_spec() 函数末尾，
    紧接在 return arm_spec 之前，插入以下代码：

    ```python
    # ── 添加触觉传感器 ──────────────────────────────────────────────────────
    from touch_sensor_builder import add_touch_sensors_to_spec
    
    touch_sensor_map = add_touch_sensors_to_spec(
        spec=arm_spec,
        hand_path=hand_path,        # 已有变量
        prefix="inspirehand_",      # 与 attach_body 时的 prefix 一致
        site_group=4,               # group 4 用于传感器可视化
        site_rgba=(1.0, 0.3, 0.0, 0.5),  # 橙色半透明
    )
    # 将 sensor_map 挂在 spec 上，方便后续使用（可选）
    arm_spec._touch_sensor_map = touch_sensor_map
    # ────────────────────────────────────────────────────────────────────────
    
    return arm_spec, touch_sensor_map
    ```

    编译后访问传感器数据：
    ```python
    model, data = load_combined_model()
    mujoco.mj_step(model, data)

    # 按 skin 名称读取所有传感器
    for skin_name, sensor_names in touch_sensor_map.items():
        for s_name in sensor_names:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, s_name)
            value = data.sensordata[sid]   # 单位：N（法向接触力）
    ```
    """
    pass  # 此函数仅作文档示例


# ====================== 独立测试入口 ======================

if __name__ == "__main__":
    """
    独立测试：加载灵巧手模型，添加传感器后编译并验证.

    测试流程：
        1. 加载灵巧手模型（不含机械臂，快速测试）。
        2. 调用 add_touch_sensors_to_spec 添加所有传感器。
        3. 编译模型，验证无错误。
        4. 统计各 skin 传感器数量。
        5. 执行一步仿真，确认 sensordata 可读。

    运行方式：
        python touch_sensor_builder.py

    预期输出：
        === 触觉传感器独立测试 ===
        手模型路径: /path/to/inspirehand.xml
        [TouchSensor] 开始添加 touch sensor，目标 skin 数量: 15
        [TouchSensor]   → 在 'skin_0_0_p' 上添加 10 * 7 = 70 个 sensor
        ...
        [TouchSensor] 完成！共添加 700 个 touch sensor，覆盖 15 块 skin mesh。
        
        编译验证中...
        编译成功！传感器总数: 700
          skin_0_0_p          :  70 个 sensor
          ...
        仿真步进成功，sensordata shape: (700,)
        === 测试通过 ===
    """
    from pathlib import Path
    import sys

    # 路径需与你的项目结构对应
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    hand_path    = PROJECT_ROOT / "models" / "inspirehand" / "inspirehand.xml"

    print("=== 触觉传感器独立测试 ===")
    print(f"手模型路径: {hand_path}")

    if not hand_path.exists():
        print(f"[错误] 找不到模型文件: {hand_path}")
        sys.exit(1)

    # 仅加载灵巧手（不含机械臂）进行快速测试
    spec = mujoco.MjSpec.from_file(str(hand_path))

    sensor_map = add_touch_sensors_to_spec(
        spec=spec,
        hand_path=hand_path,
        prefix="",   # 独立测试时无 prefix
    )

    # 编译验证
    print("\n编译验证中...")
    model = spec.compile()
    data  = mujoco.MjData(model)
    print(f"编译成功！传感器总数: {model.nsensor}")

    # 统计各 skin 传感器数
    for skin, sensors in sensor_map.items():
        print(f"  {skin:20s}: {len(sensors):3d} 个 sensor")

    # 执行一步仿真，确认 sensordata 可读
    mujoco.mj_step(model, data)
    print(f"\n仿真步进成功，sensordata shape: {data.sensordata.shape}")
    print("=== 测试通过 ===")