"""
机械臂与机械手模型合并工具模块.

该模块提供加载、修正姿态并合并机械臂 (Arm) 与机械手 (Hand) XML 模型的功能。
支持返回未编译的 MjSpec 对象以便进一步定制，或直接返回编译好的 Model/Data 对象。

核心功能：
1. 模型合并：将机械手模型挂载到机械臂指定连接点，自动处理坐标系变换
2. 姿态修正：通过欧拉角调整机械手安装姿态，适配不同抓取需求
3. 根节点修复：自动检测并重置手模型根节点偏移，确保正确附着

设计模式：
- 延迟编译：get_combined_spec 返回未编译 MjSpec 和 TactileReader，
  允许调用者继续添加物体、相机、光照等，最后手动调用 spec.compile() 生成可仿真模型。

使用方法：
    from source.robot.robot_arm_system import get_combined_spec

    # 获取合并后的规格说明和触觉读取器
    spec, reader = get_combined_spec(
        arm_path="path/to/arm.xml",
        hand_path="path/to/hand.xml",
        rot_xyz_deg=(-90, 0, 0),
        tactile_backend="physics",
    )

    # 进一步修改 spec（如添加物体、相机等）
    # ...

    # 编译模型并绑定触觉读取器
    model = spec.compile()
    reader.bind(model)

    # 创建仿真数据对象，开始仿真
    data = mujoco.MjData(model)
    # ...
"""

import traceback
from pathlib import Path
from typing import Optional, Tuple

import mujoco
from mujoco import viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

from source.sensors.tactile_sensor import TactileReader

# ====================== 路径配置 ======================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_ARM_PATH = PROJECT_ROOT / "assets" / "robots" / "rm75b" / "rm75b.xml"
DEFAULT_HAND_PATH = PROJECT_ROOT / "assets" / "grippers" / "dex_hand" / "dex_hand.xml"
DEFAULT_BASE_PATH = PROJECT_ROOT / "assets" / "bases" / "rethink_minimal_mount.xml"

PathLike = str | Path


# ====================== 公开接口 ======================
def get_combined_spec(
    arm_path: Optional[PathLike] = None,
    hand_path: Optional[PathLike] = None,
    base_path: Optional[PathLike] = None,
    rot_xyz_deg: Tuple[float, float, float] = (-90.0, 0.0, 0.0),
    attach_point_name: str = "right_hand",
    tactile_backend: str = "physics",
) -> Tuple[mujoco.MjSpec, "TactileReader"]:
    """
    加载并合并机械臂与机械手模型，返回未编译的 MjSpec 对象.

    Returns:
        Tuple[mujoco.MjSpec, TactileReader]:
            - 已合并的未编译规格对象。
              包含完整的机械臂+手爪结构，可直接编译或进一步修改。
            - 触觉传感器读取器（已 build，未 bind）。
              需在 compile 后调用 reader.bind(model)。

    Examples:
        >>> spec, reader = get_combined_spec()
        >>> model = spec.compile()
        >>> reader.bind(model)
    """
    arm_path = Path(arm_path) if arm_path else DEFAULT_ARM_PATH
    hand_path = Path(hand_path) if hand_path else DEFAULT_HAND_PATH
    base_path = Path(base_path) if base_path else DEFAULT_BASE_PATH

    if not arm_path.exists():
        raise FileNotFoundError(f"机械臂模型文件不存在: {arm_path}")
    if not hand_path.exists():
        raise FileNotFoundError(f"机械手模型文件不存在: {hand_path}")

    arm_spec = mujoco.MjSpec.from_file(str(arm_path))
    hand_spec = mujoco.MjSpec.from_file(str(hand_path))
    hand_height = 0.5  # 手模型的默认高度，确保站在底座上方

    if base_path is not None:
        base_path = Path(base_path)
        if not base_path.exists():
            raise FileNotFoundError(f"底座模型文件不存在: {base_path}")

        base_spec = mujoco.MjSpec.from_file(str(base_path))
        base_root = base_spec.worldbody.first_body()

        if base_root is None:
            raise ValueError("底座 XML 的 worldbody 下没有 body 节点。")

        wf = arm_spec.worldbody.add_frame()
        wf.attach_body(base_root, prefix="mount_", suffix="")
        # 把机械臂根节点抬高，使其站在底座顶部
        arm_root = arm_spec.worldbody.first_body()
        if arm_root is not None:
            current_z = arm_root.pos[2] if arm_root.pos is not None else 0.0
            arm_root.pos = [arm_root.pos[0], arm_root.pos[1], current_z + hand_height]

    hand_root = hand_spec.worldbody.first_body()
    if hand_root is None:
        raise ValueError("手模型 XML 缺少根节点 (worldbody 下无 body)。")

    original_pos = np.array(hand_root.pos)
    if np.linalg.norm(original_pos) > 1e-6:
        hand_root.pos = [0.0, 0.0, 0.0]

    try:
        attach_body = arm_spec.body(attach_point_name)
    except KeyError:
        available = [b.name for b in arm_spec.worldbody.bodies()]
        raise ValueError(
            f"未在机械臂模型中找到挂载点 '{attach_point_name}'。\n"
            f"可用 body 名称: {available}"
        )

    attach_frame = attach_body.add_frame()
    attached_body = attach_frame.attach_body(
        hand_root, prefix="inspirehand_", suffix=""
    )

    attach_frame.pos = [0.0, 0.0, 0.0]
    rotation = R.from_quat([0, 0, 0, 1]) * R.from_euler(
        "xyz", rot_xyz_deg, degrees=True
    )
    q_xyzw = rotation.as_quat()
    attach_frame.quat = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

    arm_spec.option.timestep = 0.001
    arm_spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    arm_spec.option.iterations = 100

    reader = TactileReader.create(tactile_backend)
    reader.build(arm_spec, hand_path, prefix="inspirehand_")

    return arm_spec, reader


def load_combined_model(
    arm_path: Optional[PathLike] = None,
    hand_path: Optional[PathLike] = None,
    base_path: Optional[PathLike] = None,
    rot_xyz_deg: Tuple[float, float, float] = (-90.0, 0.0, 0.0),
    tactile_backend: str = "simple_avg",
) -> Tuple[mujoco.MjModel, mujoco.MjData, "TactileReader"]:
    """
    便捷函数：加载、合并、编译并绑定，返回可直接仿真的三元组.

    Returns:
        Tuple[MjModel, MjData, TactileReader]:
            编译好的模型、仿真数据、已绑定的触觉读取器。
    """
    spec, reader = get_combined_spec(
        arm_path,
        hand_path,
        base_path,
        rot_xyz_deg,
        tactile_backend=tactile_backend,
    )
    # ====================== skybox ======================

    skybox_tex = spec.add_texture()
    skybox_tex.name = "skybox_tex"
    skybox_tex.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    skybox_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    skybox_tex.rgb1 = [0.3, 0.5, 0.7]
    skybox_tex.rgb2 = [0.0, 0.0, 0.0]
    skybox_tex.width = 512
    skybox_tex.height = 3072

    # ====================== ground texture ======================

    ground_tex = spec.add_texture()
    ground_tex.name = "groundplane_tex"
    ground_tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    ground_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    ground_tex.rgb1 = [0.2, 0.3, 0.4]
    ground_tex.rgb2 = [0.1, 0.2, 0.3]
    ground_tex.width = 512
    ground_tex.height = 512

    # ====================== ground material ======================

    ground_mat = spec.add_material()
    ground_mat.name = "groundplane"

    ground_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = ground_tex.name

    ground_mat.texrepeat = [5, 5]
    ground_mat.reflectance = 0.2
    ground_mat.shininess = 0.1
    ground_mat.specular = 0.1

    # ====================== light ======================

    # 主顶光
    spec.worldbody.add_light(
        name="top_light",
        pos=[0.0, 0.0, 4.0],
        dir=[0.0, 0.0, -1.0],
        diffuse=[2, 2, 2],
        ambient=[0.8, 0.8, 0.8],
        specular=[0.3, 0.3, 0.3],
    )

    # ====================== floor ======================

    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0, 0, 0.05]
    floor.material = ground_mat.name
    model = spec.compile()
    reader.bind(model)
    data = mujoco.MjData(model)
    return model, data, reader


# ====================== 独立运行入口 ======================
if __name__ == "__main__":
    """
    模块独立运行入口：可视化预览合成机械臂.

    演示功能：
    1. 加载模型
    2. 启动 MuJoCo 被动查看器
    3. 实时步进仿真，观察默认物理参数效果

    运行方式：
        python -m source.robot.robot_arm_system

    退出方式：
        关闭查看器窗口或按 Ctrl+C。
    """
    print("--- 独立运行模式：预览合成机械臂 ---")
    try:
        model, data, reader = load_combined_model()

        with viewer.launch_passive(model, data) as v:
            print("[Viewer] 查看器已启动，关闭窗口退出...")
            while v.is_running():
                mujoco.mj_step(model, data)
                v.sync()

    except FileNotFoundError as e:
        print(f"\n[错误] 文件未找到: {e}")
    except Exception as e:
        print(f"\n[错误] 发生未知异常: {e}")
        traceback.print_exc()