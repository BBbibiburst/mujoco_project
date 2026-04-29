"""
机械臂与机械手模型合并工具模块.

该模块提供加载、修正姿态并合并机械臂 (Arm) 与机械手 (Hand) XML 模型的功能。
支持返回未编译的 MjSpec 对象以便进一步定制，或直接返回编译好的 Model/Data 对象。
支持对关节物理参数（刚度、阻尼、摩擦等）进行批量配置。

核心功能：
1. 模型合并：将机械手模型挂载到机械臂指定连接点，自动处理坐标系变换
2. 姿态修正：通过欧拉角调整机械手安装姿态，适配不同抓取需求
3. 物理配置：分层级覆盖关节物理参数（全局默认→分组默认→单关节覆盖）
4. 根节点修复：自动检测并重置手模型根节点偏移，确保正确附着

设计模式：
- 使用 dataclass 实现声明式物理参数配置
- 优先级覆盖策略：per_joint_overrides > group_defaults > XML_original
- 延迟编译：get_combined_spec 返回未编译 MjSpec 和 TactileReader，
  允许调用者继续添加物体、相机、光照等，最后手动调用 spec.compile() 生成可仿真模型。
"""

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Union

import mujoco
from mujoco import viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

from src.sensors.tactile_sensor import TactileReader

# ====================== 路径配置 ======================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_ARM_PATH = PROJECT_ROOT / "models" / "rm75b" / "rm75b.xml"
DEFAULT_HAND_PATH = PROJECT_ROOT / "models" / "inspirehand" / "inspirehand.xml"

PathLike = Union[str, Path]


# ====================== 物理参数数据类 ======================
@dataclass
class JointPhysicsConfig:
    """
    单关节物理参数配置容器.
    所有字段默认为 None，表示不覆盖，沿用 XML 原始值。

    物理意义说明：
    stiffness: 弹性刚度 [N·m/rad]。正值产生恢复力矩使关节趋向 ref 角度。
               过大值会导致数值刚性，建议范围 0-1000。
    damping: 粘性阻尼 [N·m·s/rad]。与速度成正比，耗散能量。
             建议值为临界阻尼的 0.1-1 倍。
    frictionloss: 库仑摩擦损耗 [N·m]，近似静摩擦+动摩擦。
                  使关节在低速时产生恒定阻力矩。
    armature: 电机转子惯量（电枢惯量）[kg·m²]。
              增加系统惯性，改善数值稳定性但降低响应速度。
    ref: 弹性目标角度 [rad]。stiffness 不为 0 时的平衡位置。
    range: 关节运动限位 [rad]，格式 (lower, upper)。
           硬件保护参数，超出此范围会产生巨大恢复力矩。

    Attributes:
        stiffness: 弹性刚度，None 表示不修改。
        damping: 粘性阻尼，None 表示不修改。
        frictionloss: 摩擦损耗，None 表示不修改。
        armature: 转子惯量，None 表示不修改。
        ref: 弹性目标角，None 表示不修改。
        range: 关节限位元组，None 表示不修改。

    Examples:
        >>> config = JointPhysicsConfig(damping=100.0, frictionloss=0.1)
        >>> config = JointPhysicsConfig(stiffness=10.0, ref=0.0, range=(-1.57, 1.57))
    """
    stiffness: Optional[float] = None
    damping: Optional[float] = None
    frictionloss: Optional[float] = None
    armature: Optional[float] = None
    ref: Optional[float] = None
    range: Optional[Tuple[float, float]] = None


@dataclass
class PhysicsConfig:
    """
    全局物理参数配置，支持三级优先级覆盖策略.
    优先级（从高到低）：
    1. per_joint_overrides[joint_name]: 特定关节精确覆盖
    2. arm_defaults / hand_defaults: 分组默认配置
    3. XML 原始值: 模型文件中的定义

    这种分层设计允许：
    - 快速设置全系统统一参数（如统一增加阻尼）
    - 区分机械臂和手爪的不同物理特性（臂重阻尼、手轻刚度）
    - 精细调整个别关键关节（如拇指高刚度确保抓取稳定性）

    Attributes:
        arm_defaults: 应用于所有机械臂关节的默认物理参数。
                      通常设置较高阻尼保证运动平稳性。
        hand_defaults: 应用于所有机械手关节的默认物理参数。
                       通常设置较低阻尼保证抓取灵活性。
        per_joint_overrides: 按关节名称精确覆盖的字典。
                             键为关节名（自动处理带/不带 prefix 的查找）。
        geom_friction: 接触几何体摩擦系数三元组 (sliding, torsional, rolling)。
                       sliding: 滑动摩擦系数（通常 0.3-1.0）
                       torsional: 扭转摩擦（通常很小，0.001-0.01）
                       rolling: 滚动摩擦（通常极小，0.0001-0.001）
        geom_condim: 接触维度 (1/3/4/6)。
                     1: 单轴摩擦（简化计算）
                     3: 三维摩擦（标准）
                     4/6: 含扭转/滚动的完整摩擦模型

    Examples:
        >>> cfg = PhysicsConfig(
        ...     arm_defaults=JointPhysicsConfig(damping=50.0),
        ...     hand_defaults=JointPhysicsConfig(damping=2.0),
        ... )
        >>>
        >>> cfg = PhysicsConfig(
        ...     hand_defaults=JointPhysicsConfig(stiffness=1.0),
        ...     per_joint_overrides={
        ...         "thumb_joint": JointPhysicsConfig(stiffness=5.0, damping=1.0),
        ...     },
        ...     geom_friction=(0.8, 0.005, 0.0001),
        ... )
    """
    arm_defaults: JointPhysicsConfig = field(default_factory=JointPhysicsConfig)
    hand_defaults: JointPhysicsConfig = field(default_factory=JointPhysicsConfig)
    per_joint_overrides: Dict[str, JointPhysicsConfig] = field(default_factory=dict)
    geom_friction: Optional[Tuple[float, float, float]] = None
    geom_condim: Optional[int] = None


# ====================== 物理参数配置 ======================
DEFAULT_GRASP_PHYSICS = PhysicsConfig(
    arm_defaults=JointPhysicsConfig(
        damping=100.0,
        frictionloss=0.1,
        armature=0.01,
    ),
    hand_defaults=JointPhysicsConfig(
        damping=1,
        frictionloss=0.01,
        armature=0.1,
    ),
    per_joint_overrides={
        "thumb_rotate_act_push_j": JointPhysicsConfig(damping=10.0),
        "joint1": JointPhysicsConfig(damping=100.0),
        "joint2": JointPhysicsConfig(damping=50.0),
        "joint3": JointPhysicsConfig(damping=10.0),
        "joint4": JointPhysicsConfig(damping=10.0),
        "joint5": JointPhysicsConfig(damping=10.0),
        "joint6": JointPhysicsConfig(damping=5.0),
        "joint7": JointPhysicsConfig(damping=5.0),
    }
)


# ====================== 内部辅助函数 ======================
def _apply_joint_config(joint: mujoco.MjsJoint, cfg: JointPhysicsConfig) -> None:
    """
    将 JointPhysicsConfig 中非 None 的字段写入 MjsJoint 对象（in-place 修改）.
    采用显式字段检查而非循环或反射，确保：
    1. 类型安全：MuJoCo 底层 C 结构对类型敏感
    2. 性能优化：避免动态属性查找开销
    3. 可维护性：新增字段时必须显式处理，防止遗漏

    Args:
        joint: 待修改的 MjsJoint 对象（MjSpec 中的关节引用）。
        cfg: 物理参数配置，仅非 None 字段会被应用。

    Note:
        此函数直接修改输入的 joint 对象，无返回值。
        range 字段从元组转换为列表以适配 MuJoCo API。
    """
    if cfg.stiffness is not None:
        joint.stiffness = cfg.stiffness
    if cfg.damping is not None:
        joint.damping = cfg.damping
    if cfg.frictionloss is not None:
        joint.frictionloss = cfg.frictionloss
    if cfg.armature is not None:
        joint.armature = cfg.armature
    if cfg.ref is not None:
        joint.ref = cfg.ref
    if cfg.range is not None:
        joint.range = list(cfg.range)


def _apply_physics_to_spec(
    spec: mujoco.MjSpec,
    physics: PhysicsConfig,
    arm_root_name: str,
    hand_prefix: str = "inspirehand_",
) -> None:
    """
    将 PhysicsConfig 批量应用到已合并的 MjSpec 上（in-place 修改）.
    使用 spec.joints 直接获取全部关节列表，避免手动递归遍历 body 树。
    使用 spec.geoms 直接获取全部 geom 列表，效率更高。

    应用策略：
    1. 识别关节归属：通过名称前缀判断属于机械臂还是手爪
    2. 分层应用：先应用分组默认（arm_defaults/hand_defaults），
       再检查精确覆盖（per_joint_overrides）
    3. 模糊匹配：同时尝试带 prefix 和不带 prefix 的名称查找，
       提高配置灵活性

    Args:
        spec: 已合并的 MjSpec 对象（包含臂+手）。
        physics: 物理参数配置对象。
        arm_root_name: 机械臂根 body 名称（用于日志输出，当前未实际使用）。
        hand_prefix: 手爪关节名称前缀，用于识别手爪关节（默认 "inspirehand_"）。

    Note:
        此函数直接修改输入的 spec 对象，无返回值。
        修改会立即反映在所有引用该 spec 的地方。
    """
    for joint in spec.joints:
        name: str = joint.name or ""
        is_hand = name.startswith(hand_prefix)

        base_cfg = physics.hand_defaults if is_hand else physics.arm_defaults
        _apply_joint_config(joint, base_cfg)

        bare_name = name[len(hand_prefix):] if is_hand else name
        for lookup in (name, bare_name):
            if lookup in physics.per_joint_overrides:
                _apply_joint_config(joint, physics.per_joint_overrides[lookup])
                break

    if physics.geom_friction is not None or physics.geom_condim is not None:
        for geom in spec.geoms:
            if physics.geom_friction is not None:
                geom.friction = list(physics.geom_friction)
            if physics.geom_condim is not None:
                geom.condim = physics.geom_condim


# ====================== 公开接口 ======================
def get_combined_spec(
    arm_path: Optional[PathLike] = None,
    hand_path: Optional[PathLike] = None,
    rot_xyz_deg: Tuple[float, float, float] = (-90.0, 0.0, 0.0),
    attach_point_name: str = "right_hand",
    physics: Optional[PhysicsConfig] = None,
    tactile_backend: str = "physics",
) -> Tuple[mujoco.MjSpec, "TactileReader"]:
    """
    加载并合并机械臂与机械手模型，返回未编译的 MjSpec 对象.

    Returns:
        Tuple[mujoco.MjSpec, TactileReader]:
            - 已合并、可选已修改物理参数的未编译规格对象。
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

    if not arm_path.exists():
        raise FileNotFoundError(f"机械臂模型文件不存在: {arm_path}")
    if not hand_path.exists():
        raise FileNotFoundError(f"机械手模型文件不存在: {hand_path}")

    arm_spec = mujoco.MjSpec.from_file(str(arm_path))
    hand_spec = mujoco.MjSpec.from_file(str(hand_path))

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
    attached_body = attach_frame.attach_body(hand_root, prefix="inspirehand_", suffix="")

    attach_frame.pos = [0.0, 0.0, 0.0]
    rotation = R.from_quat([0, 0, 0, 1]) * R.from_euler("xyz", rot_xyz_deg, degrees=True)
    q_xyzw = rotation.as_quat()
    attach_frame.quat = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

    if physics is not None:
        _apply_physics_to_spec(arm_spec, physics, arm_root_name=attach_point_name)
    else:
        _apply_physics_to_spec(arm_spec, DEFAULT_GRASP_PHYSICS, arm_root_name=attach_point_name)

    arm_spec.option.timestep = 0.001
    arm_spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    arm_spec.option.iterations = 100

    reader = TactileReader.create(tactile_backend)
    reader.build(arm_spec, hand_path, prefix="inspirehand_")

    return arm_spec, reader


def load_combined_model(
    arm_path: Optional[PathLike] = None,
    hand_path: Optional[PathLike] = None,
    rot_xyz_deg: Tuple[float, float, float] = (-90.0, 0.0, 0.0),
    physics: Optional[PhysicsConfig] = None,
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
        rot_xyz_deg,
        physics=physics,
        tactile_backend=tactile_backend,
    )
    model = spec.compile()
    reader.bind(model)
    data = mujoco.MjData(model)
    return model, data, reader


# ====================== 独立运行入口 ======================
if __name__ == "__main__":
    """
    模块独立运行入口：可视化预览合成机械臂.

    演示功能：
    1. 使用示例物理配置加载模型
    2. 启动 MuJoCo 被动查看器
    3. 实时步进仿真，观察默认物理参数效果

    运行方式：
        python robot_arm_system.py

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
        print("请检查 'models/' 目录结构是否正确，确保包含 rm75b/ 和 inspirehand/ 子目录")
    except Exception as e:
        print(f"\n[错误] 发生未知异常: {e}")
        traceback.print_exc()