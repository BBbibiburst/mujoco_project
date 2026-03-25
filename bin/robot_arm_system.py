"""
机械臂与机械手模型合并工具模块.

该模块提供加载、修正姿态并合并机械臂 (Arm) 与机械手 (Hand) XML 模型的功能。
支持返回未编译的 MjSpec 对象以便进一步定制，或直接返回编译好的 Model/Data 对象。
支持对关节物理参数（刚度、阻尼、摩擦等）进行批量配置。
"""

import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union

import mujoco
from mujoco import viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

# ====================== 路径配置 ======================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARM_PATH  = PROJECT_ROOT / "models" / "rm75b"      / "rm75b.xml"
DEFAULT_HAND_PATH = PROJECT_ROOT / "models" / "inspirehand" / "inspirehand.xml"

PathLike = Union[str, Path]


# ====================== 物理参数数据类 ======================

@dataclass
class JointPhysicsConfig:
    """
    单关节物理参数配置.

    所有字段均带默认值（None 表示"不覆盖，沿用 XML 原始值"），
    因此可以只指定需要修改的字段，其余字段保持原样。

    Attributes:
        stiffness:  弹性刚度 [N·m/rad]。正值使关节趋向 ref 角度。
        damping:    粘性阻尼 [N·m·s/rad]。
        frictionloss: 关节摩擦损耗 [N·m]，库仑摩擦近似。
        armature:   转子惯量 / 电枢惯量 [kg·m²]。改善数值稳定性。
        ref:        弹性目标角度 [rad]（stiffness 不为 0 时生效）。
        range:      关节限位 [rad]，格式 (lower, upper)。None 表示不修改。
    """
    stiffness:    Optional[float] = None
    damping:      Optional[float] = None
    frictionloss: Optional[float] = None
    armature:     Optional[float] = None
    ref:          Optional[float] = None
    range:        Optional[Tuple[float, float]] = None


@dataclass
class PhysicsConfig:
    """
    全局物理参数配置，可按关节名称精细覆盖.

    优先级（从高到低）：
      per_joint_overrides[joint_name]  >  arm_defaults / hand_defaults  >  XML 原值

    Attributes:
        arm_defaults:          应用于所有机械臂关节的默认值。
        hand_defaults:         应用于所有机械手关节的默认值。
        per_joint_overrides:   按关节名称精细覆盖，键为关节名（不含 prefix/suffix）。
        geom_friction:         接触几何摩擦系数三元组 (sliding, torsional, rolling)。
                               None 表示不修改。
        geom_condim:           接触维度 (1/3/4/6)。None 表示不修改。
    """
    arm_defaults:        JointPhysicsConfig            = field(default_factory=JointPhysicsConfig)
    hand_defaults:       JointPhysicsConfig            = field(default_factory=JointPhysicsConfig)
    per_joint_overrides: Dict[str, JointPhysicsConfig] = field(default_factory=dict)
    geom_friction:       Optional[Tuple[float, float, float]] = None
    geom_condim:         Optional[int] = None


# ====================== 内部辅助函数 ======================

def _apply_joint_config(joint: mujoco.MjsJoint, cfg: JointPhysicsConfig) -> None:
    """将 JointPhysicsConfig 中非 None 的字段写入 MjsJoint 对象."""
    if cfg.stiffness    is not None: joint.stiffness    = cfg.stiffness
    if cfg.damping      is not None: joint.damping      = cfg.damping
    if cfg.frictionloss is not None: joint.frictionloss = cfg.frictionloss
    if cfg.armature     is not None: joint.armature     = cfg.armature
    if cfg.ref          is not None: joint.ref          = cfg.ref
    if cfg.range        is not None: joint.range        = list(cfg.range)



def _apply_physics_to_spec(
    spec: mujoco.MjSpec,
    physics: PhysicsConfig,
    arm_root_name: str,
    hand_prefix: str = "inspirehand_",
) -> None:
    """
    将 PhysicsConfig 批量应用到已合并的 MjSpec 上.

    使用 spec.joints 直接获取全部关节列表，避免手动链式遍历。
    使用 spec.geoms 直接获取全部 geom 列表。
    """
    # 1. 遍历所有关节（spec.joints 返回模型内全部 MjsJoint 的列表）
    for joint in spec.joints:
        name: str = joint.name or ""
        is_hand = name.startswith(hand_prefix)

        # 应用分组默认值
        base_cfg = physics.hand_defaults if is_hand else physics.arm_defaults
        _apply_joint_config(joint, base_cfg)

        # 应用精细覆盖（同时尝试带 prefix 和不带 prefix 的名称）
        bare_name = name[len(hand_prefix):] if is_hand else name
        for lookup in (name, bare_name):
            if lookup in physics.per_joint_overrides:
                _apply_joint_config(joint, physics.per_joint_overrides[lookup])
                break

    # 2. 遍历所有 geom（spec.geoms 返回模型内全部 MjsGeom 的列表）
    if physics.geom_friction is not None or physics.geom_condim is not None:
        for geom in spec.geoms:
            if physics.geom_friction is not None:
                geom.friction = list(physics.geom_friction)
            if physics.geom_condim is not None:
                geom.condim = physics.geom_condim

    print(f"[SpecBuilder] 物理参数已应用 (arm_defaults={physics.arm_defaults}, "
          f"hand_defaults={physics.hand_defaults}, "
          f"overrides={list(physics.per_joint_overrides.keys())})")

# ====================== 公开接口 ======================

def get_combined_spec(
    arm_path: Optional[PathLike] = None,
    hand_path: Optional[PathLike] = None,
    rot_xyz_deg: Tuple[float, float, float] = (-90.0, 0.0, 0.0),
    attach_point_name: str = "right_hand",
    physics: Optional[PhysicsConfig] = None,
) -> mujoco.MjSpec:
    """
    加载并合并机械臂与机械手模型，返回未编译的 MjSpec 对象.

    调用者可在返回的 spec 上继续修改，最后自行调用 .compile()。

    Args:
        arm_path:          机械臂 XML 路径。None → 使用默认路径。
        hand_path:         机械手 XML 路径。None → 使用默认路径。
        rot_xyz_deg:       手部安装姿态修正欧拉角 (roll, pitch, yaw) [deg]。
        attach_point_name: 机械臂中用于挂载机械手的 body 名称。
        physics:           物理参数配置对象。None → 保留 XML 原始值，不做任何修改。
                           传入 PhysicsConfig() 空对象同样不做修改（所有字段为 None）。

    Returns:
        mujoco.MjSpec: 已合并、可选已修改物理参数的未编译规格对象。

    Raises:
        FileNotFoundError: 模型文件不存在。
        ValueError:        手模型缺少根节点或找不到挂载点。

    Examples:
        # 仅调高机械手阻尼，其余保持原样
        cfg = PhysicsConfig(
            hand_defaults=JointPhysicsConfig(stiffness=2.0, damping=0.5),
            per_joint_overrides={
                "inspirehand_finger1_joint1": JointPhysicsConfig(stiffness=5.0)
            },
        )
        spec = get_combined_spec(physics=cfg)
        model = spec.compile()
    """
    # ---- 路径解析与文件检查 ----
    arm_path  = Path(arm_path)  if arm_path  else DEFAULT_ARM_PATH
    hand_path = Path(hand_path) if hand_path else DEFAULT_HAND_PATH

    if not arm_path.exists():
        raise FileNotFoundError(f"机械臂模型文件不存在: {arm_path}")
    if not hand_path.exists():
        raise FileNotFoundError(f"机械手模型文件不存在: {hand_path}")

    print(f"[SpecBuilder] 加载机械臂: {arm_path.name}")
    print(f"[SpecBuilder] 加载机械手: {hand_path.name}")

    # ---- 加载 Spec ----
    arm_spec  = mujoco.MjSpec.from_file(str(arm_path))
    hand_spec = mujoco.MjSpec.from_file(str(hand_path))

    # ---- 修复手模型根节点偏移 ----
    hand_root = hand_spec.worldbody.first_body()
    if hand_root is None:
        raise ValueError("手模型 XML 缺少根节点 (worldbody 下无 body)。")

    original_pos = np.array(hand_root.pos)
    if np.linalg.norm(original_pos) > 1e-6:
        print(f"[SpecBuilder] 检测到根节点偏移 {original_pos}，已重置为 [0, 0, 0]")
        hand_root.pos = [0.0, 0.0, 0.0]

    # ---- 寻找挂载点 ----
    try:
        attach_body = arm_spec.body(attach_point_name)
    except KeyError:
        available = [b.name for b in arm_spec.worldbody.bodies()]
        raise ValueError(
            f"未在机械臂模型中找到挂载点 '{attach_point_name}'。\n"
            f"可用 body 名称: {available}"
        )

    # ---- 挂载机械手 ----
    attach_frame = attach_body.add_frame()
    attached_body = attach_frame.attach_body(hand_root, prefix="inspirehand_", suffix="")
    print(f"[SpecBuilder] 成功挂载: '{attached_body.name}' → '{attach_body.name}'")

    # ---- 设置位姿变换 ----
    attach_frame.pos = [0.0, 0.0, 0.0]
    q_xyzw = (R.from_quat([0, 0, 0, 1]) * R.from_euler("xyz", rot_xyz_deg, degrees=True)).as_quat()
    attach_frame.quat = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
    print(f"[SpecBuilder] 姿态设定完成: Euler={rot_xyz_deg} deg")

    # ---- 应用物理参数（可选）----
    if physics is not None:
        _apply_physics_to_spec(arm_spec, physics, arm_root_name=attach_point_name)

    return arm_spec


def load_combined_model(
    arm_path: Optional[PathLike] = None,
    hand_path: Optional[PathLike] = None,
    rot_xyz_deg: Tuple[float, float, float] = (-90.0, 0.0, 0.0),
    physics: Optional[PhysicsConfig] = None,
) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    便捷函数：加载、合并并直接编译模型.

    Args:
        arm_path:    机械臂路径。
        hand_path:   机械手路径。
        rot_xyz_deg: 姿态修正欧拉角。
        physics:     物理参数配置。None → 保留 XML 原始值。

    Returns:
        (model, data): 编译好的 MuJoCo 模型与数据对象。

    Examples:
        # 使用默认物理参数
        model, data = load_combined_model()

        # 统一调高全臂阻尼，并单独加强某个手指刚度
        cfg = PhysicsConfig(
            arm_defaults=JointPhysicsConfig(damping=1.0, armature=0.01),
            hand_defaults=JointPhysicsConfig(stiffness=1.5, damping=0.3, frictionloss=0.05),
            per_joint_overrides={
                "inspirehand_thumb_proximal": JointPhysicsConfig(stiffness=8.0, damping=0.8),
            },
            geom_friction=(0.8, 0.005, 0.0001),
        )
        model, data = load_combined_model(physics=cfg)
    """
    spec = get_combined_spec(arm_path, hand_path, rot_xyz_deg, physics=physics)

    print("[SpecBuilder] 正在编译模型...")
    model = spec.compile()
    data  = mujoco.MjData(model)

    print(f"[SpecBuilder] 编译完成。自由度 (dof): {model.nv}, 执行器: {model.nu}")
    print(f"[SpecBuilder] 可用执行器: {[model.actuator(i).name for i in range(model.nu)]}")
    return model, data


# ====================== 独立运行入口 ======================

if __name__ == "__main__":
    print("--- 独立运行模式：预览合成机械臂 ---")

    # 示例：调高机械手刚度和阻尼，其余保持默认
    demo_physics = PhysicsConfig(
        arm_defaults=JointPhysicsConfig(
            damping=0.8,
            armature=0.005,
        ),
        hand_defaults=JointPhysicsConfig(
            stiffness=2.0,
            damping=0.4,
            frictionloss=0.02,
        ),
        geom_friction=(0.7, 0.005, 0.0001),
    )

    try:
        model, data = load_combined_model(physics=demo_physics)

        with viewer.launch_passive(model, data) as v:
            while v.is_running():
                mujoco.mj_step(model, data)
                v.sync()

    except FileNotFoundError as e:
        print(f"\n[错误] 文件未找到: {e}")
        print("请检查 'models/' 目录结构是否正确。")
    except Exception as e:
        print(f"\n[错误] 发生未知异常: {e}")
        traceback.print_exc()