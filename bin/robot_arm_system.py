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
    - 延迟编译：get_combined_spec 返回未编译 MjSpec，支持链式修改
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
from touch_sensor_builder import add_touch_sensors_to_spec

# ====================== 路径配置 ======================

# 项目根目录推断：假设本文件位于 <project>/src/robot_arm_system.py
# 则 PROJECT_ROOT 指向 <project>/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 默认模型路径配置（相对于项目根目录）
DEFAULT_ARM_PATH  = PROJECT_ROOT / "models" / "rm75b"       / "rm75b.xml"      # RM75B 7-DOF 机械臂
DEFAULT_HAND_PATH = PROJECT_ROOT / "models" / "inspirehand" / "inspirehand.xml"  # Inspire 灵巧手

# 类型别名：支持字符串或 Path 对象的路径参数
PathLike = Union[str, Path]


# ====================== 物理参数数据类 ======================

@dataclass
class JointPhysicsConfig:
    """
    单关节物理参数配置容器.

    采用 dataclass 实现不可变配置对象，所有字段默认为 None。
    None 表示"不覆盖，沿用 XML 原始值"，
    因此可以只指定需要修改的字段，其余字段保持原样。

    物理意义说明：
        stiffness:    弹性刚度 [N·m/rad]。正值产生恢复力矩使关节趋向 ref 角度。
                      过大值会导致数值刚性，建议范围 0-1000。
        damping:      粘性阻尼 [N·m·s/rad]。与速度成正比，耗散能量。
                      建议值为临界阻尼的 0.1-1 倍。
        frictionloss: 库仑摩擦损耗 [N·m]，近似静摩擦+动摩擦。
                      使关节在低速时产生恒定阻力矩。
        armature:     电机转子惯量（电枢惯量）[kg·m²]。
                      增加系统惯性，改善数值稳定性但降低响应速度。
        ref:          弹性目标角度 [rad]。stiffness 不为 0 时的平衡位置。
        range:        关节运动限位 [rad]，格式 (lower, upper)。
                      硬件保护参数，超出此范围会产生巨大恢复力矩。

    Attributes:
        stiffness:    弹性刚度，None 表示不修改。
        damping:      粘性阻尼，None 表示不修改。
        frictionloss: 摩擦损耗，None 表示不修改。
        armature:     转子惯量，None 表示不修改。
        ref:          弹性目标角，None 表示不修改。
        range:        关节限位元组，None 表示不修改。
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
    全局物理参数配置，支持三级优先级覆盖策略.

    优先级（从高到低）：
        1. per_joint_overrides[joint_name]: 特定关节精确覆盖
        2. arm_defaults / hand_defaults:    分组默认配置
        3. XML 原始值:                       模型文件中的定义

    这种分层设计允许：
        - 快速设置全系统统一参数（如统一增加阻尼）
        - 区分机械臂和手爪的不同物理特性（臂重阻尼、手轻刚度）
        - 精细调整个别关键关节（如拇指高刚度确保抓取稳定性）

    Attributes:
        arm_defaults:        应用于所有机械臂关节的默认物理参数。
                             通常设置较高阻尼保证运动平稳性。
        hand_defaults:       应用于所有机械手关节的默认物理参数。
                             通常设置较低阻尼保证抓取灵活性。
        per_joint_overrides: 按关节名称精确覆盖的字典。
                             键为关节名（自动处理带/不带 prefix 的查找）。
        geom_friction:       接触几何体摩擦系数三元组 (sliding, torsional, rolling)。
                             sliding: 滑动摩擦系数（通常 0.3-1.0）
                             torsional: 扭转摩擦（通常很小，0.001-0.01）
                             rolling: 滚动摩擦（通常极小，0.0001-0.001）
        geom_condim:         接触维度 (1/3/4/6)。
                             1: 单轴摩擦（简化计算）
                             3: 三维摩擦（标准）
                             4/6: 含扭转/滚动的完整摩擦模型
    """
    arm_defaults:        JointPhysicsConfig            = field(default_factory=JointPhysicsConfig)
    hand_defaults:       JointPhysicsConfig            = field(default_factory=JointPhysicsConfig)
    per_joint_overrides: Dict[str, JointPhysicsConfig] = field(default_factory=dict)
    geom_friction:       Optional[Tuple[float, float, float]] = None
    geom_condim:         Optional[int] = None


# ====================== 物理参数配置 ======================

DEFAULT_GRASP_PHYSICS = PhysicsConfig(
    # 机械臂默认物理参数：较高的阻尼确保运动平稳
    arm_defaults=JointPhysicsConfig(
        damping=100.0,        # 关节阻尼系数，抑制振荡
        frictionloss=0.1,      # 摩擦损耗，模拟关节摩擦
        armature=0.01,       # 电机惯量，影响动态响应
    ),
    # 机械手默认物理参数：低阻尼以实现灵活抓取
    hand_defaults=JointPhysicsConfig(
        damping=1,        # 手指关节低阻尼，保证灵活性
        frictionloss=0.01,   # 手指摩擦系数
        armature=0.1,       # 手指电机惯量
    ),
    # 特定关节参数覆盖：拇指旋转关节需要更高阻尼以保持稳定性
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
        cfg:   物理参数配置，仅非 None 字段会被应用。
    """
    # 按 MuJoCo 文档标准顺序应用参数
    if cfg.stiffness    is not None: joint.stiffness    = cfg.stiffness
    if cfg.damping      is not None: joint.damping      = cfg.damping
    if cfg.frictionloss is not None: joint.frictionloss = cfg.frictionloss
    if cfg.armature     is not None: joint.armature     = cfg.armature
    if cfg.ref          is not None: joint.ref          = cfg.ref
    if cfg.range        is not None: joint.range        = list(cfg.range)  # 元组转列表适配 MuJoCo API



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
        spec:          已合并的 MjSpec 对象（包含臂+手）。
        physics:       物理参数配置对象。
        arm_root_name: 机械臂根 body 名称（用于日志输出，当前未实际使用）。
        hand_prefix:   手爪关节名称前缀，用于识别手爪关节（默认 "inspirehand_"）。
    """
    # ----- 1. 遍历所有关节应用物理参数 -----
    # spec.joints 返回模型内全部 MjsJoint 的列表（包含臂和手）
    for joint in spec.joints:
        name: str = joint.name or ""
        # 通过前缀判断关节归属（简单但有效的启发式规则）
        is_hand = name.startswith(hand_prefix)

        # 应用分组默认值（机械臂或手爪）
        base_cfg = physics.hand_defaults if is_hand else physics.arm_defaults
        _apply_joint_config(joint, base_cfg)

        # 应用精细覆盖（同时尝试带 prefix 和不带 prefix 的名称）
        # 例如配置 "finger1" 可以匹配 "inspirehand_finger1"
        bare_name = name[len(hand_prefix):] if is_hand else name
        for lookup in (name, bare_name):
            if lookup in physics.per_joint_overrides:
                _apply_joint_config(joint, physics.per_joint_overrides[lookup])
                break
    # ----- 2. 遍历所有几何体应用接触参数 -----
    # 可选：修改全局接触摩擦属性
    if physics.geom_friction is not None or physics.geom_condim is not None:
        for geom in spec.geoms:
            if physics.geom_friction is not None:
                # friction 需要 list 类型，元组会被拒绝
                geom.friction = list(physics.geom_friction)
            if physics.geom_condim is not None:
                geom.condim = physics.geom_condim

    # 日志输出：确认应用情况，便于调试
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

    这是核心构建函数，执行完整的模型合并流程：
        1. 路径解析与文件验证
        2. XML 加载为 MjSpec
        3. 手模型根节点偏移修复
        4. 寻找挂载点并 attach
        5. 应用姿态变换（欧拉角转四元数）
        6. 可选：应用物理参数覆盖

    返回未编译的 MjSpec，允许调用者继续添加物体、相机、光照等，
    最后手动调用 spec.compile() 生成可仿真模型。

    Args:
        arm_path:          机械臂 XML 文件路径。None → 使用 DEFAULT_ARM_PATH。
        hand_path:         机械手 XML 文件路径。None → 使用 DEFAULT_HAND_PATH。
        rot_xyz_deg:       手部安装姿态修正欧拉角 (roll, pitch, yaw) [度]。
                           默认 (-90, 0, 0) 表示绕 X 轴旋转 -90 度（常见手爪朝前配置）。
                           按 XYZ 内旋顺序（extrinsic rotations）应用。
        attach_point_name: 机械臂中用于挂载机械手的 body 名称。
                           必须是 arm_spec 中存在的 body 名称。
        physics:           物理参数配置对象。None → 保留 XML 原始值，不做任何修改。
                           传入 PhysicsConfig() 空对象同样不做修改（所有字段为 None）。

    Returns:
        mujoco.MjSpec: 已合并、可选已修改物理参数的未编译规格对象。
                       包含完整的机械臂+手爪结构，可直接编译或进一步修改。

    Raises:
        FileNotFoundError: 模型文件不存在（路径错误或文件缺失）。
        ValueError:        手模型缺少根节点，或找不到指定的挂载点 body。

    Examples:
        >>> # 基础用法：使用默认路径和姿态
        >>> spec = get_combined_spec()
        >>> model = spec.compile()
        
        >>> # 高级用法：自定义物理参数和安装姿态
        >>> cfg = PhysicsConfig(
        ...     hand_defaults=JointPhysicsConfig(stiffness=2.0, damping=0.5),
        ...     per_joint_overrides={
        ...         "inspirehand_finger1_joint1": JointPhysicsConfig(stiffness=5.0)
        ...     },
        ... )
        >>> spec = get_combined_spec(
        ...     rot_xyz_deg=(0, -90, 0),  # 手爪朝下
        ...     physics=cfg
        ... )
        >>> # 继续添加环境物体...
        >>> spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1])
        >>> model = spec.compile()
    """
    # ----- 路径解析与文件检查 -----
    # 使用 Path 对象统一处理字符串和 Path 输入，自动适配操作系统路径分隔符
    arm_path  = Path(arm_path)  if arm_path  else DEFAULT_ARM_PATH
    hand_path = Path(hand_path) if hand_path else DEFAULT_HAND_PATH

    # 提前检查文件存在性，避免 MuJoCo 底层抛出难以理解的错误
    if not arm_path.exists():
        raise FileNotFoundError(f"机械臂模型文件不存在: {arm_path}")
    if not hand_path.exists():
        raise FileNotFoundError(f"机械手模型文件不存在: {hand_path}")

    print(f"[SpecBuilder] 加载机械臂: {arm_path.name}")
    print(f"[SpecBuilder] 加载机械手: {hand_path.name}")

    # ----- 加载 MjSpec -----
    # from_file 是 MjSpec 的静态工厂方法，解析 XML 但不编译
    arm_spec  = mujoco.MjSpec.from_file(str(arm_path))
    hand_spec = mujoco.MjSpec.from_file(str(hand_path))

    # ----- 修复手模型根节点偏移 -----
    # 某些手模型 XML 根 body 带有非零位置，会导致 attach 后位置错误
    # 规范做法：根 body 应位于原点，attach 时通过 frame 控制相对位姿
    hand_root = hand_spec.worldbody.first_body()
    if hand_root is None:
        raise ValueError("手模型 XML 缺少根节点 (worldbody 下无 body)。")

    original_pos = np.array(hand_root.pos)
    if np.linalg.norm(original_pos) > 1e-6:  # 检测到显著偏移（>1微米）
        print(f"[SpecBuilder] 检测到根节点偏移 {original_pos}，已重置为 [0, 0, 0]")
        hand_root.pos = [0.0, 0.0, 0.0]

    # ----- 寻找挂载点 -----
    # attach_point_name 必须是机械臂模型中已存在的 body 名称
    try:
        attach_body = arm_spec.body(attach_point_name)
    except KeyError:
        # 挂载点不存在时，列出所有可用 body 帮助用户排查
        available = [b.name for b in arm_spec.worldbody.bodies()]
        raise ValueError(
            f"未在机械臂模型中找到挂载点 '{attach_point_name}'。\n"
            f"可用 body 名称: {available}"
        )

    # ----- 挂载机械手 -----
    # 1. 在挂载点 body 下创建 frame（局部坐标系）
    attach_frame = attach_body.add_frame()
    # 2. 将手模型根 body attach 到该 frame，自动添加前缀避免命名冲突
    attached_body = attach_frame.attach_body(hand_root, prefix="inspirehand_", suffix="")
    print(f"[SpecBuilder] 成功挂载: '{attached_body.name}' → '{attach_body.name}'")

    # ----- 设置位姿变换（欧拉角 → 四元数）-----
    # 使用 scipy 进行旋转合成：先单位旋转，再应用指定欧拉角
    # MuJoCo 使用 [w, x, y, z] 四元数格式，scipy 输出 [x, y, z, w]，需要转换
    attach_frame.pos = [0.0, 0.0, 0.0]  # 无位置偏移，纯旋转
    # 旋转合成：R = R_initial * R_euler，默认 R_initial 为单位旋转
    q_xyzw = (R.from_quat([0, 0, 0, 1]) * R.from_euler("xyz", rot_xyz_deg, degrees=True)).as_quat()
    # 格式转换：scipy [x,y,z,w] → MuJoCo [w,x,y,z]
    attach_frame.quat = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
    print(f"[SpecBuilder] 姿态设定完成: Euler={rot_xyz_deg} deg")

    # ----- 应用物理参数（可选）-----
    if physics is not None:
        _apply_physics_to_spec(arm_spec, physics, arm_root_name=attach_point_name)
    else:
        _apply_physics_to_spec(arm_spec, DEFAULT_GRASP_PHYSICS, arm_root_name=attach_point_name)
    
    arm_spec.option.timestep = 0.001  # 适配机械臂的高频控制
    arm_spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON 
    arm_spec.option.iterations = 100  # 增加迭代次数以处理灵巧手的复杂约束
    # ── 添加触觉传感器（touch sensors on skin meshes）────────────────────────
    # 为 skin_0_0_p ~ skin_4_2_p 共 15 块 skin 在曲面上布置 touch sensor
    # 底部指节 [10,7]=70个, 中部指节 [8,5]=40个, 顶部指节 [6,5]=30个
    # 合计每根手指 140 个，4 根手指 + 拇指共 700 个 touch sensor
    from touch_sensor_builder import add_touch_sensors_to_spec
    
    touch_sensor_map = add_touch_sensors_to_spec(
        spec=arm_spec,
        hand_path=hand_path,           # get_combined_spec 已有此变量
        prefix="inspirehand_",         # 与 attach_body 时的 prefix 一致
        site_group=4,                  # group=4，可在 viewer 中单独显隐
        site_rgba=(1.0, 0.35, 0.0, 0.5),  # 橙色半透明，便于调试可视化
    )
    return arm_spec


def load_combined_model(
    arm_path: Optional[PathLike] = None,
    hand_path: Optional[PathLike] = None,
    rot_xyz_deg: Tuple[float, float, float] = (-90.0, 0.0, 0.0),
    physics: Optional[PhysicsConfig] = None,
) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    便捷函数：加载、合并并直接编译模型，返回可立即仿真的对象对.

    这是 get_combined_spec 的封装，适用于无需进一步修改模型的场景。
    内部调用 get_combined_spec → compile → MjData 创建。

    Args:
        arm_path:    机械臂 XML 路径。None → 使用默认路径。
        hand_path:   机械手 XML 路径。None → 使用默认路径。
        rot_xyz_deg: 姿态修正欧拉角 (roll, pitch, yaw) [度]。
        physics:     物理参数配置。None → 保留 XML 原始值。

    Returns:
        Tuple[mujoco.MjModel, mujoco.MjData]: 
            - model: 编译好的 MuJoCo 模型，包含完整动力学信息
            - data:  对应的仿真数据对象，已初始化但尚未步进

    Examples:
        >>> # 使用默认物理参数快速启动
        >>> model, data = load_combined_model()
        
        >>> # 自定义物理参数：统一调高全臂阻尼，并单独加强拇指刚度
        >>> cfg = PhysicsConfig(
        ...     arm_defaults=JointPhysicsConfig(damping=1.0, armature=0.01),
        ...     hand_defaults=JointPhysicsConfig(stiffness=1.5, damping=0.3, frictionloss=0.05),
        ...     per_joint_overrides={
        ...         "inspirehand_thumb_proximal": JointPhysicsConfig(stiffness=8.0, damping=0.8),
        ...     },
        ...     geom_friction=(0.8, 0.005, 0.0001),  # 高滑动摩擦，低扭转/滚动摩擦
        ... )
        >>> model, data = load_combined_model(physics=cfg)
        >>> # 现在可以开始仿真
        >>> mujoco.mj_step(model, data)
    """
    # 获取合并后的 spec（未编译）
    spec = get_combined_spec(arm_path, hand_path, rot_xyz_deg, physics=physics)

    # 编译生成可仿真模型
    print("[SpecBuilder] 正在编译模型...")
    model = spec.compile()
    # 创建对应的仿真数据对象
    data  = mujoco.MjData(model)

    # 日志输出：确认模型结构
    print(f"[SpecBuilder] 编译完成。自由度 (nv): {model.nv}, 执行器 (nu): {model.nu}")
    print(f"[SpecBuilder] 可用执行器列表: {[model.actuator(i).name for i in range(model.nu)]}")
    return model, data


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
    """
    print("--- 独立运行模式：预览合成机械臂 ---")
    try:
        # 加载并编译模型
        model, data = load_combined_model()

        # 启动被动查看器（非阻塞，允许外部控制循环）
        with viewer.launch_passive(model, data) as v:
            print("[Viewer] 查看器已启动，关闭窗口退出...")
            while v.is_running():
                # 单步物理仿真
                mujoco.mj_step(model, data)
                # 同步查看器显示
                v.sync()

    except FileNotFoundError as e:
        # 模型文件缺失（常见首次运行错误）
        print(f"\n[错误] 文件未找到: {e}")
        print("请检查 'models/' 目录结构是否正确，确保包含 rm75b/ 和 inspirehand/ 子目录")
    except Exception as e:
        # 捕获所有其他异常，打印完整堆栈
        print(f"\n[错误] 发生未知异常: {e}")
        traceback.print_exc()