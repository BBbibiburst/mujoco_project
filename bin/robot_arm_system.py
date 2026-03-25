"""
机械臂与机械手模型合并工具模块.

该模块提供加载、修正姿态并合并机械臂 (Arm) 与机械手 (Hand) XML 模型的功能。
支持返回未编译的 MjSpec 对象以便进一步定制，或直接返回编译好的 Model/Data 对象。
"""

import os
import traceback
from pathlib import Path
from typing import Optional, Tuple, List, Union

import mujoco
from mujoco import viewer, mjtGeom
import numpy as np
from scipy.spatial.transform import Rotation as R

# ====================== 路径配置 ======================
# 获取当前文件所在目录的上一级作为项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 默认模型路径 (相对于项目根目录)
DEFAULT_ARM_PATH = PROJECT_ROOT / "models" / "rm75b" / "rm75b.xml"
DEFAULT_HAND_PATH = PROJECT_ROOT / "models" / "inspirehand" / "inspirehand.xml"

# 类型别名
PathLike = Union[str, Path]


def get_combined_spec(
    arm_path: Optional[PathLike] = None,
    hand_path: Optional[PathLike] = None,
    rot_xyz_deg: Tuple[float, float, float] = (-90.0, 0.0, 0.0),
    attach_point_name: str = "right_hand",
) -> mujoco.MjSpec:
    """
    加载并合并机械臂与机械手模型，返回未编译的 MjSpec 对象.

    调用者可以在返回的 spec 对象上继续添加物体、传感器或修改物理属性，
    最后自行调用 .compile() 完成编译。

    Args:
        arm_path: 机械臂 XML 文件路径。若为 None，使用默认路径。
        hand_path: 机械手 XML 文件路径。若为 None，使用默认路径。
        rot_xyz_deg: 手部安装姿态修正欧拉角 (度)，格式为 (roll, pitch, yaw)。
                     默认 (-90, 0, 0) 通常用于将手指朝向从 Z 轴修正为 X 轴等。
        attach_point_name: 机械臂 XML 中用于挂载机械手的 body 名称。

    Returns:
        mujoco.MjSpec: 已合并机械手但尚未编译的规格对象。

    Raises:
        FileNotFoundError: 当指定的模型文件不存在时。
        ValueError: 当手模型缺少根节点或找不到指定的挂载点时。
    """
    # 1. 解析路径
    arm_path = Path(arm_path) if arm_path else DEFAULT_ARM_PATH
    hand_path = Path(hand_path) if hand_path else DEFAULT_HAND_PATH

    # 2. 文件存在性检查
    if not arm_path.exists():
        raise FileNotFoundError(f"机械臂模型文件不存在: {arm_path}")
    if not hand_path.exists():
        raise FileNotFoundError(f"机械手模型文件不存在: {hand_path}")

    print(f"[SpecBuilder] 加载机械臂: {arm_path.name}")
    print(f"[SpecBuilder] 加载机械手: {hand_path.name}")

    # 3. 加载 Spec 对象
    arm_spec = mujoco.MjSpec.from_file(str(arm_path))
    hand_spec = mujoco.MjSpec.from_file(str(hand_path))

    # 4. 修复手模型根节点偏移
    # 许多 URDF 转 XML 的模型根节点会有初始偏移，合并时需重置为 [0,0,0] 以避免双重偏移
    hand_root = hand_spec.worldbody.first_body()
    if hand_root is None:
        raise ValueError("手模型 XML 缺少根节点 (worldbody 下无 body)。")

    original_pos = np.array(hand_root.pos)
    if np.linalg.norm(original_pos) > 1e-6:
        print(f"[SpecBuilder] 检测到根节点偏移 {original_pos}，已重置为 [0, 0, 0]")
        hand_root.pos = [0.0, 0.0, 0.0]

    # 5. 寻找安装点并挂载
    try:
        attach_body = arm_spec.body(attach_point_name)
    except KeyError:
        available_bodies = [b.name for b in arm_spec.worldbody.bodies()]
        raise ValueError(
            f"未在机械臂模型中找到挂载点 '{attach_point_name}'。\n"
            f"可用的 body 名称列表: {available_bodies}"
        )

    # 创建一个临时帧 (Frame) 用于附加子模型
    attach_frame = attach_body.add_frame()
    
    # 执行附加操作
    # prefix/suffix 用于避免命名冲突
    attached_body = attach_frame.attach_body(
        hand_root,
        prefix="inspirehand_",
        suffix=""
    )
    print(f"[SpecBuilder] 成功挂载: '{attached_body.name}' 到 '{attach_body.name}'")

    # 6. 设置位姿变换 (Position & Orientation)
    attach_frame.pos = [0.0, 0.0, 0.0]  # 位置偏移通常在 attach_body 或 frame 定义中处理，此处保持相对零点
    
    # 计算旋转四元数
    # Scipy Rotation 默认顺序是 'xyz'，输出四元数为 [x, y, z, w]
    base_rot = R.from_quat([0.0, 0.0, 0.0, 1.0])  # 单位四元数
    flip_rot = R.from_euler("xyz", rot_xyz_deg, degrees=True)
    final_rot = base_rot * flip_rot
    
    # SciPy 输出: [x, y, z, w]
    q_xyzw = final_rot.as_quat()
    
    # MuJoCo XML/Array 要求顺序: [w, x, y, z]
    attach_frame.quat = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

    print(f"[SpecBuilder] 姿态设定完成: Euler={rot_xyz_deg} deg")

    return arm_spec


def load_combined_model(
    arm_path: Optional[PathLike] = None,
    hand_path: Optional[PathLike] = None,
    rot_xyz_deg: Tuple[float, float, float] = (-90.0, 0.0, 0.0),
) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    便捷函数：加载、合并并直接编译模型.

    适用于只需快速预览或仿真，不需要对场景进行额外修改的情况。

    Args:
        arm_path: 机械臂路径.
        hand_path: 机械手路径.
        rot_xyz_deg: 姿态修正欧拉角.

    Returns:
        tuple: (model, data) 编译好的 MuJoCo 模型和数据对象.
    """
    spec = get_combined_spec(arm_path, hand_path, rot_xyz_deg)
    
    print("[SpecBuilder] 正在编译模型...")
    model = spec.compile()
    data = mujoco.MjData(model)
    
    print(f"[SpecBuilder] 编译完成。自由度 (dof): {model.nv}, 执行器 (actuator): {model.nu}")
    # 打印可用执行器列表以供调试
    print(f"[SpecBuilder] 可用执行器: {[model.actuator(i).name for i in range(model.nu)]}")
    return model, data


if __name__ == "__main__":
    print("--- 独立运行模式：预览合成机械臂 ---")
    
    try:
        # 加载并编译
        model, data = load_combined_model()
        
        # 启动被动式可视化器
        with viewer.launch_passive(model, data) as v:
            # 简单的渲染循环
            while v.is_running():
                # 物理步进
                mujoco.mj_step(model, data)
                # 同步视图
                v.sync()
                
    except FileNotFoundError as e:
        print(f"\n[错误] 文件未找到: {e}")
        print("请检查 'models/' 目录结构是否正确。")
    except Exception as e:
        print(f"\n[错误] 发生未知异常: {e}")
        traceback.print_exc()