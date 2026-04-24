"""
机械臂与灵巧手底层控制接口模块.

该模块负责 MuJoCo 模型与控制器之间的底层交互，提供物理透明的力矩控制接口。
核心功能：
1. 执行器映射：解析模型中的执行器名称，建立逻辑索引（Arm/Hand分离）。
2. 物理约束提取：自动读取 XML 中定义的力矩限制和传动比。
3. 信号转换：将高层计算出的"物理力矩"转换为仿真器所需的"控制信号"，
   并处理单位换算与安全限幅。

设计原则：
- 物理透明：对外暴露力矩接口（单位 N·m），内部处理 Gear 传动比换算。
- 安全优先：在写入数据前强制执行双重限幅（力矩限幅与信号限幅）。
- 性能优化：预计算索引数组，避免控制循环中的字符串查找。

依赖：
- mujoco: MuJoCo 物理引擎接口
- numpy: 数值计算
"""

import mujoco
import numpy as np
from typing import Dict, List


class HandArmController:
    """
    机械臂与手部的底层执行器管理器.

    该类封装了 MuJoCo 的 `nu` (执行器数量) 和 `ctrl` (控制信号) 接口。
    它屏蔽了具体的执行器命名规则，向上层控制器提供统一的索引和物理限制数据。

    执行器命名约定（适配 TactileReader 输出）：
    机械臂: "torq_joint{index}" (index 0-6)
    灵巧手: 带有 "inspirehand_" 前缀的特定关键词
        - "inspirehand_hand_act_push_0" → finger_0 (食指)
        - "inspirehand_hand_act_push_1" → finger_1 (中指)
        - "inspirehand_hand_act_push_2" → finger_2 (无名指)
        - "inspirehand_hand_act_push_3" → finger_3 (小指)
        - "inspirehand_hand_thumb_grasp" → thumb_grasp (拇指弯曲)
        - "inspirehand_hand_thumb_rotate" → thumb_rotate (拇指旋转)

    物理约束说明：
    MuJoCo 中 actuator.ctrl 是控制信号，实际输出力矩为：
        torque = ctrl × gear
    本类自动处理此换算，对外接口统一为物理力矩 [N·m]。

    Attributes:
        ARM_DOF (int): 机械臂自由度数量，固定为7。
        HAND_DOF (int): 灵巧手自由度数量，固定为6。
        TOTAL_DOF (int): 总自由度数量，13。
        actuator_map (Dict[str, int]): 执行器名称到 MuJoCo 内部索引的映射表。
        arm_names (List[str]): 机械臂执行器名称列表（已按关节顺序排序）。
        hand_names (Dict[str, str]): 灵巧手执行器名称字典，键为逻辑名，值为模型名。
        arm_indices (np.ndarray): 机械臂执行器的整数索引数组，shape (7,)。
        hand_indices (np.ndarray): 灵巧手执行器的整数索引数组，shape (6,)。
        all_indices (np.ndarray): 所有受控执行器的合并索引数组，shape (13,)。
        torque_min (np.ndarray): 物理力矩下限 [N·m]，shape (nu,)。
        torque_max (np.ndarray): 物理力矩上限 [N·m]，shape (nu,)。
        ctrl_min (np.ndarray): 控制信号下限，shape (nu,)。
        ctrl_max (np.ndarray): 控制信号上限，shape (nu,)。
        gear (np.ndarray): 传动比数组，shape (nu,)。

    Examples:
        >>> # 初始化控制器
        >>> model = mujoco.MjModel.from_xml_path("robot.xml")
        >>> controller = HandArmController(model)
        >>>
        >>> # 查看物理限制
        >>> print(f"Arm torque limits: {controller.torque_max[controller.arm_indices]}")
        >>>
        >>> # 应用力矩控制（物理单位 N·m）
        >>> arm_tau = np.array([10.0, 5.0, 5.0, 2.0, 1.0, 0.5, 0.5]) # 7个关节
        >>> hand_tau = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 0.3]) # 6个手指
        >>> controller.apply_control(data, arm_tau, hand_tau)
    """
    # 类常量：自由度配置
    ARM_DOF = 7  # 7-DOF 机械臂（如 RM75B）
    HAND_DOF = 6  # 6-DOF 灵巧手（4指×1 + 拇指×2）
    TOTAL_DOF = 13  # ARM_DOF + HAND_DOF

    def __init__(self, model: mujoco.MjModel):
        """
        初始化执行器管理器.

        执行流程：
        1. 存储模型引用
        2. 构建名称到索引的映射表（_build_map）
        3. 预计算整数索引数组（_build_index_arrays）
        4. 提取执行器物理约束（_extract_actuator_limits）

        Args:
            model: MuJoCo 模型对象，需包含完整的执行器定义。
                   要求模型中执行器命名符合类文档所述的命名约定。

        Raises:
            KeyError: 如果模型中缺少预期的执行器名称。
            ValueError: 如果传动比为零（会导致除零错误）。
        """
        self.model = model
        # 初始化数据结构
        self.actuator_map: Dict[str, int] = {}
        self.arm_names: List[str] = []
        self.hand_names: Dict[str, str] = {}

        # 1. 解析名称与索引
        self._build_map()
        self._build_index_arrays()
        # 2. 提取物理约束（确保控制器知晓硬件极限）
        self._extract_actuator_limits()

    # ====================== 内部构建方法 ======================

    def _build_map(self):
        """
        遍历模型中的所有执行器，根据命名规则构建映射表.

        命名约定解析：
        - 机械臂: 以 "torq_joint" 开头，后跟数字索引
        - 灵巧手: 包含特定关键词，映射到逻辑名称
            * "hand_act_push_0" → finger_0 (食指)
            * "hand_act_push_1" → finger_1 (中指)
            * "hand_act_push_2" → finger_2 (无名指)
            * "hand_act_push_3" → finger_3 (小指)
            * "hand_thumb_grasp" → thumb_grasp (拇指弯曲)
            * "hand_thumb_rotate" → thumb_rotate (拇指旋转)

        Note:
            机械臂名称按数字索引自然排序，确保索引顺序与物理关节顺序一致。
            灵巧手使用关键词匹配以兼容不同的模型命名变体。
        """
        for i in range(self.model.nu):
            name = self.model.actuator(i).name
            self.actuator_map[name] = i

            # --- 机械臂识别 ---
            if name.startswith("torq_joint"):
                self.arm_names.append(name)

            # --- 灵巧手识别 ---
            # 使用关键词匹配以兼容不同的模型命名变体
            elif "hand_act_push_0" in name:
                self.hand_names["finger_0"] = name
            elif "hand_act_push_1" in name:
                self.hand_names["finger_1"] = name
            elif "hand_act_push_2" in name:
                self.hand_names["finger_2"] = name
            elif "hand_act_push_3" in name:
                self.hand_names["finger_3"] = name
            elif "hand_thumb_grasp" in name:
                self.hand_names["thumb_grasp"] = name
            elif "hand_thumb_rotate" in name:
                self.hand_names["thumb_rotate"] = name

        # 关键：对机械臂名称进行自然排序，确保索引顺序与物理关节顺序一致
        # 例如：torq_joint1, torq_joint2, ... torq_joint7
        self.arm_names.sort(key=lambda x: int(x.replace("torq_joint", "")))

    def _build_index_arrays(self):
        """
        预计算 NumPy 索引数组，优化控制循环性能.

        在实时控制循环中，直接使用整数数组索引比字符串查找快得多（O(1) vs O(n)）。
        同时定义了手部的逻辑顺序，确保多指协同控制时的数据一致性。

        构建的索引数组：
        - arm_indices: 机械臂7个关节的执行器索引
        - hand_indices: 灵巧手6个自由度的执行器索引（按标准顺序）
        - all_indices: 合并后的13个执行器索引（用于批量操作）

        手部标准顺序：
        [finger_0, finger_1, finger_2, finger_3, thumb_grasp, thumb_rotate]
        """
        # 机械臂索引数组
        self.arm_indices = np.array(
            [self.actuator_map[n] for n in self.arm_names], dtype=np.int32
        )

        # 定义手部自由度的标准顺序（逻辑顺序，与物理布局无关）
        self.hand_key_order = [
            "finger_0",
            "finger_1",
            "finger_2",
            "finger_3",
            "thumb_grasp",
            "thumb_rotate",
        ]

        # 手部索引数组（按标准顺序）
        self.hand_indices = np.array(
            [self.actuator_map[self.hand_names[k]] for k in self.hand_key_order if k in self.hand_names],
            dtype=np.int32,
        )

        # 合并所有受控执行器的索引，用于批量操作
        self.all_indices = np.concatenate([self.arm_indices, self.hand_indices])

    def _extract_actuator_limits(self):
        """
        从模型中提取执行器的控制范围和传动比.

        MuJoCo 的控制信号通常是归一化的或通过 Gear 缩放的。
        为了在上层进行物理意义上的力矩控制，我们需要知道：
        1. ctrlrange: 控制信号范围 [ctrl_min, ctrl_max]
        2. gear: 传动系数，实际力矩公式为 torque = ctrl × gear

        此方法计算真实的物理力矩限制，供上层控制器进行饱和处理。
        计算公式：
            torque_min = ctrl_min × gear
            torque_max = ctrl_max × gear

        Note:
            假设所有执行器为单轴（取 gear 数组的第一个元素）。
            如果 gear 为零会触发 ValueError（会导致后续除零）。
        """
        ctrl_min = []
        ctrl_max = []
        gear = []
        for i in range(self.model.nu):
            # 提取控制信号范围（来自 XML 中 actuator 的 ctrlrange 属性）
            ctrl_min.append(self.model.actuator_ctrlrange[i][0])
            ctrl_max.append(self.model.actuator_ctrlrange[i][1])
            # 提取传动比（来自 XML 中 actuator 的 gear 属性）
            # 假设单轴执行器，取第一个元素
            gear.append(self.model.actuator_gear[i][0])

        self.ctrl_min = np.array(ctrl_min)
        self.ctrl_max = np.array(ctrl_max)
        self.gear = np.array(gear)

        # 检查传动比有效性（避免除零）
        if np.any(self.gear == 0):
            zero_indices = np.where(self.gear == 0)[0]
            raise ValueError(f"执行器 {zero_indices} 的传动比为零，会导致除零错误")

        # 转换为真实物理力矩限制 [N·m]
        # 公式：Torque_Limit = Ctrl_Limit × Gear
        self.torque_min = self.ctrl_min * self.gear
        self.torque_max = self.ctrl_max * self.gear

    # ====================== 控制接口 ======================

    def _to_ctrl_signal(self, torques: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """
        物理力矩 → 控制信号转换 + 安全限幅（内部复用）.

        转换公式：
            ctrl = torque / gear

        安全限幅：
        1. 首先根据物理力矩限制裁剪输入力矩（torque_min/max）
        2. 然后转换为控制信号
        3. 最后根据控制信号范围裁剪（ctrl_min/max）

        Args:
            torques: 物理力矩数组 [N·m]，shape 与 indices 匹配。
            indices: 执行器索引数组，用于选择对应的 gear 和 limit。

        Returns:
            np.ndarray: 转换后的控制信号，已限幅，可直接写入 data.ctrl。

        Note:
            双重限幅确保即使传入力矩超出物理限制，也不会损坏仿真稳定性。
        """
        # 第一重限幅：物理力矩限制（保护硬件）
        torque_clipped = np.clip(torques, self.torque_min[indices], self.torque_max[indices])
        # 转换为控制信号
        ctrl = torque_clipped / self.gear[indices]
        # 第二重限幅：控制信号限制（保护仿真器）
        return np.clip(ctrl, self.ctrl_min[indices], self.ctrl_max[indices])

    def apply_control(self, data, arm_torques: np.ndarray, hand_torques: np.ndarray):
        """
        分别应用机械臂和灵巧手的力矩控制.

        标准接口，适用于分别计算臂和手控制律的场景。

        Args:
            data: MuJoCo MjData 对象，其 ctrl 数组将被修改。
            arm_torques: 机械臂关节力矩 [N·m]，shape (7,)。
                         顺序与 arm_names 一致（通常按关节编号 0-6）。
            hand_torques: 灵巧手关节力矩 [N·m]，shape (6,)。
                          顺序为 [finger_0, finger_1, finger_2, finger_3, thumb_grasp, thumb_rotate]。

        Raises:
            ValueError: 如果输入数组形状不匹配。

        Examples:
            >>> # PD 控制示例
            >>> arm_tau = kp_arm * (q_target - q) + kd_arm * (dq_target - dq)
            >>> hand_tau = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 0.3]) # 恒定抓握力
            >>> controller.apply_control(data, arm_tau, hand_tau)
        """
        # 输入验证
        if arm_torques.shape != (self.ARM_DOF,):
            raise ValueError(
                f"Arm torque shape mismatch: expected {(self.ARM_DOF,)}, got {arm_torques.shape}"
            )
        if hand_torques.shape != (self.HAND_DOF,):
            raise ValueError(
                f"Hand torque shape mismatch: expected {(self.HAND_DOF,)}, got {hand_torques.shape}"
            )

        # 合并力矩并应用
        all_torque = np.concatenate([arm_torques, hand_torques])
        data.ctrl[self.all_indices] = self._to_ctrl_signal(all_torque, self.all_indices)

    def apply_control_vector(self, data, control_vector: np.ndarray):
        """
        通过单一向量应用全部13个自由度的力矩控制.

        紧凑接口，适用于统一计算所有自由度的场景（如MPC、强化学习策略输出）。

        Args:
            data: MuJoCo MjData 对象，其 ctrl 数组将被修改。
            control_vector: 完整控制向量 [N·m]，shape (13,)。
                            顺序为 [arm_joint_0-6, finger_0-3, thumb_grasp, thumb_rotate]。

        Raises:
            ValueError: 如果输入向量形状不是 (13,)。

        Examples:
            >>> # 神经网络策略输出
            >>> action = policy_network(observation) # shape (13,)
            >>> controller.apply_control_vector(data, action)
        """
        if control_vector.shape != (self.TOTAL_DOF,):
            raise ValueError(
                f"Control vector shape mismatch: expected {(self.TOTAL_DOF,)}, "
                f"got {control_vector.shape}"
            )
        data.ctrl[self.all_indices] = self._to_ctrl_signal(control_vector, self.all_indices)

    # ====================== 查询接口（可选） ======================

    def get_arm_torque_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """
        获取机械臂各关节的力矩限制.

        Returns:
            tuple[np.ndarray, np.ndarray]: (min_limits, max_limits)，各 shape (7,)。
        """
        return (self.torque_min[self.arm_indices], self.torque_max[self.arm_indices])

    def get_hand_torque_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """
        获取灵巧手各自由度的力矩限制.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (min_limits, max_limits)，各 shape (6,)。
        """
        return (self.torque_min[self.hand_indices], self.torque_max[self.hand_indices])