"""
机械臂与灵巧手联合控制器模块.

提供简单的函数接口来控制组合系统中的 13 个执行器 (7个机械臂关节 + 6个手爪执行器)，
支持通过名称映射快速定位并发送控制指令。
"""

import mujoco
import numpy as np
from typing import Dict, List, Optional, Tuple


class HandArmController:
    """
    手-臂联合控制器类.

    管理 7-DOF 机械臂与多自由度灵巧手的执行器映射与控制分发，
    提供统一的控制接口，将高层语义指令转换为底层执行器信号。

    Attributes:
        model: MuJoCo MjModel 实例，用于获取执行器元数据。
        actuator_map: 执行器名称到全局索引的映射字典。
        arm_names: 机械臂关节执行器名称列表，按关节顺序排序。
        hand_names: 手爪执行器名称映射，键为语义标识，值为实际执行器名。
        arm_indices: 机械臂执行器在全局 ctrl 数组中的索引位置 (np.ndarray)。
        hand_indices: 手爪执行器在全局 ctrl 数组中的索引位置 (np.ndarray)。
        hand_key_order: 手爪控制键的顺序列表，用于数组到语义的映射。
    """

    # ====================== 常量定义 ======================
    ARM_DOF: int = 7           # 机械臂自由度
    HAND_DOF: int = 6          # 手爪执行器数量 (4指 + 拇指2DOF)
    TOTAL_DOF: int = 13        # 总执行器数量

    def __init__(self, model: mujoco.MjModel):
        """
        初始化控制器并构建执行器映射.

        Args:
            model: 已编译的 MuJoCo 模型对象，包含完整的执行器定义。

        Raises:
            ValueError: 当模型中未找到预期的执行器配置时。
        """
        self.model = model
        self.actuator_map: Dict[str, int] = {}
        self.arm_names: List[str] = []
        self.hand_names: Dict[str, str] = {}
        
        # 构建名称-索引映射表
        self._build_map()
        
        # 预计算索引数组以加速控制循环
        self._build_index_arrays()

    def _build_map(self):
        """
        构建执行器名称到索引的映射表.

        遍历模型中所有执行器，根据命名规则分类为机械臂或手爪执行器：
        - 机械臂: 以 "torq_joint" 开头的力矩控制执行器 (7个)
        - 手爪: 包含 "hand_act_push"、"hand_thumb" 等标识的执行器 (6个)
        
        Raises:
            RuntimeError: 当关键执行器缺失时记录警告信息。
        """
        for i in range(self.model.nu):
            name = self.model.actuator(i).name
            self.actuator_map[name] = i
            
            # 1. 识别机械臂关节执行器 (力矩控制)
            if name.startswith("torq_joint"):
                self.arm_names.append(name)
            
            # 2. 识别灵巧手执行器 (位置/力混合控制)
            elif "hand_act_push_0" in name:
                self.hand_names["finger_0"] = name  # 食指
            elif "hand_act_push_1" in name:
                self.hand_names["finger_1"] = name  # 中指
            elif "hand_act_push_2" in name:
                self.hand_names["finger_2"] = name  # 无名指
            elif "hand_act_push_3" in name:
                self.hand_names["finger_3"] = name  # 小指
            elif "hand_thumb_grasp" in name:
                self.hand_names["thumb_grasp"] = name   # 拇指对掌
            elif "hand_thumb_rotate" in name:
                self.hand_names["thumb_rotate"] = name  # 拇指旋转
        
        # 按关节编号排序，确保控制顺序与运动学链一致
        self.arm_names.sort(key=lambda x: int(x.replace("torq_joint", "")))
        
        # 验证映射完整性
        print(f"[Controller] 机械臂执行器: {len(self.arm_names)} 个")
        print(f"[Controller] 手爪执行器: {len(self.hand_names)} 个")
        if len(self.arm_names) != self.ARM_DOF:
            print(f"[警告] 预期 {self.ARM_DOF} 个机械臂关节，实际找到 {len(self.arm_names)} 个")

    def _build_index_arrays(self):
        """
        预计算执行器索引数组以优化控制性能.

        将名称映射转换为 NumPy 索引数组，避免在实时控制循环中进行字典查找。
        """
        # 1. 构建机械臂索引数组 (形状: [7])
        self.arm_indices = np.array(
            [self.actuator_map[name] for name in self.arm_names],
            dtype=np.int32
        )
        
        # 2. 构建手爪索引数组 (形状: [6])
        # 定义标准顺序: 四指弯曲 + 拇指对掌 + 拇指旋转
        self.hand_key_order = [
            "finger_0", "finger_1", "finger_2", "finger_3",
            "thumb_grasp", "thumb_rotate"
        ]
        
        self.hand_indices = np.array(
            [self.actuator_map[self.hand_names[key]] 
             for key in self.hand_key_order if key in self.hand_names],
            dtype=np.int32
        )
        
        # 3. 构建联合索引数组 (形状: [13])，用于批量写入
        self.all_indices = np.concatenate([self.arm_indices, self.hand_indices])

    def apply_control(self, data: mujoco.MjData, 
                      arm_torques: np.ndarray,
                      hand_commands: np.ndarray):
        """
        应用控制指令到仿真数据对象.

        将机械臂力矩和手爪控制量通过 NumPy 数组批量写入 data.ctrl，
        支持向量化操作以提升实时性能。

        Args:
            data: MuJoCo MjData 实例，其 .ctrl 数组将被修改。
            arm_torques: 形状为 (7,) 的 numpy 数组，指定各关节力矩 (Nm)。
                         顺序: [joint_0, joint_1, ..., joint_6]。
            hand_commands: 形状为 (6,) 的 numpy 数组，指定手爪控制量。
                           顺序: [finger_0, finger_1, finger_2, finger_3, 
                                  thumb_grasp, thumb_rotate]。

        Raises:
            ValueError: 当输入数组维度不匹配时。
            TypeError: 当输入不是 numpy 数组时。

        Example:
            >>> # 机械臂保持静止，手爪闭合抓取
            >>> arm_torques = np.zeros(7)
            >>> hand_commands = np.array([200., 200., 200., 200., 250., 0.])
            >>> controller.apply_control(data, arm_torques, hand_commands)
        """
        # 1. 验证输入类型与维度
        if not isinstance(arm_torques, np.ndarray):
            raise TypeError(f"arm_torques 必须是 np.ndarray，实际为 {type(arm_torques)}")
        if not isinstance(hand_commands, np.ndarray):
            raise TypeError(f"hand_commands 必须是 np.ndarray，实际为 {type(hand_commands)}")
        
        if arm_torques.shape != (self.ARM_DOF,):
            raise ValueError(
                f"arm_torques 形状必须为 ({self.ARM_DOF},)，实际为 {arm_torques.shape}"
            )
        if hand_commands.shape != (self.HAND_DOF,):
            raise ValueError(
                f"hand_commands 形状必须为 ({self.HAND_DOF},)，实际为 {hand_commands.shape}"
            )

        # 2. 合并控制指令 (向量化操作，避免 Python 循环)
        # 形状: (13,)
        all_commands = np.concatenate([arm_torques, hand_commands])
        
        # 3. 批量写入 ctrl 数组 (NumPy 高级索引)
        data.ctrl[self.all_indices] = all_commands

    def apply_control_vector(self, data: mujoco.MjData, 
                            control_vector: np.ndarray):
        """
        通过单一向量应用全部 13 个执行器的控制指令.

        适用于已有完整控制策略输出的场景，如 MPC、强化学习策略网络等。

        Args:
            data: MuJoCo MjData 实例。
            control_vector: 形状为 (13,) 的 numpy 数组，
                            顺序: [7个臂关节, 4个手指, 拇指对掌, 拇指旋转]。

        Raises:
            ValueError: 当输入维度不为 13 时。

        Example:
            >>> # 从策略网络直接输出 13-D 控制向量
            >>> ctrl = policy_network(observation)  # shape: (13,)
            >>> controller.apply_control_vector(data, ctrl)
        """
        # 1. 验证输入
        if control_vector.shape != (self.TOTAL_DOF,):
            raise ValueError(
                f"控制向量形状必须为 ({self.TOTAL_DOF},)，实际为 {control_vector.shape}"
            )
        
        # 2. 直接批量赋值 (最高效，无拼接开销)
        data.ctrl[self.all_indices] = control_vector