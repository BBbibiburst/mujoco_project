"""
机械臂与灵巧手底层控制接口模块
该模块负责 MuJoCo 模型与控制器之间的底层交互。
核心功能：
1. 执行器映射：解析模型中的执行器名称，建立逻辑索引（Arm/Hand分离）。
2. 物理约束提取：自动读取 XML 中定义的力矩限制和传动比。
3. 信号转换：将高层计算出的“物理力矩”转换为仿真器所需的“控制信号”，并处理单位换算与安全限幅。

设计原则：
- 物理透明：对外暴露力矩接口，内部处理 Gear 传动比换算。
- 安全优先：在写入数据前强制执行双重限幅（力矩限幅与信号限幅）。
"""
import mujoco
import numpy as np
from typing import Dict, List


class HandArmController:
    """
    机械臂与手部的底层执行器管理器。
    
    该类封装了 MuJoCo 的 `nu` (执行器数量) 和 `ctrl` (控制信号) 接口。
    它屏蔽了具体的执行器命名规则，向上层控制器提供统一的索引和物理限制数据。
    
    Attributes:
        ARM_DOF: 机械臂自由度数量。
        HAND_DOF: 灵巧手自由度数量。
        actuator_map: 执行器名称到 MuJoCo 内部索引的映射表。
        arm_names: 机械臂执行器名称列表（已排序）。
        hand_names: 灵巧手执行器名称字典。
        torque_min/torque_max: 物理力矩限制 [N·m]，用于上层 PD 控制器的饱和处理。
        ctrl_min/ctrl_max: 仿真器控制信号限制，用于底层写入前的最终截断。
        gear: 传动比数组，用于力矩到控制信号的线性转换。
    """
    ARM_DOF = 7
    HAND_DOF = 6
    TOTAL_DOF = 13

    def __init__(self, model: mujoco.MjModel):
        """
        初始化执行器管理器。
        
        Args:
            model: MuJoCo 模型对象，需包含完整的执行器定义。
        """
        self.model = model

        self.actuator_map: Dict[str, int] = {}
        self.arm_names: List[str] = []
        self.hand_names: Dict[str, str] = {}

        # 1. 解析名称与索引
        self._build_map()
        self._build_index_arrays()
        
        # 2. 提取物理约束（⭐ 新增：确保控制器知晓硬件极限）
        self._extract_actuator_limits()

    # -----------------------------
    # 构建映射：名称解析
    # -----------------------------
    def _build_map(self):
        """
        遍历模型中的所有执行器，根据命名规则构建映射表。
        
        命名约定：
        - 机械臂: "torq_joint{index}"
        - 灵巧手: 包含特定关键词 (如 "hand_act_push", "hand_thumb")
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
        self.arm_names.sort(key=lambda x: int(x.replace("torq_joint", "")))

    # -----------------------------
    # 索引数组：性能优化
    # -----------------------------
    def _build_index_arrays(self):
        """
        预计算 NumPy 索引数组。
        
        在控制循环中，直接使用整数数组索引比字符串查找快得多。
        同时定义了手部的逻辑顺序，确保多指协同控制时的数据一致性。
        """
        self.arm_indices = np.array(
            [self.actuator_map[n] for n in self.arm_names],
            dtype=np.int32
        )

        # 定义手部自由度的标准顺序
        self.hand_key_order = [
            "finger_0", "finger_1", "finger_2", "finger_3",
            "thumb_grasp", "thumb_rotate"
        ]

        self.hand_indices = np.array(
            [self.actuator_map[self.hand_names[k]]
             for k in self.hand_key_order if k in self.hand_names],
            dtype=np.int32
        )

        # 合并所有受控执行器的索引，用于批量操作
        self.all_indices = np.concatenate([self.arm_indices, self.hand_indices])

    # -----------------------------
    # ⭐ 提取执行器物理约束
    # -----------------------------
    def _extract_actuator_limits(self):
        """
        从模型中提取执行器的控制范围和传动比。
        
        MuJoCo 的控制信号通常是归一化的或通过 Gear 缩放的。
        为了在上层进行物理意义上的力矩控制，我们需要知道：
        1. ctrlrange: 控制信号 [ctrl_min, ctrl_max]
        2. gear: 传动系数，Torque = ctrl * gear
        
        此方法计算真实的物理力矩限制，供上层控制器进行饱和处理。
        """
        ctrl_min = []
        ctrl_max = []
        gear = []

        for i in range(self.model.nu):
            # 提取控制信号范围
            ctrl_min.append(self.model.actuator_ctrlrange[i][0])
            ctrl_max.append(self.model.actuator_ctrlrange[i][1])
            # 提取传动比 (假设单轴执行器，取第一个元素)
            gear.append(self.model.actuator_gear[i][0])

        self.ctrl_min = np.array(ctrl_min)
        self.ctrl_max = np.array(ctrl_max)
        self.gear = np.array(gear)

        # 👉 转换为真实物理力矩限制 [N·m]
        # 公式：Torque_Limit = Ctrl_Limit * Gear
        self.torque_min = self.ctrl_min * self.gear
        self.torque_max = self.ctrl_max * self.gear

    # -----------------------------
    # 控制接口：力矩分发
    # -----------------------------
    def _to_ctrl_signal(self, torques: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """力矩 → 控制信号转换 + 安全限幅（内部复用）"""
        ctrl = torques / self.gear[indices]
        return np.clip(ctrl, self.ctrl_min[indices], self.ctrl_max[indices])

    def apply_control(self, data, arm_torques, hand_torques):
        if arm_torques.shape != (self.ARM_DOF,):
            raise ValueError(f"Arm torque shape mismatch: {arm_torques.shape}")
        if hand_torques.shape != (self.HAND_DOF,):
            raise ValueError(f"Hand torque shape mismatch: {hand_torques.shape}")

        all_torque = np.concatenate([arm_torques, hand_torques])
        data.ctrl[self.all_indices] = self._to_ctrl_signal(all_torque, self.all_indices)

    def apply_control_vector(self, data, control_vector):
        if control_vector.shape != (self.TOTAL_DOF,):
            raise ValueError(f"Control vector shape mismatch: {control_vector.shape}")

        data.ctrl[self.all_indices] = self._to_ctrl_signal(control_vector, self.all_indices)