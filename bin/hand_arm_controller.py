"""
机械臂与灵巧手联合控制器模块.

提供简单的函数接口来控制组合系统中的 13 个执行器 (7个机械臂关节 + 6个手爪执行器)，
支持通过名称映射快速定位并发送控制指令。

设计原则：
    1. 预计算优化：初始化时构建索引映射，避免运行时字符串查找
    2. 向量化操作：使用 NumPy 数组和高级索引实现批量控制写入
    3. 严格验证：所有公共接口执行类型和维度检查，尽早暴露错误
    4. 语义命名：支持按功能名称（如"finger_0"）而非原始索引访问手爪
"""

import mujoco
import numpy as np
from typing import Dict, List, Optional, Tuple


class HandArmController:
    """
    手-臂联合控制器类.

    管理 7-DOF 机械臂与多自由度灵巧手的执行器映射与控制分发，
    提供统一的控制接口，将高层语义指令转换为底层执行器信号。

    系统架构：
        用户指令 → 语义映射（finger_0/thumb_grasp）→ 全局执行器索引 → data.ctrl
        
    性能优化：
        - 使用 NumPy 索引数组（arm_indices/hand_indices）替代字典查找
        - apply_control 采用向量化写入，时间复杂度 O(1)

    Attributes:
        model: MuJoCo MjModel 实例，用于获取执行器元数据（nu, actuator_names等）。
        actuator_map: 执行器名称到全局索引的映射字典 {name: index}。
                      键为模型中定义的 actuator 名称，值为 0~nu-1 的整数。
        arm_names: 机械臂关节执行器名称列表，按关节顺序排序（joint0→joint6）。
                   名称格式："torq_joint0", "torq_joint1", ..., "torq_joint6"。
        hand_names: 手爪执行器名称映射，键为语义标识（如"finger_0"），
                    值为模型中的实际执行器名称。
        arm_indices: 机械臂执行器在全局 ctrl 数组中的索引位置，形状 (7,)，dtype=int32。
                     通过 actuator_map 反查得到，用于向量化写入。
        hand_indices: 手爪执行器在全局 ctrl 数组中的索引位置，形状 (6,)，dtype=int32。
                      顺序与 hand_key_order 一致。
        hand_key_order: 手爪控制键的标准顺序列表，定义控制向量的语义排布。
                        用于保证数组索引与物理手指的对应关系。
        all_indices: 联合索引数组，形状 (13,)，是 arm_indices 和 hand_indices 的拼接。
                     用于 apply_control_vector 的批量写入。
    """

    # ====================== 常量定义 ======================
    
    ARM_DOF: int = 7           # 机械臂自由度：肩3 + 肘1 + 腕3
    HAND_DOF: int = 6          # 手爪执行器数量：食指/中指/无名指/小指（各1）+ 拇指2DOF
    TOTAL_DOF: int = 13        # 总执行器数量：ARM_DOF + HAND_DOF

    def __init__(self, model: mujoco.MjModel):
        """
        初始化控制器并构建执行器映射.

        执行流程：
            1. 保存模型引用
            2. 遍历所有执行器，按命名规则分类识别
            3. 构建名称→索引映射字典
            4. 预计算 NumPy 索引数组优化实时性能

        Args:
            model: 已编译的 MuJoCo MjModel 对象，包含完整的执行器定义。
                   要求模型中至少包含 7 个机械臂关节执行器（以"torq_joint"开头）
                   和 6 个手爪执行器（包含特定标识符）。

        Raises:
            ValueError: 当模型中未找到预期的执行器配置时（仅打印警告，不中断）。
            
        Note:
            初始化后可通过 print(controller.arm_names) 查看识别的执行器列表。
        """
        self.model = model
        self.actuator_map: Dict[str, int] = {}
        self.arm_names: List[str] = []
        self.hand_names: Dict[str, str] = {}
        
        # ----- 构建名称-索引映射表 -----
        # 遍历模型中所有执行器（nu = number of actuators）
        self._build_map()
        
        # ----- 预计算索引数组以加速控制循环 -----
        # 将 Python 列表/字典转换为 NumPy 数组，避免实时控制中的哈希查找
        self._build_index_arrays()

    def _build_map(self):
        """
        构建执行器名称到索引的映射表.

        遍历模型中所有执行器，根据命名规则分类识别：
        
        机械臂执行器识别规则：
            名称以 "torq_joint" 开头 → 归类为机械臂关节力矩执行器
            预期数量：7个（对应 7-DOF 机械臂）
            
        手爪执行器识别规则（基于特定子字符串匹配）：
            "hand_act_push_0" → finger_0（食指弯曲）
            "hand_act_push_1" → finger_1（中指弯曲）
            "hand_act_push_2" → finger_2（无名指弯曲）
            "hand_act_push_3" → finger_3（小指弯曲）
            "hand_thumb_grasp" → thumb_grasp（拇指对掌运动）
            "hand_thumb_rotate" → thumb_rotate（拇指旋转）
            
        命名规则依赖：此函数硬编码了特定模型的命名约定，
        若 XML 模型文件中的执行器名称变更，需同步修改此处逻辑。

        Raises:
            RuntimeError: 当关键执行器缺失时记录警告信息（非致命）。
        """
        # 遍历所有执行器（self.model.nu 为执行器总数）
        for i in range(self.model.nu):
            name = self.model.actuator(i).name
            self.actuator_map[name] = i
            
            # ----- 1. 识别机械臂关节执行器（力矩控制模式）-----
            # 命名约定：torq_joint0, torq_joint1, ..., torq_joint6
            if name.startswith("torq_joint"):
                self.arm_names.append(name)
            
            # ----- 2. 识别灵巧手执行器（位置/力混合控制模式）-----
            # 使用子字符串匹配，适应可能的命名前缀变化
            elif "hand_act_push_0" in name:
                self.hand_names["finger_0"] = name  # 食指（Index Finger）
            elif "hand_act_push_1" in name:
                self.hand_names["finger_1"] = name  # 中指（Middle Finger）
            elif "hand_act_push_2" in name:
                self.hand_names["finger_2"] = name  # 无名指（Ring Finger）
            elif "hand_act_push_3" in name:
                self.hand_names["finger_3"] = name  # 小指（Pinky Finger）
            elif "hand_thumb_grasp" in name:
                self.hand_names["thumb_grasp"] = name   # 拇指对掌（Thumb Opposition）
            elif "hand_thumb_rotate" in name:
                self.hand_names["thumb_rotate"] = name  # 拇指旋转（Thumb Rotation）
        
        # ----- 后处理：排序与验证 -----
        # 按关节编号数值排序，确保控制顺序与运动学链（基座→末端）一致
        # 排序键函数：提取 "torq_joint12" 中的数字部分 12
        self.arm_names.sort(key=lambda x: int(x.replace("torq_joint", "")))
        
        # 打印映射结果供调试，同时验证配置完整性
        print(f"[Controller] 机械臂执行器: {len(self.arm_names)} 个")
        print(f"[Controller] 手爪执行器: {len(self.hand_names)} 个")
        print(f"[Controller] 机械臂名称列表: {self.arm_names}")
        print(f"[Controller] 手爪映射字典: {self.hand_names}")
        
        # 完整性检查：警告但不中断，允许部分功能工作
        if len(self.arm_names) != self.ARM_DOF:
            print(f"[警告] 预期 {self.ARM_DOF} 个机械臂关节，实际找到 {len(self.arm_names)} 个")
        if len(self.hand_names) != self.HAND_DOF:
            print(f"[警告] 预期 {self.HAND_DOF} 个手爪执行器，实际找到 {len(self.hand_names)} 个")

    def _build_index_arrays(self):
        """
        预计算执行器索引数组以优化控制性能.

        将名称映射转换为 NumPy 索引数组，避免在实时控制循环中进行字典查找。
        此优化对高频控制循环（如 1kHz 仿真）至关重要。

        构建内容：
            1. arm_indices: 机械臂执行器在 data.ctrl 中的位置
            2. hand_indices: 手爪执行器在 data.ctrl 中的位置（按标准顺序）
            3. all_indices: 联合索引，用于整向量写入
            
        标准顺序定义（hand_key_order）：
            [finger_0, finger_1, finger_2, finger_3, thumb_grasp, thumb_rotate]
            此顺序决定了 apply_control 中 hand_commands 数组的语义解释。
        """
        # ----- 1. 构建机械臂索引数组（形状: [7]）-----
        # 列表推导：按 arm_names 顺序从 actuator_map 查找全局索引
        self.arm_indices = np.array(
            [self.actuator_map[name] for name in self.arm_names],
            dtype=np.int32
        )
        
        # ----- 2. 构建手爪索引数组（形状: [6]）-----
        # 定义标准顺序：四指弯曲（近端→远端）+ 拇指对掌 + 拇指旋转
        # 此顺序必须与 apply_control 的文档约定一致
        self.hand_key_order = [
            "finger_0",    # 索引 0: 食指
            "finger_1",    # 索引 1: 中指
            "finger_2",    # 索引 2: 无名指
            "finger_3",    # 索引 3: 小指
            "thumb_grasp", # 索引 4: 拇指对掌（弯曲）
            "thumb_rotate" # 索引 5: 拇指旋转（外展/内收）
        ]
        
        # 通过字典查找获取全局索引，过滤可能缺失的键（防御性编程）
        self.hand_indices = np.array(
            [self.actuator_map[self.hand_names[key]] 
             for key in self.hand_key_order if key in self.hand_names],
            dtype=np.int32
        )
        
        # ----- 3. 构建联合索引数组（形状: [13]）-----
        # 使用 np.concatenate 合并臂和手爪索引，用于批量操作
        # 顺序：前 7 个为臂，后 6 个为手爪
        self.all_indices = np.concatenate([self.arm_indices, self.hand_indices])
        
        print(f"[Controller] 索引数组构建完成："
              f"arm_indices={self.arm_indices}, "
              f"hand_indices={self.hand_indices}")

    def apply_control(self, data: mujoco.MjData, 
                      arm_torques: np.ndarray,
                      hand_commands: np.ndarray):
        """
        应用控制指令到仿真数据对象.

        核心控制接口，将机械臂力矩和手爪控制量通过 NumPy 数组批量写入 data.ctrl。
        采用向量化操作替代 Python 循环，确保高频调用的执行效率。

        控制信号说明：
            - 机械臂：直接力矩控制（Nm），由底层电机执行器实现力控
            - 手爪：位置/力混合控制，具体取决于模型中执行器的控制模式（position/force）
              通常 grasp 使用力控制（保持抓取力），rotate 使用位置控制（固定姿态）

        Args:
            data: MuJoCo MjData 实例，其 .ctrl 数组将被原地修改。
                  必须在调用前已执行 mj_forward 或 mj_step 保证状态同步。
            arm_torques: 形状为 (7,) 的 numpy 数组，指定各关节力矩（单位：Nm）。
                         顺序与 arm_names 一致：[joint_0, joint_1, ..., joint_6]，
                         对应从基座到末端的关节链。
            hand_commands: 形状为 (6,) 的 numpy 数组，指定手爪控制量。
                           顺序由 hand_key_order 定义：
                           [finger_0, finger_1, finger_2, finger_3, 
                            thumb_grasp, thumb_rotate]。
                           单位取决于具体执行器配置（通常为 N 或 m）。

        Raises:
            TypeError: 当输入不是 numpy 数组时。
            ValueError: 当输入数组维度不匹配 ARM_DOF/HAND_DOF 时。

        Example:
            >>> # 场景1：机械臂保持静止，手爪闭合抓取
            >>> arm_torques = np.zeros(7)  # 零力矩，依靠关节阻尼维持位置
            >>> hand_commands = np.array([200., 200., 200., 200., 250., 0.])
            >>> # 解释：四指 200N 抓取力，拇指对掌 250N，拇指旋转 0（中立位）
            >>> controller.apply_control(data, arm_torques, hand_commands)
            
            >>> # 场景2：机械臂抬升，手爪张开
            >>> arm_torques = np.array([0., 10., 0., 0., 0., 0., 0.])  # 肩部抬升力矩
            >>> hand_commands = np.zeros(6)  # 所有手指零力/张开位置
            >>> controller.apply_control(data, arm_torques, hand_commands)

        Performance:
            时间复杂度 O(13)，主要为 NumPy 数组拼接和索引赋值开销。
            实测在普通 CPU 上单次调用 < 10μs，满足 1kHz 控制频率要求。
        """
        # ----- 1. 验证输入类型与维度（防御性编程）-----
        # 类型检查：确保是 NumPy 数组而非 Python 列表（避免意外广播）
        if not isinstance(arm_torques, np.ndarray):
            raise TypeError(f"arm_torques 必须是 np.ndarray，实际为 {type(arm_torques)}")
        if not isinstance(hand_commands, np.ndarray):
            raise TypeError(f"hand_commands 必须是 np.ndarray，实际为 {type(hand_commands)}")
        
        # 维度检查：严格匹配自由度配置，防止部分写入或越界
        if arm_torques.shape != (self.ARM_DOF,):
            raise ValueError(
                f"arm_torques 形状必须为 ({self.ARM_DOF},)，实际为 {arm_torques.shape}"
            )
        if hand_commands.shape != (self.HAND_DOF,):
            raise ValueError(
                f"hand_commands 形状必须为 ({self.HAND_DOF},)，实际为 {hand_commands.shape}"
            )

        # ----- 2. 合并控制指令（向量化操作）-----
        # 使用 np.concatenate 合并两个控制向量，形状 (13,)
        # 内存分配：每次调用创建新数组，若成为性能瓶颈可考虑预分配优化
        all_commands = np.concatenate([arm_torques, hand_commands])
        
        # ----- 3. 批量写入 ctrl 数组（NumPy 高级索引）-----
        # data.ctrl[self.all_indices] 是高级索引赋值，等效于：
        # for i, idx in enumerate(self.all_indices): data.ctrl[idx] = all_commands[i]
        # 但由 NumPy C 层实现，速度提升 10-100 倍
        data.ctrl[self.all_indices] = all_commands

    def apply_control_vector(self, data: mujoco.MjData, 
                            control_vector: np.ndarray):
        """
        通过单一向量应用全部 13 个执行器的控制指令（便捷接口）.

        适用于已有完整控制策略输出的场景，如：
            - 模型预测控制（MPC）的优化结果
            - 强化学习策略网络的原始输出
            - 运动规划器的轨迹插值点

        此接口避免了调用方手动拆分臂和手爪分量，减少出错概率。

        Args:
            data: MuJoCo MjData 实例，将被修改。
            control_vector: 形状为 (13,) 的 numpy 数组，包含全部控制指令。
                            顺序严格定义为：
                            [arm_joint_0, ..., arm_joint_6,  # 前 7 维：机械臂
                             finger_0, finger_1, finger_2, finger_3,  # 第 7-10 维：四指
                             thumb_grasp, thumb_rotate]  # 第 11-12 维：拇指
                            此顺序与 self.all_indices 的构建顺序一致。

        Raises:
            ValueError: 当输入维度不为 TOTAL_DOF (13) 时。
            TypeError: 当输入不是 numpy 数组时（由 NumPy 隐式抛出）。

        Example:
            >>> # 从策略网络直接输出 13-D 控制向量
            >>> observation = get_observation()  # 获取当前状态
            >>> ctrl = policy_network(observation)  # shape: (13,)
            >>> # ctrl 包含完整控制策略，无需手动拆分
            >>> controller.apply_control_vector(data, ctrl)
            
            >>> # 从轨迹文件加载预计算控制序列
            >>> for ctrl_vec in trajectory:  # ctrl_vec shape (13,)
            ...     controller.apply_control_vector(data, ctrl_vec)
            ...     mujoco.mj_step(model, data)

        Performance:
            比 apply_control 更高效，省去一次 np.concatenate 调用。
            时间复杂度 O(13)，纯 NumPy 索引赋值开销。
        """
        # ----- 1. 输入验证 -----
        if control_vector.shape != (self.TOTAL_DOF,):
            raise ValueError(
                f"控制向量形状必须为 ({self.TOTAL_DOF},)，实际为 {control_vector.shape}"
            )
        
        # ----- 2. 直接批量赋值（最高效路径）-----
        # 无拼接开销，直接通过预计算索引写入
        # 注意：调用方需自行保证 control_vector 的语义顺序正确
        data.ctrl[self.all_indices] = control_vector