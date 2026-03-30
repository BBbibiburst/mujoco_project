"""
位置控制扩展模块 — HandArmController 的 PD 位置控制器.

依赖原 HandArmController 类中预计算的 arm_indices / hand_indices 索引数组，
将关节目标位置转换为力矩指令，通过原有 apply_control 接口下发至仿真器。
"""

import numpy as np
import mujoco
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PDGains:
    """
    PD 控制器增益参数容器.
    
    采用 dataclass 实现不可变配置对象，确保增益参数在控制器生命周期内保持一致性。
    各增益数组长度必须与对应自由度匹配，否则会导致广播错误。

    Attributes:
        kp_arm:   机械臂各关节位置增益，形状 (7,)，单位 Nm/rad。
                  值越大位置跟踪越刚性，但过大会引起振荡。
        kd_arm:   机械臂各关节速度增益，形状 (7,)，单位 Nm/(rad/s)。
                  提供阻尼作用，抑制超调和振动。
        kp_hand:  手爪各执行器位置增益，形状 (6,)，单位 Nm/rad 或 N/m。
                  手爪关节质量小，通常使用较小增益。
        kd_hand:  手爪各执行器速度增益，形状 (6,)，单位 Nm/(rad/s) 或 N/(m/s)。
    """
    kp_arm:  np.ndarray = field(default_factory=lambda: np.array(
        [200., 200., 150., 150., 80., 80., 40.]))  # 肩→腕递减，远端关节惯量小增益低
    kd_arm:  np.ndarray = field(default_factory=lambda: np.array(
        [20.,  20.,  15.,  15.,  8.,  8.,  4.]))   # 按 kp/10 比例配置临界阻尼
    kp_hand: np.ndarray = field(default_factory=lambda: np.full(6, 50.))   # 手爪统一中等刚度
    kd_hand: np.ndarray = field(default_factory=lambda: np.full(6, 5.))    # 手爪统一阻尼


class PositionController:
    """
    基于 PD 控制的关节空间位置控制器.

    包裹 HandArmController，在其力矩接口之上构建位置闭环。
    控制律（时域）:
        τ = Kp * (q_target - q_current) + Kd * (0 - dq_current)
          = Kp * e - Kd * dq
        
    其中 e 为位置误差，dq 为当前关节速度（目标速度假设为0）。
    
    设计特点：
        1. 零速度目标假设：适用于定点控制，不支持速度前馈
        2. 实时限幅保护：对目标位置和输出力矩进行双重饱和
        3. 内存预分配：控制循环中零动态内存分配，保证实时性
        4. 索引缓存：初始化时解析关节索引，避免运行时字符串查找

    Attributes:
        base:           被包裹的 HandArmController 实例，提供底层力矩接口。
        model:          MuJoCo MjModel，用于关节索引解析和物理查询。
        gains:          PD 增益参数对象，包含机械臂和手爪的 Kp/Kd 数组。
        arm_qpos_ids:   机械臂关节在全局 qpos 数组中的索引，形状 (7,)。
        arm_qvel_ids:   机械臂关节在全局 qvel 数组中的索引，形状 (7,)。
        hand_qpos_ids:  手爪关节在全局 qpos 数组中的索引，形状 (≤6,)。
                        长度可能小于6，取决于实际模型配置。
        hand_qvel_ids:  手爪关节在全局 qvel 数组中的索引，形状 (≤6,)。
        _torque_limits: 各执行器力矩饱和限幅，形状 (13,)，默认无限制 (inf)。
        _arm_torques:   机械臂力矩计算缓冲区，形状 (7,)，预分配避免 GC。
        _hand_torques:  手爪力矩计算缓冲区，形状 (6,)，预分配避免 GC。
    """

    def __init__(
        self,
        base_controller,          # HandArmController 实例
        model: mujoco.MjModel,
        gains: Optional[PDGains] = None,
        torque_limits: Optional[np.ndarray] = None,
    ):
        """
        初始化位置控制器.

        执行关节索引解析和缓冲区预分配，必须在仿真开始前完成。

        Args:
            base_controller: 已初始化的 HandArmController 实例。
                             必须包含 arm_names、hand_names、hand_key_order、
                             actuator_map、TOTAL_DOF、ARM_DOF、HAND_DOF 属性。
            model:           MuJoCo MjModel，用于通过 actuator_trnid 反查关节索引。
            gains:           PD 增益配置对象，为 None 时使用 PDGains 默认值。
                             可通过修改默认值实现全局调参。
            torque_limits:   各执行器力矩上限绝对值 (Nm)，形状 (13,)，
                             索引 0-6 为机械臂，7-12 为手爪。
                             为 None 时不做限幅（设为 np.inf）。
                             
        Raises:
            ValueError: 当 torque_limits 维度不匹配 TOTAL_DOF 时抛出。
            KeyError:   当 actuator_map 中找不到指定名称时抛出。
        """
        self.base = base_controller
        self.model = model
        self.gains = gains if gains is not None else PDGains()

        # ----- 建立 actuator → joint → qpos/qvel 映射 -----
        # 通过执行器名称反查其在模型中的关节索引
        # MuJoCo 中：actuator → joint (via trnid) → qposadr/qveladr
        self.arm_qpos_ids, self.arm_qvel_ids = self._resolve_joint_ids(
            base_controller.arm_names
        )
        self.hand_qpos_ids, self.hand_qvel_ids = self._resolve_joint_ids(
            [base_controller.hand_names[k]
             for k in base_controller.hand_key_order
             if k in base_controller.hand_names]
            # 注意：通过列表推导过滤确保只包含实际存在的关节
        )

        # ----- 力矩饱和配置 -----
        if torque_limits is not None:
            if torque_limits.shape != (base_controller.TOTAL_DOF,):
                raise ValueError(
                    f"torque_limits 形状应为 ({base_controller.TOTAL_DOF},)，"
                    f"实际为 {torque_limits.shape}"
                )
            self._torque_limits = torque_limits
        else:
            # 默认无限制，使用 np.inf 表示不饱和
            self._torque_limits = np.full(base_controller.TOTAL_DOF, np.inf)

        # ----- 预分配计算缓冲区 -----
        # 控制循环中复用这些数组，避免频繁的内存分配和垃圾回收
        # 这对实时控制至关重要（Python GC 可能导致时延抖动）
        self._arm_torques  = np.zeros(base_controller.ARM_DOF)
        self._hand_torques = np.zeros(base_controller.HAND_DOF)

        print(f"[PositionController] 初始化完成，"
              f"机械臂关节 {len(self.arm_qpos_ids)} 个，"
              f"手爪关节 {len(self.hand_qpos_ids)} 个，"
              f"总自由度 {base_controller.TOTAL_DOF}")

    # ------------------------------------------------------------------ #
    #  公共控制接口                                                        #
    # ------------------------------------------------------------------ #

    def set_target(
        self,
        data: mujoco.MjData,
        arm_target: np.ndarray,
        hand_target: np.ndarray,
    ):
        """
        计算 PD 力矩并写入仿真器.
        
        这是主要的控制接口，执行完整的 PD 计算流程：
        误差计算 → 增益乘法 → 阻尼补偿 → 限幅 → 下发。

        Args:
            data:        MuJoCo MjData，提供 qpos / qvel 实时反馈。
                         必须已同步当前物理状态（已执行 mj_step 或 mj_forward）。
            arm_target:  机械臂目标关节角度，形状 (7,)，单位 rad。
                         将被硬限制在 [-3.1, 3.1] 范围内（对应实机限位）。
            hand_target: 手爪目标控制量，形状 (6,)，单位视具体模型而定（通常为 m 或 rad）。
                         将被硬限制在 [0, 0.01] 范围内（对应实机行程）。
                         
        注意：
            - 目标限制在函数内部执行，修改的是局部副本，不改变输入数组。
            - 力矩计算使用向量化操作（in-place），避免临时数组分配。
        """
        # ----- 输入验证 -----
        self._validate_inputs(arm_target, hand_target)

        # ----- 目标位置安全限幅（实机保护）-----
        # 机械臂关节限位：防止指令超出物理可行范围
        # 范围 [-3.1, 3.1] 约等于 [-177°, 177°]，覆盖绝大多数关节行程
        arm_target = np.clip(arm_target, -3.1, 3.1)
        
        # 手爪行程限位：0 表示完全张开，0.01 表示闭合行程上限
        # 单位取决于 hand 模型定义（可能是线性位移或角度）
        hand_target = np.clip(hand_target, 0.0, 0.01)

        # ----- 机械臂 PD 力矩计算（in-place，复用 _arm_torques 缓冲区）-----
        # 步骤1：计算位置误差 e = q_target - q_current
        np.subtract(arm_target,                    # 被减数：目标位置
                    data.qpos[self.arm_qpos_ids],  # 减数：当前位置
                    out=self._arm_torques)         # 输出到预分配缓冲区
        
        # 步骤2：比例项 Kp * e（in-place 乘法）
        self._arm_torques *= self.gains.kp_arm
        
        # 步骤3：微分项 -Kd * dq（当前速度，目标速度为0）
        # 注意：这里使用 -= 实现累加，等价于 τ = Kp*e - Kd*dq
        self._arm_torques -= (
            self.gains.kd_arm * data.qvel[self.arm_qvel_ids]
        )

        # ----- 手爪 PD 力矩计算（同理，复用 _hand_torques）-----
        np.subtract(hand_target,
                    data.qpos[self.hand_qpos_ids],
                    out=self._hand_torques)
        self._hand_torques *= self.gains.kp_hand
        self._hand_torques -= (
            self.gains.kd_hand * data.qvel[self.hand_qvel_ids]
        )

        # ----- 力矩饱和限幅（安全保护）-----
        # 防止计算力矩超出执行器能力或安全阈值
        self._apply_saturation()

        # ----- 通过基础控制器下发力矩 -----
        # 调用 HandArmController.apply_control 将力矩写入 data.ctrl
        self.base.apply_control(data, self._arm_torques, self._hand_torques)

    def set_target_vector(
        self,
        data: mujoco.MjData,
        target: np.ndarray,
    ):
        """
        通过单一 13-D 向量设定目标位置（便捷接口）.
        
        适用于从规划器直接接收完整配置向量的场景，
        自动拆分机械臂和手爪分量。

        Args:
            data:   MuJoCo MjData。
            target: 目标配置向量，形状 (13,)，
                    前 7 维（0:7）为机械臂关节角度（rad），
                    后 6 维（7:13）为手爪控制量。
                    
        Raises:
            ValueError: 当 target 维度不为 TOTAL_DOF (13) 时抛出。
        """
        if target.shape != (self.base.TOTAL_DOF,):
            raise ValueError(
                f"target 形状应为 ({self.base.TOTAL_DOF},)，"
                f"实际为 {target.shape}"
            )
        # 切片分解并调用主控制接口
        self.set_target(
            data,
            arm_target=target[:self.base.ARM_DOF],    # 前 7 维
            hand_target=target[self.base.ARM_DOF:],   # 后 6 维
        )

    def is_converged(
        self,
        data: mujoco.MjData,
        arm_target: np.ndarray,
        hand_target: np.ndarray,
        pos_tol: float = 0.01,   # rad
        vel_tol: float = 0.05,   # rad/s
    ) -> bool:
        """
        判断系统是否收敛到目标位置（路径点切换检测）.
        
        采用最大范数（L-∞）判断，即所有关节同时满足阈值才算收敛。
        这是路径规划中的典型需求：当前点到位且静止后，再执行下一点。

        Args:
            data:        当前仿真状态（需已同步）。
            arm_target:  机械臂目标位置，形状 (7,)，单位 rad。
            hand_target: 手爪目标位置，形状 (6,)。
            pos_tol:     位置收敛阈值（rad），默认 0.01 rad ≈ 0.57°。
                         增大此值可加速路径执行，但会降低精度。
            vel_tol:     速度收敛阈值（rad/s），默认 0.05。
                         必须足够小以确保系统基本静止，避免切换时的冲击。

        Returns:
            bool: True 表示已收敛（位置误差和速度均低于阈值），
                  可安全切换至下一路径点。
                  
        注意：
            内部对目标值进行 clip 处理后再比较，确保与 set_target 逻辑一致。
            但此操作不改变输入参数。
        """
        # 安全处理：使用与 set_target 一致的限幅逻辑
        # 防止因目标值超出范围导致的误判（永远达不到非法目标）
        valid_arm_target = np.clip(arm_target, -3.1, 3.1)
        valid_hand_target = np.clip(hand_target, 0.0, 0.01)

        # 计算机械臂最大位置误差（L-∞ 范数）
        arm_pos_err = np.max(np.abs(
            valid_arm_target - data.qpos[self.arm_qpos_ids]))
        
        # 计算手爪最大位置误差
        hand_pos_err = np.max(np.abs(
            valid_hand_target - data.qpos[self.hand_qpos_ids]))
        
        # 计算最大关节速度（判断静止状态）
        arm_vel_err = np.max(np.abs(data.qvel[self.arm_qvel_ids]))
        hand_vel_err = np.max(np.abs(data.qvel[self.hand_qvel_ids]))

        # 四项指标全部满足才算收敛
        return (arm_pos_err < pos_tol and
                hand_pos_err < pos_tol and
                arm_vel_err < vel_tol and
                hand_vel_err < vel_tol)

    # ------------------------------------------------------------------ #
    #  内部辅助方法                                                        #
    # ------------------------------------------------------------------ #

    def _resolve_joint_ids(self, actuator_names: list):
        """
        通过执行器名称反查关节的 qpos / qvel 全局索引.
        
        MuJoCo 模型结构：
            actuator (通过名称) → trnid[act_id, 0] 得到 joint_id 
            → jnt_qposadr[joint_id] 得到 qpos 起始索引
            → jnt_dofadr[joint_id] 得到 qvel 起始索引
            
        对于单自由度关节，qpos_ids 和 qvel_ids 一一对应。
        对于自由关节（如物体浮动），qpos 有 7 维（3平移+4旋转四元数），
        qvel 有 6 维（3线速度+3角速度），但本控制器只处理铰链/滑动关节。

        Args:
            actuator_names: 执行器名称列表，如 ['shoulder_pan', 'shoulder_lift', ...]。

        Returns:
            Tuple[np.ndarray, np.ndarray]: (qpos_ids, qvel_ids)，均为 int32 数组。
            
        Raises:
            KeyError: 当 actuator_names 中的名称不在 actuator_map 中时抛出。
        """
        qpos_ids = []
        qvel_ids = []
        for name in actuator_names:
            # 从基础控制器的映射中获取执行器 ID
            act_id = self.base.actuator_map[name]
            # 通过 trnid 获取该执行器驱动的关节 ID（第一列）
            joint_id = self.model.actuator_trnid[act_id, 0]
            # 获取关节在全局 qpos/qvel 数组中的起始地址
            qpos_ids.append(self.model.jnt_qposadr[joint_id])
            qvel_ids.append(self.model.jnt_dofadr[joint_id])
        return (np.array(qpos_ids, dtype=np.int32),
                np.array(qvel_ids, dtype=np.int32))

    def _apply_saturation(self):
        """
        对计算得到的力矩应用对称饱和限幅（in-place 修改）.
        
        使用 np.clip 将 _arm_torques 和 _hand_torques 限制在
        [-limit, +limit] 范围内。这是执行器保护的最后防线。
        """
        # 机械臂力矩限幅（前 ARM_DOF 个限制值）
        np.clip(self._arm_torques,
                -self._torque_limits[:self.base.ARM_DOF],   # 下限
                self._torque_limits[:self.base.ARM_DOF],    # 上限
                out=self._arm_torques)                       # in-place 输出
                
        # 手爪力矩限幅（剩余部分）
        np.clip(self._hand_torques,
                -self._torque_limits[self.base.ARM_DOF:],
                self._torque_limits[self.base.ARM_DOF:],
                out=self._hand_torques)

    def _validate_inputs(self, arm_target, hand_target):
        """
        验证目标输入的类型与维度（防御性编程）.
        
        在控制循环入口处严格检查，尽早发现调用错误。

        Args:
            arm_target:  待验证的机械臂目标。
            hand_target: 待验证的手爪目标。

        Raises:
            TypeError:  当输入不是 np.ndarray 时抛出。
            ValueError: 当数组形状不匹配 DOF 配置时抛出。
        """
        # 类型检查
        if not isinstance(arm_target, np.ndarray):
            raise TypeError(f"arm_target 须为 np.ndarray，实际为 {type(arm_target)}")
        if not isinstance(hand_target, np.ndarray):
            raise TypeError(f"hand_target 须为 np.ndarray，实际为 {type(hand_target)}")
            
        # 维度检查
        if arm_target.shape != (self.base.ARM_DOF,):
            raise ValueError(
                f"arm_target 形状须为 ({self.base.ARM_DOF},)，"
                f"实际为 {arm_target.shape}"
            )
        if hand_target.shape != (self.base.HAND_DOF,):
            raise ValueError(
                f"hand_target 形状须为 ({self.base.HAND_DOF},)，"
                f"实际为 {hand_target.shape}"
            )