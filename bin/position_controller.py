"""
末端执行器与关节空间混合控制器模块
该模块实现了一个分层的机器人控制器，支持两种控制模式：
1. 末端位置/姿态控制 (IK模式)：输入目标位置和姿态，内部通过雅可比矩阵计算关节增量
2. 关节空间位置控制 (PD模式)：输入目标关节角，通过 PD 反馈计算力矩

设计特点：
- 解耦设计：IK 负责空间轨迹规划，PD 负责底层力矩执行
- 安全限制：包含关节限位保护、力矩饱和限制以及 IK 步长限制
- 灵活扩展：支持机械臂与灵巧手的独立参数配置
"""
import numpy as np
import mujoco
from dataclasses import dataclass, field


# ====================== 控制参数配置 ======================
@dataclass
class PDGains:
    """
    PD 控制器增益配置容器.
    
    采用 dataclass 实现不可变配置对象。增益值直接影响系统的刚度和阻尼特性。
    默认值设定依据：
    - 机械臂 (Arm): 通常需要较高的刚度以抵抗重力和外部扰动
    - 机械手 (Hand): 需要相对灵活，避免过大的刚度导致碰撞冲击
    
    Attributes:
        kp_arm: 机械臂比例增益 (刚度) [N·m/rad]。值越大定位越硬。
        kd_arm: 机械臂微分增益 (阻尼) [N·m·s/rad]。用于抑制运动过程中的振荡。
        kp_hand: 机械手比例增益。通常设置为与臂同量级或略低。
        kd_hand: 机械手微分增益。
    """
    kp_arm: np.ndarray = field(default_factory=lambda: np.full(7, 4000.))
    kd_arm: np.ndarray = field(default_factory=lambda: np.full(7, 400.))
    kp_hand: np.ndarray = field(default_factory=lambda: np.full(6, 4000.))
    kd_hand: np.ndarray = field(default_factory=lambda: np.full(6, 400.))


class PositionController:
    """
    混合位置控制器 (IK + PD).
    
    该控制器采用两层架构：
    1. 上层 (IK Solver): 将末端执行器 (End-Effector) 的空间目标转换为关节空间的目标增量。
    2. 下层 (PD Controller): 接收关节目标，计算所需的关节力矩并应用饱和限制。
    
    Attributes:
        base: 硬件/模型抽象接口，提供 DOF 数量、名称映射和力矩限制。
        model: MuJoCo 模型对象 (mjModel)，用于雅可比计算。
        gains: PD 增益配置。
        arm_qpos_ids: 机械臂位置自由度 (qpos) 在全局数组中的索引。
        arm_qvel_ids: 机械臂速度自由度 (qvel) 在全局数组中的索引。
        arm_range: 机械臂关节的物理限位范围。
        _torque_min/_torque_max: 读取自 base 的执行器力矩限制。
        ee_id: 末端执行器 Site 在 MuJoCo 模型中的 ID。
        jac_p/jac_r: 用于存储位置和旋转雅可比矩阵的缓冲区。
    """

    def __init__(self, base, model, gains: PDGains | None = None):
        """
        初始化控制器。
        
        Args:
            base: 包含机器人硬件参数的基础对象。
            model: MuJoCo 模型实例。
            gains: 可选的自定义 PD 增益。若为 None，则使用 PDGains 默认值。
        """
        self.base = base
        self.model = model
        self.gains = gains if gains is not None else PDGains()

        # ----- 1. 关节索引解析 -----
        # 通过 Base 接口获取关节名称，并映射为 MuJoCo 内部索引
        # 这种映射避免了硬编码索引，提高了代码对不同 URDF/XML 的适应性
        self.arm_qpos_ids, self.arm_qvel_ids = self._resolve_joint_ids(base.arm_names)
        
        # 机械手索引：按照预定义的 key_order 顺序排列，确保多指协同的一致性
        self.hand_qpos_ids, self.hand_qvel_ids = self._resolve_joint_ids(
            [base.hand_names[k] for k in base.hand_key_order if k in base.hand_names]
        )

        # ----- 2. 自动读取关节范围和限制 -----
        # 从模型中提取物理限位，用于 IK 过程中的碰撞避免和目标修正
        self.arm_range = self._get_joint_range(self.arm_qpos_ids)
        self.hand_range = self._get_joint_range(self.hand_qpos_ids)
        
        # 读取力矩限制，用于底层饱和处理
        self._torque_min = base.torque_min
        self._torque_max = base.torque_max

        # ----- 3. 初始化扭矩缓冲区 -----
        # 预分配数组以避免在控制循环中频繁分配内存（实时性优化）
        self._arm_torques = np.zeros(base.ARM_DOF)
        self._hand_torques = np.zeros(base.HAND_DOF)

        # --- IK 相关初始化 ---
        # 末端执行器的 site 名称，如果你的 model 里叫别的名字请修改
        self.ee_site_name = "right_hand_site" 
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)
        
        # 用于 IK 计算的临时变量 (预分配内存)
        self.jac_p = np.zeros((3, model.nv)) # 位置雅可比
        self.jac_r = np.zeros((3, model.nv)) # 旋转雅可比

    # -----------------------------
    # 新增：末端位置控制接口 (IK)
    # -----------------------------
    def set_ee_target(self, data, ee_pos_target, ee_quat_target=None, hand_target=None):
        """
        通过逆运动学 (IK) 控制机械臂末端位置，并协同控制手部姿态。
        
        该方法采用基于雅可比转置的数值方法求解逆运动学。
        流程：计算误差 -> 获取雅可比 -> 求解关节速度/增量 -> 更新目标 -> 调用 PD 下发
        
        Args:
            data: MuJoCo 数据对象 (mjData)。
            ee_pos_target: 3D 目标位置 [x, y, z]。
            ee_quat_target: 目标姿态四元数 [w, x, y, z] (可选)。None 表示仅位置控制。
            hand_target: 手部关节目标角度 (直接传递给 PD 控制器)。
        """
        # 1. 获取当前末端状态
        ee_pos_current = data.site_xpos[self.ee_id]
        ee_rot_current = data.site_xmat[self.ee_id].reshape(3, 3)

        # 2. 计算位置误差
        error_pos = ee_pos_target - ee_pos_current
        
        # 3. 计算旋转误差 (如果提供了四元数)
        error_rot = np.zeros(3)
        if ee_quat_target is not None:
            # 获取当前末端 site 的四元数 (MuJoCo 存储的是旋转矩阵，先转四元数)
            ee_quat_current = np.zeros(4)
            mujoco.mju_mat2Quat(ee_quat_current, data.site_xmat[self.ee_id])
            
            # 计算四元数误差 (Target * Current_inv)
            # MuJoCo 提供 mju_subQuat 得到旋转轴/速度向量
            neg_ee_quat_current = np.zeros(4)
            mujoco.mju_negQuat(neg_ee_quat_current, ee_quat_current)
            
            error_quat = np.zeros(4)
            mujoco.mju_mulQuat(error_quat, ee_quat_target, neg_ee_quat_current)
            
            # 将误差四元数转换为旋转向量 (3维)
            # 这是误差的方向和大小
            mujoco.mju_quat2Vel(error_rot, error_quat, 1.0)

        # 4. 获取雅可比矩阵
        # 计算末端 Site 相对于全局坐标系的几何雅可比
        mujoco.mj_jacSite(self.model, data, self.jac_p, self.jac_r, self.ee_id)
        
        # 只提取机械臂对应的 Dof 雅可比列 (假设机械臂对应前几个 dof)
        arm_jac_p = self.jac_p[:, self.arm_qvel_ids]
        arm_jac_r = self.jac_r[:, self.arm_qvel_ids]
        
        # 拼接雅可比 (如果只要位置就只用 jac_p)
        if ee_quat_target is not None:
            full_jac = np.vstack([arm_jac_p, arm_jac_r])
            full_error = np.concatenate([error_pos, error_rot])
        else:
            full_jac = arm_jac_p
            full_error = error_pos

        # 5. 计算关节增量 (DLS 阻尼最小二乘)
        # 解决雅可比矩阵非方阵或奇异的问题，增加数值稳定性
        lambda_sq = 0.01 
        
        # 使用 solve 通常比直接求逆更稳定
        dq = full_jac.T @ np.linalg.solve(
            full_jac @ full_jac.T + lambda_sq * np.eye(full_jac.shape[0]), 
            full_error
        )

        # ⭐ 新增：步长限制 (防止目标过远导致 dq 过大)
        # 限制单次计算的最大关节变化量（例如 0.05 弧度）
        max_dq = 0.1745
        magnitude = np.linalg.norm(dq)
        if magnitude > max_dq:
            dq = dq * (max_dq / magnitude)

        # 6. 计算新的目标关节角
        # 注意：这里我们基于当前实际位置 data.qpos 计算下一个目标点
        arm_target = data.qpos[self.arm_qpos_ids] + dq
        
        # 7. 调用原有的 set_target 进行 PD 控制和下发
        if hand_target is None:
            hand_target = data.qpos[self.hand_qpos_ids] # 保持当前手部姿态
            
        self.set_target(data, arm_target, hand_target)

    # -----------------------------
    # 原有：关节空间主控制
    # -----------------------------
    def set_target(self, data, arm_target, hand_target):
        """
        关节空间 PD 控制器。
        
        执行标准的比例-微分控制逻辑，并应用力矩饱和限制。
        
        Args:
            data: MuJoCo 数据对象。
            arm_target: 机械臂目标关节角度。
            hand_target: 机械手目标关节角度。
        """
        # --- 1. 范围限制 ---
        # 确保目标值在物理关节限位之内，防止非法指令导致仿真崩溃
        arm_target = np.clip(arm_target, self.arm_range[:, 0], self.arm_range[:, 1])
        hand_target = np.clip(hand_target, self.hand_range[:, 0], self.hand_range[:, 1])

        # --- 2. 机械臂 PD 计算 ---
        # torque = Kp * (q_target - q_current) - Kd * qvel
        np.subtract(arm_target, data.qpos[self.arm_qpos_ids], out=self._arm_torques)
        self._arm_torques *= self.gains.kp_arm
        self._arm_torques -= (self.gains.kd_arm * data.qvel[self.arm_qvel_ids])

        # --- 3. 机械手 PD 计算 ---
        np.subtract(hand_target, data.qpos[self.hand_qpos_ids], out=self._hand_torques)
        self._hand_torques *= self.gains.kp_hand
        self._hand_torques -= (self.gains.kd_hand * data.qvel[self.hand_qvel_ids])

        # --- 4. 应用限制并下发 ---
        self._apply_saturation()
        self.base.apply_control(data, self._arm_torques, self._hand_torques)

    def _apply_saturation(self):
        """
        力矩饱和限制 (In-place 修改)。
        
        将计算出的力矩限制在执行器的物理能力范围内。
        """
        np.clip(self._arm_torques, self._torque_min[:self.base.ARM_DOF], 
                self._torque_max[:self.base.ARM_DOF], out=self._arm_torques)
        np.clip(self._hand_torques, self._torque_min[self.base.ARM_DOF:], 
                self._torque_max[self.base.ARM_DOF:], out=self._hand_torques)

    def _get_joint_range(self, qpos_ids):
        """
        根据位置自由度 ID 获取关节运动范围。
        
        Args:
            qpos_ids: 位置自由度索引列表。
        
        Returns:
            2D 数组，形状为 (n_joints, 2)，包含 [下限, 上限]。
        """
        ranges = []
        for qid in qpos_ids:
            # 找到该 qpos 对应的 joint 索引
            joint_id = np.where(self.model.jnt_qposadr == qid)[0][0]
            ranges.append(self.model.jnt_range[joint_id])
        return np.array(ranges)

    def _resolve_joint_ids(self, actuator_names):
        """
        将执行器名称列表转换为 MuJoCo 内部的 qpos 和 qvel 索引。
        
        MuJoCo 中 Actuator -> Joint -> qpos/qvel 是层级关系，需要通过查表转换。
        
        Args:
            actuator_names: 执行器名称列表。
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (qpos_ids, qvel_ids)
        """
        qpos_ids = []
        qvel_ids = []
        for name in actuator_names:
            act_id = self.base.actuator_map[name] # 名称 -> 执行器 ID
            joint_id = self.model.actuator_trnid[act_id, 0] # 执行器 ID -> 关节 ID
            qpos_ids.append(self.model.jnt_qposadr[joint_id]) # 关节 ID -> 位置索引
            qvel_ids.append(self.model.jnt_dofadr[joint_id]) # 关节 ID -> 速度索引
        return (np.array(qpos_ids, dtype=np.int32), np.array(qvel_ids, dtype=np.int32))