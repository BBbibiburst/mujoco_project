import numpy as np
import mujoco
from dataclasses import dataclass, field

@dataclass
class PDGains:
    kp_arm: np.ndarray = field(default_factory=lambda: np.full(7, 4000.))
    kd_arm: np.ndarray = field(default_factory=lambda: np.full(7, 400.))
    kp_hand: np.ndarray = field(default_factory=lambda: np.full(6, 4000.))
    kd_hand: np.ndarray = field(default_factory=lambda: np.full(6, 400.))

class PositionController:
    def __init__(self, base, model, gains: PDGains | None = None):
        self.base = base
        self.model = model
        self.gains = gains if gains is not None else PDGains()

        # 关节索引解析
        self.arm_qpos_ids, self.arm_qvel_ids = self._resolve_joint_ids(base.arm_names)
        self.hand_qpos_ids, self.hand_qvel_ids = self._resolve_joint_ids(
            [base.hand_names[k] for k in base.hand_key_order if k in base.hand_names]
        )

        # 自动读取范围和限制
        self.arm_range = self._get_joint_range(self.arm_qpos_ids)
        self.hand_range = self._get_joint_range(self.hand_qpos_ids)
        self._torque_min = base.torque_min
        self._torque_max = base.torque_max

        self._arm_torques = np.zeros(base.ARM_DOF)
        self._hand_torques = np.zeros(base.HAND_DOF)

        # --- IK 相关初始化 ---
        # 末端执行器的 site 名称，如果你的 model 里叫别的名字请修改
        self.ee_site_name = "right_hand_site" 
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)
        
        # 用于 IK 计算的临时变量
        self.jac_p = np.zeros((3, model.nv)) # 位置雅可比
        self.jac_r = np.zeros((3, model.nv)) # 旋转雅可比

    # -----------------------------
    # 新增：末端位置控制接口 (IK)
    # -----------------------------
    def set_ee_target(self, data, ee_pos_target, ee_quat_target=None, hand_target=None):
        """
        通过末端位置控制机械臂
        :param ee_pos_target: 目标位置 [x, y, z]
        :param ee_quat_target: 目标姿态 [w, x, y, z] (可选，None则只追逐位置)
        :param hand_target: 手部关节目标 (复用原逻辑)
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
        # 限制范围
        arm_target = np.clip(arm_target, self.arm_range[:, 0], self.arm_range[:, 1])
        hand_target = np.clip(hand_target, self.hand_range[:, 0], self.hand_range[:, 1])

        # --- arm PD ---
        np.subtract(arm_target, data.qpos[self.arm_qpos_ids], out=self._arm_torques)
        self._arm_torques *= self.gains.kp_arm
        self._arm_torques -= (self.gains.kd_arm * data.qvel[self.arm_qvel_ids])

        # --- hand PD ---
        np.subtract(hand_target, data.qpos[self.hand_qpos_ids], out=self._hand_torques)
        self._hand_torques *= self.gains.kp_hand
        self._hand_torques -= (self.gains.kd_hand * data.qvel[self.hand_qvel_ids])

        self._apply_saturation()
        self.base.apply_control(data, self._arm_torques, self._hand_torques)

    def _apply_saturation(self):
        np.clip(self._arm_torques, self._torque_min[:self.base.ARM_DOF], 
                self._torque_max[:self.base.ARM_DOF], out=self._arm_torques)
        np.clip(self._hand_torques, self._torque_min[self.base.ARM_DOF:], 
                self._torque_max[self.base.ARM_DOF:], out=self._hand_torques)

    def _get_joint_range(self, qpos_ids):
        ranges = []
        for qid in qpos_ids:
            joint_id = np.where(self.model.jnt_qposadr == qid)[0][0]
            ranges.append(self.model.jnt_range[joint_id])
        return np.array(ranges)

    def _resolve_joint_ids(self, actuator_names):
        qpos_ids = []
        qvel_ids = []
        for name in actuator_names:
            act_id = self.base.actuator_map[name]
            joint_id = self.model.actuator_trnid[act_id, 0]
            qpos_ids.append(self.model.jnt_qposadr[joint_id])
            qvel_ids.append(self.model.jnt_dofadr[joint_id])
        return (np.array(qpos_ids, dtype=np.int32), np.array(qvel_ids, dtype=np.int32))