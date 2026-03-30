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

        # ⭐ 关键：允许外部传入 gains
        self.gains = gains if gains is not None else PDGains()

        # joint 索引
        self.arm_qpos_ids, self.arm_qvel_ids = self._resolve_joint_ids(
            base.arm_names)
        self.hand_qpos_ids, self.hand_qvel_ids = self._resolve_joint_ids(
            [base.hand_names[k]
             for k in base.hand_key_order if k in base.hand_names]
        )

        # ⭐ 自动读取 joint range
        self.arm_range = self._get_joint_range(self.arm_qpos_ids)
        self.hand_range = self._get_joint_range(self.hand_qpos_ids)

        # ⭐ 使用真实 torque 限制
        self._torque_min = base.torque_min
        self._torque_max = base.torque_max

        self._arm_torques = np.zeros(base.ARM_DOF)
        self._hand_torques = np.zeros(base.HAND_DOF)

    # -----------------------------
    # 主控制
    # -----------------------------
    def set_target(self, data, arm_target, hand_target):

        # ⭐ 使用模型范围
        arm_target = np.clip(
            arm_target,
            self.arm_range[:, 0],
            self.arm_range[:, 1]
        )

        hand_target = np.clip(
            hand_target,
            self.hand_range[:, 0],
            self.hand_range[:, 1]
        )

        # --- arm PD ---
        np.subtract(arm_target,
                    data.qpos[self.arm_qpos_ids],
                    out=self._arm_torques)

        self._arm_torques *= self.gains.kp_arm
        self._arm_torques -= (
            self.gains.kd_arm * data.qvel[self.arm_qvel_ids]
        )

        # --- hand PD ---
        np.subtract(hand_target,
                    data.qpos[self.hand_qpos_ids],
                    out=self._hand_torques)

        self._hand_torques *= self.gains.kp_hand
        self._hand_torques -= (
            self.gains.kd_hand * data.qvel[self.hand_qvel_ids]
        )

        # ⭐ torque 限幅（物理层）
        self._apply_saturation()

        # 下发
        self.base.apply_control(
            data,
            self._arm_torques,
            self._hand_torques
        )

    # -----------------------------
    # torque 限幅
    # -----------------------------
    def _apply_saturation(self):
        np.clip(
            self._arm_torques,
            self._torque_min[:self.base.ARM_DOF],
            self._torque_max[:self.base.ARM_DOF],
            out=self._arm_torques
        )

        np.clip(
            self._hand_torques,
            self._torque_min[self.base.ARM_DOF:],
            self._torque_max[self.base.ARM_DOF:],
            out=self._hand_torques
        )

    # -----------------------------
    # joint range
    # -----------------------------
    def _get_joint_range(self, qpos_ids):
        ranges = []
        for qid in qpos_ids:
            joint_id = np.where(self.model.jnt_qposadr == qid)[0][0]
            ranges.append(self.model.jnt_range[joint_id])
        return np.array(ranges)

    # -----------------------------
    # joint id
    # -----------------------------
    def _resolve_joint_ids(self, actuator_names):
        qpos_ids = []
        qvel_ids = []

        for name in actuator_names:
            act_id = self.base.actuator_map[name]
            joint_id = self.model.actuator_trnid[act_id, 0]

            qpos_ids.append(self.model.jnt_qposadr[joint_id])
            qvel_ids.append(self.model.jnt_dofadr[joint_id])

        return (np.array(qpos_ids, dtype=np.int32),
                np.array(qvel_ids, dtype=np.int32))