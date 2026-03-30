import mujoco
import numpy as np
from typing import Dict, List


class HandArmController:
    ARM_DOF = 7
    HAND_DOF = 6
    TOTAL_DOF = 13

    def __init__(self, model: mujoco.MjModel):
        self.model = model

        self.actuator_map: Dict[str, int] = {}
        self.arm_names: List[str] = []
        self.hand_names: Dict[str, str] = {}

        self._build_map()
        self._build_index_arrays()
        self._extract_actuator_limits()   # ⭐ 新增

    # -----------------------------
    # 构建映射
    # -----------------------------
    def _build_map(self):
        for i in range(self.model.nu):
            name = self.model.actuator(i).name
            self.actuator_map[name] = i

            if name.startswith("torq_joint"):
                self.arm_names.append(name)

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

        self.arm_names.sort(key=lambda x: int(x.replace("torq_joint", "")))

    # -----------------------------
    # 索引数组
    # -----------------------------
    def _build_index_arrays(self):
        self.arm_indices = np.array(
            [self.actuator_map[n] for n in self.arm_names],
            dtype=np.int32
        )

        self.hand_key_order = [
            "finger_0", "finger_1", "finger_2", "finger_3",
            "thumb_grasp", "thumb_rotate"
        ]

        self.hand_indices = np.array(
            [self.actuator_map[self.hand_names[k]]
             for k in self.hand_key_order if k in self.hand_names],
            dtype=np.int32
        )

        self.all_indices = np.concatenate([self.arm_indices, self.hand_indices])

    # -----------------------------
    # ⭐ 提取 actuator 约束
    # -----------------------------
    def _extract_actuator_limits(self):
        ctrl_min = []
        ctrl_max = []
        gear = []

        for i in range(self.model.nu):
            ctrl_min.append(self.model.actuator_ctrlrange[i][0])
            ctrl_max.append(self.model.actuator_ctrlrange[i][1])
            gear.append(self.model.actuator_gear[i][0])  # scalar actuator

        self.ctrl_min = np.array(ctrl_min)
        self.ctrl_max = np.array(ctrl_max)
        self.gear = np.array(gear)

        # 👉 转换为真实 torque 限制
        self.torque_min = self.ctrl_min * self.gear
        self.torque_max = self.ctrl_max * self.gear

    # -----------------------------
    # 控制接口（已修复）
    # -----------------------------
    def apply_control(self, data, arm_torques, hand_torques):

        if arm_torques.shape != (self.ARM_DOF,):
            raise ValueError
        if hand_torques.shape != (self.HAND_DOF,):
            raise ValueError

        # 合并 torque
        all_torque = np.concatenate([arm_torques, hand_torques])

        # ⭐ torque → ctrl
        ctrl = all_torque / self.gear[self.all_indices]

        # ⭐ ctrl 限幅（最终安全层）
        ctrl = np.clip(
            ctrl,
            self.ctrl_min[self.all_indices],
            self.ctrl_max[self.all_indices]
        )

        data.ctrl[self.all_indices] = ctrl

    def apply_control_vector(self, data, control_vector):
        if control_vector.shape != (self.TOTAL_DOF,):
            raise ValueError

        # 同样需要 torque→ctrl
        ctrl = control_vector / self.gear[self.all_indices]
        ctrl = np.clip(
            ctrl,
            self.ctrl_min[self.all_indices],
            self.ctrl_max[self.all_indices]
        )

        data.ctrl[self.all_indices] = ctrl