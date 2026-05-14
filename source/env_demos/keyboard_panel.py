"""
键盘控制面板（Tkinter）.

通过命令队列与仿真主线程解耦，支持 joint / ee 双模式。
"""

import queue
import threading
from typing import Dict

import numpy as np


# ====================== 常量 ======================

_ARM_JOINT_NAMES = [
    "J1 (Shoulder Yaw)", "J2 (Shoulder Pitch)", "J3 (Shoulder Roll)",
    "J4 (Elbow)", "J5 (Forearm)", "J6 (Wrist Pitch)", "J7 (Wrist Roll)",
]
_HAND_JOINT_NAMES = [
    "F0 (Actuator 0)", "F1 (Actuator 1)", "F2 (Actuator 2)",
    "F3 (Actuator 3)", "F4 (Actuator 4)", "F5 (Actuator 5)",
    "F2-4 Sync",
]
_EE_DOF_NAMES = [
    "EE X  (pos, m)", "EE Y  (pos, m)", "EE Z  (pos, m)",
    "EE Rx (rot, rad)", "EE Ry (rot, rad)", "EE Rz (rot, rad)",
]
_EE_ROT_NAMES = ["Roll", "Pitch", "Yaw"]


class KeyboardControlPanel:
    """
    Tkinter 键盘控制面板（joint / ee 双模式）.

    通过 cmd_queue 向仿真主线程发送命令：
        ("sel",   ±1)       切换关节
        ("delta", ±1.0)     调整当前关节
        ("reset", None)     重置回合
        ("open_hand", None) 张手
        ("close_hand", None) 握手
        ("gripper_open", None) 夹爪张开姿态
        ("quit", None)      退出
    """

    HISTORY_LEN = 200
    GRAPH_W = 460
    GRAPH_H  = 90

    def __init__(
        self,
        cmd_queue: queue.Queue,
        arm_dof: int = 7,
        hand_dof: int = 6,
        action_mode: str = "joint",
        arm_step: float = 0.05,
        hand_step: float = 0.0005,
        pos_step: float = 0.01,
        rot_step: float = 0.05,
    ):
        self.cmd_queue   = cmd_queue
        self.arm_dof     = arm_dof
        self.hand_dof    = hand_dof
        self.action_mode = action_mode

        self.ee_dof           = 6
        self.hand_display_num = hand_dof + 1  # 6 DOF + 1 Sync
        self.ctrl_dof         = (arm_dof if action_mode == "joint" else self.ee_dof) + self.hand_display_num

        self._lock         = threading.Lock()
        self._sel_idx      = 0
        self._display_vals = np.zeros(self.ctrl_dof)
        self._reward       = 0.0
        self._ep_reward    = 0.0
        self._step         = 0
        self._terminated   = False
        self._episode      = 1
        self._reward_hist: list = []
        self._info_extra: dict  = {}

        self._arm_step  = arm_step
        self._hand_step = hand_step
        self._pos_step  = pos_step
        self._rot_step  = rot_step

        self._root  = None
        self._ready = threading.Event()

    # ====================== 公开接口 ======================

    def update_state(self, sel_idx, display_vals, reward, ep_reward,
                     step, terminated, episode, info_extra):
        with self._lock:
            self._sel_idx      = sel_idx
            self._display_vals = display_vals.copy()
            self._reward       = reward
            self._ep_reward    = ep_reward
            self._step         = step
            self._terminated   = terminated
            self._episode      = episode
            self._info_extra   = dict(info_extra)
            self._reward_hist.append(reward)
            if len(self._reward_hist) > self.HISTORY_LEN:
                self._reward_hist.pop(0)

    def wait_ready(self, timeout: float = 10.0) -> None:
        self._ready.wait(timeout)

    def is_alive(self) -> bool:
        return self._root is not None and self._root.winfo_exists()

    # ====================== Tkinter 主循环 ======================

    def run(self) -> None:
        import tkinter as tk

        root = tk.Tk()
        self._root = root
        mode_label = "Joint Space" if self.action_mode == "joint" else "EE Space"
        root.title(f"Robot Arm Control [{mode_label}]  ← Click to Focus")
        root.configure(bg="#1e1e2e")
        root.resizable(False, False)

        C = dict(
            BG="#1e1e2e", BG2="#2a2a3e", FG="#cdd6f4",
            ACC="#89b4fa", ARM="#a6e3a1",
            EEP="#89dceb", EER="#cba6f7",
            HAND="#fab387", SEL="#f38ba8",
            WARN="#f9e2af", OK="#a6e3a1",
        )
        FM = ("Consolas", 11)
        FS = ("Consolas", 9)
        FT = ("Consolas", 14, "bold")

        tk.Label(root, text=f"Robot Arm + Dexterous Hand [{mode_label} Control]",
                 font=FT, bg=C["BG"], fg=C["ACC"]).pack(pady=(10, 2))
        tk.Label(root, text="Click the panel and use keyboard to control!",
                 font=FS, bg="#3e2a00", fg=C["WARN"]).pack(fill="x", padx=8, pady=(0, 2))

        hint = (" ←/→ Select Joint, ↑/↓ Adjust, R Reset, O Open, C Close, G Gripper, Q Quit "
                if self.action_mode == "joint"
                else " ←/→ Select DOF, ↑/↓ Adjust, R Reset, O Open, C Close, G Gripper, Q Quit ")
        tk.Label(root, text=hint, font=FS, bg=C["BG2"], fg=C["FG"]).pack(fill="x", padx=8, pady=(0, 4))

        # 步长设置
        sf = tk.Frame(root, bg=C["BG"])
        sf.pack(fill="x", padx=8, pady=2)
        self._build_step_controls(sf, C, FS)

        # 自由度列表
        jf = tk.Frame(root, bg=C["BG"])
        jf.pack(fill="x", padx=8, pady=4)
        self._ctrl_labels = []
        self._hand_labels = []
        self._build_dof_columns(jf, C, FM, FS)

        # 奖励/状态区
        rf = tk.Frame(root, bg=C["BG2"], padx=8, pady=6)
        rf.pack(fill="x", padx=8, pady=(4, 0))
        self._rwd_var    = tk.StringVar(value="Reward:  0.0000")
        self._ep_rwd_var = tk.StringVar(value="Cumulative:  0.0000")
        self._step_var   = tk.StringVar(value="Step:  0")
        self._ep_var     = tk.StringVar(value="Episode:  1")
        self._status_var = tk.StringVar(value="Status:  Running")
        self._extra_var  = tk.StringVar(value="")
        for var, col in [
            (self._rwd_var, C["FG"]), (self._ep_rwd_var, C["FG"]),
            (self._step_var, C["FG"]), (self._ep_var, C["ACC"]),
            (self._status_var, C["FG"]),
        ]:
            tk.Label(rf, textvariable=var, font=FM, bg=C["BG2"], fg=col, anchor="w").pack(fill="x")
        tk.Label(rf, textvariable=self._extra_var, font=FS, bg=C["BG2"],
                 fg=C["WARN"], anchor="w", wraplength=460).pack(fill="x")

        # 奖励折线图
        tk.Label(root, text="Reward History", font=FS, bg=C["BG"], fg=C["FG"]).pack(pady=(6, 0))
        self._canvas = tk.Canvas(root, width=self.GRAPH_W, height=self.GRAPH_H,
                                 bg="#11111b", highlightthickness=0)
        self._canvas.pack(padx=8, pady=(0, 8))

        self._COLORS = C
        root.bind("<KeyPress>", self._on_key)
        root.focus_set()
        self._ready.set()
        self._refresh()
        root.mainloop()
        self._root = None

    # ====================== 私有：UI 构建 ======================

    def _build_step_controls(self, sf, C, FS) -> None:
        import tkinter as tk

        def entry(label, var_name, default):
            tk.Label(sf, text=label, font=FS, bg=C["BG"], fg=C["FG"]).pack(side="left")
            setattr(self, var_name, tk.StringVar(value=str(default)))
            e = tk.Entry(sf, textvariable=getattr(self, var_name), width=6,
                         font=FS, bg=C["BG2"], fg=C["FG"], insertbackground=C["FG"])
            e.pack(side="left", padx=(2, 12))
            e.bind("<Return>", lambda *_: self._sync_steps())

        if self.action_mode == "joint":
            entry("Arm Step (rad):", "_arm_step_var", self._arm_step)
        else:
            entry("Position Step (m):", "_pos_step_var", self._pos_step)
            entry("Rotation Step (rad):", "_rot_step_var", self._rot_step)
        entry("Hand Step (m):", "_hand_step_var", self._hand_step)

    def _build_dof_columns(self, jf, C, FM, FS) -> None:
        import tkinter as tk

        lc = tk.Frame(jf, bg=C["BG2"], padx=6, pady=4)
        lc.pack(side="left", fill="y", padx=(0, 4))

        if self.action_mode == "joint":
            tk.Label(lc, text="── Arm 7 DOF [target rad] ──", font=FS, bg=C["BG2"], fg=C["ARM"]).pack()
            for i in range(self.arm_dof):
                lbl = tk.Label(lc, text=f"[{i}] {_ARM_JOINT_NAMES[i]}: 0.000",
                               font=FM, bg=C["BG2"], fg=C["ARM"], anchor="w", width=30)
                lbl.pack(fill="x")
                self._ctrl_labels.append((lbl, C["ARM"]))
        else:
            tk.Label(lc, text="── End-Effector Pose ──", font=FS, bg=C["BG2"], fg=C["EEP"]).pack()
            for i in range(3):
                lbl = tk.Label(lc, text=f"[{i}] {_EE_DOF_NAMES[i]}: 0.000",
                               font=FM, bg=C["BG2"], fg=C["EEP"], anchor="w", width=30)
                lbl.pack(fill="x")
                self._ctrl_labels.append((lbl, C["EEP"]))
            for i in range(3):
                lbl = tk.Label(lc, text=f"[{3+i}] EE {_EE_ROT_NAMES[i]:5s} (deg): 0.00°",
                               font=FM, bg=C["BG2"], fg=C["EER"], anchor="w", width=30)
                lbl.pack(fill="x")
                self._ctrl_labels.append((lbl, C["EER"]))

        rc = tk.Frame(jf, bg=C["BG2"], padx=6, pady=4)
        rc.pack(side="left", fill="y")
        ee_offset = self.arm_dof if self.action_mode == "joint" else self.ee_dof
        tk.Label(rc, text="── Hand 6 Dof [target m] ──", font=FS, bg=C["BG2"], fg=C["HAND"]).pack()
        for i in range(self.hand_display_num):
            idx = ee_offset + i
            lbl = tk.Label(rc, text=f"[{idx}] {_HAND_JOINT_NAMES[i]}: 0.00000",
                           font=FM, bg=C["BG2"], fg=C["HAND"], anchor="w", width=28)
            lbl.pack(fill="x")
            self._hand_labels.append(lbl)

    # ====================== 私有：事件处理 ======================

    def _on_key(self, event) -> None:
        key = event.keysym
        mapping = {
            "Right": ("sel", +1), "Left": ("sel", -1),
            "Up": ("delta", +1.0), "Down": ("delta", -1.0),
            "r": ("reset", None), "R": ("reset", None),
            "o": ("open_hand", None), "O": ("open_hand", None),
            "c": ("close_hand", None), "C": ("close_hand", None),
            "g": ("gripper_open", None), "G": ("gripper_open", None),
            "q": ("quit", None), "Q": ("quit", None),
        }
        if key in mapping:
            self.cmd_queue.put(mapping[key])

    def _sync_steps(self) -> None:
        def _safe_float(var_name, attr):
            try:
                setattr(self, attr, float(getattr(self, var_name).get()))
            except (ValueError, AttributeError):
                pass
        _safe_float("_hand_step_var", "_hand_step")
        if self.action_mode == "joint":
            _safe_float("_arm_step_var", "_arm_step")
        else:
            _safe_float("_pos_step_var", "_pos_step")
            _safe_float("_rot_step_var", "_rot_step")

    # ====================== 私有：刷新 ======================

    def _refresh(self) -> None:
        with self._lock:
            sel   = self._sel_idx
            vals  = self._display_vals.copy()
            rwd   = self._reward
            ep_r  = self._ep_reward
            step  = self._step
            term  = self._terminated
            ep    = self._episode
            hist  = list(self._reward_hist)
            extra = dict(self._info_extra)

        C         = self._COLORS
        ee_offset = self.arm_dof if self.action_mode == "joint" else self.ee_dof

        # 左列标签
        for i, (lbl, base_col) in enumerate(self._ctrl_labels):
            val    = vals[i] if i < len(vals) else 0.0
            is_sel = i == sel
            if self.action_mode == "joint":
                text = f"[{i}] {_ARM_JOINT_NAMES[i]}: {val:+.3f}"
            elif i < 3:
                text = f"[{i}] {_EE_DOF_NAMES[i]}: {val:+.4f} m"
            else:
                text = f"[{i}] EE {_EE_ROT_NAMES[i-3]:5s}: {val:+.2f}°"
            lbl.config(
                text=text,
                fg=C["SEL"] if is_sel else base_col,
                font=("Consolas", 11, "bold") if is_sel else ("Consolas", 11),
                bg="#3e1e2e" if is_sel else C["BG2"],
            )

        # 右列标签
        for i, lbl in enumerate(self._hand_labels):
            idx    = ee_offset + i
            is_sel = idx == sel
            if i == 6:
                v = [(vals[ee_offset + j] if (ee_offset + j) < len(vals) else 0.0) for j in [2, 3, 4]]
                text = f"[{idx}] {_HAND_JOINT_NAMES[i]}: {sum(v)/3:+.5f} (avg)"
            else:
                val  = vals[idx] if idx < len(vals) else 0.0
                text = f"[{idx}] {_HAND_JOINT_NAMES[i]}: {val:+.5f}"
            lbl.config(
                text=text,
                fg=C["SEL"] if is_sel else C["HAND"],
                font=("Consolas", 11, "bold") if is_sel else ("Consolas", 11),
                bg="#3e1e2e" if is_sel else C["BG2"],
            )

        # 状态
        self._rwd_var.set(f"Reward:  {'▲' if rwd >= 0 else '▼'} {rwd:+.4f}")
        self._ep_rwd_var.set(f"Cumulative:  {ep_r:+.4f}")
        self._step_var.set(f"Step:  {step}  (no timeout)")
        self._ep_var.set(f"Episode:  {ep}")
        self._status_var.set(
            "Status:  Success — Press R to reset" if term else "Status:  ▶ Running..."
        )

        skip = {"episode_steps", "episode_reward", "episode_count"}
        parts = []
        for k, v in extra.items():
            if k in skip:
                continue
            if isinstance(v, float):
                parts.append(f"{k}={v:.3f}")
            elif isinstance(v, bool):
                parts.append(f"{k}={'✓' if v else '✗'}")
            elif isinstance(v, np.ndarray):
                parts.append(f"{k}=[{', '.join(f'{x:.2f}' for x in v.flat)}]")
            else:
                parts.append(f"{k}={v}")
        self._extra_var.set("  ".join(parts))

        # 折线图
        cv = self._canvas
        cv.delete("all")
        W, H, pad = self.GRAPH_W, self.GRAPH_H, 6
        cv.create_line(pad, H // 2, W - pad, H // 2, fill="#313244", width=1)
        if len(hist) >= 2:
            mn, mx = min(hist), max(hist)
            span   = max(mx - mn, 1e-6)
            def _y(v): return H - pad - (v - mn) / span * (H - 2 * pad)
            pts = []
            for i, v in enumerate(hist):
                pts.extend([pad + i / (len(hist) - 1) * (W - 2 * pad), _y(v)])
            cv.create_line(*pts, fill="#89b4fa", width=1, smooth=True)
            lv = hist[-1]
            lx, ly = W - pad, _y(lv)
            cv.create_oval(lx-3, ly-3, lx+3, ly+3, fill="#f38ba8", outline="")
            cv.create_text(lx-4, ly-10, text=f"{lv:+.3f}", fill="#f38ba8",
                           font=("Consolas", 8), anchor="e")
            cv.create_text(pad+2, H-pad, text=f"min:{mn:.3f}", fill="#6c7086",
                           font=("Consolas", 8), anchor="sw")
            cv.create_text(pad+2, pad, text=f"max:{mx:.3f}", fill="#6c7086",
                           font=("Consolas", 8), anchor="nw")

        self._root.after(80, self._refresh)