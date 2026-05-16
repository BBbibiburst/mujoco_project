"""
机械臂抓取演示 - Matplotlib 3D 动画
用法:
    python -m source.env_demos.log_visualizer <data.jsonl>
"""

import json
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────── 颜色主题 ────────────────────────────
BG     = "#0d1117"
PANEL  = "#161b22"
GREEN  = "#39d353"
CYAN   = "#58d6f5"
ORANGE = "#f0883e"
RED    = "#f85149"
YELLOW = "#e3b341"
WHITE  = "#e6edf3"
GRAY   = "#8b949e"
PURPLE = "#bc8cff"


# ──────────────────────────── 数据加载 ────────────────────────────
def load_jsonl(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ──────────────────────────── 简化正向运动学（7-DOF） ────────────────────────────
def fk_positions(qpos, base=None):
    """
    7-DOF 串联机械臂简化正向运动学。
    返回从基座到末端共 8 个关节点坐标。
    连杆长度参照常见桌面机械臂比例（单位：米）。
    """
    L    = [0.0, 0.15, 0.20, 0.15, 0.20, 0.10, 0.10, 0.08]
    base = np.array([0.0, 0.0, 0.58]) if base is None else np.array(base)

    # 交替使用 Z/Y 轴（简化 DH）
    axes = [
        np.array([0, 0, 1]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
    ]

    R   = np.eye(3)
    pos = base.copy()
    pts = [base.copy()]

    for i, (q, ax) in enumerate(zip(qpos, axes)):
        c, s = np.cos(q), np.sin(q)
        K = np.array([[     0, -ax[2],  ax[1]],
                      [ ax[2],      0, -ax[0]],
                      [-ax[1],  ax[0],      0]])
        dR  = np.eye(3) + s * K + (1 - c) * (K @ K)
        R   = R @ dR
        pos = pos + R @ np.array([0, 0, L[i + 1]])
        pts.append(pos.copy())

    return pts   # 8 个 np.array([x, y, z])


# ──────────────────────────── 四元数 → 旋转矩阵 ────────────────────────────
def quat_to_rot(q):
    """[w, x, y, z] → 3×3 旋转矩阵"""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


# ──────────────────────────── 主程序 ────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("用法: python robot_arm_demo.py <data.jsonl>")
        sys.exit(1)

    records = load_jsonl(sys.argv[1])
    N = len(records)
    print(f"已加载 {N} 帧数据")

    # ── 预计算 ────────────────────────────────────────────────────
    ee_pos_all    = []
    block_pos_all = []
    arm_fk_all    = []
    rewards       = []
    heights       = []
    phase_idx_all = []
    target_pos    = None

    for rec in records:
        info  = rec["info"]
        extra = rec.get("extra", {})

        ee_pos_all.append(np.array(info["end_effector"]["position"]))
        block_pos_all.append(np.array(info["block"]["current_position"]))
        arm_fk_all.append(fk_positions(info["arm_qpos"]))
        rewards.append(info["episode_reward"])
        heights.append(info["block"]["current_height"])
        phase_idx_all.append(extra.get("phase_idx", 0))

        if target_pos is None:
            target_pos = np.array(info["target_marker"]["position"])

    ee_pos_all    = np.array(ee_pos_all)
    block_pos_all = np.array(block_pos_all)
    rewards       = np.array(rewards)
    heights       = np.array(heights)
    target_height = records[0]["info"]["block"]["target_height"]
    floor_z       = records[0]["info"]["block"]["initial_position"][2] - 0.10

    # ── 全局样式 ──────────────────────────────────────────────────
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor":   PANEL,
        "text.color":       WHITE,
        "axes.labelcolor":  WHITE,
        "xtick.color":      GRAY,
        "ytick.color":      GRAY,
        "axes.edgecolor":   GRAY,
        "grid.color":       "#21262d",
        "font.family":      "monospace",
    })

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.suptitle("■  Robot Arm Grasp  ·  Episode Replay",
                 fontsize=14, color=WHITE, fontweight="bold", y=0.98)

    gs = GridSpec(3, 3, figure=fig,
                  left=0.04, right=0.98,
                  top=0.93, bottom=0.07,
                  wspace=0.38, hspace=0.60)

    ax3d   = fig.add_subplot(gs[:, :2], projection="3d")
    ax_rwd = fig.add_subplot(gs[0, 2])
    ax_ht  = fig.add_subplot(gs[1, 2])
    ax_ee  = fig.add_subplot(gs[2, 2])

    for ax in [ax_rwd, ax_ht, ax_ee]:
        ax.set_facecolor(PANEL)
        ax.grid(True, alpha=0.3)

    # ── 3D 轴外观 ──────────────────────────────────────────────────
    ax3d.set_facecolor(BG)
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#30363d")
    ax3d.tick_params(colors=GRAY, labelsize=7)
    ax3d.set_xlabel("X (m)", color=GRAY, fontsize=8, labelpad=2)
    ax3d.set_ylabel("Y (m)", color=GRAY, fontsize=8, labelpad=2)
    ax3d.set_zlabel("Z (m)", color=GRAY, fontsize=8, labelpad=2)
    ax3d.set_title("3D Scene", color=WHITE, pad=6, fontsize=10)

    # 视角范围
    all_pts = np.vstack([ee_pos_all, block_pos_all, [target_pos]])
    margin  = 0.20
    xc, yc, zc = all_pts.mean(axis=0)
    span    = max(all_pts.ptp(axis=0)) / 2 + margin
    ax3d.set_xlim(xc - span, xc + span)
    ax3d.set_ylim(yc - span, yc + span)
    ax3d.set_zlim(floor_z, max(all_pts[:, 2]) + 0.12)
    ax3d.view_init(elev=22, azim=-55)

    # ── 静态场景元素 ──────────────────────────────────────────────
    # 地面
    gx = np.linspace(xc - span, xc + span, 8)
    gy = np.linspace(yc - span, yc + span, 8)
    GX, GY = np.meshgrid(gx, gy)
    GZ = np.full_like(GX, floor_z)
    ax3d.plot_surface(GX, GY, GZ, alpha=0.07, color=GRAY, zorder=0)

    # 基座柱
    theta = np.linspace(0, 2 * np.pi, 20)
    r = 0.04
    bx = r * np.cos(theta)
    by = r * np.sin(theta)
    bz0 = np.full_like(theta, floor_z)
    bz1 = np.full_like(theta, 0.58)
    for i in range(len(theta) - 1):
        ax3d.plot([bx[i], bx[i+1]], [by[i], by[i+1]],
                  [floor_z, floor_z], color=GRAY, alpha=0.3, lw=0.5)
    ax3d.plot_surface(
        np.array([bx, bx]),
        np.array([by, by]),
        np.array([bz0, bz1]),
        alpha=0.25, color=CYAN
    )

    # 目标标记
    d = 0.03
    ax3d.plot([target_pos[0]-d, target_pos[0]+d],
              [target_pos[1],   target_pos[1]],
              [target_pos[2],   target_pos[2]],
              color=RED, lw=2, zorder=10)
    ax3d.plot([target_pos[0],   target_pos[0]],
              [target_pos[1]-d, target_pos[1]+d],
              [target_pos[2],   target_pos[2]],
              color=RED, lw=2, zorder=10)
    ax3d.text(target_pos[0], target_pos[1], target_pos[2] + 0.018,
              "TARGET", color=RED, fontsize=6.5, ha="center", zorder=10)

    # 全轨迹底色
    ax3d.plot(ee_pos_all[:, 0], ee_pos_all[:, 1], ee_pos_all[:, 2],
              color=CYAN, alpha=0.10, lw=1, linestyle="--")
    ax3d.plot(block_pos_all[:, 0], block_pos_all[:, 1], block_pos_all[:, 2],
              color=ORANGE, alpha=0.10, lw=1, linestyle="--")

    # ── 动态句柄 ──────────────────────────────────────────────────
    arm_line, = ax3d.plot([], [], [], color=CYAN, lw=3, zorder=6,
                           marker="o", markersize=5,
                           markerfacecolor=WHITE, markeredgecolor=CYAN)

    ee_dot     = ax3d.scatter([], [], [], s=90,  c=GREEN,  zorder=8, depthshade=False)
    block_cube = ax3d.scatter([], [], [], s=220, c=ORANGE, marker="s",
                               zorder=8, depthshade=False)
    trace_ee,  = ax3d.plot([], [], [], color=CYAN,   alpha=0.7, lw=1.5)
    trace_blk, = ax3d.plot([], [], [], color=ORANGE, alpha=0.5, lw=1.2)

    ee_arrows = []   # 末端坐标轴箭头

    status_txt = ax3d.text2D(
        0.02, 0.97, "",
        transform=ax3d.transAxes, color=WHITE, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc=BG, ec=GRAY, alpha=0.85)
    )

    # ── 右侧小图 ──────────────────────────────────────────────────
    steps_arr = np.arange(N)

    ax_rwd.set_title("Cumulative Reward", color=WHITE, fontsize=9)
    ax_rwd.plot(steps_arr, rewards, color=GREEN, alpha=0.20, lw=1)
    rwd_vline = ax_rwd.axvline(0, color=GREEN, lw=1.5, alpha=0.8)
    rwd_dot,  = ax_rwd.plot([], [], "o", color=GREEN, ms=5)
    ax_rwd.set_xlim(0, N - 1)
    ax_rwd.set_ylabel("reward", color=GRAY, fontsize=7)
    ax_rwd.tick_params(labelsize=7)

    ax_ht.set_title("Block Height", color=WHITE, fontsize=9)
    ax_ht.plot(steps_arr, heights, color=ORANGE, alpha=0.20, lw=1)
    ax_ht.axhline(target_height, color=RED, lw=1.2, linestyle="--",
                  label=f"target {target_height:.3f}m")
    ax_ht.legend(fontsize=6.5, labelcolor=RED, facecolor=PANEL, edgecolor=GRAY)
    ht_vline = ax_ht.axvline(0, color=ORANGE, lw=1.5, alpha=0.8)
    ht_dot,  = ax_ht.plot([], [], "o", color=ORANGE, ms=5)
    ax_ht.set_xlim(0, N - 1)
    ax_ht.set_ylabel("height (m)", color=GRAY, fontsize=7)
    ax_ht.tick_params(labelsize=7)

    ax_ee.set_title("End-Effector XY", color=WHITE, fontsize=9)
    ax_ee.plot(ee_pos_all[:, 0], ee_pos_all[:, 1], color=CYAN, alpha=0.15, lw=1)
    ax_ee.plot(target_pos[0], target_pos[1], "x", color=RED, ms=8, mew=2)
    ee_xy_dot, = ax_ee.plot([], [], "o", color=CYAN, ms=6)
    ax_ee.set_xlabel("X (m)", color=GRAY, fontsize=7)
    ax_ee.set_ylabel("Y (m)", color=GRAY, fontsize=7)
    ax_ee.tick_params(labelsize=7)
    ax_ee.set_aspect("equal", "box")

    # ── 进度条 ──────────────────────────────────────────────────
    prog_ax = fig.add_axes([0.04, 0.015, 0.94, 0.012])
    prog_ax.set_xlim(0, N)
    prog_ax.set_ylim(0, 1)
    prog_ax.axis("off")
    prog_ax.barh(0.5, N, height=1, color="#21262d", align="center")
    prog_bar = prog_ax.barh(0.5, 0, height=1, color=PURPLE, align="center")
    prog_txt = prog_ax.text(N / 2, 0.5, f"0 / {N}",
                             ha="center", va="center", color=WHITE, fontsize=8)

    # ── 帧更新 ────────────────────────────────────────────────────
    def update(fi):
        rec   = records[fi]
        info  = rec["info"]
        extra = rec.get("extra", {})

        # 机械臂骨架
        fk = arm_fk_all[fi]
        arm_line.set_data([p[0] for p in fk], [p[1] for p in fk])
        arm_line.set_3d_properties([p[2] for p in fk])

        # 末端执行器点
        ep = ee_pos_all[fi]
        ee_dot._offsets3d = ([ep[0]], [ep[1]], [ep[2]])

        # 末端坐标轴箭头
        for a in ee_arrows:
            a.remove()
        ee_arrows.clear()
        R = quat_to_rot(info["end_effector"]["quaternion"])
        L_arr = 0.045
        for col, axis_col in zip([RED, GREEN, CYAN], R.T):
            end = ep + axis_col * L_arr
            ln, = ax3d.plot([ep[0], end[0]], [ep[1], end[1]], [ep[2], end[2]],
                             color=col, lw=1.8, zorder=9)
            ee_arrows.append(ln)

        # 方块
        bp = block_pos_all[fi]
        block_cube._offsets3d = ([bp[0]], [bp[1]], [bp[2]])
        grasped = info["block"]["grasp_success"]
        block_cube._facecolors = np.array([[
            *matplotlib.colors.to_rgb(GREEN if grasped else ORANGE), 1.0
        ]])

        # 截至当前帧的轨迹
        trace_ee.set_data(ee_pos_all[:fi+1, 0], ee_pos_all[:fi+1, 1])
        trace_ee.set_3d_properties(ee_pos_all[:fi+1, 2])
        trace_blk.set_data(block_pos_all[:fi+1, 0], block_pos_all[:fi+1, 1])
        trace_blk.set_3d_properties(block_pos_all[:fi+1, 2])

        # 右侧小图
        rwd_vline.set_xdata([fi])
        rwd_dot.set_data([fi], [rewards[fi]])
        ht_vline.set_xdata([fi])
        ht_dot.set_data([fi], [heights[fi]])
        ee_xy_dot.set_data([ep[0]], [ep[1]])

        # 进度条
        prog_bar[0].set_width(fi + 1)
        prog_txt.set_text(f"Step {fi+1:>4d} / {N}")

        # 状态文字
        ph     = extra.get("phase_idx", "?")
        succ   = "✓ SUCCESS" if extra.get("success") else ""
        lifted = "↑ LIFTED"  if info["block"]["is_lifted"] else ""
        grasp  = "GRASP: OK" if grasped else "GRASP: --"
        status_txt.set_text(
            f"Phase  : {ph}  {succ}\n"
            f"Status : {grasp}  {lifted}\n"
            f"Height : {info['block']['current_height']:.4f} m\n"
            f"Reward : {extra.get('reward', 0.0):+.3f}"
        )

        return (arm_line, ee_dot, block_cube, trace_ee, trace_blk,
                rwd_vline, rwd_dot, ht_vline, ht_dot, ee_xy_dot,
                status_txt, *ee_arrows)

    # ── 启动动画 ──────────────────────────────────────────────────
    ani = animation.FuncAnimation(
        fig, update, frames=N,
        interval=80,        # ms/frame，约 12fps；可改小加速
        blit=False, repeat=True
    )

    plt.show()
    return ani   # 防止 GC 提前回收


if __name__ == "__main__":
    ani = main()