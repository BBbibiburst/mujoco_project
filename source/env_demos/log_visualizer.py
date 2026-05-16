"""
机械臂抓取演示 - Matplotlib 3D 动画（真实 FK，RM75B + InspireHand）
用法:
    python -m source.env_demos.log_visualizer <data.jsonl>

两个 XML 均不需要 mesh/STL 文件，脚本自动剥离 geom/asset。
"""

import json
import re
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import mujoco
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────── 颜色主题 ────────────────────────────
BG     = "#0d1117"
PANEL  = "#161b22"
GREEN  = "#39d353"
CYAN   = "#58d6f5"
ORANGE = "#f0883e"
RED    = "#f85149"
WHITE  = "#e6edf3"
GRAY   = "#8b949e"
PURPLE = "#bc8cff"
YELLOW = "#e3b341"


# ──────────────────────────── XML 预处理 ──────────────────────────
def strip_mesh(xml: str) -> str:
    """去除 asset/geom，只保留运动学骨架。"""
    xml = re.sub(r'<asset>.*?</asset>', '<asset/>', xml, flags=re.DOTALL)
    xml = re.sub(r'<geom\b[^/]*/>', '', xml)
    xml = re.sub(r'<geom\b[^>]*>.*?</geom>', '', xml, flags=re.DOTALL)
    return xml


# ──────────────────────────── 机械臂 FK ───────────────────────────
# RM75B body 顺序（含世界坐标系中的基座 root）
ARM_BODIES = ["root", "Link1", "Link2", "Link3",
              "Link4", "Link5", "Link6", "Link7", "right_hand"]

_arm_model = None
_arm_data  = None
_arm_bids  = None


def init_arm(xml_path: str) -> None:
    global _arm_model, _arm_data, _arm_bids
    raw = strip_mesh(open(xml_path, encoding="utf-8").read())
    _arm_model = mujoco.MjModel.from_xml_string(raw)
    _arm_data  = mujoco.MjData(_arm_model)
    _arm_bids  = [
        mujoco.mj_name2id(_arm_model, mujoco.mjtObj.mjOBJ_BODY, n)
        for n in ARM_BODIES
    ]
    print(f"[Arm]  {_arm_model.nbody} bodies, nq={_arm_model.nq}")


def arm_fk(qpos: np.ndarray) -> np.ndarray:
    """返回机械臂各 body 世界坐标，shape (9, 3)。"""
    _arm_data.qpos[:_arm_model.nq] = qpos[:_arm_model.nq]
    mujoco.mj_kinematics(_arm_model, _arm_data)
    return np.array([_arm_data.xpos[bid].copy() for bid in _arm_bids])


# ──────────────────────────── 灵巧手 FK ───────────────────────────
# hand_qpos 6维（数据顺序）→ actuator 顺序（见 XML actuator 定义）
#   [0] act_push_0_j   (qadr=1,  slide) → finger 0 弯曲
#   [1] act_push_1_j   (qadr=6,  slide) → finger 1 弯曲
#   [2] act_push_2_j   (qadr=11, slide) → finger 2 弯曲
#   [3] act_push_3_j   (qadr=16, slide) → finger 3 弯曲
#   [4] thumb_grasp_act_push_j (qadr=26, slide) → 拇指抓握
#   [5] thumb_rotate_act_push_j(qadr=21, slide) → 拇指旋转
#
# 联动关系（equality constraint + 几何）：
#   act_push slide 0→0.01  ↔  finger_first angle 0→1.657 rad（线性）
#   thumb_grasp slide 0→0.01 ↔  thumb_second/third angle 0→0.873 rad
#   thumb_rotate slide 0→0.01 ↔ thumb_rotate_act_root 0→1.041 rad（线性）

# 四指的可视化 body：(近节, 远节)
FINGER_BODY_PAIRS = [
    ("finger_first_0_p", "finger_second_0_p"),
    ("finger_first_1_p", "finger_second_1_p"),
    ("finger_first_2_p", "finger_second_2_p"),
    ("finger_first_3_p", "finger_second_3_p"),
]
THUMB_BODIES = ["thumb_first_p", "thumb_second_p", "thumb_third_p"]
HAND_ROOT    = "hand_root"

# qpos 下标（在灵巧手单独模型里）
FINGER_FIRST_QDADR = [3, 8, 13, 18]   # finger_first_0~3_j
THUMB_GRASP_QADR   = 26               # thumb_grasp_act_push_j  (slide)
THUMB_ROTATE_QADR  = 20               # thumb_rotate_act_root_j (hinge, 0~1.041)
THUMB_SECOND_QADR  = 27               # thumb_second_j
THUMB_THIRD_QADR   = 28               # thumb_third_j

_hand_model = None
_hand_data  = None
_hand_root_bid   = None
_finger_bid_pairs = None
_thumb_bids      = None


def init_hand(xml_path: str) -> None:
    global _hand_model, _hand_data
    global _hand_root_bid, _finger_bid_pairs, _thumb_bids
    raw = strip_mesh(open(xml_path, encoding="utf-8").read())
    _hand_model = mujoco.MjModel.from_xml_string(raw)
    _hand_data  = mujoco.MjData(_hand_model)

    def bid(name):
        return mujoco.mj_name2id(_hand_model, mujoco.mjtObj.mjOBJ_BODY, name)

    _hand_root_bid   = bid(HAND_ROOT)
    _finger_bid_pairs = [(bid(a), bid(b)) for a, b in FINGER_BODY_PAIRS]
    _thumb_bids       = [bid(n) for n in THUMB_BODIES]
    print(f"[Hand] {_hand_model.nbody} bodies, nq={_hand_model.nq}")


def hand_fk(hand_qpos: np.ndarray,
            ee_pos: np.ndarray,
            ee_quat: np.ndarray):
    """
    计算灵巧手各手指关键点的世界坐标。
    hand_qpos: 6 维，[0,0.01] slide 值（数据顺序同 actuator）
    ee_pos / ee_quat: 末端执行器世界位姿（[w,x,y,z]）
    返回:
      root_world: (3,) 手根世界坐标
      fingers: list of [(近节世界坐标, 远节世界坐标), ...] × 4
      thumb:   list of [world_pos, ...] × 3 (thumb_first/second/third)
    """
    # 1. 设置四指弯曲角（slide 0→0.01 线性映射到 hinge 0→1.657）
    for i, qadr in enumerate(FINGER_FIRST_QDADR):
        slide_val = float(hand_qpos[i])
        angle = (slide_val / 0.01) * 1.657
        _hand_data.qpos[qadr] = np.clip(angle, -0.087, 1.657)

    # 2. 拇指抓握（hand_qpos[4]，slide → thumb_second/third 0→0.873）
    grasp_slide = float(hand_qpos[4])
    grasp_angle = (grasp_slide / 0.01) * 0.873
    _hand_data.qpos[THUMB_GRASP_QADR]  = np.clip(grasp_slide, 0, 0.01)
    _hand_data.qpos[THUMB_SECOND_QADR] = np.clip(grasp_angle, -0.087, 0.873)
    _hand_data.qpos[THUMB_THIRD_QADR]  = np.clip(grasp_angle, -0.087, 0.873)

    # 3. 拇指旋转（hand_qpos[5]，slide → thumb_rotate_act_root 0→1.041）
    rot_slide = float(hand_qpos[5])
    rot_angle = (rot_slide / 0.01) * 1.041
    _hand_data.qpos[THUMB_ROTATE_QADR] = np.clip(rot_angle, 0, 1.041)

    mujoco.mj_kinematics(_hand_model, _hand_data)

    # 4. 将局部坐标变换到世界坐标（ee_pos/quat 为手根附着点）
    R = _quat_to_rot(ee_quat)   # (3,3)
    # 修正：将手心从朝上翻转为朝下（绕末端执行器 X 轴旋转 -90°）
    R_fix = np.array([[1, 0, 0],
                      [0, 0, 1],
                      [0, -1, 0]])
    R = R @ R_fix

    def to_world(local_pos):
        return ee_pos + R @ local_pos

    root_local = _hand_data.xpos[_hand_root_bid].copy()
    root_world = to_world(root_local)

    fingers = []
    for (bid_a, bid_b) in _finger_bid_pairs:
        pa = to_world(_hand_data.xpos[bid_a].copy())
        pb = to_world(_hand_data.xpos[bid_b].copy())
        fingers.append((pa, pb))

    # 大拇指与其他四指一致，直接转换到世界坐标
    thumb = [to_world(_hand_data.xpos[bid].copy()) for bid in _thumb_bids]

    return root_world, fingers, thumb


# ──────────────────────────── 四元数工具 ──────────────────────────
def _quat_to_rot(q):
    """[w, x, y, z] → 3×3 旋转矩阵"""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


# ──────────────────────────── 数据加载 ───────────────────────────
def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ──────────────────────────── 主程序 ─────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("用法: python robot_arm_demo.py <data.jsonl>")
        sys.exit(1)

    records       = load_jsonl(sys.argv[1])
    arm_xml_path  = "assets/robots/rm75b/rm75b.xml"
    hand_xml_path = "assets/grippers/dex_hand/dex_hand.xml"
    N = len(records)
    print(f"已加载 {N} 帧数据")

    init_arm(arm_xml_path)
    init_hand(hand_xml_path)

    # ── 预计算 ──────────────────────────────────────────────────
    ee_pos_all    = []
    ee_quat_all   = []
    block_pos_all = []
    arm_fk_all    = []
    hand_qpos_all = []
    rewards       = []
    heights       = []
    phase_all     = []
    target_pos    = None

    for rec in records:
        info  = rec["info"]
        extra = rec.get("extra", {})
        ee_pos_all.append(np.array(info["end_effector"]["position"]))
        ee_quat_all.append(np.array(info["end_effector"]["quaternion"]))
        block_pos_all.append(np.array(info["block"]["current_position"]))
        arm_fk_all.append(arm_fk(np.array(info["arm_qpos"])))
        hand_qpos_all.append(np.array(info["hand_qpos"]))
        rewards.append(info["episode_reward"])
        heights.append(info["block"]["current_height"])
        phase_all.append(extra.get("phase_idx", 0))
        if target_pos is None:
            target_pos = np.array(info["target_marker"]["position"])

    ee_pos_all    = np.array(ee_pos_all)
    block_pos_all = np.array(block_pos_all)
    rewards       = np.array(rewards)
    heights       = np.array(heights)
    target_height = records[0]["info"]["block"]["target_height"]
    floor_z       = records[0]["info"]["block"]["initial_position"][2] - 0.10

    # ── 全局样式 ────────────────────────────────────────────────
    plt.rcParams.update({
        "figure.facecolor": BG,  "axes.facecolor": PANEL,
        "text.color": WHITE,     "axes.labelcolor": WHITE,
        "xtick.color": GRAY,     "ytick.color": GRAY,
        "axes.edgecolor": GRAY,  "grid.color": "#21262d",
        "font.family": "monospace",
    })

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.suptitle("■  RM75B + InspireHand  ·  Episode Replay  (real FK)",
                 fontsize=13, color=WHITE, fontweight="bold", y=0.98)

    gs   = GridSpec(3, 3, figure=fig,
                    left=0.04, right=0.98, top=0.93, bottom=0.07,
                    wspace=0.38, hspace=0.60)
    ax3d   = fig.add_subplot(gs[:, :2], projection="3d")
    ax_rwd = fig.add_subplot(gs[0, 2])
    ax_ht  = fig.add_subplot(gs[1, 2])
    ax_ee  = fig.add_subplot(gs[2, 2])

    for ax in [ax_rwd, ax_ht, ax_ee]:
        ax.set_facecolor(PANEL)
        ax.grid(True, alpha=0.3)

    # ── 3D 轴 ───────────────────────────────────────────────────
    ax3d.set_facecolor(BG)
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#30363d")
    ax3d.tick_params(colors=GRAY, labelsize=7)
    ax3d.set_xlabel("X (m)", color=GRAY, fontsize=8, labelpad=2)
    ax3d.set_ylabel("Y (m)", color=GRAY, fontsize=8, labelpad=2)
    ax3d.set_zlabel("Z (m)", color=GRAY, fontsize=8, labelpad=2)
    ax3d.set_title("3D Scene", color=WHITE, pad=6, fontsize=10)

    all_pts = np.vstack([ee_pos_all, block_pos_all, [target_pos]])
    margin  = 0.20
    xc, yc  = all_pts[:, 0].mean(), all_pts[:, 1].mean()
    span = max(np.ptp(all_pts, axis=0)[:2]) / 2 + margin
    ax3d.set_xlim(xc - span, xc + span)
    ax3d.set_ylim(yc - span, yc + span)
    ax3d.set_zlim(floor_z, all_pts[:, 2].max() + 0.12)
    ax3d.view_init(elev=22, azim=-55)

    # ── 静态场景 ─────────────────────────────────────────────────
    gx = np.linspace(xc-span, xc+span, 8)
    gy = np.linspace(yc-span, yc+span, 8)
    GX, GY = np.meshgrid(gx, gy)
    ax3d.plot_surface(GX, GY, np.full_like(GX, floor_z),
                      alpha=0.07, color=GRAY)

    d = 0.03
    ax3d.plot([target_pos[0]-d, target_pos[0]+d],
              [target_pos[1],   target_pos[1]],
              [target_pos[2],   target_pos[2]], color=RED, lw=2)
    ax3d.plot([target_pos[0],   target_pos[0]],
              [target_pos[1]-d, target_pos[1]+d],
              [target_pos[2],   target_pos[2]], color=RED, lw=2)
    ax3d.text(target_pos[0], target_pos[1], target_pos[2]+0.018,
              "TARGET", color=RED, fontsize=6.5, ha="center")

    ax3d.plot(ee_pos_all[:,0], ee_pos_all[:,1], ee_pos_all[:,2],
              color=CYAN,   alpha=0.10, lw=1, linestyle="--")
    ax3d.plot(block_pos_all[:,0], block_pos_all[:,1], block_pos_all[:,2],
              color=ORANGE, alpha=0.10, lw=1, linestyle="--")

    # ── 动态句柄 ─────────────────────────────────────────────────
    arm_line, = ax3d.plot([], [], [], color=CYAN, lw=3, zorder=6,
                           marker="o", markersize=4,
                           markerfacecolor=WHITE, markeredgecolor=CYAN)
    ee_dot     = ax3d.scatter([], [], [], s=60,  c=GREEN,  zorder=8, depthshade=False)
    block_cube = ax3d.scatter([], [], [], s=220, c=ORANGE, marker="s",
                               zorder=8, depthshade=False)
    trace_ee,  = ax3d.plot([], [], [], color=CYAN,   alpha=0.65, lw=1.2)
    trace_blk, = ax3d.plot([], [], [], color=ORANGE, alpha=0.45, lw=1.0)

    dynamic_lines = []   # EE 坐标轴 + 手指，每帧清空重绘

    status_txt = ax3d.text2D(
        0.02, 0.97, "", transform=ax3d.transAxes,
        color=WHITE, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc=BG, ec=GRAY, alpha=0.85)
    )

    # ── 右侧小图 ─────────────────────────────────────────────────
    steps_arr = np.arange(N)

    ax_rwd.set_title("Cumulative Reward", color=WHITE, fontsize=9)
    ax_rwd.plot(steps_arr, rewards, color=GREEN, alpha=0.20, lw=1)
    rwd_vline = ax_rwd.axvline(0, color=GREEN, lw=1.5, alpha=0.8)
    rwd_dot,  = ax_rwd.plot([], [], "o", color=GREEN, ms=5)
    ax_rwd.set_xlim(0, N-1)
    ax_rwd.set_ylabel("reward", color=GRAY, fontsize=7)
    ax_rwd.tick_params(labelsize=7)

    ax_ht.set_title("Block Height", color=WHITE, fontsize=9)
    ax_ht.plot(steps_arr, heights, color=ORANGE, alpha=0.20, lw=1)
    ax_ht.axhline(target_height, color=RED, lw=1.2, linestyle="--",
                  label=f"target {target_height:.3f}m")
    ax_ht.legend(fontsize=6.5, labelcolor=RED, facecolor=PANEL, edgecolor=GRAY)
    ht_vline = ax_ht.axvline(0, color=ORANGE, lw=1.5, alpha=0.8)
    ht_dot,  = ax_ht.plot([], [], "o", color=ORANGE, ms=5)
    ax_ht.set_xlim(0, N-1)
    ax_ht.set_ylabel("height (m)", color=GRAY, fontsize=7)
    ax_ht.tick_params(labelsize=7)

    ax_ee.set_title("End-Effector XY", color=WHITE, fontsize=9)
    ax_ee.plot(ee_pos_all[:,0], ee_pos_all[:,1], color=CYAN, alpha=0.15, lw=1)
    ax_ee.plot(target_pos[0], target_pos[1], "x", color=RED, ms=8, mew=2)
    ee_xy_dot, = ax_ee.plot([], [], "o", color=CYAN, ms=6)
    ax_ee.set_xlabel("X (m)", color=GRAY, fontsize=7)
    ax_ee.set_ylabel("Y (m)", color=GRAY, fontsize=7)
    ax_ee.tick_params(labelsize=7)
    ax_ee.set_aspect("equal", "box")

    # ── 进度条 ───────────────────────────────────────────────────
    prog_ax = fig.add_axes([0.04, 0.015, 0.94, 0.012])
    prog_ax.set_xlim(0, N); prog_ax.set_ylim(0, 1); prog_ax.axis("off")
    prog_ax.barh(0.5, N, height=1, color="#21262d", align="center")
    prog_bar = prog_ax.barh(0.5, 0, height=1, color=PURPLE, align="center")
    prog_txt = prog_ax.text(N/2, 0.5, f"0 / {N}",
                             ha="center", va="center", color=WHITE, fontsize=8)

    # ── 帧更新 ───────────────────────────────────────────────────
    def update(fi):
        rec   = records[fi]
        info  = rec["info"]
        extra = rec.get("extra", {})

        # —— 机械臂骨架 ——
        pts = arm_fk_all[fi].copy()
        # 以 EE 位置为基准，整体平移机械臂使末端对齐
        offset = ee_pos_all[fi] - pts[-1]
        pts += offset
        arm_line.set_data(pts[:, 0], pts[:, 1])
        arm_line.set_3d_properties(pts[:, 2])

        # —— EE 位姿 ——
        ep = ee_pos_all[fi]
        eq = ee_quat_all[fi]
        ee_dot._offsets3d = ([ep[0]], [ep[1]], [ep[2]])

        # 清除上帧动态线
        for ln in dynamic_lines:
            ln.remove()
        dynamic_lines.clear()

        # EE 坐标轴箭头（RGB = X/Y/Z）
        Rot = _quat_to_rot(eq)
        for col, axis_vec in zip([RED, GREEN, CYAN], Rot.T):
            end = ep + axis_vec * 0.04
            ln, = ax3d.plot([ep[0], end[0]], [ep[1], end[1]], [ep[2], end[2]],
                             color=col, lw=1.5, zorder=9)
            dynamic_lines.append(ln)

        # —— 灵巧手（真实 FK）——
        grasped = info["block"]["grasp_success"]
        f_color = GREEN if grasped else PURPLE
        hq = hand_qpos_all[fi]

        root_w, fingers, thumb = hand_fk(hq, ep, eq)

        # 手掌根部小点
        ln, = ax3d.plot([root_w[0]], [root_w[1]], [root_w[2]],
                         "o", color=WHITE, ms=3, zorder=9)
        dynamic_lines.append(ln)

        # 四指：手掌→近节→远节
        for (pa, pb) in fingers:
            # 手掌到近节
            ln, = ax3d.plot([root_w[0], pa[0]], [root_w[1], pa[1]],
                             [root_w[2], pa[2]],
                             color=GRAY, lw=1.2, alpha=0.7, zorder=7)
            dynamic_lines.append(ln)
            # 近节到远节（弯曲段）
            ln, = ax3d.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                             color=f_color, lw=2.2, zorder=8)
            dynamic_lines.append(ln)

        # 拇指：三节连线
        thumb_pts = [root_w] + thumb
        for i in range(len(thumb_pts)-1):
            ln, = ax3d.plot(
                [thumb_pts[i][0], thumb_pts[i+1][0]],
                [thumb_pts[i][1], thumb_pts[i+1][1]],
                [thumb_pts[i][2], thumb_pts[i+1][2]],
                color=YELLOW if not grasped else f_color,
                lw=2.2, zorder=8
            )
            dynamic_lines.append(ln)

        # —— 方块 ——
        bp = block_pos_all[fi]
        block_cube._offsets3d = ([bp[0]], [bp[1]], [bp[2]])
        block_cube._facecolors = np.array([[
            *matplotlib.colors.to_rgb(GREEN if grasped else ORANGE), 1.0
        ]])

        # —— 轨迹 ——
        trace_ee.set_data(ee_pos_all[:fi+1, 0], ee_pos_all[:fi+1, 1])
        trace_ee.set_3d_properties(ee_pos_all[:fi+1, 2])
        trace_blk.set_data(block_pos_all[:fi+1, 0], block_pos_all[:fi+1, 1])
        trace_blk.set_3d_properties(block_pos_all[:fi+1, 2])

        # —— 右侧图表 ——
        rwd_vline.set_xdata([fi]); rwd_dot.set_data([fi], [rewards[fi]])
        ht_vline.set_xdata([fi]);  ht_dot.set_data([fi], [heights[fi]])
        ee_xy_dot.set_data([ep[0]], [ep[1]])

        # —— 进度条 ——
        prog_bar[0].set_width(fi + 1)
        prog_txt.set_text(f"Step {fi+1:>4d} / {N}")

        # —— 状态文字 ——
        ph   = extra.get("phase_idx", "?")
        succ = "✓ SUCCESS" if extra.get("success") else ""
        lift = "↑ LIFTED"  if info["block"]["is_lifted"] else ""
        status_txt.set_text(
            f"Phase  : {ph}  {succ}\n"
            f"Grasp  : {'OK' if grasped else '--'}  {lift}\n"
            f"Height : {info['block']['current_height']:.4f} m\n"
            f"Reward : {extra.get('reward', 0.0):+.3f}"
        )

        return (arm_line, ee_dot, block_cube, trace_ee, trace_blk,
                rwd_vline, rwd_dot, ht_vline, ht_dot,
                ee_xy_dot, status_txt, *dynamic_lines)

    ani = animation.FuncAnimation(
        fig, update, frames=N,
        interval=80, blit=False, repeat=False
    )

    plt.show()
    return ani


if __name__ == "__main__":
    ani = main()