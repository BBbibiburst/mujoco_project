"""
Episode 统计工具 - 方块初始位置 × 成功/失败/超时 散点图
用法:
    python -m source.env_demos.strategy_stat.block_lifting.episode_stats <log_dir>
    python -m source.env_demos.strategy_stat.block_lifting.episode_stats source/env_demos/log
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

# ──────────────────────────── 颜色主题 ────────────────────────────
BG     = "#0d1117"
PANEL  = "#161b22"
GREEN  = "#3fb950"
RED    = "#f85149"
YELLOW = "#d29922"
WHITE  = "#e6edf3"
GRAY   = "#8b949e"
DGRAY  = "#30363d"
CYAN   = "#58d6f5"
PURPLE = "#bc8cff"


# ──────────────────────────── 数据读取 ────────────────────────────
def classify(last: dict) -> str:
    """从最后一帧判断结果。"""
    extra = last.get("extra", {})
    if extra.get("success"):
        return "success"
    if extra.get("truncated"):
        return "timeout"
    if extra.get("terminated"):
        return "failure"
    return "timeout"   # 默认归入超时


def collect_files(log_dir: Path):
    """
    递归收集 log_dir 下所有 .jsonl 文件。
    如果 log_dir 下直接有文件，则返回这些文件；
    如果 log_dir 下只有子文件夹，则递归遍历所有子文件夹收集文件。
    """
    # 先尝试直接收集当前目录下的 .jsonl 文件
    files = sorted(log_dir.glob("*.jsonl"))
    if not files:
        # 也支持无扩展名或 .log
        files = sorted(log_dir.glob("*.log")) + sorted(log_dir.glob("*"))
        files = [f for f in files if f.is_file()]

    if files:
        return files

    # 当前目录没有文件，递归遍历子目录
    all_files = []
    for subdir in sorted(log_dir.iterdir()):
        if subdir.is_dir():
            all_files.extend(collect_files(subdir))

    return all_files


def load_log_dir(log_dir: Path):
    """
    读取目录及其子目录下所有 .jsonl 文件，返回记录列表。
    每条记录: {x, y, z, result, reward, steps, filename, subdir}
    """
    records = []
    files = collect_files(log_dir)

    print(f"找到 {len(files)} 个文件")

    for fpath in files:
        try:
            lines = [l.strip() for l in fpath.read_text().splitlines() if l.strip()]
            if not lines:
                continue
            last = json.loads(lines[-1])
            first = json.loads(lines[0])

            info = first.get("info", {})
            block = info.get("block", {})
            init_pos = block.get("initial_position", [None, None, None])
            if init_pos[0] is None:
                continue

            result = classify(last)
            reward = last["info"].get("episode_reward", 0)
            steps  = last["info"].get("episode_steps", len(lines))
            max_h  = last["info"]["block"].get("max_height", 0)

            # 记录文件所在的子目录名（用于分类显示）
            try:
                subdir = fpath.relative_to(log_dir).parent.as_posix()
                if subdir == ".":
                    subdir = ""
            except ValueError:
                subdir = ""

            records.append({
                "x":       init_pos[0],
                "y":       init_pos[1],
                "z":       init_pos[2],
                "result":  result,
                "reward":  reward,
                "steps":   steps,
                "max_h":   max_h,
                "file":    fpath.name,
                "subdir":  subdir,
            })
        except Exception as e:
            print(f"  跳过 {fpath.name}: {e}")

    return records


# ──────────────────────────── 绘图 ────────────────────────────────
RESULT_STYLE = {
    "success": dict(color=GREEN,  marker="o", label="Success",  zorder=5),
    "timeout": dict(color=YELLOW, marker="^", label="Timeout",  zorder=4),
    "failure": dict(color=RED,    marker="X", label="Failure",  zorder=4),
}


def plot(records):
    if not records:
        print("没有有效记录，退出。")
        return

    n_total   = len(records)
    n_success = sum(1 for r in records if r["result"] == "success")
    n_timeout = sum(1 for r in records if r["result"] == "timeout")
    n_failure = sum(1 for r in records if r["result"] == "failure")
    sr = n_success / n_total * 100

    xs = np.array([r["x"] for r in records])
    ys = np.array([r["y"] for r in records])

    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor":   PANEL,
        "text.color":       WHITE,
        "axes.labelcolor":  WHITE,
        "xtick.color":      GRAY,
        "ytick.color":      GRAY,
        "axes.edgecolor":   DGRAY,
        "grid.color":       DGRAY,
        "font.family":      "monospace",
    })

    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    fig.suptitle(
        f"Episode Outcome  ·  Block Initial Position",
        fontsize=14, color=WHITE, fontweight="bold", y=0.97
    )

    gs = GridSpec(2, 3, figure=fig,
                  left=0.07, right=0.97, top=0.90, bottom=0.10,
                  wspace=0.38, hspace=0.48)

    ax_main = fig.add_subplot(gs[:, :2])   # 主散点图
    ax_bar  = fig.add_subplot(gs[0, 2])    # 数量条形
    ax_rwd  = fig.add_subplot(gs[1, 2])    # 奖励分布

    for ax in [ax_main, ax_bar, ax_rwd]:
        ax.set_facecolor(PANEL)
        ax.grid(True, alpha=0.25, linestyle="--")

    # ── 主散点图 ─────────────────────────────────────────────────
    ax_main.set_title("Block Initial XY  (color = episode outcome)",
                       color=WHITE, fontsize=10, pad=8)

    for result, style in RESULT_STYLE.items():
        sub = [r for r in records if r["result"] == result]
        if not sub:
            continue
        ax_main.scatter(
            [r["x"] for r in sub],
            [r["y"] for r in sub],
            c=style["color"],
            marker=style["marker"],
            s=90, alpha=0.82, linewidths=0.4,
            edgecolors="white",
            label=f"{style['label']}  ({len(sub)})",
            zorder=style["zorder"],
        )

    # 机器人工作区参考圆（RM75B 工作半径 610mm）
    theta = np.linspace(0, 2*np.pi, 200)
    ax_main.plot(0.61*np.cos(theta), 0.61*np.sin(theta),
                 color=CYAN, lw=0.8, alpha=0.25, linestyle=":")
    ax_main.plot(0, 0, "+", color=CYAN, ms=10, mew=1.2, alpha=0.4)
    ax_main.text(0.005, 0.005, "base", color=CYAN, fontsize=7, alpha=0.5)

    ax_main.set_xlabel("X (m)", fontsize=9)
    ax_main.set_ylabel("Y (m)", fontsize=9)
    ax_main.tick_params(labelsize=8)
    ax_main.set_aspect("equal", "box")

    # 图例
    legend = ax_main.legend(
        fontsize=8.5, facecolor="#1c2128", edgecolor=DGRAY,
        labelcolor=WHITE, loc="upper right",
        title=f"Total: {n_total}  SR: {sr:.1f}%",
        title_fontsize=8,
    )
    legend.get_title().set_color(GRAY)

    # ── 数量条形图 ──────────────────────────────────────────────
    ax_bar.set_title("Outcome Count", color=WHITE, fontsize=9, pad=6)
    labels = ["Success", "Timeout", "Failure"]
    counts = [n_success, n_timeout, n_failure]
    colors = [GREEN, YELLOW, RED]
    bars = ax_bar.barh(labels, counts, color=colors, height=0.5, alpha=0.85)
    for bar, cnt in zip(bars, counts):
        ax_bar.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    str(cnt), va="center", ha="left", color=WHITE, fontsize=9)
    ax_bar.set_xlim(0, max(counts) * 1.22)
    ax_bar.tick_params(labelsize=8)
    ax_bar.invert_yaxis()

    # ── 奖励分布（按结果分组 KDE/hist） ──────────────────────────
    ax_rwd.set_title("Episode Reward Distribution", color=WHITE, fontsize=9, pad=6)
    for result, style in RESULT_STYLE.items():
        sub_r = [r["reward"] for r in records if r["result"] == result]
        if len(sub_r) < 2:
            if sub_r:
                ax_rwd.axvline(sub_r[0], color=style["color"], lw=1.5, alpha=0.7)
            continue
        ax_rwd.hist(sub_r, bins=min(20, max(5, len(sub_r)//2)),
                    color=style["color"], alpha=0.55, label=style["label"],
                    density=True, edgecolor="none")
    ax_rwd.set_xlabel("reward", fontsize=8)
    ax_rwd.tick_params(labelsize=7)
    ax_rwd.legend(fontsize=7, facecolor="#1c2128", edgecolor=DGRAY,
                  labelcolor=WHITE)

    # ── 底部摘要文字 ──────────────────────────────────────────────
    summary = (f"Episodes: {n_total}   "
               f"Success: {n_success} ({sr:.1f}%)   "
               f"Timeout: {n_timeout}   "
               f"Failure: {n_failure}   "
               f"Avg reward: {np.mean([r['reward'] for r in records]):.1f}")
    fig.text(0.5, 0.02, summary, ha="center", va="bottom",
             color=GRAY, fontsize=8)

    plt.show()


# ──────────────────────────── 入口 ───────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("用法: python episode_stats.py <log_dir>")
        sys.exit(1)

    log_dir = Path(sys.argv[1])
    if not log_dir.is_dir():
        print(f"错误：{log_dir} 不是目录")
        sys.exit(1)

    records = load_log_dir(log_dir)
    print(f"有效 episode: {len(records)}")
    for k in ("success", "timeout", "failure"):
        n = sum(1 for r in records if r["result"] == k)
        print(f"  {k:8s}: {n}")

    plot(records)


if __name__ == "__main__":
    main()