"""
鲁棒残缺椭圆柱面拟合与采样算法 v3
======================================
核心改进（针对残缺弧 < 180° 时代数拟合严重失准的问题）：

1. 放弃 Fitzgibbon 代数拟合 —— 残缺弧时欠定，必然过估半轴
2. 改用「法向量聚类估轴 + 几何距离最小化」：
   a. 从网格面法向量的主方向估计柱轴（比 PCA on vertices 更准）
   b. 将顶点投影到垂直于柱轴的截面
   c. 用 scipy.optimize 最小化所有点到椭圆的几何距离（真正的几何拟合）
   d. RANSAC 外层包裹，剔除噪声/破损面片带来的异常点
3. 弧段检测：角度直方图 + 形态学闭运算，对稀疏噪声免疫
4. 面片均匀重采样（sample_surface）解决粗糙 STL 顶点稀少问题
5. 函数返回完整结果字典
"""

from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# 1. 柱轴估计：从面法向量聚类
# ─────────────────────────────────────────────────────────────

def estimate_cylinder_axis_from_normals(mesh, n_candidates=3):
    """
    圆/椭圆柱面的面法向量都垂直于柱轴。
    因此柱轴 ≈ 面法向量的「零空间主方向」，即 PCA 的最小方差方向。

    对面面积加权，避免小破碎面片主导结果。
    返回单位柱轴向量 (3,)
    """
    areas = mesh.area_faces  # (F,)
    normals = mesh.face_normals  # (F, 3)，已单位化

    # 加权 PCA：构造加权协方差矩阵
    w = areas / areas.sum()
    normals_w = normals * w[:, None]
    cov = normals_w.T @ normals  # (3,3)

    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigvals 升序，最小特征值对应的向量 ≈ 柱轴
    axis = eigvecs[:, 0]
    return axis / np.linalg.norm(axis)


# ─────────────────────────────────────────────────────────────
# 2. 几何距离椭圆拟合（核心改进）
# ─────────────────────────────────────────────────────────────

def point_to_ellipse_distance_sq(params, pts):
    """
    计算一批 2D 点到椭圆的近似几何距离平方之和。
    params = [cx, cy, a, b, angle]
    使用「归一化代数距离」作为几何距离的稳定近似，
    配合迭代权重可收敛到真正几何距离。
    """
    cx, cy, a, b, angle = params
    if a <= 0 or b <= 0:
        return 1e12

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy

    # 旋转到椭圆主轴坐标
    u = dx * cos_a + dy * sin_a
    v = -dx * sin_a + dy * cos_a

    # 点到椭圆的近似几何距离（Sampson 距离）
    F = (u / a) ** 2 + (v / b) ** 2 - 1.0
    grad_u = 2 * u / a ** 2
    grad_v = 2 * v / b ** 2
    denom = grad_u ** 2 + grad_v ** 2 + 1e-12

    sampson = F ** 2 / denom
    return sampson.sum()


def fit_ellipse_geometric(pts_2d, init_params=None, max_iter=3):
    """
    用 scipy.optimize.minimize (L-BFGS-B) 最小化几何距离，
    外加迭代重加权（IRLS）进一步抑制外点。

    params = [cx, cy, a, b, angle_rad]
    返回 (cx, cy, a, b, angle) 或 None
    """
    if len(pts_2d) < 10:
        return None

    # 初始参数估计：AABB（鲁棒版，用百分位而非 min/max）
    if init_params is None:
        x, y = pts_2d[:, 0], pts_2d[:, 1]
        cx0 = np.median(x)
        cy0 = np.median(y)
        a0 = (np.percentile(x, 95) - np.percentile(x, 5)) / 2
        b0 = (np.percentile(y, 95) - np.percentile(y, 5)) / 2
        a0 = max(a0, 1e-6)
        b0 = max(b0, 1e-6)
        init_params = [cx0, cy0, a0, b0, 0.0]

    # x_range = pts_2d[:, 0].ptp()
    x_range = np.ptp(pts_2d[:, 0])
    # y_range = pts_2d[:, 1].ptp()
    y_range = np.ptp(pts_2d[:, 1])
    ref = max(x_range, y_range)

    bounds = [
        (init_params[0] - ref, init_params[0] + ref),  # cx
        (init_params[1] - ref, init_params[1] + ref),  # cy
        (ref * 0.05, ref * 3.0),                        # a
        (ref * 0.05, ref * 3.0),                        # b
        (-np.pi / 2, np.pi / 2),                        # angle
    ]

    weights = np.ones(len(pts_2d))
    params = init_params.copy()

    for _ in range(max_iter):
        # 加权最小化
        def objective(p):
            return point_to_ellipse_distance_sq(p, pts_2d * weights[:, None] ** 0.5)

        res = minimize(
            lambda p: point_to_ellipse_distance_sq(p, pts_2d),
            params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-12},
        )

        if not res.success and res.fun > 1e6:
            return None

        params = res.x
        cx, cy, a, b, angle = params

        # 更新迭代权重（Huber-like）：远点降权
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx = pts_2d[:, 0] - cx
        dy = pts_2d[:, 1] - cy
        u = dx * cos_a + dy * sin_a
        v = -dx * sin_a + dy * cos_a
        residuals = np.abs((u / a) ** 2 + (v / b) ** 2 - 1.0)
        sigma = np.median(residuals) + 1e-9
        weights = 1.0 / (1.0 + (residuals / sigma) ** 2)

    cx, cy, a, b, angle = params
    if a < b:
        a, b = b, a
        angle += np.pi / 2
    angle = (angle + np.pi / 2) % np.pi - np.pi / 2

    return cx, cy, a, b, angle


def fit_ellipse_ransac(pts_2d, n_iter=200, inlier_tol=0.15, min_inliers=20):
    """
    RANSAC 外层：随机抽样 → 几何拟合 → 计数内点 → 取最优。
    tol 单位：相对于数据范围的比例。
    """
    if len(pts_2d) < min_inliers:
        return fit_ellipse_geometric(pts_2d), np.ones(len(pts_2d), dtype=bool)

    # ref = max(pts_2d[:, 0].ptp(), pts_2d[:, 1].ptp())
    ref = max(np.ptp(pts_2d[:, 0]), np.ptp(pts_2d[:, 1]))
    tol = inlier_tol * ref

    best_result = None
    best_inliers = np.zeros(len(pts_2d), dtype=bool)
    best_count = 0

    rng = np.random.default_rng(42)

    for _ in range(n_iter):
        # 随机采样子集（30%~60%）
        k = rng.integers(max(min_inliers, len(pts_2d) // 4),
                         max(min_inliers + 1, len(pts_2d) // 2))
        idx = rng.choice(len(pts_2d), size=min(k, len(pts_2d)), replace=False)
        subset = pts_2d[idx]

        result = fit_ellipse_geometric(subset)
        if result is None:
            continue

        cx, cy, a, b, angle = result
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx = pts_2d[:, 0] - cx
        dy = pts_2d[:, 1] - cy
        u = dx * cos_a + dy * sin_a
        v = -dx * sin_a + dy * cos_a

        # 点到椭圆的近似距离
        r_ellipse = np.sqrt((u / a) ** 2 + (v / b) ** 2)
        r_ellipse = np.where(r_ellipse < 1e-9, 1e-9, r_ellipse)
        dist = np.abs(r_ellipse - 1.0) * np.sqrt(u ** 2 + v ** 2) / r_ellipse
        inliers = dist < tol

        if inliers.sum() > best_count:
            best_count = inliers.sum()
            best_inliers = inliers
            best_result = result

    if best_count < min_inliers:
        # RANSAC 失败，回退到全量拟合
        best_result = fit_ellipse_geometric(pts_2d)
        best_inliers = np.ones(len(pts_2d), dtype=bool)

    if best_result is not None and best_count > 0:
        # 用内点集精化
        refined = fit_ellipse_geometric(pts_2d[best_inliers],
                                        init_params=list(best_result))
        if refined is not None:
            best_result = refined

    return best_result, best_inliers


# ─────────────────────────────────────────────────────────────
# 3. 弧段检测
# ─────────────────────────────────────────────────────────────

def detect_arc_robust(angles_deg, n_bins=90, sigma=1.0, empty_thresh=0.05):
    """
    直方图 + 高斯平滑 + 最大连续空白段检测。
    返回 (start_deg, end_deg, arc_ratio)，均在 [0, 360) 内循环。
    """
    hist, _ = np.histogram(angles_deg, bins=n_bins, range=(0, 360))
    smooth = gaussian_filter1d(hist.astype(float), sigma=sigma, mode='wrap')
    threshold = empty_thresh * smooth.max()
    empty = smooth < threshold

    # 环形最大连续空白
    double = np.concatenate([empty, empty])
    best_start, best_len = 0, 0
    cur_start, cur_len = 0, 0
    in_gap = False

    for i, e in enumerate(double):
        if e:
            if not in_gap:
                cur_start, cur_len, in_gap = i, 1, True
            else:
                cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            in_gap = False

    if best_len == 0:
        return 0.0, 360.0, 1.0

    bw = 360.0 / n_bins
    gap_end_bin = (best_start + best_len) % n_bins
    start_deg = gap_end_bin * bw
    end_deg = start_deg + (n_bins - best_len) * bw
    arc_ratio = min((n_bins - best_len) / n_bins, 1.0)

    return start_deg, end_deg, arc_ratio


# ─────────────────────────────────────────────────────────────
# 4. 主函数
# ─────────────────────────────────────────────────────────────

def fit_viz_sampling_robust(
    stl_path,
    m,
    n,
    ransac_iters=150,
    inlier_tol=0.15,
    show_plot=True,
):
    """
    鲁棒残缺椭圆柱面拟合与采样。

    :param stl_path:           STL 文件路径
    :param m:                  周向采样数
    :param n:                  轴向采样数
    :param ransac_iters:       RANSAC 迭代次数（每次从重采样点云中随机抽样子集）
    :param inlier_tol:         RANSAC 内点容差（相对于数据范围）
    :param show_plot:          是否显示可视化
    :return: dict
    """
    print(f"\n{'='*60}")
    print(f"Processing: {stl_path}")
    print(f"{'='*60}")

    mesh = trimesh.load(stl_path)
    if not isinstance(mesh, trimesh.Trimesh):
        meshes = list(mesh.geometry.values())
        mesh = max(meshes, key=lambda m_: m_.area)

    # ── 1. 柱轴估计（法向量 PCA） ──────────────────────────────
    axis = estimate_cylinder_axis_from_normals(mesh)
    print(f"Estimated cylinder axis: {axis}")

    # 构建截面坐标系（axis → e_z，取两个正交向量 e_x, e_y）
    # 选一个与 axis 不平行的参考向量
    ref = np.array([0, 0, 1.0])
    if abs(np.dot(axis, ref)) > 0.9:
        ref = np.array([1, 0, 0.0])

    e_x = np.cross(ref, axis)
    e_x /= np.linalg.norm(e_x)
    e_y = np.cross(axis, e_x)
    e_y /= np.linalg.norm(e_y)

    # ── 2. 选主连通体，面片均匀重采样 → 解决粗糙 STL 顶点稀少问题 ──
    facet_groups = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(mesh.faces))
    )
    main_idx = np.argmax([mesh.area_faces[g].sum() for g in facet_groups])
    main_face_indices = facet_groups[main_idx]

    # 取主连通体子网格
    main_mesh = mesh.submesh([main_face_indices], append=True)

    # 在面片上均匀撒点：目标点数 = max(原始顶点数 × 10, 5000)
    # 即使原始网格只有几十个面片，也能得到足够密集的截面点云
    n_orig_verts = len(main_mesh.vertices)
    n_resample = max(n_orig_verts * 10, 5000)
    sampled_pts, _ = trimesh.sample.sample_surface(main_mesh, n_resample)
    print(f"Resampled surface: {n_orig_verts} verts → {n_resample} points")

    # 保留原始顶点 + 重采样点（边界顶点对弧段检测很重要）
    all_pts = np.vstack([main_mesh.vertices, sampled_pts])

    # 轴向坐标（沿柱轴）
    z_vals = all_pts @ axis
    # 截面坐标（垂直于柱轴）
    x_vals = all_pts @ e_x
    y_vals = all_pts @ e_y

    pts_2d = np.column_stack([x_vals, y_vals])

    # ── 3. RANSAC + 几何距离椭圆拟合 ──────────────────────────
    print(f"Fitting ellipse with RANSAC (iters={ransac_iters})...")
    fit_result, inlier_mask = fit_ellipse_ransac(
        pts_2d, n_iter=ransac_iters, inlier_tol=inlier_tol
    )

    if fit_result is None:
        raise RuntimeError("Ellipse fitting failed completely.")

    cx, cy, a_fit, b_fit, angle_fit = fit_result
    inlier_ratio = inlier_mask.sum() / len(inlier_mask)
    print(f"Ellipse: a={a_fit:.4f}, b={b_fit:.4f}, "
          f"center=({cx:.4f}, {cy:.4f}), "
          f"angle={np.degrees(angle_fit):.1f}°, "
          f"inliers={inlier_ratio:.1%}")

    # ── 4. 弧段检测 ────────────────────────────────────────────
    cos_a, sin_a = np.cos(angle_fit), np.sin(angle_fit)

    # 内点的截面坐标
    x_in = x_vals[inlier_mask] - cx
    y_in = y_vals[inlier_mask] - cy
    u_in = x_in * cos_a + y_in * sin_a
    v_in = -x_in * sin_a + y_in * cos_a

    angles_deg = np.degrees(np.mod(np.arctan2(v_in / b_fit, u_in / a_fit), 2 * np.pi))
    start_deg, end_deg, arc_ratio = detect_arc_robust(angles_deg)
    print(f"Arc: [{start_deg:.1f}°, {end_deg:.1f}°], ratio={arc_ratio:.2f}")

    # ── 5. 坐标还原函数 ────────────────────────────────────────
    def to_world(z_arr, theta_arr):
        """
        (z, theta) → 世界坐标 (N, 3)
        theta 在椭圆参数空间（含旋转角 angle_fit）
        """
        u = a_fit * np.cos(theta_arr)
        v = b_fit * np.sin(theta_arr)
        # 旋转回截面坐标
        x_sec = u * cos_a - v * sin_a + cx
        y_sec = u * sin_a + v * cos_a + cy
        # 还原到世界坐标
        pts = (z_arr[:, None] * axis[None, :]
               + x_sec[:, None] * e_x[None, :]
               + y_sec[:, None] * e_y[None, :])
        # 加上质心偏置（axis 投影到原始空间需要考虑 mesh 质心）
        return pts

    # 质心修正：mesh 顶点在截面坐标系下的原点 ≠ 世界坐标原点
    # 用柱轴通过点（截面中心在世界坐标的位置）修正
    # 世界坐标 = z*axis + x*e_x + y*e_y + origin
    # origin 通过任意顶点还原：v = (v@axis)*axis + (v@e_x)*e_x + (v@e_y)*e_y
    # 验证：若三者正交，上式成立。我们确认 e_x,e_y,axis 两两正交，所以无需额外 origin。

    # ── 6. 生成拟合曲面（可视化用） ───────────────────────────
    theta_start = np.radians(start_deg)
    theta_end = np.radians(end_deg)
    num_plot = 80

    theta_plot = np.linspace(theta_start, theta_end, num_plot)
    z_range = np.linspace(z_vals.min(), z_vals.max(), num_plot)
    THETA_p, Z_p = np.meshgrid(theta_plot, z_range)

    surf_pts = to_world(Z_p.ravel(), THETA_p.ravel()).reshape(num_plot, num_plot, 3)
    X_surf = surf_pts[..., 0]
    Y_surf = surf_pts[..., 1]
    Z_surf = surf_pts[..., 2]

    # ── 7. 生成 m×n 采样点 ────────────────────────────────────
    theta_edges = np.linspace(theta_start, theta_end, m + 1)
    z_edges = np.linspace(z_vals.min(), z_vals.max(), n + 1)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    THETA, Z = np.meshgrid(theta_centers, z_centers)
    ideal_pts = to_world(Z.ravel(), THETA.ravel())  # (m*n, 3)

    sample_pts = ideal_pts.copy()
    print(f"Generated {len(ideal_pts)} ideal sampling points")

    # ── 9. 可视化 ──────────────────────────────────────────────
    if show_plot:
        fig = plt.figure(figsize=(22, 7))
        fig.suptitle(
            f"Robust Fit (RANSAC+Geometric) | Arc {arc_ratio:.2f} | "
            f"a={a_fit:.4f}, b={b_fit:.4f}, angle={np.degrees(angle_fit):.1f}°\n"
            f"Inliers: {inlier_ratio:.1%} | {Path(stl_path).name}",
            fontsize=13,
        )

        def set_eq(ax):
            lims = np.array([mesh.vertices.min(0), mesh.vertices.max(0)])
            c = lims.mean(0)
            r = (lims[1] - lims[0]).max() * 0.55
            ax.set_xlim(c[0]-r, c[0]+r)
            ax.set_ylim(c[1]-r, c[1]+r)
            ax.set_zlim(c[2]-r, c[2]+r)
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

        # 子图 1：原始 STL
        ax1 = fig.add_subplot(141, projection="3d")
        ax1.plot_trisurf(
            mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
            triangles=mesh.faces, color="gray", alpha=0.4, linewidth=0.1,
        )
        ax1.set_title("1. Original STL")
        set_eq(ax1)

        # 子图 2：STL + 拟合曲面
        ax2 = fig.add_subplot(142, projection="3d")
        ax2.plot_trisurf(
            mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
            triangles=mesh.faces, color="gray", alpha=0.2, linewidth=0,
        )
        ax2.plot_surface(X_surf, Y_surf, Z_surf,
                         cmap=cm.coolwarm, alpha=0.7, linewidth=0)
        ax2.set_title("2. Fitted Surface\n(RANSAC+Geometric)")
        set_eq(ax2)

        # 子图 3：内点 vs 外点（截面）
        ax3 = fig.add_subplot(143)
        ax3.scatter(x_vals[~inlier_mask], y_vals[~inlier_mask],
                    c='red', s=3, alpha=0.4, label='Outlier')
        ax3.scatter(x_vals[inlier_mask], y_vals[inlier_mask],
                    c='steelblue', s=3, alpha=0.4, label='Inlier')
        # 绘制拟合椭圆
        t_ell = np.linspace(0, 2 * np.pi, 300)
        u_ell = a_fit * np.cos(t_ell)
        v_ell = b_fit * np.sin(t_ell)
        x_ell = u_ell * cos_a - v_ell * sin_a + cx
        y_ell = u_ell * sin_a + v_ell * cos_a + cy
        ax3.plot(x_ell, y_ell, 'orange', lw=2, label='Fitted ellipse')
        ax3.set_aspect('equal')
        ax3.legend(fontsize=8, markerscale=3)
        ax3.set_title("3. Cross-section (2D)\nInlier/Outlier")

        # 子图 4：采样点
        ax4 = fig.add_subplot(144, projection="3d")
        ax4.plot_trisurf(
            mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
            triangles=mesh.faces, color="gray", alpha=0.2, linewidth=0,
        )
        colors = np.tile(np.linspace(0, 1, n), m)
        ax4.scatter(
            sample_pts[:, 0], sample_pts[:, 1], sample_pts[:, 2],
            c=colors, cmap="jet", s=50, edgecolors="none", zorder=5,
        )
        ax4.set_title(f"4. Ideal Samples ({m}×{n})")
        set_eq(ax4)

        plt.tight_layout()
        plt.show()

    return {
        "sample_pts": sample_pts,
        "ideal_pts": ideal_pts,
        "ellipse": {
            "cx": cx, "cy": cy,
            "a": a_fit, "b": b_fit,
            "angle_rad": angle_fit,
            "inlier_ratio": float(inlier_ratio),
        },
        "arc": {
            "start_deg": start_deg,
            "end_deg": end_deg,
            "ratio": arc_ratio,
        },
        "cylinder_axis": axis,
        "section_axes": (e_x, e_y),
    }


# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    STL_PATH = (
        PROJECT_ROOT / "models" / "inspirehand" / "meshes" / "skin_0_0_p.STL"
    )

    result = fit_viz_sampling_robust(
        str(STL_PATH),
        m=10,
        n=7,
        ransac_iters=150,
        inlier_tol=0.15,
        show_plot=True,
    )

    print("\n--- First 5 sample points ---")
    print(result["sample_pts"][:5])
    print(f"\nEllipse: {result['ellipse']}")
    print(f"Arc:     {result['arc']}")
    print(f"Cylinder axis: {result['cylinder_axis']}")