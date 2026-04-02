"""
鲁棒厚壁椭圆柱面拟合与采样算法 v4
======================================
核心改进（相对 v3）：
1. 自动分离内外表面：
   - 先用 v3 做整体粗拟合，得到椭圆中心与法向
   - 按每个面片法向量与"径向朝外"方向的夹角 → 分为内表面（朝内）/ 外表面（朝外）
   - 对稀疏面片补充用"点到粗拟合椭圆的有符号距离"做分类
2. 分别对内外两层做独立的 RANSAC + 几何距离椭圆拟合
3. 壁厚估计：沿外椭圆法向方向量测到内椭圆的距离，取中位数作为全局壁厚
4. 采样策略：在外表面椭圆柱均匀采样（m×n 网格）
5. 输出完整结果字典，包含内外椭圆参数、壁厚、采样点
"""

from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# 1. 柱轴估计：从面法向量 PCA（最小方差方向）
# ─────────────────────────────────────────────────────────────

def estimate_cylinder_axis_from_normals(mesh):
    areas = mesh.area_faces
    normals = mesh.face_normals
    w = areas / areas.sum()
    normals_w = normals * w[:, None]
    cov = normals_w.T @ normals
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, 0]
    return axis / np.linalg.norm(axis)


# ─────────────────────────────────────────────────────────────
# 2. 几何距离椭圆拟合（Sampson + L-BFGS-B + IRLS）
# ─────────────────────────────────────────────────────────────

def point_to_ellipse_sampson(params, pts):
    cx, cy, a, b, angle = params
    if a <= 0 or b <= 0:
        return 1e12
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    u = dx * cos_a + dy * sin_a
    v = -dx * sin_a + dy * cos_a
    F = (u / a) ** 2 + (v / b) ** 2 - 1.0
    grad_u = 2 * u / a ** 2
    grad_v = 2 * v / b ** 2
    denom = grad_u ** 2 + grad_v ** 2 + 1e-12
    return (F ** 2 / denom).sum()


def fit_ellipse_geometric(pts_2d, init_params=None, max_iter=3):
    if len(pts_2d) < 10:
        return None
    if init_params is None:
        x, y = pts_2d[:, 0], pts_2d[:, 1]
        cx0, cy0 = np.median(x), np.median(y)
        a0 = max((np.percentile(x, 95) - np.percentile(x, 5)) / 2, 1e-6)
        b0 = max((np.percentile(y, 95) - np.percentile(y, 5)) / 2, 1e-6)
        init_params = [cx0, cy0, a0, b0, 0.0]

    ref = max(np.ptp(pts_2d[:, 0]), np.ptp(pts_2d[:, 1]))
    bounds = [
        (init_params[0] - ref, init_params[0] + ref),
        (init_params[1] - ref, init_params[1] + ref),
        (ref * 0.02, ref * 5.0),
        (ref * 0.02, ref * 5.0),
        (-np.pi / 2, np.pi / 2),
    ]

    weights = np.ones(len(pts_2d))
    params = list(init_params)

    for _ in range(max_iter):
        res = minimize(
            lambda p: point_to_ellipse_sampson(p, pts_2d),
            params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not res.success and res.fun > 1e6:
            return None
        params = list(res.x)
        cx, cy, a, b, angle = params
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx, dy = pts_2d[:, 0] - cx, pts_2d[:, 1] - cy
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


def fit_ellipse_ransac(pts_2d, n_iter=200, inlier_tol=0.15, min_inliers=15):
    if len(pts_2d) < min_inliers:
        result = fit_ellipse_geometric(pts_2d)
        return result, np.ones(len(pts_2d), dtype=bool)

    ref = max(np.ptp(pts_2d[:, 0]), np.ptp(pts_2d[:, 1]))
    tol = inlier_tol * ref
    best_result, best_inliers, best_count = None, np.zeros(len(pts_2d), dtype=bool), 0
    rng = np.random.default_rng(42)

    for _ in range(n_iter):
        k = rng.integers(max(min_inliers, len(pts_2d) // 4),
                         max(min_inliers + 1, len(pts_2d) // 2))
        idx = rng.choice(len(pts_2d), size=min(k, len(pts_2d)), replace=False)
        result = fit_ellipse_geometric(pts_2d[idx])
        if result is None:
            continue
        cx, cy, a, b, angle = result
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx, dy = pts_2d[:, 0] - cx, pts_2d[:, 1] - cy
        u = dx * cos_a + dy * sin_a
        v = -dx * sin_a + dy * cos_a
        r = np.sqrt((u / a) ** 2 + (v / b) ** 2)
        r = np.where(r < 1e-9, 1e-9, r)
        dist = np.abs(r - 1.0) * np.sqrt(u ** 2 + v ** 2) / r
        inliers = dist < tol
        if inliers.sum() > best_count:
            best_count, best_inliers, best_result = inliers.sum(), inliers, result

    if best_count < min_inliers:
        best_result = fit_ellipse_geometric(pts_2d)
        best_inliers = np.ones(len(pts_2d), dtype=bool)

    if best_result is not None and best_inliers.sum() > 0:
        refined = fit_ellipse_geometric(pts_2d[best_inliers], init_params=list(best_result))
        if refined is not None:
            best_result = refined

    return best_result, best_inliers


# ─────────────────────────────────────────────────────────────
# 3. 弧段检测
# ─────────────────────────────────────────────────────────────

def detect_arc_robust(angles_deg, n_bins=90, sigma=1.0, empty_thresh=0.05):
    hist, _ = np.histogram(angles_deg, bins=n_bins, range=(0, 360))
    smooth = gaussian_filter1d(hist.astype(float), sigma=sigma, mode='wrap')
    threshold = empty_thresh * smooth.max()
    empty = smooth < threshold
    double = np.concatenate([empty, empty])
    best_start, best_len, cur_start, cur_len, in_gap = 0, 0, 0, 0, False
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
# 4. 内外表面分离
# ─────────────────────────────────────────────────────────────

def separate_inner_outer_surfaces(mesh, axis, e_x, e_y,
                                   coarse_cx, coarse_cy,
                                   coarse_a, coarse_b, coarse_angle):
    """
    策略：
    - 计算每个面片质心在截面坐标系下的位置 (u, v)
    - 计算该点在粗拟合椭圆上的"标准化半径" r = sqrt((u/a)^2 + (v/b)^2)
      r > 1 → 点在椭圆外侧 → 外表面候选
      r < 1 → 点在椭圆内侧 → 内表面候选
    - 同时用面法向量与径向的点积符号做辅助校验

    返回 outer_mask, inner_mask（面片级别的布尔掩码）
    """
    face_centers = mesh.vertices[mesh.faces].mean(axis=1)  # (F, 3)
    face_normals = mesh.face_normals  # (F, 3)

    # 截面坐标
    cos_a, sin_a = np.cos(coarse_angle), np.sin(coarse_angle)
    cx_3d = coarse_cx * e_x + coarse_cy * e_y  # 椭圆中心在截面平面的世界偏移

    dx = (face_centers @ e_x) - coarse_cx
    dy = (face_centers @ e_y) - coarse_cy

    # 旋转到椭圆主轴
    u = dx * cos_a + dy * sin_a
    v = -dx * sin_a + dy * cos_a

    # 标准化半径（相对于拟合椭圆）
    r_norm = np.sqrt((u / coarse_a) ** 2 + (v / coarse_b) ** 2)

    # 径向方向（从椭圆中心指向面片质心，截面内）
    radial_x = dx
    radial_y = dy
    radial_len = np.sqrt(radial_x ** 2 + radial_y ** 2) + 1e-12
    radial_x /= radial_len
    radial_y /= radial_len

    # 面法向量在截面内的径向分量
    fn_radial = (face_normals @ e_x) * radial_x + (face_normals @ e_y) * radial_y

    # ── 分类逻辑 ──
    # 主判据：r_norm（位置）
    # 辅助：fn_radial（法向方向）
    # 外表面：位置偏外(r>1) 或 法向朝外(fn_radial>0)
    # 内表面：位置偏内(r<1) 或 法向朝内(fn_radial<0)

    # 用中位数分割更鲁棒（避免薄壁时 r≈1 分不开）
    r_median = np.median(r_norm)
    spread = np.std(r_norm)

    if spread < 0.02:
        # 壁极薄，主要靠法向分类
        print("  Thin wall detected — using normal direction for separation")
        outer_mask = fn_radial > 0
        inner_mask = fn_radial < 0
    else:
        # 位置 + 法向联合投票
        pos_vote = (r_norm > r_median).astype(float)   # 1=外, 0=内
        nor_vote = (fn_radial > 0).astype(float)       # 1=外, 0=内
        score = 0.6 * pos_vote + 0.4 * nor_vote
        outer_mask = score > 0.5
        inner_mask = ~outer_mask

    outer_count = outer_mask.sum()
    inner_count = inner_mask.sum()
    print(f"  Surface separation: {outer_count} outer faces, {inner_count} inner faces")

    # 若有一侧极少（< 5%），可能 STL 只有单面 → 回退到全量
    total = len(outer_mask)
    if outer_count < 0.05 * total or inner_count < 0.05 * total:
        print("  WARNING: separation failed (one side < 5%) — using all faces as outer")
        outer_mask = np.ones(total, dtype=bool)
        inner_mask = np.ones(total, dtype=bool)

    return outer_mask, inner_mask


# ─────────────────────────────────────────────────────────────
# 5. 壁厚估计
# ─────────────────────────────────────────────────────────────

def estimate_wall_thickness(outer_fit, inner_fit, n_samples=360):
    """
    沿外椭圆的法向方向（向内），量测到内椭圆的距离。
    返回 (mean_thickness, std_thickness, thickness_array)
    """
    cx_o, cy_o, a_o, b_o, ang_o = outer_fit
    cx_i, cy_i, a_i, b_i, ang_i = inner_fit

    cos_o, sin_o = np.cos(ang_o), np.sin(ang_o)
    cos_i, sin_i = np.cos(ang_i), np.sin(ang_i)

    thetas = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)

    # 外椭圆上的点
    u_o = a_o * np.cos(thetas)
    v_o = b_o * np.sin(thetas)
    x_o = u_o * cos_o - v_o * sin_o + cx_o
    y_o = u_o * sin_o + v_o * cos_o + cy_o

    # 外椭圆上各点的法向（向内，指向中心）
    # 外椭圆法向（参数化法向）在椭圆坐标系下：(b*cos(t)/a, a*sin(t)/b)，归一化
    nx_u = b_o * np.cos(thetas) / a_o
    nx_v = a_o * np.sin(thetas) / b_o
    norm_len = np.sqrt(nx_u ** 2 + nx_v ** 2) + 1e-12
    # 转回截面坐标，取向内方向（朝中心）
    nx_sec = (nx_u * cos_o - nx_v * sin_o) / norm_len
    ny_sec = (nx_u * sin_o + nx_v * cos_o) / norm_len
    # 向内 = 朝中心方向
    cx_dir = cx_i - cx_o  # 内外中心偏移（通常很小）
    cy_dir = cy_i - cy_o
    # 确保法向朝内：若与"从外椭圆中心指向内椭圆中心"方向一致则保留
    # 简单做法：令法向指向椭圆中心
    to_center_x = cx_o - x_o  # 从外椭圆点指向外椭圆中心
    to_center_y = cy_o - y_o
    dot = nx_sec * to_center_x + ny_sec * to_center_y
    flip = np.where(dot < 0, -1.0, 1.0)
    nx_sec *= flip
    ny_sec *= flip

    # 沿法向射线与内椭圆求交（用二分法）
    thicknesses = []
    for i in range(n_samples):
        # 射线：P(t) = (x_o[i] + t*nx_sec[i], y_o[i] + t*ny_sec[i])
        # 在内椭圆坐标系下
        def signed_dist_to_inner(t):
            px = x_o[i] + t * nx_sec[i]
            py = y_o[i] + t * ny_sec[i]
            dx = px - cx_i
            dy = py - cy_i
            u_i = dx * cos_i + dy * sin_i
            v_i = -dx * sin_i + dy * cos_i
            return (u_i / a_i) ** 2 + (v_i / b_i) ** 2 - 1.0

        # 粗估：在 [0, max_t] 上搜索符号变化
        max_t = (a_o + b_o)  # 上限设为外椭圆半轴和
        t_lo, t_hi = 0.0, max_t
        f_lo = signed_dist_to_inner(t_lo)
        f_hi = signed_dist_to_inner(t_hi)

        if f_lo * f_hi > 0:
            # 没有符号变化：取最小值处
            ts = np.linspace(0, max_t, 200)
            fs = np.abs([signed_dist_to_inner(t) for t in ts])
            thicknesses.append(ts[np.argmin(fs)])
            continue

        # 二分求根
        for _ in range(50):
            t_mid = (t_lo + t_hi) / 2
            f_mid = signed_dist_to_inner(t_mid)
            if f_lo * f_mid <= 0:
                t_hi = t_mid
                f_hi = f_mid
            else:
                t_lo = t_mid
                f_lo = f_mid

        thicknesses.append((t_lo + t_hi) / 2)

    thicknesses = np.array(thicknesses)
    # 剔除异常值（> 3σ）
    mask = np.abs(thicknesses - np.median(thicknesses)) < 3 * thicknesses.std() + 1e-9
    return float(np.mean(thicknesses[mask])), float(np.std(thicknesses[mask])), thicknesses


# ─────────────────────────────────────────────────────────────
# 6. 主函数
# ─────────────────────────────────────────────────────────────

def fit_thick_elliptic_cylinder(
    stl_path,
    m,
    n,
    ransac_iters=150,
    inlier_tol=0.15,
    show_plot=True,
):
    """
    厚壁椭圆柱面拟合与外表面采样。

    Parameters
    ----------
    stl_path      : STL 文件路径
    m             : 周向采样数
    n             : 轴向采样数
    ransac_iters  : RANSAC 迭代次数
    inlier_tol    : RANSAC 内点容差（相对数据范围）
    show_plot     : 是否显示可视化

    Returns
    -------
    dict 包含:
        sample_pts    : (m*n, 3) 外表面采样点
        outer_ellipse : 外层椭圆参数
        inner_ellipse : 内层椭圆参数
        wall_thickness: {'mean', 'std', 'array'}
        arc           : 弧段信息
        cylinder_axis : 柱轴向量
    """
    print(f"\n{'='*60}")
    print(f"Processing (thick-wall): {stl_path}")
    print(f"{'='*60}")

    mesh = trimesh.load(stl_path)
    if not isinstance(mesh, trimesh.Trimesh):
        meshes = list(mesh.geometry.values())
        mesh = max(meshes, key=lambda m_: m_.area)

    # ── 1. 柱轴估计 ───────────────────────────────────────────
    axis = estimate_cylinder_axis_from_normals(mesh)
    print(f"Cylinder axis: {axis}")

    ref_v = np.array([0, 0, 1.0])
    if abs(np.dot(axis, ref_v)) > 0.9:
        ref_v = np.array([1, 0, 0.0])
    e_x = np.cross(ref_v, axis); e_x /= np.linalg.norm(e_x)
    e_y = np.cross(axis, e_x); e_y /= np.linalg.norm(e_y)

    # ── 2. 面片均匀重采样 ─────────────────────────────────────
    facet_groups = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(mesh.faces))
    )
    main_idx = np.argmax([mesh.area_faces[g].sum() for g in facet_groups])
    main_mesh = mesh.submesh([facet_groups[main_idx]], append=True)

    n_resample = max(len(main_mesh.vertices) * 10, 8000)
    sampled_pts, sampled_face_idx = trimesh.sample.sample_surface(main_mesh, n_resample)
    all_pts = np.vstack([main_mesh.vertices, sampled_pts])
    print(f"Resampled: {len(main_mesh.vertices)} verts → {n_resample} pts")

    z_vals = all_pts @ axis
    x_vals = all_pts @ e_x
    y_vals = all_pts @ e_y
    pts_2d = np.column_stack([x_vals, y_vals])

    # ── 3. 粗拟合（全部点）→ 得到大致椭圆参数 ────────────────
    print("Step 1/3: Coarse fit (all points)...")
    coarse_result, _ = fit_ellipse_ransac(pts_2d, n_iter=100, inlier_tol=0.2)
    if coarse_result is None:
        raise RuntimeError("Coarse ellipse fitting failed.")
    c_cx, c_cy, c_a, c_b, c_angle = coarse_result
    print(f"  Coarse: a={c_a:.4f}, b={c_b:.4f}, center=({c_cx:.4f},{c_cy:.4f}), "
          f"angle={np.degrees(c_angle):.1f}°")

    # ── 4. 内外表面分离（面片级别） ───────────────────────────
    print("Step 2/3: Separating inner/outer surfaces...")
    outer_face_mask, inner_face_mask = separate_inner_outer_surfaces(
        main_mesh, axis, e_x, e_y, c_cx, c_cy, c_a, c_b, c_angle
    )

    # 获取内外面片对应的顶点
    def get_pts_for_faces(mesh_, face_mask):
        face_indices = np.where(face_mask)[0]
        vert_indices = np.unique(mesh_.faces[face_indices])
        verts = mesh_.vertices[vert_indices]
        # 补充：在这些面上重采样
        n_s = max(len(vert_indices) * 8, 3000)
        if face_indices.size > 0:
            # 在子网格上采样
            sub = mesh_.submesh([face_indices], append=True)
            pts_s, _ = trimesh.sample.sample_surface(sub, n_s)
            return np.vstack([verts, pts_s])
        return verts

    outer_pts_3d = get_pts_for_faces(main_mesh, outer_face_mask)
    inner_pts_3d = get_pts_for_faces(main_mesh, inner_face_mask)
    print(f"  Outer pts: {len(outer_pts_3d)}, Inner pts: {len(inner_pts_3d)}")

    def to_2d(pts):
        return np.column_stack([pts @ e_x, pts @ e_y])

    outer_2d = to_2d(outer_pts_3d)
    inner_2d = to_2d(inner_pts_3d)

    # ── 5. 分别精拟合内外椭圆 ─────────────────────────────────
    print("Step 3/3: Fine-fitting outer and inner ellipses...")
    outer_result, outer_inliers = fit_ellipse_ransac(
        outer_2d, n_iter=ransac_iters, inlier_tol=inlier_tol
    )
    inner_result, inner_inliers = fit_ellipse_ransac(
        inner_2d, n_iter=ransac_iters, inlier_tol=inlier_tol
    )

    if outer_result is None:
        print("  WARNING: outer fit failed, using coarse result")
        outer_result = coarse_result
    if inner_result is None:
        print("  WARNING: inner fit failed, using coarse result with slight shrink")
        cx, cy, a, b, ang = coarse_result
        inner_result = (cx, cy, a * 0.9, b * 0.9, ang)

    o_cx, o_cy, o_a, o_b, o_ang = outer_result
    i_cx, i_cy, i_a, i_b, i_ang = inner_result

    # 确保 outer a/b >= inner a/b（若分类导致反转则交换）
    if (o_a + o_b) < (i_a + i_b):
        print("  Swapping inner/outer (size inversion detected)")
        outer_result, inner_result = inner_result, outer_result
        outer_pts_3d, inner_pts_3d = inner_pts_3d, outer_pts_3d
        outer_2d, inner_2d = inner_2d, outer_2d
        o_cx, o_cy, o_a, o_b, o_ang = outer_result
        i_cx, i_cy, i_a, i_b, i_ang = inner_result

    print(f"  Outer ellipse: a={o_a:.4f}, b={o_b:.4f}, "
          f"center=({o_cx:.4f},{o_cy:.4f}), angle={np.degrees(o_ang):.1f}°, "
          f"inliers={outer_inliers.sum()/len(outer_inliers):.1%}")
    print(f"  Inner ellipse: a={i_a:.4f}, b={i_b:.4f}, "
          f"center=({i_cx:.4f},{i_cy:.4f}), angle={np.degrees(i_ang):.1f}°, "
          f"inliers={inner_inliers.sum()/len(inner_inliers):.1%}")

    # ── 6. 壁厚估计 ───────────────────────────────────────────
    t_mean, t_std, t_arr = estimate_wall_thickness(outer_result, inner_result)
    print(f"  Wall thickness: mean={t_mean:.4f}, std={t_std:.4f}")

    # ── 7. 弧段检测（基于外表面） ─────────────────────────────
    o_cos, o_sin = np.cos(o_ang), np.sin(o_ang)
    dx_o = outer_2d[outer_inliers, 0] - o_cx
    dy_o = outer_2d[outer_inliers, 1] - o_cy
    u_o = dx_o * o_cos + dy_o * o_sin
    v_o = -dx_o * o_sin + dy_o * o_cos
    angles_deg = np.degrees(np.mod(np.arctan2(v_o / o_b, u_o / o_a), 2 * np.pi))
    start_deg, end_deg, arc_ratio = detect_arc_robust(angles_deg)
    print(f"  Arc (outer): [{start_deg:.1f}°, {end_deg:.1f}°], ratio={arc_ratio:.2f}")

    # ── 8. 坐标还原（外表面采样）──────────────────────────────
    def to_world_outer(z_arr, theta_arr):
        u = o_a * np.cos(theta_arr)
        v = o_b * np.sin(theta_arr)
        x_sec = u * o_cos - v * o_sin + o_cx
        y_sec = u * o_sin + v * o_cos + o_cy
        return (z_arr[:, None] * axis[None, :]
                + x_sec[:, None] * e_x[None, :]
                + y_sec[:, None] * e_y[None, :])

    # z 范围用全部点
    z_all = all_pts @ axis

    # ========== 修改：按弧长采样 ==========
    def ellipse_arc_length_parameterization(a, b, n_samples, start_deg, end_deg):
        """
        将角度区间 [start_deg, end_deg] 按弧长均匀采样，返回采样角度数组
        使用数值积分计算椭圆弧长
        """
        start_rad = np.radians(start_deg)
        end_rad = np.radians(end_deg)
        
        # 椭圆参数方程导数的模（弧长微分）
        def arc_integrand(theta):
            return np.sqrt((a * np.sin(theta))**2 + (b * np.cos(theta))**2)
        
        # 数值积分计算累积弧长
        n_integral = 1000  # 积分精度
        thetas_integral = np.linspace(start_rad, end_rad, n_integral)
        ds = arc_integrand(thetas_integral)
        cumulative_arc = np.cumsum(ds) * (end_rad - start_rad) / (n_integral - 1)
        total_arc = cumulative_arc[-1]
        
        # 等弧长分布的目标弧长位置
        target_arcs = np.linspace(0, total_arc, n_samples)
        
        # 通过插值找到对应的角度
        sample_angles = np.interp(target_arcs, cumulative_arc, thetas_integral)
        
        return sample_angles, total_arc

    # 定义弧段起止角度（从弧段检测结果）
    theta_start = np.radians(start_deg)
    theta_end = np.radians(end_deg)

    # 生成周向采样角度（按弧长均匀）
    theta_c, total_arc_length = ellipse_arc_length_parameterization(
        o_a, o_b, m, start_deg, end_deg
    )

    # 轴向采样（保持均匀）
    z_edges = np.linspace(z_all.min(), z_all.max(), n + 1)
    z_c = 0.5 * (z_edges[:-1] + z_edges[1:])

    THETA, Z = np.meshgrid(theta_c, z_c)
    sample_pts = to_world_outer(Z.ravel(), THETA.ravel())
    print(f"Generated {len(sample_pts)} sample points on outer surface ({m}×{n})")
    print(f"  Total arc length: {total_arc_length:.4f}, arc per cell: {total_arc_length/m:.4f}")

    # ── 9. 可视化 ──────────────────────────────────────────────
    if show_plot:
        # 生成外/内拟合曲面（用于绘图）
        num_plot = 60
        theta_plot = np.linspace(theta_start, theta_end, num_plot)
        z_plot = np.linspace(z_all.min(), z_all.max(), num_plot)
        THETA_p, Z_p = np.meshgrid(theta_plot, z_plot)

        def make_surf(cx_, cy_, a_, b_, ang_):
            cos_, sin_ = np.cos(ang_), np.sin(ang_)
            u_ = a_ * np.cos(THETA_p.ravel())
            v_ = b_ * np.sin(THETA_p.ravel())
            xs = u_ * cos_ - v_ * sin_ + cx_
            ys = u_ * sin_ + v_ * cos_ + cy_
            pts_ = (Z_p.ravel()[:, None] * axis[None, :]
                    + xs[:, None] * e_x[None, :]
                    + ys[:, None] * e_y[None, :])
            return pts_.reshape(num_plot, num_plot, 3)

        outer_surf = make_surf(o_cx, o_cy, o_a, o_b, o_ang)
        inner_surf = make_surf(i_cx, i_cy, i_a, i_b, i_ang)

        fig = plt.figure(figsize=(26, 7))
        fig.suptitle(
            f"Thick-wall Elliptic Cylinder Fit\n"
            f"Outer: a={o_a:.4f}, b={o_b:.4f}  |  "
            f"Inner: a={i_a:.4f}, b={i_b:.4f}  |  "
            f"Wall thickness: {t_mean:.4f}±{t_std:.4f}  |  "
            f"Arc: {arc_ratio:.2f}",
            fontsize=12,
        )

        def set_eq(ax):
            lims = np.array([mesh.vertices.min(0), mesh.vertices.max(0)])
            c = lims.mean(0)
            r = (lims[1] - lims[0]).max() * 0.55
            ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r)
            ax.set_zlim(c[2]-r, c[2]+r)
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

        # 子图 1：原始 STL
        ax1 = fig.add_subplot(151, projection="3d")
        ax1.plot_trisurf(mesh.vertices[:,0], mesh.vertices[:,1], mesh.vertices[:,2],
                         triangles=mesh.faces, color="gray", alpha=0.4, linewidth=0.1)
        ax1.set_title("1. Original STL")
        set_eq(ax1)

        # 子图 2：内外拟合曲面叠加
        ax2 = fig.add_subplot(152, projection="3d")
        ax2.plot_trisurf(mesh.vertices[:,0], mesh.vertices[:,1], mesh.vertices[:,2],
                         triangles=mesh.faces, color="gray", alpha=0.15, linewidth=0)
        ax2.plot_surface(outer_surf[...,0], outer_surf[...,1], outer_surf[...,2],
                         color='steelblue', alpha=0.6, linewidth=0, label='Outer')
        ax2.plot_surface(inner_surf[...,0], inner_surf[...,1], inner_surf[...,2],
                         color='salmon', alpha=0.6, linewidth=0, label='Inner')
        ax2.set_title("2. Outer (blue) + Inner (red)\nFitted surfaces")
        set_eq(ax2)

        # 子图 3：截面 2D 内外点云 + 拟合椭圆
        ax3 = fig.add_subplot(153)
        ax3.scatter(outer_2d[::5, 0], outer_2d[::5, 1],
                    c='steelblue', s=3, alpha=0.4, label='Outer pts')
        ax3.scatter(inner_2d[::5, 0], inner_2d[::5, 1],
                    c='salmon', s=3, alpha=0.4, label='Inner pts')
        t_ell = np.linspace(0, 2*np.pi, 300)
        for (cx_, cy_, a_, b_, ang_, col, lbl) in [
            (o_cx, o_cy, o_a, o_b, o_ang, 'navy', 'Outer ellipse'),
            (i_cx, i_cy, i_a, i_b, i_ang, 'darkred', 'Inner ellipse'),
        ]:
            cos_, sin_ = np.cos(ang_), np.sin(ang_)
            u_ = a_ * np.cos(t_ell)
            v_ = b_ * np.sin(t_ell)
            x_ = u_*cos_ - v_*sin_ + cx_
            y_ = u_*sin_ + v_*cos_ + cy_
            ax3.plot(x_, y_, color=col, lw=2, label=lbl)
        ax3.set_aspect('equal')
        ax3.legend(fontsize=7, markerscale=3)
        ax3.set_title("3. Cross-section (2D)\nInner / Outer separation")

        # 子图 4：壁厚分布（极坐标直方图）
        ax4 = fig.add_subplot(154, projection='polar')
        theta_polar = np.linspace(0, 2*np.pi, len(t_arr), endpoint=False)
        ax4.plot(theta_polar, t_arr, 'steelblue', lw=1.5)
        ax4.fill(theta_polar, t_arr, alpha=0.3, color='steelblue')
        ax4.axhline(t_mean, color='red', lw=1.5, linestyle='--', label=f'mean={t_mean:.3f}')
        ax4.set_title(f"4. Wall Thickness\n(polar, mean={t_mean:.4f})")

        # 子图 5：外表面采样点（保持弧长均匀的颜色映射）
        ax5 = fig.add_subplot(155, projection="3d")
        ax5.plot_trisurf(mesh.vertices[:,0], mesh.vertices[:,1], mesh.vertices[:,2],
                        triangles=mesh.faces, color="gray", alpha=0.15, linewidth=0)
        
        # 颜色按弧长位置映射（更直观）
        arc_positions = np.linspace(0, 1, m)
        colors = np.tile(arc_positions, n)
        
        ax5.scatter(sample_pts[:,0], sample_pts[:,1], sample_pts[:,2],
                    c=colors, cmap='jet', s=50, edgecolors='none', zorder=5)
        ax5.set_title(f"5. Outer Surface Samples (Arc-length)\n({m}×{n}={m*n} pts)")
        set_eq(ax5)

        plt.tight_layout()
        # out_fig = Path(stl_path).with_suffix('.fit_thick.png')
        # plt.savefig(str(out_fig), dpi=150, bbox_inches='tight')
        # print(f"Figure saved: {out_fig}")
        plt.show()

    return {
        "sample_pts": sample_pts,
        "outer_ellipse": {
            "cx": o_cx, "cy": o_cy, "a": o_a, "b": o_b,
            "angle_rad": o_ang,
            "inlier_ratio": float(outer_inliers.sum() / len(outer_inliers)),
        },
        "inner_ellipse": {
            "cx": i_cx, "cy": i_cy, "a": i_a, "b": i_b,
            "angle_rad": i_ang,
            "inlier_ratio": float(inner_inliers.sum() / len(inner_inliers)),
        },
        "wall_thickness": {
            "mean": t_mean,
            "std": t_std,
            "array": t_arr,
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

    result = fit_thick_elliptic_cylinder(
        str(STL_PATH),
        m=10,
        n=7,
        ransac_iters=150,
        inlier_tol=0.15,
        show_plot=True,
    )

    print("\n--- Results ---")
    print(f"Outer ellipse : {result['outer_ellipse']}")
    print(f"Inner ellipse : {result['inner_ellipse']}")
    print(f"Wall thickness: mean={result['wall_thickness']['mean']:.4f}, "
          f"std={result['wall_thickness']['std']:.4f}")
    print(f"Arc           : {result['arc']}")
    print(f"Cylinder axis : {result['cylinder_axis']}")
    print(f"\nFirst 5 sample points:\n{result['sample_pts'][:5]}")