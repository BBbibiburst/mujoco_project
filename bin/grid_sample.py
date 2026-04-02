from pathlib import Path

import trimesh
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings

warnings.filterwarnings("ignore", message="Signature.*longdouble")


def detect_axis_alignment(pca_components, threshold=0.95):
    """
    检测 PCA 主轴是否近似于标准 xyz 轴。
    
    :param pca_components: PCA 的主成分矩阵 (3x3)，每行是一个主方向向量
    :param threshold: 对齐判断阈值（主轴与标准轴的 cos 相似度）
    :return: (is_aligned, axis_labels, aligned_axes)
        - is_aligned: 是否检测到近似对齐
        - axis_labels: 各 PCA 分量对应的轴名称列表，如 ['Z', 'X', 'Y']
        - aligned_axes: 各 PCA 分量对应的标准轴索引（0=X,1=Y,2=Z），None 表示未对齐
    """
    standard_axes = np.eye(3)  # X=[1,0,0], Y=[0,1,0], Z=[0,0,1]
    axis_names = ['X', 'Y', 'Z']

    aligned_axes = []
    axis_labels = []
    is_aligned = True

    for i, component in enumerate(pca_components):
        # 计算与三个标准轴的绝对余弦相似度（方向可正可负）
        similarities = np.abs(standard_axes @ component)
        best_axis = np.argmax(similarities)
        best_sim = similarities[best_axis]

        if best_sim >= threshold:
            aligned_axes.append(best_axis)
            axis_labels.append(axis_names[best_axis])
        else:
            aligned_axes.append(None)
            axis_labels.append(f'PCA_{i}')
            is_aligned = False

    # 检查是否有重复对齐轴（避免两个 PCA 轴都对齐到同一标准轴）
    valid = [a for a in aligned_axes if a is not None]
    if len(valid) != len(set(valid)):
        is_aligned = False
        axis_labels = [f'PCA_{i}' for i in range(3)]
        aligned_axes = [None, None, None]

    return is_aligned, axis_labels, aligned_axes


def fit_viz_sampling_pca_locked(stl_path, m, n, alignment_threshold=0.95):
    """
    基于 PCA 锁轴的残缺椭圆柱面拟合与采样算法
    :param stl_path: 输入 STL 文件路径
    :param m: 周向采样点数量 (Theta/Arc direction) → 生成 m 个中心采样点
    :param n: 轴向采样点数量 (Z/Height direction) → 生成 n 个中心采样点
    :param alignment_threshold: PCA 轴与标准轴对齐的余弦相似度阈值 (0~1)
    """
    print(f"--- Processing model: {stl_path} ---")
    mesh = trimesh.load(stl_path)

    # ==========================================
    # 1. PCA 锁定主方向
    # ==========================================
    facet_groups = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(mesh.faces))
    )
    main_facet_indices = facet_groups[
        np.argmax([mesh.area_faces[g].sum() for g in facet_groups])
    ]
    main_vertices = mesh.vertices[np.unique(mesh.faces[main_facet_indices])]

    pca = PCA(n_components=3)
    pts_pca = pca.fit_transform(main_vertices)

    print(f"--- PCA Axis Locked. Variance Ratios: {pca.explained_variance_ratio_} ---")

    # ==========================================
    # 1.5 检测 PCA 主轴是否近似于标准 xyz 轴
    # ==========================================
    is_aligned, axis_labels, aligned_axes = detect_axis_alignment(
        pca.components_, threshold=alignment_threshold
    )

    if is_aligned:
        print(f"--- PCA axes approximately aligned to standard axes: "
              f"PCA_0→{axis_labels[0]}, PCA_1→{axis_labels[1]}, PCA_2→{axis_labels[2]} ---")
        print(f"--- Using standard XYZ coordinates directly (skipping PCA transform) ---")

        # 直接使用对应的标准轴坐标，无需 PCA 变换
        # PCA 第 0 分量（方差最大）→ 柱轴方向
        col_z = aligned_axes[0]  # 轴向对应的原始坐标列索引
        col_x = aligned_axes[1]  # 截面 X 对应的原始坐标列索引
        col_y = aligned_axes[2]  # 截面 Y 对应的原始坐标列索引

        z_pca_data = main_vertices[:, col_z]
        x_pca_data = main_vertices[:, col_x]
        y_pca_data = main_vertices[:, col_y]

        # 检查分量方向符号是否需要翻转（与 PCA 方向一致）
        for i, (col, pca_col) in enumerate(zip([col_z, col_x, col_y], [0, 1, 2])):
            std_axis = np.zeros(3)
            std_axis[col] = 1.0
            dot = np.dot(pca.components_[pca_col], std_axis)
            if dot < 0:
                if i == 0:
                    z_pca_data = -z_pca_data
                elif i == 1:
                    x_pca_data = -x_pca_data
                else:
                    y_pca_data = -y_pca_data

        use_pca_transform = False
    else:
        print(f"--- PCA axes not aligned to standard axes. Using PCA transform. ---")
        z_pca_data = pts_pca[:, 0]  # 轴向
        x_pca_data = pts_pca[:, 1]  # 截面 X
        y_pca_data = pts_pca[:, 2]  # 截面 Y
        use_pca_transform = True

    print(f"--- Cylinder axis direction: {axis_labels[0]}, "
          f"Cross-section axes: {axis_labels[1]}, {axis_labels[2]} ---")

    # ==========================================
    # 2. 截面全量拟合
    # ==========================================
    x_min, x_max = x_pca_data.min(), x_pca_data.max()
    y_min, y_max = y_pca_data.min(), y_pca_data.max()

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    a_fit = (x_max - x_min) / 2
    b_fit = (y_max - y_min) / 2

    print(f"--- Full Ellipse Params: a={a_fit:.2f}, b={b_fit:.2f}, "
          f"Center=({center_x:.2f}, {center_y:.2f}) ---")

    # ==========================================
    # 3. 投影与分布分析
    # ==========================================
    x_centered = x_pca_data - center_x
    y_centered = y_pca_data - center_y

    angles = np.arctan2(y_centered / b_fit, x_centered / a_fit)
    angles_deg = np.degrees(np.mod(angles, 2 * np.pi))

    sorted_angles = np.sort(angles_deg)
    gaps = np.diff(sorted_angles)

    wrap_gap = 360 - (sorted_angles[-1] - sorted_angles[0])
    gaps = np.append(gaps, wrap_gap)

    max_gap_idx = np.argmax(gaps)

    if max_gap_idx == len(gaps) - 1:
        start_angle_deg = sorted_angles[0]
        end_angle_deg = sorted_angles[-1]
    else:
        start_angle_deg = sorted_angles[(max_gap_idx + 1) % len(sorted_angles)]
        end_angle_deg = sorted_angles[max_gap_idx]
        if start_angle_deg > end_angle_deg:
            end_angle_deg += 360

    covered_angle_deg = end_angle_deg - start_angle_deg
    arc_ratio = covered_angle_deg / 360.0

    print(f"--- Detected Arc: [{start_angle_deg:.1f}°, {end_angle_deg:.1f}°], "
          f"Ratio: {arc_ratio:.2f} ---")

    # ==========================================
    # 4. 生成拟合曲面与网格 (用于显示)
    # ==========================================
    theta_start = np.radians(start_angle_deg)
    theta_end = np.radians(end_angle_deg)

    num_plot = 80
    z_grid_plot = np.linspace(z_pca_data.min(), z_pca_data.max(), num_plot)
    theta_grid_plot = np.linspace(theta_start, theta_end, num_plot)

    Z_mesh, THETA_mesh = np.meshgrid(z_grid_plot, theta_grid_plot)

    X_mesh = center_x + a_fit * np.cos(THETA_mesh)
    Y_mesh = center_y + b_fit * np.sin(THETA_mesh)

    def to_original_coords(z_vals, x_vals, y_vals):
        """将拟合坐标系中的点转回原始坐标系"""
        pts_local = np.stack([z_vals, x_vals, y_vals], axis=-1)
        if use_pca_transform:
            # PCA 逆变换
            pts_flat = pts_local.reshape(-1, 3)
            pts_orig = pca.inverse_transform(pts_flat)
            return pts_orig.reshape(pts_local.shape)
        else:
            # 直接映射回原始轴（考虑符号）
            pts_orig = np.zeros((*pts_local.shape[:-1], 3))
            # 还原时注意之前可能做了符号翻转，需反向处理
            for i, (col, pca_col) in enumerate(zip([col_z, col_x, col_y], [0, 1, 2])):
                std_axis = np.zeros(3)
                std_axis[col] = 1.0
                dot = np.dot(pca.components_[pca_col], std_axis)
                sign = 1.0 if dot >= 0 else -1.0
                pts_orig[..., col] = pts_local[..., i] * sign
            # 加上均值（PCA inverse_transform 会加 mean_，对齐情况下需手动补偿）
            pts_orig += pca.mean_
            return pts_orig

    surf_local = np.stack([Z_mesh, X_mesh, Y_mesh], axis=-1)
    surf_orig = to_original_coords(Z_mesh, X_mesh, Y_mesh)

    X_surf = surf_orig[..., 0]
    Y_surf = surf_orig[..., 1]
    Z_surf = surf_orig[..., 2]

    # ==========================================
    # 5. 生成 m×n 个规则中心采样点
    # ==========================================
    theta_edges = np.linspace(theta_start, theta_end, m + 1)
    z_edges = np.linspace(z_pca_data.min(), z_pca_data.max(), n + 1)

    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    THETA, Z = np.meshgrid(theta_centers, z_centers)

    X = center_x + a_fit * np.cos(THETA)
    Y = center_y + b_fit * np.sin(THETA)

    ideal_pts_orig = to_original_coords(Z, X, Y).reshape(-1, 3)

    print(f"--- Generated Center Grid: {len(ideal_pts_orig)} points (m={m} × n={n}) ---")

    # ==========================================
    # 6. 可视化
    # ==========================================
    axis_info = (f"Cylinder Axis≈{axis_labels[0]}" if is_aligned
                 else "Cylinder Axis=PCA_0")

    fig = plt.figure(figsize=(20, 6))
    fig.suptitle(
        f"PCA-Locked Fitting (Ratio {arc_ratio:.2f}) | {axis_info}\n"
        f"{stl_path.split('/')[-1]}",
        fontsize=16,
    )

    def set_axes_equal(ax):
        limits = np.array([mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)])
        center = limits.mean(axis=0)
        max_range = (limits[1] - limits[0]).max() * 0.5
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)

    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot_trisurf(
        mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
        triangles=mesh.faces, color="gray", alpha=0.3, linewidth=0.1, edgecolors="k",
    )
    ax1.set_title("1. Original STL")
    set_axes_equal(ax1)

    ax2 = fig.add_subplot(132, projection="3d")
    ax2.plot_trisurf(
        mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
        triangles=mesh.faces, color="gray", alpha=0.2, linewidth=0,
    )
    ax2.plot_surface(X_surf, Y_surf, Z_surf, cmap=cm.coolwarm, alpha=0.6, linewidth=0)
    ax2.set_title(f"2. Fitted Surface ({axis_info})")
    set_axes_equal(ax2)

    ax3 = fig.add_subplot(133, projection="3d")
    ax3.plot_trisurf(
        mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
        triangles=mesh.faces, color="gray", alpha=0.2, linewidth=0,
    )

    z_colors = np.linspace(0, 1, n)
    colors = np.repeat(z_colors, m)

    ax3.scatter(
        ideal_pts_orig[:, 0], ideal_pts_orig[:, 1], ideal_pts_orig[:, 2],
        c=colors, cmap="jet", s=40, edgecolors="none",
    )
    ax3.set_title(f"3. Sampling Grid ({m}×{n})")
    set_axes_equal(ax3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DEFAULT_STL_PATH = (
        PROJECT_ROOT / "models" / "inspirehand" / "meshes" / "skin_0_2_p.STL"
    )
    path = str(DEFAULT_STL_PATH)
    fit_viz_sampling_pca_locked(path, m=10, n=7, alignment_threshold=0.95)