import trimesh
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings

warnings.filterwarnings("ignore", message="Signature.*longdouble")

def fit_viz_sampling_pca_locked(stl_path, m, n):
    """
    基于 PCA 锁轴的残缺椭圆柱面拟合与采样算法
    :param stl_path: 输入 STL 文件路径
    :param m: 周向采样点数量 (Theta/Arc direction) - [已修正]
    :param n: 轴向采样点数量 (Z/Height direction) - [已修正]
    """
    print(f"--- Processing model: {stl_path} ---")
    mesh = trimesh.load(stl_path)
    
    # ==========================================
    # 1. PCA 锁定主方向
    # ==========================================
    facet_groups = trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(mesh.faces)))
    main_facet_indices = facet_groups[np.argmax([mesh.area_faces[g].sum() for g in facet_groups])]
    main_vertices = mesh.vertices[np.unique(mesh.faces[main_facet_indices])]
    
    pca = PCA(n_components=3)
    pts_pca = pca.fit_transform(main_vertices)
    
    z_pca_data = pts_pca[:, 0] # 轴向
    x_pca_data = pts_pca[:, 1] # 截面 X
    y_pca_data = pts_pca[:, 2] # 截面 Y
    
    print(f"--- PCA Axis Locked. Variance Ratios: {pca.explained_variance_ratio_} ---")

    # ==========================================
    # 2. 截面全量拟合
    # ==========================================
    x_min, x_max = x_pca_data.min(), x_pca_data.max()
    y_min, y_max = y_pca_data.min(), y_pca_data.max()
    
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    a_fit = (x_max - x_min) / 2
    b_fit = (y_max - y_min) / 2
    
    print(f"--- Full Ellipse Params: a={a_fit:.2f}, b={b_fit:.2f}, Center=({center_x:.2f}, {center_y:.2f}) ---")

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
    
    print(f"--- Detected Arc: [{start_angle_deg:.1f}°, {end_angle_deg:.1f}°], Ratio: {arc_ratio:.2f} ---")

    # ==========================================
    # 4. 生成拟合曲面与网格
    # ==========================================
    theta_start = np.radians(start_angle_deg)
    theta_end = np.radians(end_angle_deg)
    
    # 绘图用网格 (保持 80x80 用于显示，不受 m,n 影响)
    num_plot = 80
    z_grid_plot = np.linspace(z_pca_data.min(), z_pca_data.max(), num_plot)
    theta_grid_plot = np.linspace(theta_start, theta_end, num_plot)
    
    Z_mesh, THETA_mesh = np.meshgrid(z_grid_plot, theta_grid_plot)
    
    X_mesh = center_x + a_fit * np.cos(THETA_mesh)
    Y_mesh = center_y + b_fit * np.sin(THETA_mesh)
    
    surf_pca = np.vstack([Z_mesh.ravel(), X_mesh.ravel(), Y_mesh.ravel()]).T
    surf_orig = pca.inverse_transform(surf_pca)
    
    X_surf = surf_orig[:, 0].reshape(num_plot, num_plot)
    Y_surf = surf_orig[:, 1].reshape(num_plot, num_plot)
    Z_surf = surf_orig[:, 2].reshape(num_plot, num_plot)

    # ==========================================
    # 5. 生成采样点 (已修正 m, n 对应关系)
    # ==========================================
    # m -> 周向 (Theta)
    # n -> 轴向 (Z)
    theta_grid_sampling = np.linspace(theta_start, theta_end, m)
    z_grid_sampling = np.linspace(z_pca_data.min(), z_pca_data.max(), n)
    
    sampled_pts_pca = []
    # 先遍历 Z，再遍历 Theta (或者反过来，只要对应即可，这里保持逻辑一致)
    for zi in z_grid_sampling:
        for theta in theta_grid_sampling:
            xi = center_x + a_fit * np.cos(theta)
            yi = center_y + b_fit * np.sin(theta)
            sampled_pts_pca.append([zi, xi, yi])
            
    ideal_pts_orig = pca.inverse_transform(np.array(sampled_pts_pca))

    # ==========================================
    # 5. 可视化
    # ==========================================
    fig = plt.figure(figsize=(20, 6))
    fig.suptitle(f"PCA-Locked Fitting (Ratio {arc_ratio:.2f})\n{stl_path.split('/')[-1]}", fontsize=16)

    def set_axes_equal(ax):
        limits = np.array([mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)])
        center = limits.mean(axis=0)
        max_range = (limits[1] - limits[0]).max() * 0.5
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)

    # View 1: Original
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], 
                    triangles=mesh.faces, color='gray', alpha=0.3, linewidth=0.1, edgecolors='k')
    ax1.set_title("1. Original STL")
    set_axes_equal(ax1)

    # View 2: Fitted Surface
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], 
                    triangles=mesh.faces, color='gray', alpha=0.2, linewidth=0)
    ax2.plot_surface(X_surf, Y_surf, Z_surf, cmap=cm.coolwarm, alpha=0.6, linewidth=0)
    ax2.set_title(f"2. PCA-Locked Fitted Surface")
    set_axes_equal(ax2)

    # View 3: Sampling Grid
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], 
                    triangles=mesh.faces, color='gray', alpha=0.2, linewidth=0)
    
    colors = np.tile(np.linspace(0, 1, n), m)
    ax3.scatter(ideal_pts_orig[:, 0], ideal_pts_orig[:, 1], ideal_pts_orig[:, 2], 
               c=colors, cmap='jet', s=40, edgecolors='none')
    ax3.set_title(f"3. Sampling Grid ({m}x{n})")
    set_axes_equal(ax3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    path = "/home/zmy/MyProject/models/inspirehand/meshes/skin_0_0_p.STL"
    fit_viz_sampling_pca_locked(path, m=11, n=8)