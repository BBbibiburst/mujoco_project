import trimesh
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings

# Ignore interference warnings
warnings.filterwarnings("ignore", message="Signature.*longdouble")

def fit_viz_sampling_three_views_analytic(stl_path, m, n):
    print(f"--- Processing model: {stl_path} ---")
    mesh = trimesh.load(stl_path)
    
    # 1. Automatically identify the main surface (filter out cross-section noise to ensure accurate fitting)
    facet_groups = trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(mesh.faces)))
    main_facet_indices = facet_groups[np.argmax([mesh.area_faces[g].sum() for g in facet_groups])]
    main_vertices = mesh.vertices[np.unique(mesh.faces[main_facet_indices])]
    
    # 2. PCA to establish local coordinate system
    pca = PCA(n_components=3)
    pts_pca = pca.fit_transform(main_vertices)
    
    # PC0: Axial(z), PC1: Long axis(x), PC2: Short axis(y)
    z_pca_data = pts_pca[:, 0]
    x_pca_data = pts_pca[:, 1]
    y_pca_data = pts_pca[:, 2]
    
    # 3. Fit ellipse parameters (a, b)
    a_fit = (x_pca_data.max() - x_pca_data.min()) / 2
    b_fit = y_pca_data.max() - y_pca_data.min() 
    y_base = y_pca_data.min()
    
    # 4. Generate high-density "fitted surface" for plotting (Fig 2 background)
    num_plot = 80 # Plotting density
    z_grid_plot = np.linspace(z_pca_data.min(), z_pca_data.max(), num_plot)
    theta_grid_plot = np.linspace(0, np.pi, num_plot)
    Z_mesh, THETA_mesh = np.meshgrid(z_grid_plot, theta_grid_plot)

    # Ellipse parametric equation
    X_mesh = a_fit * np.cos(THETA_mesh)
    Y_mesh = y_base + b_fit * np.sin(THETA_mesh)
    
    # Inverse transform back to original space
    surf_pca = np.vstack([Z_mesh.ravel(), X_mesh.ravel(), Y_mesh.ravel()]).T
    surf_orig = pca.inverse_transform(surf_pca)
    
    # Reshape into 2D matrix for plot_surface
    X_surf = surf_orig[:, 0].reshape(num_plot, num_plot)
    Y_surf = surf_orig[:, 1].reshape(num_plot, num_plot)
    Z_surf = surf_orig[:, 2].reshape(num_plot, num_plot)

    # 5. Core modification: Generate m*n analytic ideal sampling grid (Fig 3 protagonist)
    z_grid_sampling = np.linspace(z_pca_data.min(), z_pca_data.max(), m)
    theta_grid_sampling = np.linspace(0, np.pi, n) 
    
    # Analytic sampling logic: Force movement along the elliptical arc
    sampled_pts_pca = []
    for zi in z_grid_sampling:
        for theta in theta_grid_sampling:
            xi = a_fit * np.cos(theta)
            yi = y_base + b_fit * np.sin(theta)
            sampled_pts_pca.append([zi, xi, yi])

    ideal_pts_orig = pca.inverse_transform(np.array(sampled_pts_pca))
    
    # 6. Save data to MuJoCo simulation environment directory
    # np.save("/home/zmy/MyProject/sampled_coords_mn_analytic.npy", ideal_pts_orig)
    # print(f"Analytic coordinate data saved to: /home/zmy/MyProject/sampled_coords_mn_analytic.npy")

    # 7. Create three-view visualization (Subplots)
    fig = plt.figure(figsize=(20, 7))
    # --- Title Changed to English ---
    fig.suptitle(f"VTLA E-Skin Modeling Pipeline - Analytic Grid Display ({m}x{n})\n{stl_path.split('/')[-1]}", fontsize=16)

    # Helper function to set equal aspect ratio axes
    def set_axes_equal(ax):
        limits = np.array([mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)])
        center = limits.mean(axis=0)
        max_range = (limits[1] - limits[0]).max() * 0.5
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)

    # Fig 1: Original Mesh
    ax1 = fig.add_subplot(131, projection='3d')
    # Draw high-density wireframe to reflect STL topology
    ax1.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], 
                    triangles=mesh.faces, color='gray', alpha=0.3, linewidth=0.1, edgecolors='k')
    ax1.set_title("1. Original STL Mesh Model")
    set_axes_equal(ax1)

    # Fig 2: Mesh and Fitted Elliptical Cylindrical Surface
    ax2 = fig.add_subplot(132, projection='3d')
    # Mesh as semi-transparent background
    ax2.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], 
                    triangles=mesh.faces, color='gray', alpha=0.05, linewidth=0)
    # Draw fitted analytic surface (use coolwarm cmap to reflect curvature)
    ax2.plot_surface(X_surf, Y_surf, Z_surf, cmap=cm.coolwarm,
                     linewidth=0, antialiased=False, alpha=0.5)
    ax2.set_title("2. Fitted Elliptical Cylindrical Surface")
    set_axes_equal(ax2)

    # Fig 3: Mesh and Analytic Sampled Point Lattice
    ax3 = fig.add_subplot(133, projection='3d')
    # Mesh as background
    ax3.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], 
                    triangles=mesh.faces, color='gray', alpha=0.05, linewidth=0)
    
    # Core modification: Directly plot the "ideal sampled lattice" calculated based on analytic parametric equations
    # These points will perfectly present an elliptical sector distribution, reflecting mathematical equidistance, floating above the background Mesh
    # Color by theta index row to facilitate continuity verification
    colors = np.tile(np.arange(n), m)
    scatter = ax3.scatter(ideal_pts_orig[:, 0], ideal_pts_orig[:, 1], ideal_pts_orig[:, 2], 
                         c=colors, cmap='jet', s=35, edgecolors='none')
    ax3.set_title(f"3. Analytic Parametric Sampling Grid ({m}x{n})")
    
    plt.colorbar(scatter, ax=ax3, label='Theta Index (Latitudinal)', fraction=0.046, pad=0.04)
    set_axes_equal(ax3)

    plt.tight_layout()
    print("--- Three-view generation complete ---")
    plt.show()

# Execute visualization
if __name__ == "__main__":
    path = "/home/zmy/MyProject/models/inspirehand/meshes/skin_0_0_p.STL"
    fit_viz_sampling_three_views_analytic(path, m=11, n=8)