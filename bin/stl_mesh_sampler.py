import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 内部辅助方法 (Internal Helpers)
# ─────────────────────────────────────────────────────────────

def _estimate_axis(mesh):
    areas = mesh.area_faces
    normals = mesh.face_normals
    w = areas / areas.sum()
    cov = (normals * w[:, None]).T @ normals
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, 0]
    return axis / np.linalg.norm(axis)

def _fit_ellipse_geometric(pts_2d, init_params=None):
    if len(pts_2d) < 10: return None
    if init_params is None:
        x, y = pts_2d[:, 0], pts_2d[:, 1]
        init_params = [np.median(x), np.median(y), np.ptp(x)/2, np.ptp(y)/2, 0.0]

    def sampson_dist(p, pts):
        cx, cy, a, b, ang = p
        if a <= 0 or b <= 0: return 1e12
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
        u, v = dx * cos_a + dy * sin_a, -dx * sin_a + dy * cos_a
        f = (u/a)**2 + (v/b)**2 - 1.0
        grad = (2*u/a**2)**2 + (2*v/b**2)**2 + 1e-12
        return (f**2 / grad).sum()

    res = minimize(sampson_dist, init_params, args=(pts_2d,), method="L-BFGS-B")
    if not res.success: return None
    cx, cy, a, b, ang = res.x
    if a < b: a, b, ang = b, a, ang + np.pi/2
    return cx, cy, a, b, (ang + np.pi/2) % np.pi - np.pi/2

def _fit_ellipse_ransac(pts_2d, n_iter=100, tol_ratio=0.15):
    if len(pts_2d) < 15: return _fit_ellipse_geometric(pts_2d), np.ones(len(pts_2d), dtype=bool)
    ref = max(np.ptp(pts_2d[:, 0]), np.ptp(pts_2d[:, 1]))
    tol = tol_ratio * ref
    best_res, best_inliers, max_in = None, np.zeros(len(pts_2d), dtype=bool), 0
    rng = np.random.default_rng(42)

    for _ in range(n_iter):
        idx = rng.choice(len(pts_2d), size=min(20, len(pts_2d)), replace=False)
        res = _fit_ellipse_geometric(pts_2d[idx])
        if res is None: continue
        cx, cy, a, b, ang = res
        dx, dy = pts_2d[:, 0] - cx, pts_2d[:, 1] - cy
        u, v = dx*np.cos(ang) + dy*np.sin(ang), -dx*np.sin(ang) + dy*np.cos(ang)
        r = np.sqrt((u/a)**2 + (v/b)**2 + 1e-12)
        dist = np.abs(r - 1.0) * np.sqrt(u**2 + v**2) / r
        inliers = dist < tol
        if inliers.sum() > max_in:
            max_in, best_inliers, best_res = inliers.sum(), inliers, res
    
    if best_res:
        refined = _fit_ellipse_geometric(pts_2d[best_inliers], init_params=list(best_res))
        if refined: best_res = refined
    return best_res, best_inliers

def _detect_arc_range(angles_deg):
    hist, _ = np.histogram(angles_deg, bins=90, range=(0, 360))
    smooth = gaussian_filter1d(hist.astype(float), sigma=1.0, mode="wrap")
    empty = smooth < (0.05 * smooth.max())
    double = np.concatenate([empty, empty])
    b_start, b_len, c_start, c_len, in_g = 0, 0, 0, 0, False
    for i, e in enumerate(double):
        if e:
            if not in_g: c_start, c_len, in_g = i, 1, True
            else: c_len += 1
            if c_len > b_len: b_len, b_start = c_len, c_start
        else: in_g = False
    start_deg = ((b_start + b_len) % 90) * 4.0
    return start_deg, start_deg + (90 - b_len) * 4.0

# ─────────────────────────────────────────────────────────────
# 核心功能方法 (Core Method)
# ─────────────────────────────────────────────────────────────

def generate_surface_mesh_points_from_stl(stl_path, m, n, ransac_iters=100, inlier_tol=0.15):
    """
    Fits a thick-walled elliptic cylinder to the STL and generates grid points on the outer surface.
    
    Parameters:
    - stl_path: Path to the STL file.
    - m: Number of samples along the circumferential arc.
    - n: Number of samples along the cylinder axis.
    
    Returns:
    - sample_pts: np.ndarray of shape (m*n, 3).
    """
    # 1. Load mesh
    mesh = trimesh.load(stl_path)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = max(mesh.geometry.values(), key=lambda x: x.area)

    # 2. Establish local coordinate system
    axis = _estimate_axis(mesh)
    ref = np.array([1, 0, 0.0]) if abs(np.dot(axis, [0,0,1])) > 0.9 else np.array([0,0,1.0])
    e_x = np.cross(ref, axis); e_x /= np.linalg.norm(e_x)
    e_y = np.cross(axis, e_x)

    # 3. Surface separation and fitting
    pts_all = mesh.vertices
    pts_2d_all = np.column_stack([pts_all @ e_x, pts_all @ e_y])
    c_res, _ = _fit_ellipse_ransac(pts_2d_all, n_iter=50)
    
    # Outer surface mask based on coarse fit
    f_centers = mesh.vertices[mesh.faces].mean(axis=1)
    dx, dy = (f_centers @ e_x) - c_res[0], (f_centers @ e_y) - c_res[1]
    r_norm = np.sqrt(((dx*np.cos(c_res[4]) + dy*np.sin(c_res[4]))/c_res[2])**2 + 
                     ((-dx*np.sin(c_res[4]) + dy*np.cos(c_res[4]))/c_res[3])**2)
    o_mesh = mesh.submesh([np.where(r_norm > np.median(r_norm))[0]], append=True)
    
    # 4. Refined fitting for outer ellipse
    o_pts_3d = np.vstack([o_mesh.vertices, trimesh.sample.sample_surface(o_mesh, 5000)[0]])
    o_pts_2d = np.column_stack([o_pts_3d @ e_x, o_pts_3d @ e_y])
    o_res, o_inliers = _fit_ellipse_ransac(o_pts_2d, n_iter=ransac_iters, tol_ratio=inlier_tol)
    cx, cy, a, b, ang = o_res

    # 5. Arc detection and grid sampling
    dx_in, dy_in = o_pts_2d[o_inliers, 0] - cx, o_pts_2d[o_inliers, 1] - cy
    angles = np.degrees(np.arctan2((-dx_in*np.sin(ang)+dy_in*np.cos(ang))/b, 
                                   (dx_in*np.cos(ang)+dy_in*np.sin(ang))/a))
    s_deg, e_deg = _detect_arc_range(np.mod(angles, 360))
    
    z_vals = pts_all @ axis
    z_grid = np.linspace(z_vals.min(), z_vals.max(), n)
    t_grid = np.radians(np.linspace(s_deg, e_deg, m))
    
    T, Z = np.meshgrid(t_grid, z_grid)
    T_flat = T.ravel()
    Z_flat = Z.ravel()
    
    u = a * np.cos(T_flat)
    v = b * np.sin(T_flat)
    
    # 2D Section to 3D World conversion
    x_s = u * np.cos(ang) - v * np.sin(ang) + cx
    y_s = u * np.sin(ang) + v * np.cos(ang) + cy
    
    # 正确的广播方式：(N, 1) * (1, 3) -> (N, 3)
    sample_pts = (Z_flat[:, None] * axis[None, :] + 
                  x_s[:, None] * e_x[None, :] + 
                  y_s[:, None] * e_y[None, :])
    
    return sample_pts

# ─────────────────────────────────────────────────────────────
# 测试入口 (Test Entry)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example usage for testing
    PATH_TO_FILE = "/home/zmy/MyProject/models/inspirehand/meshes/skin_0_0_p.STL"
    
    if Path(PATH_TO_FILE).exists():
        print(f"Testing generate_surface_mesh_points_from_stl with: {PATH_TO_FILE}")
        pts = generate_surface_mesh_points_from_stl(PATH_TO_FILE, m=10, n=7)
        print(f"Generated point cloud shape: {pts.shape}")
    else:
        print("Script ready. Please import 'generate_surface_mesh_points_from_stl' into your project.")