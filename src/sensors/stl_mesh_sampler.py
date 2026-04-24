"""
STL厚壁椭圆柱表面网格点云生成模块.

该模块提供从STL网格文件拟合厚壁椭圆柱体，并在其外表面生成规则采样点云的功能。
采用基于RANSAC的鲁棒椭圆拟合算法，自动识别圆柱弧段范围，支持结果磁盘缓存。

核心功能：
1. 椭圆柱拟合：从三角网格估计主轴方向，拟合椭圆截面参数
2. 厚壁分离：区分内外表面，提取外表面进行精确拟合
3. 弧段检测：基于角度直方图分析自动识别有效圆柱弧段
4. 网格采样：在参数空间生成规则网格，映射回三维表面
5. 智能缓存：基于文件内容MD5的缓存机制，避免重复计算

算法流程：
1. 加载STL → 2. 估计主轴 → 3. 粗拟合椭圆 → 4. 分离外表面 → 
5. 精拟合外椭圆(RANSAC) → 6. 检测弧段范围 → 7. 生成网格点

依赖库：
- trimesh: STL文件加载与网格处理
- scipy: 优化求解与信号处理
- joblib: 结果序列化缓存
"""

import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import joblib
import hashlib
import warnings
from typing import Tuple, Optional, List

warnings.filterwarnings("ignore")

# ====================== 缓存配置 ======================

# 缓存目录：模块同级目录下的 stl_cache 文件夹
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache" / "stl_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_key(stl_path: Path, m: int, n: int, ransac_iters: int, inlier_tol: float) -> str:
    """
    生成缓存键，基于文件内容MD5和采样参数.

    使用文件内容哈希而非路径或修改时间，确保：
    1. 文件内容变化时缓存自动失效
    2. 跨机器/路径移动后缓存仍有效（只要内容相同）
    3. 避免文件系统时间戳不可靠问题

    Args:
        stl_path: STL文件路径。
        m: 圆周方向采样数。
        n: 轴向采样数。
        ransac_iters: RANSAC迭代次数。
        inlier_tol: 内点容差比例。

    Returns:
        str: 格式为 "{md5}_m{m}_n{n}_iter{iters}_tol{tol}" 的缓存键字符串。
    """
    md5 = hashlib.md5(Path(stl_path).read_bytes()).hexdigest()
    return f"{md5}_m{m}_n{n}_iter{ransac_iters}_tol{inlier_tol}"


def _cache_path(key: str) -> Path:
    """
    将缓存键转换为完整的文件路径.

    Args:
        key: 缓存键字符串。

    Returns:
        Path: 指向缓存文件的完整路径（.joblib后缀）。
    """
    return _CACHE_DIR / f"{key}.joblib"


# ====================== 内部辅助方法 ======================

def _estimate_axis(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    基于面法向量的加权协方差分析估计圆柱主轴方向.

    算法原理：
    圆柱侧面法向量应垂直于主轴，因此法向量分布的协方差矩阵
    的最小特征值对应的特征向量即为主轴方向。

    Args:
        mesh: 输入三角网格对象。

    Returns:
        np.ndarray: 单位化的主轴方向向量，shape (3,)。
    """
    areas = mesh.area_faces
    normals = mesh.face_normals
    w = areas / areas.sum()
    
    # 加权协方差矩阵
    cov = (normals * w[:, None]).T @ normals
    
    # 特征值分解
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # 最小特征值对应的特征向量为主轴（法向量垂直于主轴）
    # 确保按特征值排序（eigh通常已排序，但显式确认更安全）
    axis = eigvecs[:, 0]
    return axis / np.linalg.norm(axis)


def _fit_ellipse_geometric(pts_2d: np.ndarray, init_params: Optional[List[float]] = None) -> Optional[Tuple[float, float, float, float, float]]:
    """
    使用Sampson距离最小化进行几何椭圆拟合.

    求解优化问题： min Σ (Sampson_distance(p_i, ellipse))²
    其中Sampson距离是几何距离的一阶近似，比代数距离更稳定。

    Args:
        pts_2d: 二维点集，shape (N, 2)。
        init_params: 初始参数 [cx, cy, a, b, angle]，None时使用数据包围盒估计。

    Returns:
        tuple: (cx, cy, a, b, angle) 椭圆参数，或 None 如果优化失败。
               - cx, cy: 椭圆中心坐标
               - a, b: 长半轴、短半轴（保证 a >= b）
               - angle: 长轴与x轴夹角 [弧度]，范围 [-π/2, π/2]
    """
    if len(pts_2d) < 10:
        return None

    if init_params is None:
        x, y = pts_2d[:, 0], pts_2d[:, 1]
        init_params = [np.median(x), np.median(y), np.ptp(x) / 2, np.ptp(y) / 2, 0.0]

    def sampson_dist(p, pts):
        """计算所有点到椭圆的Sampson距离平方和."""
        cx, cy, a, b, ang = p
        if a <= 0 or b <= 0:
            return 1e12  # 无效参数，返回大值避免选择
        
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
        
        # 旋转到椭圆主轴坐标系
        u, v = dx * cos_a + dy * sin_a, -dx * sin_a + dy * cos_a
        
        # 椭圆代数距离 f = (u/a)^2 + (v/b)^2 - 1
        f = (u / a) ** 2 + (v / b) ** 2 - 1.0
        
        # 梯度模平方（用于Sampson距离归一化）
        # ∇f = [2u/a^2, 2v/b^2]
        grad_norm_sq = (2 * u / a ** 2) ** 2 + (2 * v / b ** 2) ** 2 + 1e-12
        
        # Sampson距离 ≈ |f| / |∇f|
        return (f ** 2 / grad_norm_sq).sum()

    res = minimize(sampson_dist, init_params, args=(pts_2d,), method="L-BFGS-B")
    
    if not res.success:
        return None

    cx, cy, a, b, ang = res.x
    
    # 规范化：确保 a >= b，否则交换并调整角度
    if a < b:
        a, b, ang = b, a, ang + np.pi / 2
    
    # 将角度规范化到 [-π/2, π/2] 范围
    ang = (ang + np.pi / 2) % np.pi - np.pi / 2
    
    return cx, cy, a, b, ang


def _fit_ellipse_ransac(
    pts_2d: np.ndarray, 
    n_iter: int = 100, 
    tol_ratio: float = 0.15
) -> Tuple[Optional[Tuple[float, float, float, float, float]], np.ndarray]:
    """
    使用RANSAC鲁棒拟合椭圆，处理噪声和离群点.

    RANSAC流程：
    1. 随机采样最小子集（20点）进行几何拟合
    2. 计算所有点的几何距离，统计内点
    3. 保留内点最多的模型
    4. 用所有内点重新精修模型

    Args:
        pts_2d: 二维点集，shape (N, 2)。
        n_iter: RANSAC迭代次数，默认100。
        tol_ratio: 内点距离容差相对于点云范围的比例，默认0.15。

    Returns:
        tuple: (ellipse_params, inlier_mask)
               - ellipse_params: (cx, cy, a, b, angle) 或 None
               - inlier_mask: 布尔数组，标记内点，shape (N,)
    """
    if len(pts_2d) < 15:
        # 点数不足，直接拟合不分内外点
        return _fit_ellipse_geometric(pts_2d), np.ones(len(pts_2d), dtype=bool)

    # 根据点云范围计算绝对容差
    ref = max(np.ptp(pts_2d[:, 0]), np.ptp(pts_2d[:, 1]))
    tol = tol_ratio * ref
    
    best_res, best_inliers, max_in = None, np.zeros(len(pts_2d), dtype=bool), 0
    rng = np.random.default_rng(42)  # 固定种子保证可复现性

    for _ in range(n_iter):
        # 随机采样子集
        idx = rng.choice(len(pts_2d), size=min(20, len(pts_2d)), replace=False)
        res = _fit_ellipse_geometric(pts_2d[idx])
        
        if res is None:
            continue
            
        cx, cy, a, b, ang = res
        
        # 计算所有点到当前椭圆的几何距离
        dx, dy = pts_2d[:, 0] - cx, pts_2d[:, 1] - cy
        u, v = dx * np.cos(ang) + dy * np.sin(ang), -dx * np.sin(ang) + dy * np.cos(ang)
        
        # 归一化半径
        r = np.sqrt((u / a) ** 2 + (v / b) ** 2 + 1e-12)
        
        # 几何距离 = |r - 1| * 实际距离，其中r是归一化半径
        dist = np.abs(r - 1.0) * np.sqrt(u ** 2 + v ** 2) / r
        
        inliers = dist < tol
        
        # 更新最佳模型
        if inliers.sum() > max_in:
            max_in, best_inliers, best_res = inliers.sum(), inliers, res

    # 用所有内点重新精修
    if best_res is not None:
        refined = _fit_ellipse_geometric(pts_2d[best_inliers], init_params=list(best_res))
        if refined:
            best_res = refined
            
    return best_res, best_inliers


def _detect_arc_range(angles_deg: np.ndarray) -> Tuple[float, float]:
    """
    基于角度直方图检测圆柱有效弧段范围.

    算法步骤：
    1. 构建90-bin的角度直方图（覆盖0-360度）
    2. 高斯平滑消除噪声
    3. 标记低密度区域（<5%峰值）作为间隙
    4. 在循环边界上寻找最大连续间隙
    5. 返回间隙的补集作为有效弧段

    Args:
        angles_deg: 采样点角度数组 [度]，范围 [0, 360)。

    Returns:
        tuple: (start_deg, end_deg) 有效弧段起始和结束角度。
               始终满足 end_deg > start_deg，弧段跨度 <= 360度。
    """
    # 构建90-bin直方图（每bin 4度）
    hist, _ = np.histogram(angles_deg, bins=90, range=(0, 360))
    
    # 高斯平滑，sigma=1.0 bin，mode='wrap'处理循环边界
    smooth = gaussian_filter1d(hist.astype(float), sigma=1.0, mode="wrap")
    
    # 标记低密度区域（<5%峰值）为间隙
    empty = smooth < (0.05 * smooth.max())
    
    # 循环拼接以处理0/360度边界
    double = np.concatenate([empty, empty])
    
    # 寻找最长连续间隙
    best_start, best_len, curr_start, curr_len, in_gap = 0, 0, 0, 0, False
    
    for i, is_empty in enumerate(double):
        if is_empty:
            if not in_gap:
                # 进入新间隙
                curr_start, curr_len, in_gap = i, 1, True
            else:
                # 延续当前间隙
                curr_len += 1
        else:
            # 离开间隙
            if in_gap and curr_len > best_len:
                best_len, best_start = curr_len, curr_start
            in_gap = False
            
    # 处理跨越末尾的情况
    if in_gap and curr_len > best_len:
        best_len, best_start = curr_len, curr_start

    # 计算有效弧段（间隙的补集）
    # 注意：直方图bin索引转角度，每个bin代表4度
    start_deg = ((best_start + best_len) % 90) * 4.0
    # 有效弧段 = 总圆周 - 间隙长度
    return start_deg, start_deg + (90 - best_len) * 4.0


# ====================== 核心功能方法 ======================

def generate_surface_mesh_points_from_stl(
    stl_path: Path,
    m: int,
    n: int,
    ransac_iters: int = 100,
    inlier_tol: float = 0.15,
    use_cache: bool = True,
) -> np.ndarray:
    """
    拟合厚壁椭圆柱并在其外表面生成规则网格采样点.

    完整处理流程：
    1. 缓存检查：基于文件内容MD5的缓存机制
    2. 网格加载：支持单网格或场景文件（自动选择最大网格）
    3. 坐标系建立：基于法向量协方差估计主轴
    4. 粗椭圆拟合：全部点参与，获取初始截面参数
    5. 内外分离：基于到椭圆边界的距离分离外表面
    6. 精椭圆拟合：RANSAC鲁棒拟合外表面，获取精确参数
    7. 弧段检测：识别有效圆柱弧段范围
    8. 网格生成：在参数空间(m×n)均匀采样，映射回三维

    Args:
        stl_path: STL文件路径（字符串或Path对象）。
        m: 圆周方向采样点数（沿弧段均匀分布）。
        n: 轴向（圆柱高度方向）采样点数。
        ransac_iters: RANSAC迭代次数，默认100。增大可提高鲁棒性但增加计算时间。
        inlier_tol: RANSAC内点容差比例，默认0.15（相对于点云范围）。
                   减小可提高精度但可能丢失有效点。
        use_cache: 是否启用磁盘缓存，默认True。
                   缓存键基于文件内容MD5和采样参数。

    Returns:
        np.ndarray: 采样点云坐标数组，shape (m*n, 3)。
                    点按 [z_grid, theta_grid] 的row-major顺序排列，
                    即先遍历圆周方向，再遍历轴向。

    Raises:
        FileNotFoundError: STL文件不存在。
        ValueError: 网格为空或无法拟合有效椭圆。

    Examples:
        >>> # 基础用法：生成20×10的网格点云
        >>> pts = generate_surface_mesh_points_from_stl("finger.stl", m=20, n=10)
        >>> print(pts.shape) # (200, 3)
        
        >>> # 高精度模式：更多RANSAC迭代，更严格容差
        >>> pts = generate_surface_mesh_points_from_stl(
        ...     "finger.stl", m=50, n=20,
        ...     ransac_iters=500, inlier_tol=0.05
        ... )
    """
    stl_path = Path(stl_path)
    
    # ----- 缓存读取 -----
    if use_cache:
        key = _get_cache_key(stl_path, m, n, ransac_iters, inlier_tol)
        fpath = _cache_path(key)
        if fpath.exists():
            print(f"[STLSampler] 命中缓存: {stl_path.name} (m={m}, n={n})")
            return joblib.load(fpath)

    # ----- 实际计算 -----
    print(f"[STLSampler] 计算采样点云: {stl_path.name} (m={m}, n={n})...")

    # 1. 加载网格
    mesh = trimesh.load(stl_path)
    if not isinstance(mesh, trimesh.Trimesh):
        # 场景文件：选择面积最大的几何体
        mesh = max(mesh.geometry.values(), key=lambda x: x.area)
    
    if mesh.is_empty:
        raise ValueError("Loaded mesh is empty.")

    # 2. 建立局部坐标系
    # e_z = 主轴（圆柱方向），e_x, e_y = 截面平面基向量
    axis = _estimate_axis(mesh)
    
    # 避免axis与参考向量平行导致叉积为零
    ref = np.array([1, 0, 0.0]) if abs(np.dot(axis, [0, 0, 1])) > 0.9 else np.array([0, 0, 1.0])
    e_x = np.cross(ref, axis)
    e_x /= np.linalg.norm(e_x)
    e_y = np.cross(axis, e_x)  # 自动单位化，因为axis和e_x正交且单位长度

    # 3. 表面分离与拟合
    # 投影所有顶点到截面平面
    pts_all = mesh.vertices
    pts_2d_all = np.column_stack([pts_all @ e_x, pts_all @ e_y])
    
    # 粗拟合：获取初始椭圆参数用于分离内外表面
    c_res, _ = _fit_ellipse_ransac(pts_2d_all, n_iter=50)
    if c_res is None:
        raise ValueError("Failed to fit initial ellipse.")
        
    # 计算面片中心到椭圆边界的归一化距离
    f_centers = mesh.vertices[mesh.faces].mean(axis=1)
    dx, dy = (f_centers @ e_x) - c_res[0], (f_centers @ e_y) - c_res[1]
    
    # 旋转到椭圆主轴坐标系计算归一化半径
    cos_ang, sin_ang = np.cos(c_res[4]), np.sin(c_res[4])
    r_norm = np.sqrt(
        ((dx * cos_ang + dy * sin_ang) / c_res[2]) ** 2 +
        ((-dx * sin_ang + dy * cos_ang) / c_res[3]) ** 2
    )
    
    # 分离外表面：归一化半径大于中位数的视为外表面
    outer_face_idx = np.where(r_norm > np.median(r_norm))[0]
    o_mesh = mesh.submesh([outer_face_idx], append=True)
    
    # 4. 外表面精拟合
    # 增加表面采样点密度以提高拟合精度
    o_pts_3d = np.vstack([
        o_mesh.vertices,
        trimesh.sample.sample_surface(o_mesh, 5000)[0]
    ])
    o_pts_2d = np.column_stack([o_pts_3d @ e_x, o_pts_3d @ e_y])
    
    # RANSAC鲁棒拟合
    o_res, o_inliers = _fit_ellipse_ransac(
        o_pts_2d, n_iter=ransac_iters, tol_ratio=inlier_tol
    )
    
    if o_res is None:
        raise ValueError("Failed to fit outer ellipse with RANSAC.")
        
    cx, cy, a, b, ang = o_res

    # 5. 弧段检测与网格采样
    # 计算内点的归一化角度
    dx_in, dy_in = o_pts_2d[o_inliers, 0] - cx, o_pts_2d[o_inliers, 1] - cy
    angles = np.degrees(np.arctan2(
        (-dx_in * np.sin(ang) + dy_in * np.cos(ang)) / b,
        (dx_in * np.cos(ang) + dy_in * np.sin(ang)) / a,
    ))
    
    # 检测有效弧段范围
    s_deg, e_deg = _detect_arc_range(np.mod(angles, 360))
    
    # 轴向（高度）范围
    z_vals = pts_all @ axis
    z_grid = np.linspace(z_vals.min(), z_vals.max(), n)
    
    # 圆周方向参数网格
    t_grid = np.radians(np.linspace(s_deg, e_deg, m))
    
    # 生成二维参数网格
    T, Z = np.meshgrid(t_grid, z_grid)
    T_flat = T.ravel()
    Z_flat = Z.ravel()
    
    # 椭圆参数方程计算截面坐标
    u = a * np.cos(T_flat)
    v = b * np.sin(T_flat)
    
    # 旋转和平移到实际位置
    x_s = u * np.cos(ang) - v * np.sin(ang) + cx
    y_s = u * np.sin(ang) + v * np.cos(ang) + cy
    
    # 映射回三维空间：Z*axis + x_s*e_x + y_s*e_y
    sample_pts = (
        Z_flat[:, None] * axis[None, :] +
        x_s[:, None] * e_x[None, :] +
        y_s[:, None] * e_y[None, :]
    )

    # ----- 缓存写入 -----
    if use_cache:
        joblib.dump(sample_pts, fpath)
        print(f"[STLSampler] 已写入缓存: {fpath.name}")
        
    return sample_pts


# ====================== 测试入口 ======================

if __name__ == "__main__":
    """
    模块独立测试入口.
    
    测试内容：
    1. 首次调用：计算点云并写入缓存
    2. 二次调用：命中缓存，验证结果一致性
    3. 输出诊断信息：点云形状、缓存命中率
    
    使用方法：
        python stl_sampler.py
        
    预期输出：
        [STLSampler] 计算采样点云: skin_0_0_p.STL (m=10, n=7)...
        [STLSampler] 已写入缓存: {md5}_m10_n7_iter100_tol0.15.joblib
        Generated point cloud shape: (70, 3)
        [STLSampler] 命中缓存: skin_0_0_p.STL (m=10, n=7)
        Cached point cloud shape: (70, 3)
        Results identical: True
    """
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    PATH_TO_FILE = PROJECT_ROOT / "models" / "inspirehand" / "meshes" / "skin_0_0_p.STL"
    
    if Path(PATH_TO_FILE).exists():
        print(f"Testing generate_surface_mesh_points_from_stl with: {PATH_TO_FILE}")
        
        # 第一次：计算并写入缓存
        pts = generate_surface_mesh_points_from_stl(PATH_TO_FILE, m=10, n=7)
        print(f"Generated point cloud shape: {pts.shape}")
        
        # 第二次：命中缓存
        pts_cached = generate_surface_mesh_points_from_stl(PATH_TO_FILE, m=10, n=7)
        print(f"Cached point cloud shape: {pts_cached.shape}")
        print(f"Results identical: {np.allclose(pts, pts_cached)}")
    else:
        print("Script ready. Please import 'generate_surface_mesh_points_from_stl' into your project.")