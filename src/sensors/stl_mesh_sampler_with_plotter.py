"""
鲁棒厚壁椭圆柱面拟合与采样算法 v4.

该模块实现从STL网格自动分离并拟合厚壁椭圆柱内外表面，生成外表面规则采样点云。
采用基于RANSAC的鲁棒椭圆拟合、法向量辅助的内外表面分离、以及弧长均匀的网格采样策略。

核心改进（相对 v3）：
    1. 自动分离内外表面：
        - 先用 v3 做整体粗拟合，得到椭圆中心与法向
        - 按每个面片法向量与"径向朝外"方向的夹角 → 分为内表面（朝内）/ 外表面（朝外）
        - 对稀疏面片补充用"点到粗拟合椭圆的有符号距离"做分类
    2. 分别对内外两层做独立的 RANSAC + 几何距离椭圆拟合
    3. 壁厚估计：沿外椭圆法向方向量测到内椭圆的距离，取中位数作为全局壁厚
    4. 采样策略：在外表面椭圆柱均匀采样（m×n 网格），采用弧长均匀分割而非角度均匀
    5. 输出完整结果字典，包含内外椭圆参数、壁厚、采样点

算法流程：
    1. 柱轴估计：面法向量PCA（最小方差方向）
    2. 粗拟合：全部点参与RANSAC椭圆拟合
    3. 内外分离：基于位置+法向联合投票
    4. 精拟合：分别对内外表面点云RANSAC拟合
    5. 壁厚估计：沿法向射线求交
    6. 弧段检测：基于外表面角度直方图
    7. 网格采样：弧长均匀分割的矩形网格中点采样
    8. 可视化：5子图展示拟合过程与结果

依赖库：
    - trimesh: STL加载与网格处理
    - scipy: 优化求解与信号处理
    - matplotlib: 可视化
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


# ====================== 1. 柱轴估计 ======================

def estimate_cylinder_axis_from_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    基于面法向量的加权PCA估计圆柱主轴方向.

    算法原理：
        圆柱侧面法向量应垂直于主轴，因此法向量分布的协方差矩阵
        的最小特征值对应的特征向量即为主轴方向。

    Args:
        mesh: 输入三角网格对象。

    Returns:
        np.ndarray: 单位化的主轴方向向量，shape (3,)。

    Note:
        使用面片面积加权，大面对方向估计贡献更大，提高鲁棒性。
        适用于圆柱侧面占主导的网格。
    """
    areas = mesh.area_faces
    normals = mesh.face_normals
    w = areas / areas.sum()
    normals_w = normals * w[:, None]
    cov = normals_w.T @ normals
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, 0]
    return axis / np.linalg.norm(axis)


# ====================== 2. 几何距离椭圆拟合 ======================

def point_to_ellipse_sampson(params: tuple, pts: np.ndarray) -> float:
    """
    计算所有点到椭圆的几何距离平方和（Sampson距离）.

    Sampson距离是几何距离的一阶近似，比代数距离更稳定。
    对于椭圆 F(x,y) = (u/a)² + (v/b)² - 1 = 0，
    Sampson距离 ≈ |F| / |∇F|。

    Args:
        params: 椭圆参数 (cx, cy, a, b, angle)。
        pts: 二维点集，shape (N, 2)。

    Returns:
        float: Sampson距离平方和，作为优化目标函数值。
            若a<=0或b<=0返回大值(1e12)以惩罚无效参数。
    """
    cx, cy, a, b, angle = params
    if a <= 0 or b <= 0:
        return 1e12
    
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    
    # 旋转到椭圆主轴坐标系
    u = dx * cos_a + dy * sin_a
    v = -dx * sin_a + dy * cos_a
    
    # 椭圆代数距离
    F = (u / a) ** 2 + (v / b) ** 2 - 1.0
    
    # 梯度模平方（用于Sampson距离归一化）
    grad_u = 2 * u / a**2
    grad_v = 2 * v / b**2
    denom = grad_u**2 + grad_v**2 + 1e-12
    
    # Sampson距离平方和
    return (F**2 / denom).sum()


def fit_ellipse_geometric(
    pts_2d: np.ndarray, 
    init_params: list = None, 
    max_iter: int = 3
) -> tuple:
    """
    使用Sampson距离最小化进行几何椭圆拟合，带IRLS迭代重加权.

    求解优化问题：
        min Σ w_i * Sampson_distance(p_i, ellipse)²
    其中权重 w_i 通过迭代重加权最小二乘(IRLS)更新，降低离群点影响。

    Args:
        pts_2d: 二维点集，shape (N, 2)。至少需要10个点。
        init_params: 初始参数 [cx, cy, a, b, angle]。
            None时使用数据分位数估计（5%-95%范围）。
        max_iter: IRLS迭代次数，默认3。

    Returns:
        tuple: (cx, cy, a, b, angle) 椭圆参数，或 None 如果优化失败。
            - cx, cy: 椭圆中心坐标
            - a, b: 长半轴、短半轴（保证 a >= b）
            - angle: 长轴与x轴夹角 [弧度]，范围 [-π/2, π/2]

    Note:
        使用L-BFGS-B求解器，带边界约束防止参数发散。
        通过交换a,b和旋转90度确保 a >= b 的规范形式。
    """
    if len(pts_2d) < 10:
        return None
    
    if init_params is None:
        x, y = pts_2d[:, 0], pts_2d[:, 1]
        cx0, cy0 = np.median(x), np.median(y)
        # 使用5%-95%分位数避免离群点影响
        a0 = max((np.percentile(x, 95) - np.percentile(x, 5)) / 2, 1e-6)
        b0 = max((np.percentile(y, 95) - np.percentile(y, 5)) / 2, 1e-6)
        init_params = [cx0, cy0, a0, b0, 0.0]

    # 设置参数边界，防止优化过程中出现无效值
    ref = max(np.ptp(pts_2d[:, 0]), np.ptp(pts_2d[:, 1]))
    bounds = [
        (init_params[0] - ref, init_params[0] + ref),  # cx
        (init_params[1] - ref, init_params[1] + ref),  # cy
        (ref * 0.02, ref * 5.0),                       # a
        (ref * 0.02, ref * 5.0),                       # b
        (-np.pi / 2, np.pi / 2),                       # angle
    ]

    weights = np.ones(len(pts_2d))
    params = list(init_params)

    # IRLS迭代：交替优化参数和更新权重
    for _ in range(max_iter):
        # 加权优化
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
        
        # 计算残差并更新权重（Tukey双权重函数）
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx, dy = pts_2d[:, 0] - cx, pts_2d[:, 1] - cy
        u = dx * cos_a + dy * sin_a
        v = -dx * sin_a + dy * cos_a
        residuals = np.abs((u / a) ** 2 + (v / b) ** 2 - 1.0)
        sigma = np.median(residuals) + 1e-9  # 鲁棒尺度估计
        weights = 1.0 / (1.0 + (residuals / sigma) ** 2)

    # 规范化：确保 a >= b，否则交换并调整角度
    cx, cy, a, b, angle = params
    if a < b:
        a, b = b, a
        angle += np.pi / 2
    
    # 将角度规范化到 [-π/2, π/2] 范围
    angle = (angle + np.pi / 2) % np.pi - np.pi / 2
    
    return cx, cy, a, b, angle


def fit_ellipse_ransac(
    pts_2d: np.ndarray,
    n_iter: int = 200,
    inlier_tol: float = 0.15,
    min_inliers: int = 15,
) -> tuple:
    """
    使用RANSAC鲁棒拟合椭圆，处理噪声和离群点.

    RANSAC流程：
        1. 随机采样子集（大小为总点数的1/4到1/2，至少min_inliers）
        2. 几何拟合椭圆
        3. 计算所有点的几何距离，统计内点
        4. 保留内点最多的模型
        5. 用所有内点重新精修模型

    Args:
        pts_2d: 二维点集，shape (N, 2)。
        n_iter: RANSAC迭代次数，默认200。
        inlier_tol: 内点距离容差相对于点云范围的比例，默认0.15。
        min_inliers: 最小内点数阈值，默认15。

    Returns:
        tuple: (ellipse_params, inlier_mask)
            - ellipse_params: (cx, cy, a, b, angle) 或 None
            - inlier_mask: 布尔数组，标记内点，shape (N,)

    Note:
        当点数<min_inliers时退化为普通几何拟合（无RANSAC）。
        使用固定随机种子(42)保证结果可复现。
        最终使用所有内点进行精修，提高拟合精度。
    """
    if len(pts_2d) < min_inliers:
        # 点数不足，直接拟合不分内外点
        result = fit_ellipse_geometric(pts_2d)
        return result, np.ones(len(pts_2d), dtype=bool)

    # 根据点云范围计算绝对容差
    ref = max(np.ptp(pts_2d[:, 0]), np.ptp(pts_2d[:, 1]))
    tol = inlier_tol * ref
    
    best_result, best_inliers, best_count = None, np.zeros(len(pts_2d), dtype=bool), 0
    rng = np.random.default_rng(42)  # 固定种子保证可复现性

    for _ in range(n_iter):
        # 随机采样大小为总点数1/4到1/2的子集
        k = rng.integers(
            max(min_inliers, len(pts_2d) // 4), 
            max(min_inliers + 1, len(pts_2d) // 2)
        )
        idx = rng.choice(len(pts_2d), size=min(k, len(pts_2d)), replace=False)
        result = fit_ellipse_geometric(pts_2d[idx])
        
        if result is None:
            continue
        
        cx, cy, a, b, angle = result
        
        # 计算所有点到当前椭圆的几何距离
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx, dy = pts_2d[:, 0] - cx, pts_2d[:, 1] - cy
        u = dx * cos_a + dy * sin_a
        v = -dx * sin_a + dy * cos_a
        r = np.sqrt((u / a) ** 2 + (v / b) ** 2)
        r = np.where(r < 1e-9, 1e-9, r)  # 避免除零
        
        # 几何距离 = |r - 1| * 实际距离，其中r是归一化半径
        dist = np.abs(r - 1.0) * np.sqrt(u**2 + v**2) / r
        inliers = dist < tol
        
        # 更新最佳模型
        if inliers.sum() > best_count:
            best_count, best_inliers, best_result = inliers.sum(), inliers, result

    # 若内点不足，退化为全量拟合
    if best_count < min_inliers:
        best_result = fit_ellipse_geometric(pts_2d)
        best_inliers = np.ones(len(pts_2d), dtype=bool)

    # 用所有内点重新精修
    if best_result is not None and best_inliers.sum() > 0:
        refined = fit_ellipse_geometric(
            pts_2d[best_inliers], init_params=list(best_result)
        )
        if refined is not None:
            best_result = refined

    return best_result, best_inliers


# ====================== 3. 弧段检测 ======================

def detect_arc_robust(
    angles_deg: np.ndarray,
    n_bins: int = 90,
    sigma: float = 1.0,
    empty_thresh: float = 0.05,
) -> tuple:
    """
    基于角度直方图检测圆柱有效弧段范围.

    算法步骤：
        1. 构建n_bins-bin的角度直方图（覆盖0-360度）
        2. 高斯平滑消除噪声
        3. 标记低密度区域（<empty_thresh*峰值）作为间隙
        4. 在循环边界上寻找最大连续间隙
        5. 返回间隙的补集作为有效弧段

    Args:
        angles_deg: 采样点角度数组 [度]，范围 [0, 360)。
        n_bins: 直方图bin数量，默认90（每bin 4度）。
        sigma: 高斯平滑核标准差，默认1.0 bin。
        empty_thresh: 间隙判定阈值（相对于峰值比例），默认0.05。

    Returns:
        tuple: (start_deg, end_deg, arc_ratio)
            - start_deg: 有效弧段起始角度 [度]
            - end_deg: 有效弧段结束角度 [度]，始终满足 end_deg > start_deg
            - arc_ratio: 弧段占完整圆周的比例，范围 (0, 1]

    Note:
        使用循环数组处理角度环绕问题（0°/360°边界）。
        适用于检测部分圆柱（C型截面）的有效表面范围。
        若未检测到间隙（全圆周），返回 (0, 360, 1.0)。
    """
    # 构建直方图
    hist, _ = np.histogram(angles_deg, bins=n_bins, range=(0, 360))
    
    # 高斯平滑，mode='wrap'处理循环边界
    smooth = gaussian_filter1d(hist.astype(float), sigma=sigma, mode="wrap")
    
    # 标记低密度区域为间隙
    threshold = empty_thresh * smooth.max()
    empty = smooth < threshold
    
    # 循环拼接以处理0/360度边界
    double = np.concatenate([empty, empty])
    
    # 寻找最长连续间隙
    best_start, best_len, cur_start, cur_len, in_gap = 0, 0, 0, 0, False
    
    for i, is_empty in enumerate(double):
        if is_empty:
            if not in_gap:
                # 进入新间隙
                cur_start, cur_len, in_gap = i, 1, True
            else:
                # 延续当前间隙
                cur_len += 1
            
            # 更新最佳间隙
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            # 离开间隙
            in_gap = False
    
    # 若未检测到间隙，返回全圆周
    if best_len == 0:
        return 0.0, 360.0, 1.0
    
    # 计算有效弧段（间隙的补集）
    bw = 360.0 / n_bins
    gap_end_bin = (best_start + best_len) % n_bins
    start_deg = gap_end_bin * bw
    end_deg = start_deg + (n_bins - best_len) * bw
    arc_ratio = min((n_bins - best_len) / n_bins, 1.0)
    
    return start_deg, end_deg, arc_ratio


# ====================== 4. 内外表面分离 ======================

def separate_inner_outer_surfaces(
    mesh: trimesh.Trimesh,
    axis: np.ndarray,
    e_x: np.ndarray,
    e_y: np.ndarray,
    coarse_cx: float,
    coarse_cy: float,
    coarse_a: float,
    coarse_b: float,
    coarse_angle: float,
) -> tuple:
    """
    基于位置+法向联合投票的内外表面分离算法.

    分类策略：
        1. 计算每个面片质心在截面坐标系下的位置 (u, v)
        2. 计算该点在粗拟合椭圆上的"标准化半径" r = sqrt((u/a)² + (v/b)²)
           - r > 1 → 点在椭圆外侧 → 外表面候选
           - r < 1 → 点在椭圆内侧 → 内表面候选
        3. 计算面法向量与径向方向的点积符号作为辅助判据
           - fn_radial > 0 → 法向朝外 → 外表面
           - fn_radial < 0 → 法向朝内 → 内表面
        4. 联合投票：0.6*位置投票 + 0.4*法向投票
        5. 薄壁特殊情况（std(r) < 0.02）：主要依赖法向分类

    Args:
        mesh: 输入三角网格。
        axis: 圆柱主轴方向（已单位化）。
        e_x, e_y: 截面平面基向量（正交且单位化）。
        coarse_cx, coarse_cy: 粗拟合椭圆中心在截面坐标系中的坐标。
        coarse_a, coarse_b: 粗拟合椭圆半轴长度。
        coarse_angle: 粗拟合椭圆旋转角度 [弧度]。

    Returns:
        tuple: (outer_mask, inner_mask)
            - outer_mask: 布尔数组，标记外表面面片，shape (F,)
            - inner_mask: 布尔数组，标记内表面面片，shape (F,)

    Note:
        若任一侧面片数<5%，判定为分离失败，回退到全量作为外表面。
        适用于厚壁圆管类零件，要求内外表面几何可区分。
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
    radial_len = np.sqrt(radial_x**2 + radial_y**2) + 1e-12
    radial_x /= radial_len
    radial_y /= radial_len

    # 面法向量在截面内的径向分量
    fn_radial = (face_normals @ e_x) * radial_x + (face_normals @ e_y) * radial_y

    # ----- 分类逻辑 -----
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
        pos_vote = (r_norm > r_median).astype(float)  # 1=外, 0=内
        nor_vote = (fn_radial > 0).astype(float)      # 1=外, 0=内
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


# ====================== 5. 壁厚估计 ======================

def estimate_wall_thickness(
    outer_fit: tuple,
    inner_fit: tuple,
    n_samples: int = 360,
) -> tuple:
    """
    沿外椭圆法向方向量测到内椭圆的距离，估计壁厚.

    算法步骤：
        1. 在外椭圆上均匀采样n_samples个点（参数空间均匀）
        2. 计算每个点的外法向（向内，指向中心）
        3. 沿法向射线与内椭圆求交（二分法）
        4. 统计壁厚分布，剔除3σ异常值

    Args:
        outer_fit: 外椭圆参数 (cx, cy, a, b, angle)。
        inner_fit: 内椭圆参数 (cx, cy, a, b, angle)。
        n_samples: 采样点数，默认360（每度一个点）。

    Returns:
        tuple: (mean_thickness, std_thickness, thickness_array)
            - mean_thickness: 平均壁厚（剔除异常值后）
            - std_thickness: 壁厚标准差
            - thickness_array: 原始壁厚数组，shape (n_samples,)

    Note:
        使用二分法求交，精度取决于迭代次数（固定50次）。
        异常值剔除使用3σ准则，提高统计鲁棒性。
        适用于壁厚相对均匀的厚壁管状结构。
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
    norm_len = np.sqrt(nx_u**2 + nx_v**2) + 1e-12
    
    # 转回截面坐标
    nx_sec = (nx_u * cos_o - nx_v * sin_o) / norm_len
    ny_sec = (nx_u * sin_o + nx_v * cos_o) / norm_len
    
    # 确保法向朝内：若与"从外椭圆点指向外椭圆中心"方向一致则保留
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
        # 在内椭圆坐标系下的有符号距离函数
        def signed_dist_to_inner(t):
            px = x_o[i] + t * nx_sec[i]
            py = y_o[i] + t * ny_sec[i]
            dx = px - cx_i
            dy = py - cy_i
            u_i = dx * cos_i + dy * sin_i
            v_i = -dx * sin_i + dy * cos_i
            return (u_i / a_i) ** 2 + (v_i / b_i) ** 2 - 1.0

        # 粗估：在 [0, max_t] 上搜索符号变化
        max_t = a_o + b_o  # 上限设为外椭圆半轴和
        t_lo, t_hi = 0.0, max_t
        f_lo = signed_dist_to_inner(t_lo)
        f_hi = signed_dist_to_inner(t_hi)

        if f_lo * f_hi > 0:
            # 没有符号变化：取最小值处
            ts = np.linspace(0, max_t, 200)
            fs = np.abs([signed_dist_to_inner(t) for t in ts])
            thicknesses.append(ts[np.argmin(fs)])
            continue

        # 二分求根（50次迭代）
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
    
    return (
        float(np.mean(thicknesses[mask])),
        float(np.std(thicknesses[mask])),
        thicknesses,
    )


# ====================== 6. 主函数 ======================

def fit_thick_elliptic_cylinder(
    stl_path: Path,
    m: int,
    n: int,
    ransac_iters: int = 150,
    inlier_tol: float = 0.15,
    show_plot: bool = True,
) -> dict:
    """
    厚壁椭圆柱面拟合与外表面采样主函数.

    完整处理流程：
        1. 加载STL并估计柱轴
        2. 建立截面坐标系（axis, e_x, e_y）
        3. 面片重采样与粗拟合
        4. 内外表面分离
        5. 分别精拟合内外椭圆（RANSAC）
        6. 壁厚估计（沿法向射线求交）
        7. 弧段检测（基于外表面角度直方图）
        8. 弧长均匀网格采样（矩形网格中点）
        9. 可视化（5子图）

    Args:
        stl_path: STL文件路径（字符串或Path对象）。
        m: 周向采样点数（沿弧段均匀分布）。
        n: 轴向（圆柱高度方向）采样点数。
        ransac_iters: RANSAC迭代次数，默认150。
        inlier_tol: RANSAC内点容差比例，默认0.15。
        show_plot: 是否显示可视化，默认True。

    Returns:
        dict: 包含以下键值：
            - sample_pts: (m*n, 3) 外表面采样点坐标
            - outer_ellipse: 外椭圆参数字典（cx, cy, a, b, angle_rad, inlier_ratio）
            - inner_ellipse: 内椭圆参数字典（cx, cy, a, b, angle_rad, inlier_ratio）
            - wall_thickness: 壁厚字典（mean, std, array）
            - arc: 弧段信息字典（start_deg, end_deg, ratio）
            - cylinder_axis: 柱轴方向向量 (3,)
            - section_axes: 截面基向量元组 (e_x, e_y)

    Raises:
        FileNotFoundError: STL文件不存在。
        RuntimeError: 粗拟合失败或内外拟合均失败。

    Examples:
        >>> # 基础用法：生成10×7网格
        >>> result = fit_thick_elliptic_cylinder("finger.stl", m=10, n=7)
        >>> print(result['outer_ellipse'])
        >>> print(result['wall_thickness']['mean'])
        
        >>> # 高精度模式
        >>> result = fit_thick_elliptic_cylinder(
        ...     "finger.stl", m=20, n=10, 
        ...     ransac_iters=300, inlier_tol=0.05
        ... )

    Note:
        采样策略采用弧长均匀分割而非角度均匀，确保网格单元在曲率大的区域
        （短半轴附近）更密集，曲率小的区域更稀疏，符合触觉传感器的布置需求。
        可视化包含5个子图：原始STL、内外拟合曲面、截面2D、壁厚极坐标、采样网格。
    """
    print(f"\n{'='*60}")
    print(f"Processing (thick-wall): {stl_path}")
    print(f"{'='*60}")

    # ----- 加载网格 -----
    mesh = trimesh.load(stl_path)
    if not isinstance(mesh, trimesh.Trimesh):
        # 场景文件：选择面积最大的几何体
        meshes = list(mesh.geometry.values())
        mesh = max(meshes, key=lambda m_: m_.area)

    # ----- 1. 柱轴估计 -----
    axis = estimate_cylinder_axis_from_normals(mesh)
    print(f"Cylinder axis: {axis}")

    # 建立截面坐标系
    ref_v = np.array([0, 0, 1.0])
    if abs(np.dot(axis, ref_v)) > 0.9:
        ref_v = np.array([1, 0, 0.0])
    e_x = np.cross(ref_v, axis)
    e_x /= np.linalg.norm(e_x)
    e_y = np.cross(axis, e_x)
    e_y /= np.linalg.norm(e_y)

    # ----- 2. 面片均匀重采样 -----
    # 获取最大连通分量（去除噪声孤岛）
    facet_groups = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(mesh.faces))
    )
    main_idx = np.argmax([mesh.area_faces[g].sum() for g in facet_groups])
    main_mesh = mesh.submesh([facet_groups[main_idx]], append=True)

    # 在主导连通分量上重采样
    n_resample = max(len(main_mesh.vertices) * 10, 8000)
    sampled_pts, sampled_face_idx = trimesh.sample.sample_surface(main_mesh, n_resample)
    all_pts = np.vstack([main_mesh.vertices, sampled_pts])
    print(f"Resampled: {len(main_mesh.vertices)} verts → {n_resample} pts")

    # 投影到截面坐标系
    z_vals = all_pts @ axis
    x_vals = all_pts @ e_x
    y_vals = all_pts @ e_y
    pts_2d = np.column_stack([x_vals, y_vals])

    # ----- 3. 粗拟合（全部点）→ 得到大致椭圆参数 -----
    print("Step 1/3: Coarse fit (all points)...")
    coarse_result, _ = fit_ellipse_ransac(pts_2d, n_iter=100, inlier_tol=0.2)
    if coarse_result is None:
        raise RuntimeError("Coarse ellipse fitting failed.")
    c_cx, c_cy, c_a, c_b, c_angle = coarse_result
    print(
        f"  Coarse: a={c_a:.4f}, b={c_b:.4f}, center=({c_cx:.4f},{c_cy:.4f}), "
        f"angle={np.degrees(c_angle):.1f}°"
    )

    # ----- 4. 内外表面分离（面片级别） -----
    print("Step 2/3: Separating inner/outer surfaces...")
    outer_face_mask, inner_face_mask = separate_inner_outer_surfaces(
        main_mesh, axis, e_x, e_y, c_cx, c_cy, c_a, c_b, c_angle
    )

    # 获取内外面片对应的顶点
    def get_pts_for_faces(mesh_, face_mask):
        """提取面片对应的顶点并补充采样."""
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
        """将3D点投影到截面2D坐标."""
        return np.column_stack([pts @ e_x, pts @ e_y])

    outer_2d = to_2d(outer_pts_3d)
    inner_2d = to_2d(inner_pts_3d)

    # ----- 5. 分别精拟合内外椭圆 -----
    print("Step 3/3: Fine-fitting outer and inner ellipses...")
    outer_result, outer_inliers = fit_ellipse_ransac(
        outer_2d, n_iter=ransac_iters, inlier_tol=inlier_tol
    )
    inner_result, inner_inliers = fit_ellipse_ransac(
        inner_2d, n_iter=ransac_iters, inlier_tol=inlier_tol
    )

    # 失败回退策略
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

    print(
        f"  Outer ellipse: a={o_a:.4f}, b={o_b:.4f}, "
        f"center=({o_cx:.4f},{o_cy:.4f}), angle={np.degrees(o_ang):.1f}°, "
        f"inliers={outer_inliers.sum()/len(outer_inliers):.1%}"
    )
    print(
        f"  Inner ellipse: a={i_a:.4f}, b={i_b:.4f}, "
        f"center=({i_cx:.4f},{i_cy:.4f}), angle={np.degrees(i_ang):.1f}°, "
        f"inliers={inner_inliers.sum()/len(inner_inliers):.1%}"
    )

    # ----- 6. 壁厚估计 -----
    t_mean, t_std, t_arr = estimate_wall_thickness(outer_result, inner_result)
    print(f"  Wall thickness: mean={t_mean:.4f}, std={t_std:.4f}")

    # ----- 7. 弧段检测（基于外表面） -----
    o_cos, o_sin = np.cos(o_ang), np.sin(o_ang)
    dx_o = outer_2d[outer_inliers, 0] - o_cx
    dy_o = outer_2d[outer_inliers, 1] - o_cy
    u_o = dx_o * o_cos + dy_o * o_sin
    v_o = -dx_o * o_sin + dy_o * o_cos
    angles_deg = np.degrees(np.mod(np.arctan2(v_o / o_b, u_o / o_a), 2 * np.pi))
    start_deg, end_deg, arc_ratio = detect_arc_robust(angles_deg)
    print(f"  Arc (outer): [{start_deg:.1f}°, {end_deg:.1f}°], ratio={arc_ratio:.2f}")

    # ----- 8. 坐标还原（外表面采样） -----
    def to_world_outer(z_arr: np.ndarray, theta_arr: np.ndarray) -> np.ndarray:
        """
        将柱坐标（z, theta）映射回世界坐标3D点.
        
        Args:
            z_arr: 轴向坐标数组，shape (N,)。
            theta_arr: 角度坐标数组（弧度），shape (N,)。
        
        Returns:
            np.ndarray: 世界坐标3D点，shape (N, 3)。
        """
        u = o_a * np.cos(theta_arr)
        v = o_b * np.sin(theta_arr)
        x_sec = u * o_cos - v * o_sin + o_cx
        y_sec = u * o_sin + v * o_cos + o_cy
        return (
            z_arr[:, None] * axis[None, :]
            + x_sec[:, None] * e_x[None, :]
            + y_sec[:, None] * e_y[None, :]
        )

    # z 范围用全部点
    z_all = all_pts @ axis

    # ========== 矩形网格中点采样（弧长均匀） ==========
    def ellipse_arc_length_divisions(
        a: float, b: float, n_divisions: int, start_deg: float, end_deg: float
    ) -> tuple:
        """
        将角度区间按弧长均匀分割为 n_divisions 段.

        通过数值积分计算椭圆弧长累积分布，然后等弧长插值得到角度分割线。

        Args:
            a: 椭圆长半轴。
            b: 椭圆短半轴。
            n_divisions: 分割段数。
            start_deg: 起始角度 [度]。
            end_deg: 结束角度 [度]。

        Returns:
            tuple: (division_angles, total_arc_length)
                - division_angles: (n_divisions+1,) 分割线角度数组 [弧度]
                - total_arc_length: 总弧长
        """
        start_rad = np.radians(start_deg)
        end_rad = np.radians(end_deg)

        # 椭圆参数方程导数的模（弧长微分）
        def arc_integrand(theta):
            return np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)

        # 数值积分计算累积弧长
        n_integral = 1000
        thetas_integral = np.linspace(start_rad, end_rad, n_integral)
        ds = arc_integrand(thetas_integral)
        cumulative_arc = np.cumsum(ds) * (end_rad - start_rad) / (n_integral - 1)
        total_arc = cumulative_arc[-1]

        # 等弧长分布的分割线位置（n_divisions+1 条线）
        target_arcs = np.linspace(0, total_arc, n_divisions + 1)

        # 通过插值找到对应的角度
        division_angles = np.interp(target_arcs, cumulative_arc, thetas_integral)

        return division_angles, total_arc

    # 定义弧段起止角度
    theta_start = np.radians(start_deg)
    theta_end = np.radians(end_deg)

    # 生成周向分割线：(m+1) 条弧长均匀分割线
    theta_divisions, total_arc_length = ellipse_arc_length_divisions(
        o_a, o_b, m, start_deg, end_deg
    )
    # 取 m 个区间中点作为采样角度
    theta_c = 0.5 * (theta_divisions[:-1] + theta_divisions[1:])

    # 生成轴向分割线：(n+1) 条均匀分割线
    z_divisions = np.linspace(z_all.min(), z_all.max(), n + 1)
    # 取 n 个区间中点作为采样高度
    z_c = 0.5 * (z_divisions[:-1] + z_divisions[1:])

    # 生成二维参数网格并映射到3D
    THETA, Z = np.meshgrid(theta_c, z_c)
    sample_pts = to_world_outer(Z.ravel(), THETA.ravel())
    
    print(f"Generated {len(sample_pts)} sample points on outer surface ({m}×{n})")
    print(f"  Grid: {m} arc-length divisions × {n} axial divisions")
    print(f"  Total arc length: {total_arc_length:.4f}, arc per cell: {total_arc_length/m:.4f}")
    print(f"  Axial range: [{z_all.min():.4f}, {z_all.max():.4f}], height per cell: {(z_all.max()-z_all.min())/n:.4f}")

    # ----- 9. 可视化 -----
    if show_plot:
        # 生成外/内拟合曲面（用于绘图）
        num_plot = 60
        theta_plot = np.linspace(theta_start, theta_end, num_plot)
        z_plot = np.linspace(z_all.min(), z_all.max(), num_plot)
        THETA_p, Z_p = np.meshgrid(theta_plot, z_plot)

        def make_surf(cx_, cy_, a_, b_, ang_):
            """生成椭圆柱曲面网格点."""
            cos_, sin_ = np.cos(ang_), np.sin(ang_)
            u_ = a_ * np.cos(THETA_p.ravel())
            v_ = b_ * np.sin(THETA_p.ravel())
            xs = u_ * cos_ - v_ * sin_ + cx_
            ys = u_ * sin_ + v_ * cos_ + cy_
            pts_ = (
                Z_p.ravel()[:, None] * axis[None, :]
                + xs[:, None] * e_x[None, :]
                + ys[:, None] * e_y[None, :]
            )
            return pts_.reshape(num_plot, num_plot, 3)

        outer_surf = make_surf(o_cx, o_cy, o_a, o_b, o_ang)
        inner_surf = make_surf(i_cx, i_cy, i_a, i_b, i_ang)

        # 创建5子图布局
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
            """设置等比例坐标轴范围."""
            lims = np.array([mesh.vertices.min(0), mesh.vertices.max(0)])
            c = lims.mean(0)
            r = (lims[1] - lims[0]).max() * 0.55
            ax.set_xlim(c[0] - r, c[0] + r)
            ax.set_ylim(c[1] - r, c[1] + r)
            ax.set_zlim(c[2] - r, c[2] + r)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

        # 子图 1：原始 STL
        ax1 = fig.add_subplot(151, projection="3d")
        ax1.plot_trisurf(
            mesh.vertices[:, 0],
            mesh.vertices[:, 1],
            mesh.vertices[:, 2],
            triangles=mesh.faces,
            color="gray",
            alpha=0.4,
            linewidth=0.1,
        )
        ax1.set_title("1. Original STL")
        set_eq(ax1)

        # 子图 2：内外拟合曲面叠加
        ax2 = fig.add_subplot(152, projection="3d")
        ax2.plot_trisurf(
            mesh.vertices[:, 0],
            mesh.vertices[:, 1],
            mesh.vertices[:, 2],
            triangles=mesh.faces,
            color="gray",
            alpha=0.15,
            linewidth=0,
        )
        ax2.plot_surface(
            outer_surf[..., 0],
            outer_surf[..., 1],
            outer_surf[..., 2],
            color="steelblue",
            alpha=0.6,
            linewidth=0,
            label="Outer",
        )
        ax2.plot_surface(
            inner_surf[..., 0],
            inner_surf[..., 1],
            inner_surf[..., 2],
            color="salmon",
            alpha=0.6,
            linewidth=0,
            label="Inner",
        )
        ax2.set_title("2. Outer (blue) + Inner (red)\nFitted surfaces")
        set_eq(ax2)

        # 子图 3：截面 2D 内外点云 + 拟合椭圆
        ax3 = fig.add_subplot(153)
        ax3.scatter(
            outer_2d[::5, 0],
            outer_2d[::5, 1],
            c="steelblue",
            s=3,
            alpha=0.4,
            label="Outer pts",
        )
        ax3.scatter(
            inner_2d[::5, 0],
            inner_2d[::5, 1],
            c="salmon",
            s=3,
            alpha=0.4,
            label="Inner pts",
        )
        t_ell = np.linspace(0, 2 * np.pi, 300)
        for cx_, cy_, a_, b_, ang_, col, lbl in [
            (o_cx, o_cy, o_a, o_b, o_ang, "navy", "Outer ellipse"),
            (i_cx, i_cy, i_a, i_b, i_ang, "darkred", "Inner ellipse"),
        ]:
            cos_, sin_ = np.cos(ang_), np.sin(ang_)
            u_ = a_ * np.cos(t_ell)
            v_ = b_ * np.sin(t_ell)
            x_ = u_ * cos_ - v_ * sin_ + cx_
            y_ = u_ * sin_ + v_ * cos_ + cy_
            ax3.plot(x_, y_, color=col, lw=2, label=lbl)
        ax3.set_aspect("equal")
        ax3.legend(fontsize=7, markerscale=3)
        ax3.set_title("3. Cross-section (2D)\nInner / Outer separation")

        # 子图 4：壁厚分布（极坐标直方图）
        ax4 = fig.add_subplot(154, projection="polar")
        theta_polar = np.linspace(0, 2 * np.pi, len(t_arr), endpoint=False)
        ax4.plot(theta_polar, t_arr, "steelblue", lw=1.5)
        ax4.fill(theta_polar, t_arr, alpha=0.3, color="steelblue")
        ax4.axhline(
            t_mean, color="red", lw=1.5, linestyle="--", label=f"mean={t_mean:.3f}"
        )
        ax4.set_title(f"4. Wall Thickness\n(polar, mean={t_mean:.4f})")

        # 子图 5：外表面采样点 + 完整 (m+1)×(n+1) 分割线网格
        ax5 = fig.add_subplot(155, projection="3d")
        ax5.plot_trisurf(
            mesh.vertices[:, 0],
            mesh.vertices[:, 1],
            mesh.vertices[:, 2],
            triangles=mesh.faces,
            color="gray",
            alpha=0.15,
            linewidth=0,
        )

        # 周向分割线：n+1 条椭圆环线
        for z_div in z_divisions:
            theta_line = np.linspace(theta_start, theta_end, 100)
            line_pts = to_world_outer(np.full_like(theta_line, z_div), theta_line)
            ax5.plot(
                line_pts[:, 0],
                line_pts[:, 1],
                line_pts[:, 2],
                "gray",
                alpha=0.6,
                linewidth=1.0,
                linestyle="-",
            )

        # 轴向分割线：m+1 条竖直母线
        for theta_div in theta_divisions:
            z_line = np.linspace(z_all.min(), z_all.max(), 50)
            line_pts = to_world_outer(z_line, np.full_like(z_line, theta_div))
            ax5.plot(
                line_pts[:, 0],
                line_pts[:, 1],
                line_pts[:, 2],
                "gray",
                alpha=0.6,
                linewidth=1.0,
                linestyle="-",
            )

        # 网格交点：(m+1)×(n+1) 个
        THETA_grid, Z_grid = np.meshgrid(theta_divisions, z_divisions)
        grid_intersections = to_world_outer(Z_grid.ravel(), THETA_grid.ravel())
        ax5.scatter(
            grid_intersections[:, 0],
            grid_intersections[:, 1],
            grid_intersections[:, 2],
            c="black",
            s=20,
            marker="+",
            alpha=0.8,
            linewidth=1.5,
            label=f"Grid intersections ({(m+1)*(n+1)})",
        )

        # 采样点：m×n 个
        arc_positions = np.linspace(0, 1, m)
        colors = np.tile(arc_positions, n)
        ax5.scatter(
            sample_pts[:, 0],
            sample_pts[:, 1],
            sample_pts[:, 2],
            c=colors,
            cmap="jet",
            s=80,
            edgecolors="black",
            linewidth=0.5,
            zorder=5,
            label=f"Sample points ({m}×{n})",
        )

        ax5.set_title(
            f"5. Sampling Grid with Divisions\n"
            f"({m+1}×{n+1}={(m+1)*(n+1)} intersections, {m}×{n}={m*n} cell centers)"
        )
        set_eq(ax5)
        ax5.legend(loc="upper left", fontsize=7)

        plt.tight_layout()
        plt.show()

    return {
        "sample_pts": sample_pts,
        "outer_ellipse": {
            "cx": o_cx,
            "cy": o_cy,
            "a": o_a,
            "b": o_b,
            "angle_rad": o_ang,
            "inlier_ratio": float(outer_inliers.sum() / len(outer_inliers)),
        },
        "inner_ellipse": {
            "cx": i_cx,
            "cy": i_cy,
            "a": i_a,
            "b": i_b,
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


# ====================== 入口 ======================

if __name__ == "__main__":
    """
    模块独立测试入口.

    测试内容：
        1. 加载示例STL文件（InspireHand皮肤网格）
        2. 执行完整拟合与采样流程
        3. 输出关键结果参数
        4. 显示5子图可视化

    运行方式：
        python thick_wall_ellipse_fit.py

    预期输出：
        ============================================================
        Processing (thick-wall): /path/to/skin_0_0_p.STL
        ============================================================
        Cylinder axis: [x x x]
        Resampled: X verts → Y pts
        Step 1/3: Coarse fit (all points)...
          Coarse: a=X.X, b=X.X, center=(X,X), angle=X.X°
        Step 2/3: Separating inner/outer surfaces...
          Surface separation: X outer faces, Y inner faces
          Outer pts: X, Inner pts: Y
        Step 3/3: Fine-fitting outer and inner ellipses...
          Outer ellipse: a=X, b=X, center=(X,X), angle=X°, inliers=X%
          Inner ellipse: a=X, b=X, center=(X,X), angle=X°, inliers=X%
          Wall thickness: mean=X, std=X
          Arc (outer): [X°, X°], ratio=X
        Generated X sample points on outer surface (10×7)
          Grid: 10 arc-length divisions × 7 axial divisions
          ...
    """
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    STL_PATH = PROJECT_ROOT / "models" / "inspirehand" / "meshes" / "skin_0_0_p.STL"

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
    print(
        f"Wall thickness: mean={result['wall_thickness']['mean']:.4f}, "
        f"std={result['wall_thickness']['std']:.4f}"
    )
    print(f"Arc           : {result['arc']}")
    print(f"Cylinder axis : {result['cylinder_axis']}")
    print(f"\nFirst 5 sample points:\n{result['sample_pts'][:5]}")