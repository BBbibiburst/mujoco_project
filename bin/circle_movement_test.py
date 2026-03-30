import mujoco
from mujoco import mjtGeom, mjtJoint
from robot_arm_system import get_combined_spec, PhysicsConfig, JointPhysicsConfig
from position_controller import PositionController
from typing import Tuple
import numpy as np
import time

# 内部模块导入
from hand_arm_controller import HandArmController

# ====================== 仿真配置 ======================
TARGET_CENTER = np.array([1,1,1])  # 圆周运动的圆心
CIRCLE_RADIUS = 1                     # 圆周半径 (10cm)
CIRCLE_SPEED  = 1.5                      # 旋转角速度 (rad/s)
HAND_RANDOM_INTERVAL = 1.0               # 手部随机运动变换间隔 (秒)

# ====================== 仿真常量配置 ======================

# 相机分辨率配置 (宽 x 高)，影响渲染质量和计算开销
CAM_WIDTH  = 320   # 相机图像宽度（像素）
CAM_HEIGHT = 240   # 相机图像高度（像素）

# 目标物体初始位置 [x, y, z]（单位：米）
# 位于机械臂前方0.4米，高度0.025米处
TARGET_POS = [0.4, 0.0, 0.025]


# ====================== 物理参数配置 ======================

DEFAULT_GRASP_PHYSICS = PhysicsConfig(
    # 机械臂默认物理参数：较高的阻尼确保运动平稳
    arm_defaults=JointPhysicsConfig(
        damping=100.0,        # 关节阻尼系数，抑制振荡
        frictionloss=0.1,      # 摩擦损耗，模拟关节摩擦
        armature=0.01,       # 电机惯量，影响动态响应
    ),
    # 机械手默认物理参数：低阻尼以实现灵活抓取
    hand_defaults=JointPhysicsConfig(
        damping=0.01,        # 手指关节低阻尼，保证灵活性
        frictionloss=0.01,   # 手指摩擦系数
        armature=0.01,       # 手指电机惯量
    ),
    # 特定关节参数覆盖：拇指旋转关节需要更高阻尼以保持稳定性
    per_joint_overrides={
        "thumb_rotate_act_push_j": JointPhysicsConfig(damping=10.0),
    }
)


# ====================== 环境构建 ======================

def build_custom_grasp_environment(
    physics: PhysicsConfig = DEFAULT_GRASP_PHYSICS,
) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    构建并编译抓取仿真环境，添加外部环境物体与光照.
    
    Args:
        physics: 物理参数配置对象，包含机械臂和机械手的关节物理属性。
                默认为 DEFAULT_GRASP_PHYSICS。
    
    Returns:
        Tuple[mujoco.MjModel, mujoco.MjData]: 
            - model: 编译后的 MuJoCo 模型对象
            - data:  对应的仿真数据对象，包含状态信息
    
    环境组成：
        1. 机械臂+机械手组合体（通过 get_combined_spec 导入）
        2. 顶部定向光源照明
        3. 可抓取的红色立方体目标物体
        4. 俯视全局相机，自动计算视野角度
    """
    print("=== [EnvBuilder] 开始构建自定义抓取环境 ===")

    # 获取组合机器人模型规格，机械手安装在"right_hand"连接点
    # rot_xyz_deg: 机械手相对机械臂的旋转角度（度）
    spec = get_combined_spec(
        rot_xyz_deg=(-90, 0, 0),           # 绕X轴旋转-90度，调整机械手姿态
        attach_point_name="right_hand",     # 机械臂腕部连接点名称
        physics=physics,                    # 应用物理参数配置
    )
    worldbody = spec.worldbody

    # ----- 配置环境光照 -----
    # 添加顶部平行光源，提供均匀照明
    worldbody.add_light(
        name="top_light",                   # 光源标识名称
        pos=[0.0, 0.0, 2.0],               # 光源位置：场景正上方2米处
        dir=[0.0, 0.0, -1.0],              # 照射方向：垂直向下
        diffuse=[1.0, 1.0, 1.0],           # 漫反射颜色：白色
    )

    # ----- 添加抓取目标物体 -----
    # 创建可自由移动的立方体作为抓取目标
    obj_body = worldbody.add_body(
        name="target_box",                  # 物体名称
        pos=TARGET_POS,                     # 初始位置
    )
    # 添加立方体几何形状和物理属性
    obj_body.add_geom(
        type=mjtGeom.mjGEOM_BOX,           # 几何类型：立方体
        size=[0.025, 0.025, 0.025],        # 半边长：5cm立方体
        rgba=[1.0, 0.2, 0.2, 1.0],         # 颜色：红色不透明
        mass=0.2,                          # 质量：200克
    )
    # 添加6自由度自由关节，允许物体在空间中自由移动和旋转
    obj_body.add_joint(
        name="box_free", 
        type=mjtJoint.mjJNT_FREE           # 自由关节类型：6DOF
    )

    # ----- 自动定位俯视相机 -----
    # 计算机械臂基座与目标物体的中点位置
    base_pos   = np.array([0.0, 0.0, 0.0])  # 机械臂基座位置（假设在原点）
    target_pos = np.array(TARGET_POS)
    mid_point  = (base_pos + target_pos) / 2.0  # 场景中心点
    
    # 相机高度设置
    cam_height = 3.0  # 相机距离地面3米

    # 根据场景几何自动计算合适的视野角度(FOVY)
    # 确保相机能同时看到机械臂基座和目标物体
    dist_to_cover = np.linalg.norm(target_pos - base_pos)  # 基座到目标的水平距离
    # 三角函数计算：fovy = 2 * arctan(覆盖范围 / 相机高度)
    calc_fovy = 2 * np.degrees(np.arctan2((dist_to_cover / 2) * 2.0, cam_height))

    # 添加俯视相机到场景
    worldbody.add_camera(
        name="downward_cam",                # 相机名称
        pos=[mid_point[0], mid_point[1], cam_height],  # 位置：中点正上方
        xyaxes=[0, 1, 0, -1, 0, 0],        # 相机坐标系朝向：俯视-Z轴
        fovy=calc_fovy,                     # 垂直视野角度（自动计算）
    )

    print("[EnvBuilder] 模型构建完成，正在编译并生成仿真对象...")
    # 编译模型规格，生成可仿真的模型对象
    model = spec.compile()
    # 创建对应的仿真数据对象，存储状态变量
    data  = mujoco.MjData(model)
    return model, data

def main():
    try:
        # 1. 初始化模型和数据
        model, data = build_custom_grasp_environment()
        
        # 2. 初始化控制器
        # 注意：HandArmController 是底层，PositionController 是封装了 IK 和 PD 的上层
        base_controller = HandArmController(model)
        pos_controller = PositionController(base_controller, model)

        # 3. 计算运动到位时间 (到达圆周起点)
        # 获取当前末端位置
        ee_id = pos_controller.ee_id
        current_ee_pos = data.site_xpos[ee_id].copy()
        # 设定圆周起点 (angle = 0)
        start_target = TARGET_CENTER + np.array([CIRCLE_RADIUS, 0, 0])
        
        # 估算时间：基于距离和平均移动速度
        # 假设关节步长限制为 0.05 rad/step，仿真频率为 1/timestep
        # 这是一个经验估算值，实际受 PD 参数影响
        dist = np.linalg.norm(start_target - current_ee_pos)
        estimated_arrival_time = dist / 0.1  # 假设平均末端移动速度 0.1m/s
        print(f"\n[规划] 目标圆心: {TARGET_CENTER}, 半径: {CIRCLE_RADIUS}m")
        print(f"[规划] 预计机械臂运动到位时间: 约 {estimated_arrival_time:.2f} 秒\n")

        # 4. 启动可视化
        with mujoco.viewer.launch_passive(model, data) as viewer:
            sim_time = 0.0
            last_hand_update = -HAND_RANDOM_INTERVAL
            hand_target = np.zeros(6)

            while viewer.is_running():
                step_start = time.time()
                
                # --- A. 机械臂圆周运动逻辑 ---
                # 计算当前圆周角度
                angle = sim_time * CIRCLE_SPEED
                ee_target_pos = TARGET_CENTER + np.array([
                    CIRCLE_RADIUS * np.cos(angle),
                    CIRCLE_RADIUS * np.sin(angle),
                    0  # 保持在固定高度
                ])
                
                # 保持手的姿势
                # 如果你的模型姿态不同，请调整此四元数
                ee_target_quat = np.array([0, 0, 0, 0]) 

                # --- B. 机械手随机运动逻辑 ---
                if sim_time - last_hand_update > HAND_RANDOM_INTERVAL:
                    # 为 6 个手部关节生成随机目标 (参考 self.hand_range)
                    low = pos_controller.hand_range[:, 0]
                    high = pos_controller.hand_range[:, 1]
                    hand_target = np.random.uniform(low, high)
                    last_hand_update = sim_time

                # --- C. 调用 IK 接口下发控制 ---
                pos_controller.set_ee_target(
                    data,
                    ee_pos_target=ee_target_pos,
                    ee_quat_target=ee_target_quat,
                    hand_target=hand_target
                )

                # --- D. 物理步进 ---
                mujoco.mj_step(model, data)
                viewer.sync()

                # 更新仿真时间
                sim_time += model.opt.timestep
                
                # 实时状态打印
                if int(sim_time * 10) % 2 == 0: # 降低打印频率
                    print(f"\r[Sim {sim_time:.2f}s] EE_Pos: {data.site_xpos[ee_id].round(3)} | "
                          f"Hand_Move: 随机中", end="", flush=True)

                # 维持仿真频率 (可选)
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n=== 仿真结束 ===")

if __name__ == "__main__":
    main()