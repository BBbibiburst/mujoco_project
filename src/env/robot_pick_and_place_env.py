import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import viewer

# 导入你的核心模块
from src.robot.robot_arm_system import get_combined_spec, PhysicsConfig
from src.controllers.hand_arm_controller import HandArmController
from src.controllers.position_controller import OSC_PositionController as PositionController, OSCGains
from src.sensors.tactile_sensor import TactileReader

class RobotPickAndPlaceEnv(gym.Env):
    def __init__(self, render_mode="human"):
        super(RobotPickAndPlaceEnv, self).__init__()
        
        # 1. 模型合成
        physics_cfg = PhysicsConfig() # 使用默认物理参数
        self.spec, self.sensor_map = get_combined_spec(
            arm_xml="arm.xml",   # 请替换为你实际的路径
            hand_xml="hand.xml", 
            config=physics_cfg
        )
        
        # 在此处添加目标物体(Box)到场景中
        worldbody = self.spec.worldbody
        self.obj = worldbody.add_body(name="object", pos=[0.5, 0, 0.05])
        self.obj.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.02], rgba=[1, 0, 0, 1], condim=4)
        self.obj.add_freejoint()

        # 编译模型
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        
        # 2. 初始化你的控制器与感知器
        self.hw_interface = HandArmController(self.model, self.data)
        self.osc_controller = PositionController(self.model, self.data, OSCGains())
        self.tactile_reader = TactileReader(self.model, self.data, self.sensor_map)
        
        # 3. 定义 RL 空间 (动作: 末端 dx, dy, dz + 手指抓取力度)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # 观测空间: [末端位姿(7), 物体位姿(7), 触觉特征(15)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32)

        self.render_mode = render_mode
        self.viewer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # 随机化物体位置
        self.data.body("object").xpos[:2] = np.random.uniform([-0.1, -0.1], [0.1, 0.1]) + [0.5, 0]
        
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        # 提取末端位置
        ee_pos = self.data.site("endpoint").xpos
        obj_pos = self.data.body("object").xpos
        # 获取结构化的触觉数据并展平
        tactile_data = self.tactile_reader.get_structured_data()
        tactile_flat = np.array([np.mean(v) for v in tactile_data.values()])
        
        return np.concatenate([ee_pos, obj_pos, tactile_flat]).astype(np.float32)

    def step(self, action):
        # action: [dx, dy, dz, grasp_strength]
        
        # 1. 运动控制逻辑
        target_pos = self.data.site("endpoint").xpos + action[:3] * 0.05
        # 调用你的 PositionController 计算 OSC 力矩
        arm_torques = self.osc_controller.compute(target_pos, target_quat=None)
        
        # 2. 抓取逻辑 (简化处理)
        hand_torques = np.ones(6) * action[3] * 2.0 
        
        # 3. 通过你的底层控制器执行
        full_torques = np.concatenate([arm_torques, hand_torques])
        self.hw_interface.apply_torque(full_torques)
        
        # 4. 仿真步进
        mujoco.mj_step(self.model, self.data)
        
        # 5. 计算奖励 (抓取放置逻辑)
        obj_pos = self.data.body("object").xpos
        ee_pos = self.data.site("endpoint").xpos
        dist = np.linalg.norm(obj_pos - ee_pos)
        
        reward = -dist # 靠近奖励
        if obj_pos[2] > 0.1: # 成功抬起
            reward += 10.0
            
        terminated = False
        truncated = False
        obs = self._get_obs()
        
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.viewer is None:
            self.viewer = viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

# --- 运行示例 ---
if __name__ == "__main__":
    # 实例化你要求的通用接口环境
    env = RobotPickAndPlaceEnv(render_mode="human")
    
    obs, info = env.reset()
    for _ in range(1000):
        # 这里可以用真正的 RL 算法 (如 PPO) 的 model.predict(obs)
        action = env.action_space.sample() 
        obs, reward, done, _, _ = env.step(action)
        
        if done:
            obs, info = env.reset()