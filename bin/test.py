import mujoco
import mujoco.viewer
import time

# 1. 设置模型路径
MODEL_PATH = "/home/zmy/MyProject/models/test_models/touchtest.xml"

def main():
    # 2. 加载模型和数据
    try:
        print(f"正在加载模型: {MODEL_PATH} ...")
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data = mujoco.MjData(model)
        print("模型加载成功！")
    except Exception as e:
        print(f"加载失败: {e}")
        return

    # 输出模型的可控制执行器数量和关节数量
    print(f"模型包含 {model.nu} 个可控制执行器和 {model.njnt} 个关节。")
    
    # 输出执行器的名称 (使用 model.actuator(i).name)
    print("执行器名称:")
    for i in range(model.nu):
        print(f"  {i}: {model.actuator(i).name}")
    
    # 输出传感器数量，并输出传感器的名称 (使用 model.sensor(i).name)
    print(f"模型包含 {model.nsensor} 个传感器。")
    print("传感器名称:")
    for i in range(model.nsensor):
        print(f"  {i}: {model.sensor(i).name}")

    # 3. 启动查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        timestep = model.opt.timestep
        print("查看器已启动。按 ESC 关闭窗口。")
        
        # 4. 仿真主循环
        while viewer.is_running():
            step_start = time.time()

            # 物理步进
            mujoco.mj_step(model, data)

            # 同步查看器
            viewer.sync()

            # 速度控制
            time_until_next_step = timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()