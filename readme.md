# 项目名称：VTLA 机器人仿真控制系统 (MuJoCo)

## 1. 项目概述

本项目是一个基于 **MuJoCo** 物理引擎的机器人仿真平台，专门用于 **Vision-Tactile-Language-Action (VTLA)** 多模态模型的研究与开发[cite: 1]。系统集成了 **RM75-B 7自由度机械臂** 与 **Inspire 灵巧手**，实现了从底层力矩控制到高层操作空间控制（OSC）的全栈功能[cite: 1]。

### 核心特性

- **多模态融合控制**：支持机械臂与 6 自由度灵巧手的协同控制[cite: 1]。
- **物理透明接口**：底层控制器自动处理 MuJoCo 执行器传动比（Gear），提供直观的物理力矩（N·m）接口[cite: 1]。
- **高性能控制算法**：实现基于 SVD 截断伪逆的**操作空间控制器 (OSC)**，有效处理冗余自由度与奇异构型[cite: 1]。
- **触觉仿真集成**：内置基于物理的弹性触觉传感器阵列，支持实时触觉热力图可视化[cite: 1]。
- **灵活的模型组装**：提供动态 XML 模型合并工具，支持自定义安装姿态与物理参数覆盖[cite: 1]。

---

## 2. 目录结构与文件说明

当前合并报告包含 10 个核心文件，前 5 个关键模块如下：

| 模块路径                                 | 功能描述                                                                               |
| :--------------------------------------- | :------------------------------------------------------------------------------------- |
| `src/controllers/hand_arm_controller.py` | **底层执行器管理器**：负责执行器映射、物理约束提取及力矩控制信号转换[cite: 1]。        |
| `src/controllers/position_controller.py` | **运动控制器库**：包含高性能 OSC 控制器及兼容旧版的 IK/PD 控制实现[cite: 1]。          |
| `src/simulation/robot_arm_system.py`     | **模型组装工具**：实现机械臂与灵巧手的 XML 合并、姿态修正及物理参数批量配置[cite: 1]。 |
| `src/simulation/circle_movement_test.py` | **运动演示模块**：圆周轨迹跟踪测试，包含实时目标与轨迹可视化工具[cite: 1]。            |
| `src/simulation/grasp_task_env.py`       | **抓取任务环境**：集成了相机、目标物体及触觉传感器热力图显示的综合演示环境[cite: 1]。  |

---

## 3. 核心算法实现

### 3.1 操作空间控制 (OSC)

控制器直接在笛卡尔空间定义期望加速度 $\ddot{x}_{des}$，并通过操作空间惯量矩阵 $\Lambda$ 映射到关节力矩 $\tau$[cite: 1]：
$$\tau = J^T \Lambda \ddot{x}_{des} + \tau_{bias} + \tau_{null}$$

- **零空间控制**：利用机械臂的冗余自由度，在不影响末端任务的前提下将姿态拉回参考构型[cite: 1]。
- **数值稳定性**：采用截断 SVD 处理雅可比矩阵，防止在奇异构型附近产生过大力矩[cite: 1]。

### 3.2 触觉传感器仿真

系统在灵巧手指节表面部署了 `touch_grid` 传感器网格[cite: 1]：

- **实时读取**：通过 `read_all_tactile` 接口获取各指节受力数据[cite: 1]。
- **热力图可视化**：将触觉信号处理为 $160 \times 120$ 的热力图，并按手指/指节逻辑顺序拼图显示[cite: 1]。

---

## 4. 快速开始

### 环境依赖

- Ubuntu 22.04+ (支持 WSL2)[cite: 1]
- Python 3.10+
- MuJoCo 3.0+
- NumPy, OpenCV, SciPy

### 单独展示robot的功能

```bash
python -m src.simulation.robot_arm_system
```

该脚本将加载机械臂与灵巧手的合成模型，并在 MuJoCo Viewer 中展示初始姿态。

### 运行圆周轨迹演示

```bash
python -m src.simulation.circle_movement_test
```

- **青色球体**：代表机械臂末端实际位置[cite: 1]。
- **红色球体**：代表实时更新的指令目标位置[cite: 1]。

### 运行抓取与触觉演示

```bash
python -m src.simulation.grasp_task_env
```

该脚本将加载 `data/position_log.csv` 中的预录制轨迹，并实时显示五指触觉热力图窗口[cite: 1]。

---
