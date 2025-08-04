#!/usr/bin/env python3
"""
在 Libero 环境中实时可视化策略轨迹
"""

import dataclasses
import pathlib
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
from PIL import Image
import tyro

# 导入Libero相关模块
try:
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    LIBERO_AVAILABLE = True
except ImportError:
    print("Warning: Libero not available. Only offline visualization will work.")
    LIBERO_AVAILABLE = False


@dataclasses.dataclass
class Args:
    """可视化参数"""
    
    # 模式: 'offline' 从保存的记录可视化, 'online' 在环境中实时可视化
    mode: str = "offline"
    # 记录数据路径（offline模式）
    records_path: str = "policy_records"
    # 输出路径
    output_path: str = "visualizations"
    # Libero任务套件（online模式）
    task_suite_name: str = "libero_spatial"
    # 任务ID（online模式）
    task_id: int = 0
    # 试验次数（online模式）
    episode_id: int = 0
    # 是否保存图片
    save_plots: bool = True
    # 是否显示实时图像
    show_images: bool = True
    # 更新频率（online模式）
    update_interval: int = 100  # milliseconds


class LiberoVisualizationEnv:
    """Libero环境可视化包装器"""
    
    def __init__(self, task_suite_name: str, task_id: int, episode_id: int = 0):
        if not LIBERO_AVAILABLE:
            raise ImportError("Libero is not available")
        
        # 初始化任务
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        self.task = task_suite.get_task(task_id)
        self.initial_states = task_suite.get_task_init_states(task_id)
        
        # 初始化环境
        self.env = self._create_env()
        self.episode_id = episode_id
        
        # 记录数据
        self.trajectory_history = []
        self.action_history = []
        self.image_history = []
        self.current_step = 0
        
    def _create_env(self):
        """创建Libero环境"""
        task_description = self.task.language
        task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / self.task.problem_folder / self.task.bddl_file
        env_args = {
            "bddl_file_name": task_bddl_file, 
            "camera_heights": 256, 
            "camera_widths": 256
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(42)
        return env
    
    def reset(self):
        """重置环境"""
        self.env.reset()
        obs = self.env.set_init_state(self.initial_states[self.episode_id])
        self.trajectory_history = []
        self.action_history = []
        self.image_history = []
        self.current_step = 0
        
        # 记录初始状态
        self._record_step(obs, None)
        return obs
    
    def step(self, action):
        """执行动作并记录"""
        obs, reward, done, info = self.env.step(action)
        self._record_step(obs, action)
        self.current_step += 1
        return obs, reward, done, info
    
    def _record_step(self, obs, action):
        """记录步骤数据"""
        # 提取机械臂状态
        state = np.concatenate([
            obs["robot0_eef_pos"],
            self._quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"]
        ])
        
        self.trajectory_history.append(state)
        
        if action is not None:
            self.action_history.append(action)
        
        # 处理图像（旋转180度匹配训练数据）
        main_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        
        self.image_history.append({
            'main': main_img,
            'wrist': wrist_img
        })
    
    def _quat2axisangle(self, quat):
        """四元数转轴角表示"""
        import math
        
        if quat[3] > 1.0:
            quat[3] = 1.0
        elif quat[3] < -1.0:
            quat[3] = -1.0
        
        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            return np.zeros(3)
        
        return (quat[:3] * 2.0 * math.acos(quat[3])) / den
    
    def get_task_description(self):
        """获取任务描述"""
        return self.task.language


class RealTimeVisualizer:
    """实时可视化器"""
    
    def __init__(self, show_images: bool = True):
        self.show_images = show_images
        self.fig, self.axes = self._setup_plot()
        self.trajectory_line = None
        self.current_point = None
        
    def _setup_plot(self):
        """设置绘图"""
        if self.show_images:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        return fig, axes
    
    def update(self, env: LiberoVisualizationEnv):
        """更新可视化"""
        if len(env.trajectory_history) == 0:
            return
        
        # 清除所有轴
        for ax in self.axes.flat:
            ax.clear()
        
        trajectory = np.array(env.trajectory_history)
        
        if self.show_images and len(env.image_history) > 0:
            # 主相机图像
            self.axes[0, 0].imshow(env.image_history[-1]['main'])
            self.axes[0, 0].set_title(f'Main Camera - Step {env.current_step}')
            self.axes[0, 0].axis('off')
            
            # 手腕相机图像
            self.axes[0, 1].imshow(env.image_history[-1]['wrist'])
            self.axes[0, 1].set_title(f'Wrist Camera - Step {env.current_step}')
            self.axes[0, 1].axis('off')
            
            # 3D轨迹
            ax_3d = self.axes[0, 2]
            ax_3d.remove()
            ax_3d = self.fig.add_subplot(2, 3, 3, projection='3d')
            self.axes[0, 2] = ax_3d
            
            ax_3d.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', linewidth=2)
            if len(trajectory) > 0:
                ax_3d.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                             color='red', s=100)
            ax_3d.set_xlabel('X')
            ax_3d.set_ylabel('Y')
            ax_3d.set_title('3D Trajectory')
            
            # XY平面轨迹
            self.axes[1, 0].plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2)
            if len(trajectory) > 0:
                self.axes[1, 0].scatter(trajectory[-1, 0], trajectory[-1, 1], 
                                       color='red', s=100)
            self.axes[1, 0].set_xlabel('X Position (m)')
            self.axes[1, 0].set_ylabel('Y Position (m)')
            self.axes[1, 0].set_title('XY Trajectory')
            self.axes[1, 0].axis('equal')
            self.axes[1, 0].grid(True)
            
            # 状态演化
            if len(trajectory) > 1:
                steps = range(len(trajectory))
                self.axes[1, 1].plot(steps, trajectory[:, 2], label='Z position', linewidth=2)
                self.axes[1, 1].plot(steps, trajectory[:, 6], label='Gripper width', linewidth=2)
                self.axes[1, 1].set_xlabel('Step')
                self.axes[1, 1].set_ylabel('Value')
                self.axes[1, 1].set_title('State Evolution')
                self.axes[1, 1].legend()
                self.axes[1, 1].grid(True)
            
            # 任务信息
            task_info = f"""Task: {env.get_task_description()}
Step: {env.current_step}
Position: ({trajectory[-1, 0]:.3f}, {trajectory[-1, 1]:.3f}, {trajectory[-1, 2]:.3f})
Gripper: {trajectory[-1, 6]:.4f}"""n            \n            self.axes[1, 2].text(0.1, 0.5, task_info, transform=self.axes[1, 2].transAxes,\n                                fontsize=10, verticalalignment='center',\n                                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))\n            self.axes[1, 2].set_xlim(0, 1)\n            self.axes[1, 2].set_ylim(0, 1)\n            self.axes[1, 2].axis('off')\n            self.axes[1, 2].set_title('Info')\n            \n        else:\n            # 简化版本，只显示轨迹\n            # XY轨迹\n            self.axes[0].plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2)\n            if len(trajectory) > 0:\n                self.axes[0].scatter(trajectory[-1, 0], trajectory[-1, 1], \n                                    color='red', s=100)\n            self.axes[0].set_xlabel('X Position (m)')\n            self.axes[0].set_ylabel('Y Position (m)')\n            self.axes[0].set_title(f'XY Trajectory - Step {env.current_step}')\n            self.axes[0].axis('equal')\n            self.axes[0].grid(True)\n            \n            # 状态演化\n            if len(trajectory) > 1:\n                steps = range(len(trajectory))\n                self.axes[1].plot(steps, trajectory[:, 2], label='Z position', linewidth=2)\n                self.axes[1].plot(steps, trajectory[:, 6], label='Gripper width', linewidth=2)\n                self.axes[1].set_xlabel('Step')\n                self.axes[1].set_ylabel('Value')\n                self.axes[1].set_title('State Evolution')\n                self.axes[1].legend()\n                self.axes[1].grid(True)\n        \n        plt.tight_layout()\n        plt.pause(0.01)  # 短暂暂停以更新显示\n\n\ndef offline_visualization(args: Args):\n    \"\"\"离线可视化已保存的记录\"\"\"\n    # 使用之前创建的可视化脚本\n    from scripts.visualize_libero_policy_fixed import load_policy_records, main as viz_main\n    \n    # 创建临时参数对象\n    @dataclasses.dataclass\n    class TempArgs:\n        records_path: str = args.records_path\n        output_path: str = args.output_path\n        viz_type: str = \"all\"\n        save_plots: bool = args.save_plots\n        dpi: int = 150\n        fps: int = 5\n    \n    temp_args = TempArgs()\n    viz_main(temp_args)\n\n\ndef online_visualization(args: Args):\n    \"\"\"在线可视化（需要Libero环境）\"\"\"\n    if not LIBERO_AVAILABLE:\n        print(\"Error: Libero is not available for online visualization\")\n        return\n    \n    # 创建环境\n    env = LiberoVisualizationEnv(args.task_suite_name, args.task_id, args.episode_id)\n    visualizer = RealTimeVisualizer(args.show_images)\n    \n    print(f\"Task: {env.get_task_description()}\")\n    print(\"Starting online visualization...\")\n    print(\"Close the plot window to stop.\")\n    \n    # 重置环境\n    obs = env.reset()\n    \n    try:\n        # 显示初始状态\n        visualizer.update(env)\n        plt.show(block=False)\n        \n        # 模拟步骤（这里你需要集成你的策略）\n        max_steps = 100\n        for step in range(max_steps):\n            # 这里应该是你的策略推理代码\n            # action = your_policy.predict(obs)\n            # 现在我们使用随机动作作为示例\n            action = np.random.randn(7) * 0.1  # 小幅随机动作\n            \n            obs, reward, done, info = env.step(action)\n            \n            # 更新可视化\n            if step % (args.update_interval // 100) == 0:  # 控制更新频率\n                visualizer.update(env)\n            \n            if done:\n                print(f\"Task completed at step {step}\")\n                break\n        \n        # 保存最终结果\n        if args.save_plots:\n            pathlib.Path(args.output_path).mkdir(parents=True, exist_ok=True)\n            plt.savefig(f\"{args.output_path}/online_visualization_final.png\", \n                       dpi=150, bbox_inches='tight')\n            print(f\"Final visualization saved to {args.output_path}/online_visualization_final.png\")\n        \n        # 保持窗口开启\n        plt.show()\n        \n    except KeyboardInterrupt:\n        print(\"\\nVisualization stopped by user\")\n    except Exception as e:\n        print(f\"Error during visualization: {e}\")\n\n\ndef main(args: Args) -> None:\n    \"\"\"主函数\"\"\"\n    if args.mode == \"offline\":\n        print(\"Starting offline visualization...\")\n        offline_visualization(args)\n    elif args.mode == \"online\":\n        print(\"Starting online visualization...\")\n        online_visualization(args)\n    else:\n        raise ValueError(f\"Unknown mode: {args.mode}. Choose 'offline' or 'online'\")\n\n\nif __name__ == \"__main__\":\n    tyro.cli(main)
