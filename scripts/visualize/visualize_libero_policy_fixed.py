#!/usr/bin/env python3
"""
可视化 Libero 环境中保存的策略轨迹数据
"""

import dataclasses
import pathlib
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from PIL import Image
import tyro


@dataclasses.dataclass
class Args:
    """可视化参数"""
    
    # 记录数据路径
    records_path: str = "policy_records"
    # 输出路径
    output_path: str = "visualizations"
    # 可视化类型: ['trajectory', 'actions', 'images', 'animation', 'all']
    viz_type: str = "all"
    # 是否保存图片
    save_plots: bool = True
    # 图片DPI
    dpi: int = 150
    # 动画帧率
    fps: int = 5


def load_policy_records(records_path: str) -> List[dict]:
    """加载所有记录的策略数据"""
    record_path = pathlib.Path(records_path)
    
    if not record_path.exists():
        raise FileNotFoundError(f"Records path {records_path} does not exist")
    
    # 获取所有step文件
    step_files = sorted(record_path.glob("step_*.npy"), 
                       key=lambda x: int(x.stem.split('_')[1]))
    
    records = []
    for step_file in step_files:
        try:
            record = np.load(step_file, allow_pickle=True).item()
            records.append(record)
        except Exception as e:
            print(f"Warning: Could not load {step_file}: {e}")
            continue
    
    print(f"Loaded {len(records)} policy records")
    return records


def analyze_data_structure(records: List[dict]) -> None:
    """分析数据结构"""
    if not records:
        print("No records to analyze")
        return
    
    print("\n=== Data Structure Analysis ===")
    print(f"Total steps: {len(records)}")
    print(f"Keys in each record: {list(records[0].keys())}")
    
    for key, value in records[0].items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"{key}: type {type(value)}, value: {value}")


def visualize_trajectory_3d(records: List[dict], output_path: str, save: bool = True) -> None:
    """可视化3D机械臂轨迹"""
    # 提取末端执行器位置 (前3维是位置)
    positions = np.array([record['inputs/observation/state'][:3] for record in records])
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制轨迹
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            'b-', linewidth=2, alpha=0.7, label='End-effector trajectory')
    
    # 标记起始点和结束点
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
              color='green', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
              color='red', s=100, label='End')
    
    # 每隔几步标记一个点
    step_interval = max(1, len(positions) // 20)
    for i in range(0, len(positions), step_interval):
        ax.scatter(positions[i, 0], positions[i, 1], positions[i, 2], 
                  color='orange', s=30, alpha=0.6)
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    # ax.set_zlabel('Z Position (m)')  # Some versions of matplotlib don't support this
    ax.set_title('Robot End-Effector 3D Trajectory')
    ax.legend()
    
    plt.tight_layout()
    
    if save:
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{output_path}/trajectory_3d.png", dpi=150, bbox_inches='tight')
        print(f"Saved 3D trajectory plot to {output_path}/trajectory_3d.png")
    
    plt.show()


def visualize_state_evolution(records: List[dict], output_path: str, save: bool = True) -> None:
    """可视化状态演化"""
    # 提取状态信息
    states = np.array([record['inputs/observation/state'] for record in records])
    
    # 状态维度：[x, y, z, rx, ry, rz, gripper_width, gripper_state]
    state_names = ['X pos', 'Y pos', 'Z pos', 'Roll', 'Pitch', 'Yaw', 'Gripper Width', 'Gripper State']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes, state_names)):
        ax.plot(states[:, i], linewidth=2)
        ax.set_title(f'{name}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Robot State Evolution Over Time', fontsize=16)
    plt.tight_layout()
    
    if save:
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{output_path}/state_evolution.png", dpi=150, bbox_inches='tight')
        print(f"Saved state evolution plot to {output_path}/state_evolution.png")
    
    plt.show()


def visualize_actions(records: List[dict], output_path: str, save: bool = True) -> None:
    """可视化动作序列"""
    # 提取动作信息 (形状为 [time_steps, prediction_horizon, action_dim])
    actions_data = []
    for record in records:
        actions = record['outputs/actions']  # shape: (10, 7)
        # 只取第一个预测步骤的动作
        actions_data.append(actions[0])
    
    actions = np.array(actions_data)
    
    # 动作维度：[x, y, z, rx, ry, rz, gripper]
    action_names = ['X action', 'Y action', 'Z action', 'Roll action', 'Pitch action', 'Yaw action', 'Gripper action']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes[:7], action_names)):
        ax.plot(actions[:, i], linewidth=2, color=f'C{i}')
        ax.set_title(f'{name}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Action Value')
        ax.grid(True, alpha=0.3)
    
    # 最后一个子图显示动作幅度
    ax = axes[7]
    action_magnitude = np.linalg.norm(actions[:, :6], axis=1)  # 不包括gripper
    ax.plot(action_magnitude, linewidth=2, color='black')
    ax.set_title('Action Magnitude')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Magnitude')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Action Sequence Over Time', fontsize=16)
    plt.tight_layout()
    
    if save:
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{output_path}/actions.png", dpi=150, bbox_inches='tight')
        print(f"Saved actions plot to {output_path}/actions.png")
    
    plt.show()


def visualize_images_grid(records: List[dict], output_path: str, save: bool = True, max_images: int = 16) -> None:
    """可视化图像网格"""
    num_steps = min(len(records), max_images)
    step_interval = max(1, len(records) // num_steps)
    
    # 创建主相机和手腕相机的图像网格
    for camera_type in ['inputs/observation/image', 'inputs/observation/wrist_image']:
        camera_name = 'main_camera' if 'wrist' not in camera_type else 'wrist_camera'
        
        rows = int(np.sqrt(num_steps))
        cols = int(np.ceil(num_steps / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_steps):
            step_idx = i * step_interval
            if step_idx >= len(records):
                break
                
            row, col = divmod(i, cols)
            ax = axes[row, col]
            
            # 获取图像
            img = records[step_idx][camera_type]
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            
            ax.imshow(img)
            ax.set_title(f'Step {step_idx}')
            ax.axis('off')
        
        # 隐藏多余的子图
        for i in range(num_steps, rows * cols):
            row, col = divmod(i, cols)
            axes[row, col].axis('off')
        
        plt.suptitle(f'{camera_name.replace("_", " ").title()} Images', fontsize=16)
        plt.tight_layout()
        
        if save:
            pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{output_path}/{camera_name}_grid.png", dpi=150, bbox_inches='tight')
            print(f"Saved {camera_name} grid to {output_path}/{camera_name}_grid.png")
        
        plt.show()


def create_animation(records: List[dict], output_path: str, fps: int = 5) -> None:
    """创建动画（简化版本，只保存图像序列）"""
    # 准备数据
    positions = np.array([record['inputs/observation/state'][:3] for record in records])
    main_images = [record['inputs/observation/image'] for record in records]
    wrist_images = [record['inputs/observation/wrist_image'] for record in records]
    
    # 确保图像是uint8格式
    for i in range(len(main_images)):
        if main_images[i].dtype != np.uint8:
            main_images[i] = (main_images[i] * 255).astype(np.uint8)
        if wrist_images[i].dtype != np.uint8:
            wrist_images[i] = (wrist_images[i] * 255).astype(np.uint8)
    
    # 创建输出目录
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    animation_dir = pathlib.Path(output_path) / "animation_frames"
    animation_dir.mkdir(exist_ok=True)
    
    # 保存每一帧的组合图像
    for frame in range(len(records)):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # 子图1: 到当前帧的轨迹
        current_positions = positions[:frame+1]
        ax1.plot(current_positions[:, 0], current_positions[:, 1], 'b-', linewidth=2)
        if len(current_positions) > 0:
            ax1.scatter(current_positions[-1, 0], current_positions[-1, 1], color='red', s=50)
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title(f'2D Trajectory (Step {frame})')
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 主相机
        ax2.imshow(main_images[frame])
        ax2.set_title('Main Camera')
        ax2.axis('off')
        
        # 子图3: 手腕相机
        ax3.imshow(wrist_images[frame])
        ax3.set_title('Wrist Camera')
        ax3.axis('off')
        
        plt.tight_layout()
        plt.savefig(animation_dir / f"frame_{frame:04d}.png", dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {len(records)} animation frames to {animation_dir}")
    print(f"You can create a video using: ffmpeg -r {fps} -i {animation_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {output_path}/policy_animation.mp4")


def print_summary_statistics(records: List[dict]) -> None:
    """打印汇总统计信息"""
    print("\n=== Summary Statistics ===")
    
    # 状态统计
    states = np.array([record['inputs/observation/state'] for record in records])
    state_names = ['X pos', 'Y pos', 'Z pos', 'Roll', 'Pitch', 'Yaw', 'Gripper Width', 'Gripper State']
    
    print("\nState Statistics:")
    for i, name in enumerate(state_names):
        print(f"{name:15s}: mean={states[:, i].mean():.4f}, std={states[:, i].std():.4f}, "
              f"min={states[:, i].min():.4f}, max={states[:, i].max():.4f}")
    
    # 动作统计
    actions_data = []
    for record in records:
        actions = record['outputs/actions']
        actions_data.append(actions[0])  # 只取第一个预测步骤
    actions = np.array(actions_data)
    
    action_names = ['X action', 'Y action', 'Z action', 'Roll action', 'Pitch action', 'Yaw action', 'Gripper action']
    
    print("\nAction Statistics:")
    for i, name in enumerate(action_names):
        print(f"{name:15s}: mean={actions[:, i].mean():.4f}, std={actions[:, i].std():.4f}, "
              f"min={actions[:, i].min():.4f}, max={actions[:, i].max():.4f}")
    
    # 推理时间统计
    inference_times = [record['outputs/policy_timing/infer_ms'] for record in records]
    print(f"\nInference Time Statistics:")
    print(f"Mean: {np.mean(inference_times):.2f} ms")
    print(f"Std:  {np.std(inference_times):.2f} ms")
    print(f"Min:  {np.min(inference_times):.2f} ms")
    print(f"Max:  {np.max(inference_times):.2f} ms")
    
    # 任务信息
    prompt = records[0]['inputs/prompt']
    print(f"\nTask Prompt: {prompt}")


def main(args: Args) -> None:
    """主函数"""
    print("Loading policy records...")
    records = load_policy_records(args.records_path)
    
    if not records:
        print("No records found!")
        return
    
    # 分析数据结构
    analyze_data_structure(records)
    
    # 打印统计信息
    print_summary_statistics(records)
    
    # 创建输出目录
    pathlib.Path(args.output_path).mkdir(parents=True, exist_ok=True)
    
    # 根据可视化类型进行可视化
    if args.viz_type in ['trajectory', 'all']:
        print("\nGenerating 3D trajectory visualization...")
        visualize_trajectory_3d(records, args.output_path, args.save_plots)
    
    if args.viz_type in ['trajectory', 'all']:
        print("\nGenerating state evolution visualization...")
        visualize_state_evolution(records, args.output_path, args.save_plots)
    
    if args.viz_type in ['actions', 'all']:
        print("\nGenerating actions visualization...")
        visualize_actions(records, args.output_path, args.save_plots)
    
    if args.viz_type in ['images', 'all']:
        print("\nGenerating images visualization...")
        visualize_images_grid(records, args.output_path, args.save_plots)
    
    if args.viz_type in ['animation', 'all']:
        print("\nGenerating animation frames...")
        create_animation(records, args.output_path, args.fps)
    
    print(f"\nVisualization complete! Results saved to {args.output_path}")


if __name__ == "__main__":
    tyro.cli(main)
