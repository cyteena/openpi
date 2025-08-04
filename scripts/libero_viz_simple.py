#!/usr/bin/env python3
"""
在 Libero 环境中实时可视化策略轨迹（简化版本）
"""

import dataclasses
import pathlib
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tyro


@dataclasses.dataclass
class Args:
    """可视化参数"""
    
    # 记录数据路径
    records_path: str = "policy_records"
    # 输出路径
    output_path: str = "visualizations"
    # 是否保存图片
    save_plots: bool = True


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


def create_comprehensive_visualization(records: List[dict], output_path: str, save: bool = True):
    """创建综合可视化"""
    
    # 提取数据
    positions = np.array([record['inputs/observation/state'][:3] for record in records])
    states = np.array([record['inputs/observation/state'] for record in records])
    actions = np.array([record['outputs/actions'][0] for record in records])
    
    # 创建大型图形
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 3D轨迹
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, alpha=0.7)
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='green', s=100, label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='red', s=100, label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    
    # 2. XY平面轨迹
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
    ax2.scatter(positions[0, 0], positions[0, 1], color='green', s=100, label='Start')
    ax2.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, label='End')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Trajectory')
    ax2.legend()
    ax2.axis('equal')
    ax2.grid(True)
    
    # 3. Z轴高度变化
    ax3 = fig.add_subplot(2, 4, 3)
    ax3.plot(range(len(positions)), positions[:, 2], 'b-', linewidth=2)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Z Position (m)')
    ax3.set_title('Height Over Time')
    ax3.grid(True)
    
    # 4. 夹爪状态
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.plot(range(len(states)), states[:, 6], 'g-', linewidth=2, label='Gripper Width')
    ax4.plot(range(len(states)), states[:, 7], 'r-', linewidth=2, label='Gripper State')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Value')
    ax4.set_title('Gripper States')
    ax4.legend()
    ax4.grid(True)
    
    # 5. 位置动作对比
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.plot(range(len(actions)), actions[:, 0], 'r-', linewidth=2, label='X action')
    ax5.plot(range(len(actions)), actions[:, 1], 'g-', linewidth=2, label='Y action')
    ax5.plot(range(len(actions)), actions[:, 2], 'b-', linewidth=2, label='Z action')
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Action Value')
    ax5.set_title('Position Actions')
    ax5.legend()
    ax5.grid(True)
    
    # 6. 旋转动作
    ax6 = fig.add_subplot(2, 4, 6)
    ax6.plot(range(len(actions)), actions[:, 3], 'r-', linewidth=2, label='Roll')
    ax6.plot(range(len(actions)), actions[:, 4], 'g-', linewidth=2, label='Pitch')
    ax6.plot(range(len(actions)), actions[:, 5], 'b-', linewidth=2, label='Yaw')
    ax6.set_xlabel('Time Step')
    ax6.set_ylabel('Action Value')
    ax6.set_title('Rotation Actions')
    ax6.legend()
    ax6.grid(True)
    
    # 7. 动作幅度分析
    ax7 = fig.add_subplot(2, 4, 7)
    position_magnitude = np.linalg.norm(actions[:, :3], axis=1)
    rotation_magnitude = np.linalg.norm(actions[:, 3:6], axis=1)
    ax7.plot(range(len(actions)), position_magnitude, 'b-', linewidth=2, label='Position')
    ax7.plot(range(len(actions)), rotation_magnitude, 'r-', linewidth=2, label='Rotation')
    ax7.set_xlabel('Time Step')
    ax7.set_ylabel('Magnitude')
    ax7.set_title('Action Magnitudes')
    ax7.legend()
    ax7.grid(True)
    
    # 8. 推理时间
    ax8 = fig.add_subplot(2, 4, 8)
    inference_times = [record['outputs/policy_timing/infer_ms'] for record in records]
    ax8.plot(range(len(inference_times)), inference_times, 'orange', linewidth=2)
    ax8.set_xlabel('Time Step')
    ax8.set_ylabel('Inference Time (ms)')
    ax8.set_title('Inference Time')
    ax8.grid(True)
    
    # 添加总标题
    task_prompt = records[0]['inputs/prompt']
    plt.suptitle(f'Libero Policy Visualization: {task_prompt}', fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    if save:
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{output_path}/comprehensive_analysis.png", dpi=300, bbox_inches='tight')
        print(f"Comprehensive analysis saved to {output_path}/comprehensive_analysis.png")
    
    plt.show()


def create_image_sequence_visualization(records: List[dict], output_path: str, save: bool = True):
    """创建图像序列可视化"""
    
    # 选择几个关键帧
    num_frames = min(8, len(records))
    frame_indices = np.linspace(0, len(records)-1, num_frames, dtype=int)
    
    fig, axes = plt.subplots(2, num_frames, figsize=(num_frames * 3, 6))
    
    for i, frame_idx in enumerate(frame_indices):
        record = records[frame_idx]
        
        # 主相机
        axes[0, i].imshow(record['inputs/observation/image'])
        axes[0, i].set_title(f'Main Camera\nStep {frame_idx}')
        axes[0, i].axis('off')
        
        # 手腕相机
        axes[1, i].imshow(record['inputs/observation/wrist_image'])
        axes[1, i].set_title(f'Wrist Camera\nStep {frame_idx}')
        axes[1, i].axis('off')
    
    plt.suptitle('Key Frames from Policy Execution', fontsize=16)
    plt.tight_layout()
    
    if save:
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{output_path}/key_frames.png", dpi=200, bbox_inches='tight')
        print(f"Key frames saved to {output_path}/key_frames.png")
    
    plt.show()


def create_trajectory_evolution_gif(records: List[dict], output_path: str):
    """创建轨迹演化的gif动画"""
    
    positions = np.array([record['inputs/observation/state'][:3] for record in records])
    
    # 创建帧序列
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    frames_dir = pathlib.Path(output_path) / "trajectory_frames"
    frames_dir.mkdir(exist_ok=True)
    
    for i in range(0, len(positions), 5):  # 每5步一帧
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 当前轨迹
        current_traj = positions[:i+1]
        
        # XY轨迹
        ax1.plot(current_traj[:, 0], current_traj[:, 1], 'b-', linewidth=2, alpha=0.7)
        if len(current_traj) > 0:
            ax1.scatter(current_traj[-1, 0], current_traj[-1, 1], color='red', s=100)
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title(f'XY Trajectory (Step {i})')
        ax1.axis('equal')
        ax1.grid(True)
        
        # 设置固定的轴范围
        ax1.set_xlim(positions[:, 0].min() - 0.1, positions[:, 0].max() + 0.1)
        ax1.set_ylim(positions[:, 1].min() - 0.1, positions[:, 1].max() + 0.1)
        
        # Z轴高度
        ax2.plot(range(len(current_traj)), current_traj[:, 2], 'b-', linewidth=2)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Z Position (m)')
        ax2.set_title('Height Evolution')
        ax2.grid(True)
        ax2.set_xlim(0, len(positions))
        ax2.set_ylim(positions[:, 2].min() - 0.1, positions[:, 2].max() + 0.1)
        
        plt.tight_layout()
        plt.savefig(frames_dir / f"frame_{i:04d}.png", dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"Animation frames saved to {frames_dir}")
    print(f"Create GIF using: ffmpeg -r 10 -i {frames_dir}/frame_%04d.png -vf 'palettegen' {output_path}/trajectory_evolution.gif")


def print_analysis_summary(records: List[dict]):
    """打印分析摘要"""
    
    states = np.array([record['inputs/observation/state'] for record in records])
    actions = np.array([record['outputs/actions'][0] for record in records])
    positions = states[:, :3]
    
    print("\n" + "="*60)
    print("LIBERO POLICY ANALYSIS SUMMARY")
    print("="*60)
    
    # 基本信息
    print(f"Task: {records[0]['inputs/prompt']}")
    print(f"Total steps: {len(records)}")
    
    # 轨迹分析
    start_pos = positions[0]
    end_pos = positions[-1]
    total_distance = np.linalg.norm(end_pos - start_pos)
    path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    
    print(f"\nTrajectory Analysis:")
    print(f"  Start position: ({start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f})")
    print(f"  End position: ({end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f})")
    print(f"  Straight-line distance: {total_distance:.3f} m")
    print(f"  Actual path length: {path_length:.3f} m")
    print(f"  Path efficiency: {total_distance/path_length:.3f}")
    
    # 动作分析
    action_magnitude = np.linalg.norm(actions[:, :6], axis=1)
    print(f"\nAction Analysis:")
    print(f"  Mean action magnitude: {np.mean(action_magnitude):.4f}")
    print(f"  Max action magnitude: {np.max(action_magnitude):.4f}")
    print(f"  Action smoothness (std): {np.std(action_magnitude):.4f}")
    
    # 夹爪分析
    gripper_width = states[:, 6]
    gripper_changes = np.abs(np.diff(gripper_width))
    significant_changes = np.sum(gripper_changes > 0.005)
    
    print(f"\nGripper Analysis:")
    print(f"  Initial gripper width: {gripper_width[0]:.4f}")
    print(f"  Final gripper width: {gripper_width[-1]:.4f}")
    print(f"  Significant state changes: {significant_changes}")
    
    # 性能指标
    inference_times = [record['outputs/policy_timing/infer_ms'] for record in records]
    print(f"\nPerformance:")
    print(f"  Mean inference time: {np.mean(inference_times):.2f} ms")
    print(f"  Max inference time: {np.max(inference_times):.2f} ms")
    
    print("="*60)


def main(args: Args) -> None:
    """主函数"""
    print("Loading policy records...")
    records = load_policy_records(args.records_path)
    
    if not records:
        print("No records found!")
        return
    
    # 打印分析摘要
    print_analysis_summary(records)
    
    # 创建可视化
    print("\nCreating comprehensive visualization...")
    create_comprehensive_visualization(records, args.output_path, args.save_plots)
    
    print("\nCreating image sequence visualization...")
    create_image_sequence_visualization(records, args.output_path, args.save_plots)
    
    print("\nCreating trajectory evolution animation frames...")
    create_trajectory_evolution_gif(records, args.output_path)
    
    print(f"\nAll visualizations saved to {args.output_path}")


if __name__ == "__main__":
    tyro.cli(main)
