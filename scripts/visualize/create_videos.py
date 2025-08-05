#!/usr/bin/env python3
"""
将动画帧转换为高质量视频的脚本
"""

import pathlib
import subprocess
import sys
from typing import Optional

def create_video_from_frames(
    frames_dir: str,
    output_path: str,
    fps: int = 10,
    quality: str = "high",
    codec: str = "mpeg4"
) -> bool:
    """
    从帧图像创建视频
    
    Args:
        frames_dir: 帧图像目录
        output_path: 输出视频路径
        fps: 帧率
        quality: 质量 ("low", "medium", "high", "best")
        codec: 编码器 ("mpeg4", "libx264")
    """
    frames_path = pathlib.Path(frames_dir)
    if not frames_path.exists():
        print(f"Error: Frames directory {frames_dir} does not exist")
        return False
    
    # 质量设置
    quality_settings = {
        "low": {"bitrate": "1M", "scale": "640:-1"},
        "medium": {"bitrate": "2M", "scale": "800:-1"},
        "high": {"bitrate": "4M", "scale": "1200:-1"},
        "best": {"bitrate": "8M", "scale": "-1:-1"}
    }
    
    settings = quality_settings.get(quality, quality_settings["high"])
    
    # 构建ffmpeg命令
    cmd = [
        "ffmpeg", "-y",  # -y to overwrite output file
        "-r", str(fps),
        "-i", f"{frames_dir}/frame_%04d.png",
        "-c:v", codec,
        "-b:v", settings["bitrate"],
        "-vf", f"scale={settings['scale']}:flags=lanczos",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    
    try:
        print(f"Creating {quality} quality video...")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            output_size = pathlib.Path(output_path).stat().st_size / (1024 * 1024)
            print(f"✓ Successfully created video: {output_path} ({output_size:.1f} MB)")
            return True
        else:
            print(f"✗ Error creating video: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def create_gif_from_frames(
    frames_dir: str,
    output_path: str,
    fps: int = 8,
    width: int = 800,
    optimize: bool = True
) -> bool:
    """创建优化的GIF动画"""
    
    cmd = [
        "ffmpeg", "-y",
        "-r", str(fps),
        "-i", f"{frames_dir}/frame_%04d.png",
    ]
    
    # 添加滤镜来优化GIF
    filters = [f"scale={width}:-1:flags=lanczos"]
    
    if optimize:
        # 使用调色板优化来减小文件大小
        palette_cmd = cmd + ["-vf", f"{','.join(filters)},palettegen", f"{output_path}.palette.png"]
        
        try:
            # 第一步：生成调色板
            subprocess.run(palette_cmd, capture_output=True, check=True)
            
            # 第二步：使用调色板生成GIF
            gif_cmd = cmd + [
                "-i", f"{output_path}.palette.png",
                "-lavfi", f"{','.join(filters)}[x];[x][1:v]paletteuse",
                "-loop", "0",
                output_path
            ]
            
            result = subprocess.run(gif_cmd, capture_output=True, text=True)
            
            # 清理临时调色板文件
            pathlib.Path(f"{output_path}.palette.png").unlink(missing_ok=True)
            
        except subprocess.CalledProcessError as e:
            print(f"Error in palette generation: {e}")
            return False
            
    else:
        # 简单方式生成GIF
        cmd.extend([
            "-vf", ",".join(filters),
            "-loop", "0",
            output_path
        ])
        result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        output_size = pathlib.Path(output_path).stat().st_size / (1024 * 1024)
        print(f"✓ Successfully created GIF: {output_path} ({output_size:.1f} MB)")
        return True
    else:
        print(f"✗ Error creating GIF: {result.stderr}")
        return False


def main():
    """主函数"""
    frames_dir = "visualizations/animation_frames"
    output_dir = "visualizations"
    
    print("=== Converting Animation Frames to Videos ===")
    print(f"Input frames: {frames_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # 创建多种质量的MP4视频
    video_configs = [
        {"suffix": "_low", "fps": 5, "quality": "low"},
        {"suffix": "_medium", "fps": 8, "quality": "medium"},
        {"suffix": "_high", "fps": 10, "quality": "high"},
        {"suffix": "_best", "fps": 15, "quality": "best"}
    ]
    
    for config in video_configs:
        output_path = f"{output_dir}/policy_animation{config['suffix']}.mp4"
        create_video_from_frames(
            frames_dir, 
            output_path, 
            fps=config['fps'], 
            quality=config['quality']
        )
    
    print()
    
    # 创建多种尺寸的GIF
    gif_configs = [
        {"suffix": "_small", "fps": 6, "width": 400},
        {"suffix": "_medium", "fps": 8, "width": 600},
        {"suffix": "_large", "fps": 10, "width": 800}
    ]
    
    for config in gif_configs:
        output_path = f"{output_dir}/policy_animation{config['suffix']}.gif"
        create_gif_from_frames(
            frames_dir,
            output_path,
            fps=config['fps'],
            width=config['width'],
            optimize=True
        )
    
    print()
    print("=== Summary ===")
    
    # 显示所有生成的文件
    output_path = pathlib.Path(output_dir)
    video_files = list(output_path.glob("policy_animation*"))
    
    if video_files:
        print("Generated files:")
        for file in sorted(video_files):
            size = file.stat().st_size / (1024 * 1024)
            print(f"  {file.name}: {size:.1f} MB")
    else:
        print("No files were generated")


if __name__ == "__main__":
    main()
