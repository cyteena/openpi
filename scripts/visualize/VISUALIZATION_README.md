# Libero 策略轨迹可视化工具

本文档介绍如何使用提供的工具来可视化和分析在 Libero 环境中记录的策略行为数据。

## 工具概览

我们提供了三种可视化工具：

1. **基础可视化脚本** (`visualize_libero_policy_fixed.py`) - 生成基本的轨迹、状态和动作图表
2. **详细分析脚本** (`libero_viz_simple.py`) - 生成综合分析报告和高质量图表
3. **交互式 Jupyter Notebook** (`libero_policy_visualization.ipynb`) - 提供交互式分析界面

## 数据要求

这些工具适用于使用 `serve_policy.py` 脚本的 `--record` 参数记录的数据。记录的数据应包含以下信息：

- `inputs/observation/image`: 主相机图像 (224x224x3)
- `inputs/observation/wrist_image`: 手腕相机图像 (224x224x3)
- `inputs/observation/state`: 机器人状态 (8维向量：x,y,z,roll,pitch,yaw,gripper_width,gripper_state)
- `inputs/prompt`: 任务描述
- `outputs/actions`: 策略输出的动作序列 (10x7 矩阵)
- `outputs/policy_timing/infer_ms`: 推理时间

## 使用方法

### 1. 基础可视化

```bash
python scripts/visualize_libero_policy_fixed.py \\
    --args.records-path policy_records \\
    --args.output-path visualizations \\
    --args.viz-type all
```

生成的文件：
- `trajectory_3d.png`: 3D机械臂轨迹
- `state_evolution.png`: 状态演化图表
- `actions.png`: 动作序列分析
- `main_camera_grid.png`: 主相机图像网格
- `wrist_camera_grid.png`: 手腕相机图像网格
- `animation_frames/`: 动画帧序列

### 2. 详细分析

```bash
python scripts/libero_viz_simple.py \\
    --args.records-path policy_records \\
    --args.output-path visualizations_detailed
```

生成的文件：
- `comprehensive_analysis.png`: 8合1综合分析图表
- `key_frames.png`: 关键帧图像序列
- `trajectory_frames/`: 轨迹演化动画帧

### 3. 交互式分析

在 Jupyter 环境中打开 `libero_policy_visualization.ipynb`：

```bash
jupyter notebook libero_policy_visualization.ipynb
```

该 notebook 提供：
- 交互式步骤查看器
- 实时数据统计
- 可定制的可视化选项
- 详细的性能分析

## 可视化内容说明

### 轨迹分析
- **3D轨迹**: 机械臂末端执行器在3D空间中的运动轨迹
- **XY平面投影**: 从上往下看的运动轨迹
- **Z轴变化**: 机械臂高度随时间的变化

### 状态分析
- **位置状态**: X, Y, Z 坐标的演化
- **姿态状态**: Roll, Pitch, Yaw 角度的变化
- **夹爪状态**: 夹爪开合度和状态的变化

### 动作分析
- **位置动作**: X, Y, Z 方向的控制信号
- **旋转动作**: Roll, Pitch, Yaw 的控制信号
- **夹爪动作**: 夹爪控制信号
- **动作幅度**: 动作向量的模长分析

### 性能指标
- **轨迹效率**: 实际路径长度 vs 直线距离
- **动作平滑度**: 动作序列的连续性分析
- **推理时间**: 策略计算耗时统计
- **夹爪使用**: 抓取操作的频率和模式

## 分析结果解读

### 轨迹效率 (Path Efficiency)
- **1.0**: 完美的直线轨迹
- **0.5-1.0**: 较为高效的轨迹
- **0.1-0.5**: 中等效率，有一些绕行
- **<0.1**: 低效率，存在大量不必要的运动

### 动作平滑度
- **低标准差**: 动作平滑，控制稳定
- **高标准差**: 动作不稳定，可能存在震荡

### 夹爪分析
- **显著状态变化数**: 抓取和释放操作的次数
- **最终夹爪宽度**: 任务结束时的夹爪状态

## 自定义 Checkpoint 测试

要测试你的自定义 checkpoint，使用以下命令：

```bash
# 1. 启动策略服务器并记录数据
python scripts/serve_policy.py \\
    --env libero \\
    --policy.config pi0_dfm_libero \\
    --policy.dir checkpoints/pi0_dfm_libero/default_beta_1_1.5_mask_action_len_30/29999 \\
    --record \\
    --port 8000

# 2. 运行 Libero 环境连接到服务器
cd examples/libero
python main.py --host localhost --port 8000

# 3. 生成可视化分析
python ../../scripts/libero_viz_simple.py \\
    --args.records-path ../../policy_records \\
    --args.output-path ../../my_checkpoint_analysis
```

## 高级功能

### 创建动画视频

使用 ffmpeg 将动画帧转换为视频：

```bash
# 创建轨迹演化动画
ffmpeg -r 10 -i visualizations_detailed/trajectory_frames/frame_%04d.png \\
    -c:v libx264 -pix_fmt yuv420p visualizations_detailed/trajectory_evolution.mp4

# 创建高质量GIF
ffmpeg -r 10 -i visualizations_detailed/trajectory_frames/frame_%04d.png \\
    -vf "palettegen" visualizations_detailed/palette.png

ffmpeg -r 10 -i visualizations_detailed/trajectory_frames/frame_%04d.png \\
    -i visualizations_detailed/palette.png -lavfi "paletteuse" \\
    visualizations_detailed/trajectory_evolution.gif
```

### 批量分析多个checkpoint

```bash
#!/bin/bash
# 批量分析脚本示例

CHECKPOINTS=(
    "checkpoints/pi0_dfm_libero/default_beta_1_1.5_mask_action_len_30/29999"
    "checkpoints/pi0_dfm_libero/other_experiment/best_model"
    # 添加更多checkpoint路径
)

for checkpoint in "${CHECKPOINTS[@]}"; do
    echo "Testing checkpoint: $checkpoint"
    
    # 清理之前的记录
    rm -rf policy_records/*
    
    # 运行策略记录 (需要另一个终端运行Libero环境)
    echo "Start serve_policy.py in another terminal, then press Enter"
    read
    
    # 生成分析
    checkpoint_name=$(basename "$checkpoint")
    python scripts/libero_viz_simple.py \\
        --args.records-path policy_records \\
        --args.output-path "analysis_results/$checkpoint_name"
done
```

## 故障排除

### 常见问题

1. **找不到记录文件**
   - 确保运行 `serve_policy.py` 时使用了 `--record` 参数
   - 检查 `policy_records` 目录是否存在且包含 `.npy` 文件

2. **图像显示异常**
   - 确保环境中安装了合适的 matplotlib 后端
   - 在无GUI环境中，将 `plt.show()` 改为只保存图片

3. **内存不足**
   - 减少分析的步数或使用采样
   - 关闭不必要的可视化选项

4. **Libero环境问题**
   - 确保已正确安装 Libero
   - 检查 CUDA/GPU 配置

## 结果解释指南

生成的分析结果可以帮助你评估策略的：

1. **运动效率**: 是否采用了合理的运动路径
2. **控制稳定性**: 动作是否平滑连续
3. **任务完成度**: 是否成功完成了预期的操作
4. **计算性能**: 推理时间是否在可接受范围内

通过对比不同checkpoint的分析结果，你可以：
- 选择表现最佳的模型
- 识别训练中的问题
- 优化策略参数
- 改进网络架构

## 扩展功能

你可以基于提供的工具进行扩展：

1. **添加新的可视化指标**
2. **集成更多的分析方法**
3. **支持其他机器人环境**
4. **实现实时监控界面**

如有问题或需要帮助，请参考代码注释或创建issue。
