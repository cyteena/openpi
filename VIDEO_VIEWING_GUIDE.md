# VS Code 视频查看解决方案

## 🎬 问题说明
VS Code 默认不支持直接播放MP4视频文件，这是正常现象。以下是几种解决方案：

## 🔧 解决方案

### 1. 使用HTML预览页面 (推荐)
我已经为您创建了一个专门的HTML页面来查看所有视频：

**文件位置**: `visualizations/video_viewer.html`

**使用方法**:
1. 在VS Code中右键点击 `video_viewer.html`
2. 选择 "Open with Live Server" 或 "Preview in Browser"
3. 如果没有这些选项，可以直接双击文件在浏览器中打开

### 2. 安装VS Code扩展

推荐安装以下扩展来增强视频查看功能：

#### A. Media Preview (推荐)
```
扩展名: Media Preview
发布者: suchaaver
功能: 支持在VS Code中预览视频、音频和图片
安装: 在扩展市场搜索 "Media Preview"
```

#### B. Video Preview
```
扩展名: Video Preview  
发布者: Dima Kornilov
功能: 专门用于在VS Code中预览视频文件
安装: 在扩展市场搜索 "Video Preview"
```

#### C. Live Server (配合HTML使用)
```
扩展名: Live Server
发布者: Ritwick Dey  
功能: 启动本地服务器，用于预览HTML页面
安装: 在扩展市场搜索 "Live Server"
```

### 3. 使用系统默认程序

#### Windows:
```bash
start visualizations/policy_animation_medium.mp4
```

#### macOS:
```bash
open visualizations/policy_animation_medium.mp4
```

#### Linux:
```bash
xdg-open visualizations/policy_animation_medium.mp4
# 或者使用特定播放器
vlc visualizations/policy_animation_medium.mp4
mpv visualizations/policy_animation_medium.mp4
```

### 4. 在VS Code终端中查看视频信息

可以使用 ffprobe 查看视频的详细信息：

```bash
ffprobe -v quiet -print_format json -show_format -show_streams visualizations/policy_animation_medium.mp4
```

## 📱 HTML预览页面功能

创建的 `video_viewer.html` 包含以下功能：

✅ **多个视频质量选项**: 低、中、高、最佳质量  
✅ **播放控制**: 统一播放、暂停、重置  
✅ **视频同步**: 多个视频可以同步播放  
✅ **GIF备选方案**: 如果视频播放有问题  
✅ **详细信息显示**: 文件大小、帧率等  
✅ **响应式设计**: 适配不同屏幕尺寸  

## 🎯 推荐使用方法

1. **日常查看**: 使用HTML预览页面
2. **快速预览**: 安装 Media Preview 扩展
3. **专业分析**: 使用系统的专业视频播放器
4. **分享演示**: 直接分享HTML文件

## 🚨 常见问题

### Q: HTML页面显示"视频无法加载"
**A**: 确保视频文件与HTML文件在同一目录下，或检查文件路径是否正确

### Q: VS Code中视频显示为乱码
**A**: 这是正常现象，VS Code会尝试用文本编辑器打开二进制文件

### Q: 想要更好的视频查看体验
**A**: 推荐使用专业的视频播放器如VLC、MPV等

## 📋 文件清单

当前已生成的视频文件：
- `policy_animation_low.mp4` (1.9 MB) - 快速预览
- `policy_animation_medium.mp4` (2.4 MB) - 推荐日常使用  
- `policy_animation_high.mp4` (3.9 MB) - 高质量版本
- `policy_animation_best.mp4` (4.8 MB) - 最高质量
- `policy_animation_simple.gif` (6.2 MB) - GIF动画
- `video_viewer.html` - 专用预览页面

选择最适合您需求的方式来查看这些精彩的策略执行视频吧！
