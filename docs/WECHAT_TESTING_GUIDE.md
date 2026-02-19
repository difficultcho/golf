# 微信小程序测试指南

## 📱 测试准备

### 1. 确认后端服务运行

```bash
# 检查服务器进程
ps aux | grep uvicorn

# 应该看到：
# uvicorn app.main:app --host 127.0.0.1 --port 8888
```

**服务器信息：**
- 地址：`http://127.0.0.1:8888`
- API 文档：`http://127.0.0.1:8888/docs`
- 健康检查：`http://127.0.0.1:8888/health`

### 2. 打开微信开发者工具

1. 启动微信开发者工具
2. 导入项目：
   - 项目路径：`/Users/mzhang/Workspaces/golf`
   - AppID：使用测试号或你的 AppID（在 `project.config.json` 中配置）
   - 项目名称：AI Golf Swing Analysis

### 3. 配置检查

已自动配置完成 ✅：
- API 地址：`utils/config.js` → `http://127.0.0.1:8888`
- 开发环境：`ENV = 'development'`

---

## 🧪 测试流程

### 测试场景 1：完整上传分析流程

**步骤：**

1. **进入首页** (`pages/index/index`)
   - 点击"开始录制"按钮
   - 或点击"上传视频"按钮

2. **录制视频** (`pages/record/record`)
   - 方式 A：使用相机录制新视频
     - 点击"开始录制"
     - 录制 3-10 秒挥杆动作
     - 点击"停止录制"
     - 确认使用该视频

   - 方式 B：从相册选择
     - 点击"从相册选择"
     - 选择已有视频（支持 .mp4, .mov, .avi）

3. **上传处理** (`pages/upload/upload`)
   - 自动开始上传
   - 观察上传进度条（0-100%）
   - 等待处理状态（"视频处理中..."）
   - 处理完成后显示"处理完成！"

4. **查看结果** (`pages/result/result`)
   - 播放处理后的视频
   - 查看元数据（点击"查看数据"）
   - 下载视频到相册（点击"保存到相册"）

**预期结果：**
- ✅ 视频成功上传（进度 100%）
- ✅ 服务器处理完成（status: "done"）
- ✅ 可以播放处理后的视频
- ✅ 元数据包含视频信息（fps, duration, resolution等）

---

### 测试场景 2：错误处理

**测试点：**

1. **无效视频格式**
   - 尝试上传非视频文件
   - 预期：显示"Invalid video format"错误

2. **视频过大**
   - 上传超过限制大小的视频（默认 100MB）
   - 预期：显示"File too large"错误

3. **网络中断**
   - 上传过程中关闭服务器
   - 预期：显示"网络错误"，可以重试

4. **无效视频ID**
   - 手动修改 URL 访问不存在的视频
   - 预期：显示"视频ID无效"并返回

---

## 🔍 调试技巧

### 1. 查看控制台日志

微信开发者工具 → 控制台（Console）标签页

**关键日志：**
```javascript
// 上传开始
=== uploadVideo called ===
Video path: wxfile://...
API URL: http://127.0.0.1:8888/api/video/upload

// 上传成功
=== Upload SUCCESS ===
Status code: 200
Video ID: xxx-xxx-xxx

// 状态轮询
=== pollStatus called ===
Current status: processing / done

// 查看结果
=== viewResult called ===
Navigating to: /pages/result/result?videoId=...
```

### 2. 网络请求监控

微信开发者工具 → Network 标签页

**检查请求：**
- ✅ `POST /api/video/upload` - 200 OK
- ✅ `GET /api/video/status/{id}` - 200 OK, {"status": "done"}
- ✅ `GET /api/video/result/{id}` - 200 OK (返回视频文件)
- ✅ `GET /api/video/data/{id}` - 200 OK (返回 JSON)

### 3. 常见问题排查

**问题 1：请求失败 (fail)**
```
原因：后端服务器未启动或地址错误
解决：
1. 检查服务器进程：ps aux | grep uvicorn
2. 确认配置文件：utils/config.js → API_BASE
3. 重启服务器：cd backend && source venv/bin/activate && uvicorn app.main:app --host 127.0.0.1 --port 8888
```

**问题 2：上传成功但一直"处理中"**
```
原因：Phase 2A 模块未正常工作
解决：
1. 查看后端日志：tail -f backend/server_8888.log
2. 检查处理状态：curl http://127.0.0.1:8888/api/video/status/{video_id}
3. 运行模块测试：cd backend && python test_phase2a_modules.py
```

**问题 3：视频无法播放**
```
原因：视频编码格式不兼容或文件损坏
解决：
1. 检查处理后的文件：ls -lh backend/data/processed_videos/
2. 尝试用 VLC 播放器打开该文件
3. 查看处理日志中的 FFmpeg 错误信息
```

**问题 4：权限错误**
```
原因：相册或相机权限未授予
解决：
1. 微信开发者工具 → 详情 → 本地设置 → 不校验合法域名...（勾选）
2. 模拟器 → 权限管理 → 授予相机和相册权限
```

---

## 📊 Phase 2A 功能验证

**当前状态：**
- ✅ 3D 姿态估计：Dummy 模式（生成模拟数据）
- ✅ MuJoCo 物理仿真：正常工作
- ✅ 生物力学分析：正常工作
- ⚠️ 真实 AI 模型：未下载（可选）

**验证方法：**

```bash
# 测试 Phase 2A 模块
cd backend
python test_phase2a_modules.py

# 预期输出：
✓ PASS     Pose Estimator
✓ PASS     Mujoco Simulator
✓ PASS     Physics Analyzer
✓ PASS     Complete Pipeline
Total: 4/4 tests passed
```

**查看分析结果：**

上传一个视频后，查看生成的元数据：

```bash
# 查看元数据文件
cat backend/data/metadata/{video_id}.json

# 应包含 Phase 2A 分析数据：
{
  "video_info": { ... },
  "pose_data": {
    "joints_3d": [...],
    "confidence": [...],
    "timestamps": [...]
  },
  "physics_data": {
    "club_head_speed_mph": 85.5,
    "x_factor": 45.2,
    "energy_efficiency": 0.78,
    "balance_score": 82.3
  }
}
```

---

## 🎯 测试检查清单

### 基础功能
- [ ] 首页加载正常
- [ ] 可以进入录制页面
- [ ] 相机权限授予成功
- [ ] 可以录制新视频
- [ ] 可以从相册选择视频
- [ ] 视频预览正常

### 上传功能
- [ ] 视频上传进度条显示
- [ ] 上传成功返回 video_id
- [ ] 状态轮询工作正常
- [ ] 处理完成后自动跳转

### 结果展示
- [ ] 处理后视频可以播放
- [ ] 元数据加载成功
- [ ] 可以查看分析数据
- [ ] 可以下载到相册
- [ ] 返回首页功能正常

### Phase 2A 集成
- [ ] 生成了姿态数据（pose_data）
- [ ] 生成了物理数据（physics_data）
- [ ] 杆头速度在合理范围（50-150 mph）
- [ ] 分析指标完整（X-Factor, 能效, 平衡等）

### 错误处理
- [ ] 无效格式提示正确
- [ ] 文件过大提示正确
- [ ] 网络错误可以重试
- [ ] 无效 ID 处理正确

---

## 🚀 下一步（可选）

### 1. 安装 Phase 2A 依赖

```bash
cd backend
pip install -r requirements_phase2.txt
```

安装后重启服务器，姿态估计将使用 MediaPipe 模型。

### 2. 集成到生产环境

1. 配置生产域名（`utils/config.js` → `production.API_BASE`）
2. 部署后端到云服务器
3. 配置 HTTPS 证书
4. 在微信公众平台配置服务器域名
5. 提交小程序审核

---

## 📞 遇到问题？

**快速诊断：**

```bash
# 1. 检查后端健康
curl http://127.0.0.1:8888/health

# 2. 查看 API 文档
open http://127.0.0.1:8888/docs

# 3. 查看服务器日志
tail -f backend/server_8888.log

# 4. 运行模块测试
cd backend && python test_phase2a_modules.py

# 5. 检查依赖
cd backend && python check_phase2a_status.py
```

**联系方式：**
- 查看 PRD.md 了解项目架构
- 查看 VISUALIZATION_GUIDE.md 了解可视化功能
- 运行测试脚本获取详细错误信息

---

## ✅ 测试完成标志

当你完成以下操作，说明测试成功：

1. ✅ 成功上传一个视频
2. ✅ 视频处理完成（status: "done"）
3. ✅ 可以播放处理后的视频
4. ✅ 元数据包含 Phase 2A 分析数据
5. ✅ 所有功能按钮都正常工作

**恭喜！你的高尔夫挥杆分析系统已经可以正常工作了！** 🎉⛳
