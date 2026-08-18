# VSS 每日活动与连续动作助手（DGX Spark）

本项目分析单摄像头内 1–3 名人员的连续活动。RT-CV 的 NvDCF 负责人物框、track、ReID 与 BodyPose3DNet 34 个 2D/3D 关键点；独立 motion worker 在 8 秒窗口内计算关节角度、速度、画面内方向与相对 z 变化，并为 Cosmos3 生成同步的人物/环境故事板。MediaPipe 只对人物关键帧补充 21 点手部细节。Cosmos3 根据这些事实生成保守的自由中文描述，椅子 ROI 仅继续服务于旧的在座/日报统计。

本扩展在 NVIDIA VSS `dev-profile-alerts` 之上增加单路 USB 摄像头、自动人员库、连续动作窗口、活动时间线和内网 HTTPS 入口。VSS 核心服务保持原样，便于定位官方组件问题。

## 1. Spark 前置条件

- DGX OS 7.4、NVIDIA 驱动 580.95.05 或兼容版本。
- Docker 28.3.3+ 且低于 29.5.0、Docker Compose 2.39.1+、NVIDIA Container Toolkit 1.17.8+。
- `git-lfs`、`curl`、NVIDIA NGC CLI（`ngc`）、`v4l2-ctl`；首次下载模型至少预留 40 GiB 可用空间。
- 一个可用的 NVIDIA NGC API key，以及允许下载所需模型的凭据。
- 一台支持 1920×1080 MJPEG 或 raw 输出的 USB 摄像头。

## 2. 下载模型

`.env` 中默认使用 NVIDIA 官方仓库并固定到已验证提交：

```dotenv
COSMOS3_MODEL_REPO=nvidia/Cosmos3-Nano
COSMOS3_MODEL_REVISION=411f42a8fdfb8c5b2583cb8786e0938f49796eaa
COSMOS3_MODEL_DIR=/home/shiyiming/models/Cosmos3-Nano
```

执行 `bash ./scripts/download-cosmos3-nano.sh` 可单独下载。模型约 32.6 GiB，断线后再次执行会自动续传；权重保存在宿主机，停止或重建容器不会重新下载。`install.sh` 默认自动执行这一步；已有完整权重时可设置 `COSMOS3_AUTO_DOWNLOAD=false`。

执行 `bash ./scripts/download-motion-models.sh` 会下载固定版本的 BodyPose3DNet accuracy ONNX 与 MediaPipe Hand Landmarker，并校验官方对象摘要。它们保存在 `data/models`；`MOTION_MODELS_AUTO_DOWNLOAD=false` 可关闭自动下载。Office 镜像固定使用提供 Linux aarch64/Python 3.12 wheel 的 MediaPipe 0.10.18，以兼容 DGX Spark。

服务使用 DGX Spark 可用的多架构 NGC vLLM 26.07 镜像，并安装 NVIDIA Cosmos3 Reasoner 插件。它监听宿主机回环地址 `127.0.0.1:8018`，不会暴露给办公网。旧 VSS `vss-rtvi-vlm` 会停止，避免两套 VLM 同时占用统一内存；RT-CV、VST、Kafka、Elasticsearch 等 VSS 服务继续使用。

## 3. 配置

```bash
cp .env.example .env
cp config/office-config.example.yaml config/office-config.yaml
```

编辑 `.env`，设置摄像头设备、NGC key 和 Hugging Face token。Web 入口不要求用户名或密码，因此只能部署在受控办公内网，不能暴露到公网。

编辑 `config/office-config.yaml`，确认工作时间、节假日、人数上限和 ROI。ROI 坐标以画面左上角为 `(0,0)`、右下角为 `(1,1)`。示例 ROI 只是占位值，正式告警前必须现场标定。

RT-CV 对 10 FPS 输入每隔一帧执行一次 BodyPose，motion worker 直接消费 Kafka `mdx-raw`，避免人物框、关键点与故事板时间错位。每 2 秒落一份运动事实，通常每 10 秒或在明显姿态变化时请求一次 Cosmos3；同一人员最短 5 秒请求一次。单目 z 只解释为人物局部相对前后变化，未标定时移动方向只写“画面中向左/向右”。多摄像头环境必须填写 `camera.vss_sensor_id`。

安装后打开 `/office`，首页按日期和人员展示连续活动时间线，可用关键词筛选。展开“人员与摄像头设置”，在“椅子 ROI 标定”中只框住椅面和正常坐姿区域。活动主类固定为电脑、阅读、书写、手机、交谈、吃东西、休息和无法判断；模型只生成画面证据支持的简短描述，不推测屏幕内容或业务目的。相同事件会持续延长，只有连续两次确认变化才会拆分成新的时间段。

## 4. 安装与验证

```bash
./scripts/preflight.sh
./scripts/install.sh
./scripts/smoke-test.sh
./scripts/status.sh
```

默认入口为 `https://<SPARK_IP>:8443/office`，VSS 聊天位于同一入口根路径。Caddy 使用本地 CA；将 `deploy/docker/developer-profiles/office-assistant/caddy-data/caddy/pki/authorities/local/root.crt` 导入受信任办公终端，或替换 Caddyfile 使用组织签发的证书。

办公面板在查看当天时自动刷新；点击事件可查看动作事实、不确定性以及人物/环境故事板，点击人员照片可查看最多 5 张参考图。列表接口为 `GET /office-api/api/activity/events`，详情接口为 `GET /office-api/api/activity/events/{id}`，人员图库为 `GET /office-api/api/people/{id}/images`。旧的工位、人数、ROI 与日报接口保持兼容。Agent 可先查询活动，再用 `event_id` 读取证据回答“为什么这样判断”。

安装脚本会尝试通过 VSS Agent API 自动注册 `rtsp://127.0.0.1:8554/office-main`。如果 VSS 启动较慢导致注册失败，可在 VSS 的 Video Management 页面手动添加同一 URL。

## 5. 数据与隐私

- 自动人员库使用 RT-CV ReID 召回候选，并由 Cosmos3 对参考截图做两次保守确认；它不是生物识别认证，不能用于权限或执法决定。
- 只保存最多 5 张达到尺寸、置信度、关键点可见率和清晰度阈值的参考图；不推断姓名、民族、健康、情绪等敏感属性。
- Cosmos3 不猜测屏幕内容、文件名、谈话主题或业务目的；分类仅用于统计，网页以保守自由描述为主。
- MediaMTX 只保留 120 秒滚动 fMP4；人物/环境故事板和运动窗口保留 7 天，活动事件与统计保留 365 天，人员参考图库随人员档案保留。
- 身份不确定时显示“待确认人员”，优先避免错误合并；关闭 `workstation.motion_pipeline.enabled` 可回退旧单图活动分析。
- VSS Elasticsearch ILM 在部署时设置为 7 天。VIOS 的临时录像策略仍应在上线前通过 VSS 配置和磁盘检查确认。

## 6. 网络安全

Web 入口没有登录密码。只向受信任办公网开放 HTTPS 端口（默认 8443），SSH 仅向管理员网段开放；严禁将 8443 暴露到公网。端口 7777、8000、8090、8554、9080、9200、9901、30888 和模型服务端口不得暴露给不受信任网络。宿主机防火墙因环境差异不会由安装脚本自动修改。

## 7. 发布与回滚

```bash
git tag -a office-assistant-v0.1.0 -m 'DGX Spark office assistant v0.1.0'
git push origin codex/office-assistant
git push origin office-assistant-v0.1.0
```

Spark 只部署不可变标签。回滚时检出上一个标签并重新运行 `./scripts/install.sh`；配置和 `data/office-assistant` 不会被卸载脚本删除。
