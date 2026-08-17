# VSS 单工位专注助手（DGX Spark）

本项目只分析一个固定椅子 ROI 内的匿名使用者。RT-CV 持续检测人物；只有最新帧中人物框底部中心位于椅子 ROI 时，Office API 才每 20 秒调用一次本地 `nvidia/Cosmos3-Nano` 16B Reasoner。离开椅子后不会调用 Cosmos3，也不做人脸识别或跨天身份关联。

本扩展在 NVIDIA VSS `dev-profile-alerts` 之上增加单路 USB 摄像头、匿名办公事件分类、事件面板和内网 HTTPS 入口。VSS 核心服务保持原样，便于定位官方组件问题。

## 1. Spark 前置条件

- DGX OS 7.4、NVIDIA 驱动 580.95.05 或兼容版本。
- Docker 28.3.3+ 且低于 29.5.0、Docker Compose 2.39.1+、NVIDIA Container Toolkit 1.17.8+。
- `git-lfs`、`curl`、NVIDIA NGC CLI（`ngc`）、`v4l2-ctl`；首次下载模型至少预留 40 GiB 可用空间。
- 一个可用的 NVIDIA NGC API key，以及允许下载所需模型的凭据。
- 一台支持 1920×1080 MJPEG 或 raw 输出的 USB 摄像头。

## 2. 下载 Cosmos3-Nano 16B

`.env` 中默认使用 NVIDIA 官方仓库并固定到已验证提交：

```dotenv
COSMOS3_MODEL_REPO=nvidia/Cosmos3-Nano
COSMOS3_MODEL_REVISION=411f42a8fdfb8c5b2583cb8786e0938f49796eaa
COSMOS3_MODEL_DIR=/home/shiyiming/models/Cosmos3-Nano
```

执行 `bash ./scripts/download-cosmos3-nano.sh` 可单独下载。模型约 32.6 GiB，断线后再次执行会自动续传；权重保存在宿主机，停止或重建容器不会重新下载。`install.sh` 默认自动执行这一步；已有完整权重时可设置 `COSMOS3_AUTO_DOWNLOAD=false`。

服务使用 DGX Spark 可用的多架构 NGC vLLM 26.07 镜像，并安装 NVIDIA Cosmos3 Reasoner 插件。它监听宿主机回环地址 `127.0.0.1:8018`，不会暴露给办公网。旧 VSS `vss-rtvi-vlm` 会停止，避免两套 VLM 同时占用统一内存；RT-CV、VST、Kafka、Elasticsearch 等 VSS 服务继续使用。

## 3. 配置

```bash
cp .env.example .env
cp config/office-config.example.yaml config/office-config.yaml
```

编辑 `.env`，设置摄像头设备、NGC key 和 Hugging Face token。Web 入口不要求用户名或密码，因此只能部署在受控办公内网，不能暴露到公网。

编辑 `config/office-config.yaml`，确认工作时间、节假日、人数上限和 ROI。ROI 坐标以画面左上角为 `(0,0)`、右下角为 `(1,1)`。示例 ROI 只是占位值，正式告警前必须现场标定。

RT-CV 工位检测默认每 2 秒读取 Elasticsearch 最新的 `mdx-frames-*` 帧。人物框底部中心进入椅子 ROI 后立即显示在座；离开椅子超过 `workstation.departure_seconds`（默认 60 秒）才记录一次离座。多摄像头环境必须填写 `camera.vss_sensor_id`，避免统计其他摄像头。

安装后打开 `/office`，首页按日期和人员展示连续活动时间线，可用关键词筛选。展开“人员与摄像头设置”，在“椅子 ROI 标定”中只框住椅面和正常坐姿区域。活动主类固定为电脑、阅读、书写、手机、交谈、吃东西、休息和无法判断；模型只生成画面证据支持的简短描述，不推测屏幕内容或业务目的。相同事件会持续延长，只有连续两次确认变化才会拆分成新的时间段。

## 4. 安装与验证

```bash
./scripts/preflight.sh
./scripts/install.sh
./scripts/smoke-test.sh
./scripts/status.sh
```

默认入口为 `https://<SPARK_IP>:8443/office`，VSS 聊天位于同一入口根路径。Caddy 使用本地 CA；将 `deploy/docker/developer-profiles/office-assistant/caddy-data/caddy/pki/authorities/local/root.crt` 导入受信任办公终端，或替换 Caddyfile 使用组织签发的证书。

办公面板在查看当天时自动刷新活动事件与当前人数；历史日期保持静态。活动日志接口为 `GET /office-api/api/activity/events?date=YYYY-MM-DD`，也可使用 `start`、`end`、`person_id` 和 `q` 参数。原有工位结构化接口 `GET /office-api/api/workstation/live` 和 `GET /office-api/api/workstation/reports` 保持兼容。VSS Agent 的 `office_activity_query` 工具可按日期、时间、人员或关键词读取同一份日志。

安装脚本会尝试通过 VSS Agent API 自动注册 `rtsp://127.0.0.1:8554/office-main`。如果 VSS 启动较慢导致注册失败，可在 VSS 的 Video Management 页面手动添加同一 URL。

## 5. 数据与隐私

- 不启用人脸识别，不保存人脸模板，不推断人员身份或敏感属性。
- RT-CV 只负责人物检测、跟踪和椅子 ROI 门控；只有椅子当前有人时才调用 Cosmos3-Nano 进行行为分类。
- Cosmos3 只返回受控行为主类及简短可见动作描述；不读取屏幕私密内容，也不推断身份或敏感属性。
- VIOS 循环缓冲默认限制为 128 MB；Office API 每分钟归档新事件片段到 `data/office-assistant/clips`。
- `data/office-assistant` 保存人工确认和事件片段；本地清理线程删除超过 7 天的片段。
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
