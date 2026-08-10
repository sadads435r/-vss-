# VSS 智能办公助手（DGX Spark）

本扩展在 NVIDIA VSS `dev-profile-alerts` 之上增加单路 USB 摄像头、匿名办公事件分类、事件面板和内网 HTTPS 入口。VSS 核心服务保持原样，便于定位官方组件问题。

## 1. Spark 前置条件

- DGX OS 7.4、NVIDIA 驱动 580.95.05 或兼容版本。
- Docker 28.3.3+ 且低于 29.5.0、Docker Compose 2.39.1+、NVIDIA Container Toolkit 1.17.8+。
- `git-lfs`、`curl`、NVIDIA NGC CLI（`ngc`）、`v4l2-ctl`、至少 30 GB 可用空间。
- 一个可用的 NVIDIA NGC API key，以及允许下载所需模型的凭据。
- 一台支持 1920×1080 MJPEG 或 raw 输出的 USB 摄像头。

## 2. 配置

```bash
cp .env.example .env
cp config/office-config.example.yaml config/office-config.yaml
```

编辑 `.env`，设置摄像头设备、NGC key 和 Hugging Face token。Web 入口不要求用户名或密码，因此只能部署在受控办公内网，不能暴露到公网。

编辑 `config/office-config.yaml`，确认工作时间、节假日、人数上限和 ROI。ROI 坐标以画面左上角为 `(0,0)`、右下角为 `(1,1)`。示例 ROI 只是占位值，正式告警前必须现场标定。

人数统计默认每 2 秒读取 Elasticsearch 最新的 `mdx-frames-*` 帧。人员连续 10 秒未被同一摄像头看到后记为离开；这两个值可以通过 `occupancy.poll_seconds` 和 `occupancy.departure_timeout_seconds` 调整。多摄像头环境必须填写 `camera.vss_sensor_id`，避免统计其他摄像头。

## 3. 安装与验证

```bash
./scripts/preflight.sh
./scripts/install.sh
./scripts/smoke-test.sh
./scripts/status.sh
```

默认入口为 `https://<SPARK_IP>:8443/office`，VSS 聊天位于同一入口根路径。Caddy 使用本地 CA；将 `deploy/docker/developer-profiles/office-assistant/caddy-data/caddy/pki/authorities/local/root.crt` 导入受信任办公终端，或替换 Caddyfile 使用组织签发的证书。

办公面板每 2 秒刷新当前可见人数，显示每个匿名 Track ID 的出现时间、最后看到时间、离开时间和停留时长。结构化接口为 `GET /office-api/api/occupancy/current`。

安装脚本会尝试通过 VSS Agent API 自动注册 `rtsp://127.0.0.1:8554/office-main`。如果 VSS 启动较慢导致注册失败，可在 VSS 的 Video Management 页面手动添加同一 URL。

## 4. 数据与隐私

- 不启用人脸识别，不保存人脸模板，不推断人员身份或敏感属性。
- DeepStream 产生人员候选事件，VLM 仅验证画面中是否确有真人。
- Office API 根据配置的时区、时间表、区域、人数和持续时间对已有事件分类。
- VIOS 循环缓冲默认限制为 128 MB；Office API 每分钟归档新事件片段到 `data/office-assistant/clips`。
- `data/office-assistant` 保存人工确认和事件片段；本地清理线程删除超过 7 天的片段。
- VSS Elasticsearch ILM 在部署时设置为 7 天。VIOS 的临时录像策略仍应在上线前通过 VSS 配置和磁盘检查确认。

## 5. 网络安全

Web 入口没有登录密码。只向受信任办公网开放 HTTPS 端口（默认 8443），SSH 仅向管理员网段开放；严禁将 8443 暴露到公网。端口 7777、8000、8090、8554、9080、9200、9901、30888 和模型服务端口不得暴露给不受信任网络。宿主机防火墙因环境差异不会由安装脚本自动修改。

## 6. 发布与回滚

```bash
git tag -a office-assistant-v0.1.0 -m 'DGX Spark office assistant v0.1.0'
git push origin codex/office-assistant
git push origin office-assistant-v0.1.0
```

Spark 只部署不可变标签。回滚时检出上一个标签并重新运行 `./scripts/install.sh`；配置和 `data/office-assistant` 不会被卸载脚本删除。
