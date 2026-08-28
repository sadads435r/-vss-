# 项目简历素材：办公室多模态行为分析与数据飞轮

更新时间：2026-08-25

## 一句话项目介绍

基于 NVIDIA VSS、DeepStream、GDINO、MediaPipe 与 Cosmos3 构建办公室多人行为分析系统，实现人物识别、动作候选挖掘、事件视频裁剪、人工标注、数据增强、SFT LoRA 微调及 Base/LoRA 对照评估的数据闭环。

## 中文简历表述

- 设计并落地办公室多模态行为分析数据飞轮，串联 GDINO 目标检测、MediaPipe 姿态特征、人物身份匹配、规则候选生成、事件片段裁剪、人工标注和 Cosmos3 VLM 二次判定。
- 基于 45 条人工确认喝水正样本、90 条困难负样本和 32 条轻度增强样本，构建 167 条无验证/测试增强泄漏的 SFT 数据集；将视频统一裁剪并预抽取为 4 帧 224×224 输入。
- 在 NVIDIA GB10 统一内存平台上完成 Cosmos3-Nano Reasoner BF16 LoRA 监督微调；采用 LoRA Rank 16，训练 43,646,976 个参数，50 个优化步、400 个微步，峰值显存 17.51 GiB。
- 建立无训练样本泄漏的 137 条扩展压力测试集（5 正、132 负），在完全相同输入上进行 Base/LoRA 成对评估：LoRA 将误报从 17 条降至 6 条，负样本误报率由 12.9% 降至 4.5%，相对下降 64.7%。
- LoRA 在扩展集上保持 5/5 喝水样本检出，Precision 从 22.7% 提升至 45.5%，F1 从 0.370 提升至 0.625，并将结构化 JSON 输出合规率从 0% 提升至 100%；逐条对比修正 11 个 Base 误报且未出现新增回退。
- 解决 Cosmos3 BF16 训练在 GB10/cuDNN 上的视觉编码器 Conv3D 执行引擎兼容问题，切换至 PyTorch 原生 CUDA convolution 路径，完成一阶显存探测后再启动正式训练。
- 设计训练与线上推理服务的显存调度流程：训练前评估统一内存，按依赖顺序暂停和恢复 Cosmos/Nemotron VLLM 服务，避免 KV Cache 竞争，并保持 Office API、GDINO、姿态采集链路持续运行。

## 精简版（适合一段项目经历）

构建基于 NVIDIA VSS、DeepStream、GDINO、MediaPipe 和 Cosmos3 的办公室行为分析与数据飞轮系统，完成候选动作挖掘、人物视频裁剪、人工标注、数据增强及 BF16 LoRA SFT。基于 167 条训练数据在 GB10 上微调 Cosmos3-Nano Reasoner（LoRA Rank 16，峰值显存 17.51 GiB），并建立 137 条无训练泄漏的 Base/LoRA 成对测试集。LoRA 将喝水识别误报由 17 条降至 6 条，负样本误报率下降 64.7%，Precision 由 22.7% 提升至 45.5%，F1 由 0.370 提升至 0.625，结构化输出合规率提升至 100%。

## English resume bullet

Built an office behavior-analysis data flywheel using NVIDIA VSS, DeepStream, GDINO, MediaPipe, and Cosmos3, covering candidate mining, identity-aware clip extraction, human labeling, augmentation, and BF16 LoRA SFT. Fine-tuned a Cosmos3-Nano Reasoner with 43.6M trainable LoRA parameters on NVIDIA GB10 and created a leakage-free 137-clip paired benchmark. Reduced false positives from 17 to 6 (64.7% relative reduction), improved precision from 22.7% to 45.5% and F1 from 0.370 to 0.625, while achieving 100% structured-JSON compliance.

## 面试可以展开的技术难点

1. DeepStream/NvDCF PoseEstimator 没有向 `Object.pose` 附着关节，如何通过 verbose 日志、类别绑定和引擎输出逐层排查。
2. 为什么使用 MediaPipe 姿态作为独立数据流，而不是继续依赖 NvDCF 内部姿态附着。
3. 如何从规则候选、目标框、姿态轨迹和视频片段构建“候选生成 → VLM 确认 → 人工标注 → 再训练”的闭环。
4. 小样本视频训练如何避免数据泄漏：按原始事件划分、增强只进入训练集、Base/LoRA 使用完全相同的盲测样本。
5. 为什么 14 条测试集会产生误导，以及扩大到 137 条后如何使用 Precision、Recall、F1 和负样本误报率解释模型收益。
6. GB10 统一内存环境中如何评估训练显存、处理 BF16 Conv3D 兼容性、协调多个 VLLM 服务的 KV Cache。

## 需要诚实说明的边界

- 当前扩展测试集只有 5 条未见正样本，因此 100% Recall 不能视为稳定的生产结论；后续至少需要补充 20–30 条训练后采集的独立喝水正样本。
- 132 条负样本中多数为“未分类”负样本，剩余 6 个 LoRA 误报应先人工复核标签，再作为下一轮 hard negative。
- 当前 LoRA 已完成离线训练和对照评估，尚不等同于已在生产 Cosmos 推理服务中灰度部署。

## 关键产物

- LoRA Adapter：`/home/shiyiming/cosmos3-lora/runs/drinking-bf16-lora-baseline-v0`
- 扩展测试集：`/home/shiyiming/cosmos3-lora/data/drinking-test-v1-136`
- Base/LoRA 报告：`/home/shiyiming/cosmos3-lora/runs/drinking-expanded-base-vs-lora-v1.json`
- Adapter SHA-256：`c84f7b5bd3c3b3889ba8cd3c76af6204346466249e14020309f6a800813fe587`
- 报告 SHA-256：`26d1bab2a79c5cf945173188db908e9197ce0a63055e33e090210a38fef7078a`

## 项目中遇到的问题与解决过程

### 1. NvDCF PoseEstimator 不输出 `Object.pose`

- 现象：BodyPose 引擎显示加载完成，但目标对象始终没有附着关节数据，导致依赖姿态的行为事件窗口无法生成。
- 排查：开启 `debugVerboseLevel: 2`，确认引擎输出为 34 关键点，并确认 VPI 裁剪和推理输出内存已经分配；但运行期没有“对目标执行姿态推理并附着关节”的日志。
- 根因线索：NvDCF 内部绑定 `target-class: 0`，而 RT-DETR 标签中的 person 为 class 3；修改 `operateOnClassIds` 仍无法纠正内部目标类别绑定。
- 处理：确认该问题位于 NvMultiObjectTracker 内部姿态附着层，配置层无法继续修复；将姿态链路拆出，使用 MediaPipe 构建独立 `mdx-office-pose` 数据流，避免阻塞整个项目。
- 状态：通过替代架构绕开；NvDCF 内部问题如需根治，仍需要 NVIDIA 源码级排查或官方支持。

### 2. BodyPose TensorRT 引擎构建失败

- 现象：BodyPose 模型可以读取，但 NvDCF 无法生成推理计划文件。
- 根因：模型目录 `/opt/storage/bodypose3dnet` 以只读方式挂载，NvDCF 尝试在该目录写 TensorRT plan。
- 处理：使用 `trtexec` 将引擎预构建到可写的 `/opt/engines/`，并验证引擎推理通过、输出为 34 个关键点。
- 结果：身体姿态模型和 VPI 推理基础设施正常加载。

### 3. GB10 统一内存竞争导致 GDINO/Cosmos 服务异常

- 现象：GDINO 出现 CUDA OOM，`mdx-raw` 停止增长；训练模型时可用显存不足。
- 根因：Cosmos3、Nemotron、DeepStream、Triton 和 GDINO 共用 GB10 的 121 GiB 统一内存，多个 VLLM 服务预留了较大的模型与 KV Cache 空间。
- 处理：训练或排障前先读取 `free` 与 `nvidia-smi`，只暂停占用最大的 Cosmos/Nemotron 推理服务，保持摄像头、姿态和数据采集链路运行；任务结束后按 Nemotron → Cosmos 的顺序恢复。
- 结果：GDINO 恢复生产，LoRA 训练峰值控制在 17.51 GiB。

### 4. VLLM 服务恢复顺序造成 KV Cache 初始化失败

- 现象：训练结束后同时启动 Cosmos 和 Nemotron，Nemotron 反复重启并提示没有可用 KV Cache 内存。
- 根因：Cosmos 先启动后抢占了约 46–52 GiB，导致 Nemotron 计算出的可用 KV Cache 为负数。
- 处理：停止两者，先单独启动 Nemotron并等待健康检查通过，再启动 Cosmos。
- 结果：Nemotron稳定占用约24.6 GiB，Cosmos随后正常启动，两套服务重新共存。

### 5. `mdx-raw` 本身不包含人体姿态数据

- 现象：Behavior Analytics 只能接收到检测框、跟踪和基础对象元数据，无法直接进行喝水等细粒度动作规则识别。
- 根因：原有 `mdx-raw` 流没有 MediaPipe 关键点、手腕到嘴部距离、动作速度等特征。
- 处理：新建 `mdx-office-pose`，为每个人输出姿态关键点和派生特征，由飞轮 worker 订阅并产生喝水候选。
- 结果：形成“姿态规则召回候选 + 目标物证据 + VLM 强化判断”的分层识别链路。

### 6. 喝水规则在召回率与误报率之间反复失衡

- 现象：规则严格时真实喝水无法进入候选；放宽后摸脸、挠脸、托腮等动作大量进入候选。
- 根因：单纯依靠手腕靠近嘴部无法区分杯子、手部和手机，同时摄像机视角和人物框裁剪会造成关键点抖动。
- 处理：组合手腕—嘴部距离、持续时间、轨迹变化、GDINO 杯子证据和 VLM 判断；将挠脸、摸脸、手机靠近脸等样本作为 hard negative。
- 结果：建立可持续迭代的数据飞轮，但仍需继续补充困难负样本和杯子证据。

### 7. Tracker ID 频繁切换导致身份和事件碎片化

- 现象：同一个人在连续时间段被分配多个 tracker ID，产生多个“人员 N”，连续使用电脑或离开/返回事件无法自然合并。
- 原因：遮挡、检测框置信度波动、ROI 边界、人物短暂离开画面以及 NvDCF 跟踪生命周期结束。
- 处理：将 tracker ID 视为短期轨迹标识，再通过人物库/ReID 完成长期身份归并；提供人物手动合并和错误人物删除能力。
- 状态：身份层已经与 tracker 层解耦，但跨遮挡和跨时间的自动合并仍需持续优化。

### 8. 连续活动被切成大量短事件

- 现象：看电脑、键盘输入、身体前倾、鼠标操作等被分别标为“电脑操作”和“其他动作”，时间线上出现多个相邻短片段。
- 根因：VLM 对动作描述过细，事件聚合仅按原始标签和 tracker ID 合并，未考虑语义同类和短间隔。
- 处理：建立“使用电脑”上位类别，将看屏幕、键鼠操作、坐姿微调统一归类；增加允许短暂空洞和 tracker 切换的时间合并逻辑，并提供细分时间线展开按钮。
- 结果：主页显示人物级事件段，用户需要时再查看原始细分片段。

### 9. 离开工位与返回工位事件缺失

- 现象：人物离开座位一段时间后，时间线只显示离开前后的活动，没有明确的“离开工位”和“返回工位”证据。
- 根因：系统只记录有检测框或动作窗口的片段，将“持续缺席”当成数据空洞；tracker 变化也会破坏前后状态关联。
- 处理：引入座位 ROI 与人物状态机，将在岗 → 缺席持续超过阈值 → 再次出现分别生成离开和返回事件，并关联离开前后的身份而非 tracker ID。
- 状态：已补充状态逻辑，仍需用更多实际离开/返回样本验证边界时间。

### 10. 人物库匹配、删除与手动合并不稳定

- 现象：同一人出现多个临时人物，错误人物无法删除，手动合并功能曾失效。
- 根因：人物库身份、实时 tracker 和事件数据库之间存在多层 ID；合并操作未完全做到幂等，部分事件仍引用旧人物 ID。
- 处理：以人物库 ID 作为长期主键，tracker 只作为证据；补充删除、手动合并和历史事件引用迁移，并增加幂等处理。
- 结果：人员状态和全天事件线可以按人物库身份聚合，而不是按短期 tracker 展示。

### 11. Cosmos3 没有可直接使用的官方 Reasoner LoRA 训练路径

- 现象：官方 Cosmos Framework 的 LoRA 配置主要面向 VFM，无法直接完成当前 Reasoner 视频分类 SFT。
- 处理：将 Cosmos3 Reasoner 权重流式转换为 Transformers/Qwen3-VL 兼容结构，对 750 个 tensor 进行键名和 shape 全量校验，再使用 PEFT 实现 BF16 LoRA。
- 结果：生成独立的 8.81B Reasoner 基座训练目录，未覆盖或修改线上原始模型。

### 12. GB10 上 BF16 视频 Conv3D 找不到 cuDNN 执行引擎

- 现象：完整模型成功加载，但第一次视觉前向在 Qwen3-VL temporal patch embedding 的 `Conv3D` 报错：`GET was unable to find an engine to execute this computation`。
- 根因：GB10 当前 cuDNN 对该 BF16 Conv3D 形状无法选择可用执行引擎。
- 处理：关闭该训练进程的 cuDNN convolution 路径，切换到 PyTorch 原生 CUDA convolution；先跑 1 个优化步探测显存和数值稳定性，再执行正式训练。
- 结果：1 步探测与 50 步正式训练均成功，未发生 OOM。

### 13. 视频容器缺少 H.264 解码能力

- 现象：训练镜像中的安全版 FFmpeg 无法直接解码采集到的 H.264 事件视频。
- 处理：在数据构建阶段使用 Office 视频容器统一抽取 4 帧 224×224 RGB，并保存为压缩 NPZ；训练容器只读取预抽帧。
- 结果：训练不再依赖容器运行时视频编解码能力，数据加载更稳定、可复现。

### 14. 小样本测试集产生误导性结论

- 现象：最初 14 条测试集上 Base 与 LoRA 的语义 F1 都为 0.833，看起来 LoRA 只改善了 JSON 格式。
- 根因：测试集只有 5 正、9 负，且两个模型恰好误报相同的两条挠脸视频，样本量不足以观察差异。
- 处理：从飞轮中筛选完全未参与训练的样本，构建 137 条扩展压力测试集（5 正、132 负），进行相同输入、同一模型进程的 Base/LoRA 成对评估。
- 结果：Base 误报17条、LoRA误报6条；LoRA将负样本误报率由12.9%降至4.5%，修正11条误报且无新增回退。

### 15. 严格 JSON 解析掩盖原模型的真实语义能力

- 现象：初始评估显示 Base Recall/F1 为0。
- 根因：评估器只识别 `"confirmed": true`，而 Base 输出 `is喝水`、`is饮水`、`is_sipping_water` 等不同字段；判断语义正确但被计为错误。
- 处理：同时实现“严格结构化指标”和“语义指标”，并单独统计 JSON schema 合规率。
- 结果：避免把格式不一致误判为视觉识别失败，也明确证明 LoRA 将 schema 合规率从0%提升至100%。

### 16. 在线 SQLite WAL/SHM 权限被离线数据构建影响

- 现象：扩展测试集构建结束后，`office-flywheel-worker` 因 `attempt to write a readonly database` 反复重启。
- 根因：离线只读脚本以 `shiyiming` 用户连接在线 WAL 数据库，重新创建的 `office.db-shm`/`office.db-wal` 权限暂时阻止以 `chi` 用户运行的飞轮写入。
- 处理：恢复 WAL/SHM 写权限并确认 flywheel worker 重新订阅 `mdx-office-pose`、继续生成候选。
- 后续改进：离线训练和评估数据构建必须使用 SQLite 快照，不再直接连接在线数据库的 WAL/SHM。

## 面试总结表达

这个项目的主要难点不只是训练一个动作分类模型，而是把检测、跟踪、身份、姿态、时序事件、视频裁剪、标注、训练和线上资源管理组成一个可靠闭环。过程中既处理了 DeepStream/NvDCF 的底层姿态附着问题，也处理了小样本数据泄漏、评估口径、统一内存调度、VLLM KV Cache 竞争和在线数据库权限等工程问题。最终通过扩大无泄漏测试集证明 LoRA 不仅规范了输出格式，还将负样本误报率相对降低了64.7%。
