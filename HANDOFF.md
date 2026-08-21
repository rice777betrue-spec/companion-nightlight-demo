# 陪伴小夜灯 Demo 交接文档

更新时间：2026-08-19  
项目目录：`C:\Users\DELL\Documents\ChatGPT\公司\companion-nightlight-demo`  
GitHub：<https://github.com/rice777betrue-spec/companion-nightlight-demo>

## 给新任务的开场说明

请先完整阅读本文件，再查看工作区和 `git diff`。当前工作区包含尚未提交的 3B 对话模型、Faster-Whisper Small、回答相关性和上下文清理等重要修改；这些修改是有效工作成果，不能重置或覆盖。除非用户明确要求，不要推送 GitHub、重新打包模型或删除缓存。

## 一、产品目标

本项目是“独居人士专属陪伴小夜灯”的电脑端验证框架，目标是验证以下链路：

```text
麦克风
  → VAD 自动分句
  → 本地 Whisper 语音识别
  → 唤醒词与连续对话窗口
  → 声纹身份与隐私门控
  → 灯光指令解析/执行
  → 本地 Qwen 陪伴回复
  → 本地中文 TTS
  → 扬声器播放
```

电脑 Demo 采用端口化结构，界面、核心对话流程和电脑硬件适配器已经分离，方便后续把音频、灯控、模型推理替换为 RK3576 设备实现。

## 二、当前模型和运行方式

### 语音识别

- 当前：Faster-Whisper Small。
- 旧版：Whisper Base。
- 本地目录：`.cache\models\faster-whisper-small`。
- 当前电脑使用 CUDA + FP16；无 CUDA 时需要调整配置。
- 使用中文、beam size 5、VAD、唤醒词/灯控热词提示。
- 默认离线运行，不调用 OpenAI 语音识别 API。

### 对话模型

- 当前：Qwen2.5-3B-Instruct。
- 旧版：Qwen2.5-1.5B-Instruct。
- 本地目录：`.cache\models\Qwen2.5-3B-Instruct`。
- 通过 Transformers + bitsandbytes NF4 4-bit 加载，当前电脑约占 2GB 显存。
- 生成参数：`max_new_tokens=96`、`do_sample=False`、`repetition_penalty=1.06`。
- 模型在服务启动时预热并常驻内存/显存；每轮只传入 Token，不会重新下载或重新加载。
- 这是本地推理，不是训练；聊天不会自动修改模型权重。

### 其他模块

- TTS：Windows SAPI 本地中文语音；不可用时程序可退回 Edge TTS。
- 声纹：本地轻量基线，默认录入 3 段，阈值 0.82，档案位于 `.cache\voiceprint`。
- 唤醒词：默认“小夜灯”，可在页面修改并持久化；唤醒后 30 秒内可连续聊天。
- 灯光：电脑端使用 `VirtualLightDriver`，只改变页面上的虚拟亮度；尚未连接实体 PWM/GPIO。

模型、声纹、输出音频、`.env` 和缓存都被 `.gitignore` 排除，不会上传 GitHub。

## 三、已经完成的功能

- Gradio 操作界面与灯光明暗对比展示。
- 手动录音调试模式。
- 麦克风免按键持续监听、VAD 自动开始/结束录音。
- 自定义唤醒词、待机、唤醒和 30 秒连续对话窗口。
- 声纹录入、验证和访客隐私隔离。
- 设备运行状态机、错误恢复和打断旧轮次。
- 0～100% 灯光亮度、相对调亮/调暗、关灯和预设模式。
- 灯控动作由确定性规则执行，不让大模型虚构“已经关灯”。
- 普通聊天、情绪陪伴和多轮上下文。
- “我要睡了/准备睡觉”先确认，明确同意后才开启 10% 睡眠模式。
- 清空对话上下文按钮。
- 文字先显示、TTS 随后生成，降低主观等待时间。

## 四、本轮回答相关性优化

用户反馈 Whisper 已经正确转写，但 Qwen 经常答非所问。定位后确认主要问题在旧 1.5B 模型能力、提示词设计、随机采样和历史上下文污染，不是文字编码。

已完成：

- Qwen 1.5B 升级为 Qwen2.5-3B-Instruct 4-bit。
- Whisper Base 升级为 Faster-Whisper Small CUDA。
- 系统提示词要求优先回应最后一句、否定、因果和具体问题。
- 普通分歧不再自动升级成争吵或断绝关系。
- Qwen 实际只读取清理后的最近 8 条有效历史。
- 普通聊天直接传入 ASR 原文，不再把冗长策略附加在用户原话后。
- 关闭随机采样，提高回答稳定性。
- 没听清时清空旧识别结果，避免上一轮文字残留。
- 页面增加“清空对话上下文”。

在当前电脑上曾测得：ASR 约 0.35～0.61 秒，Qwen 约 0.7～2.7 秒。实际时间会随录音长度和硬件变化。

## 五、本轮完成：睡前自然意图二次确认

复现文本：

```text
小夜灯，我今天想早点睡觉，我准备睡觉了。
```

当前行为：

- “我要睡了”“准备睡觉了”等临近入睡表达只询问是否开启睡眠模式，灯光保持不变。
- 默认 30 秒确认窗口与 `WAKE_SESSION_SECONDS` 一致；明确回答“要/好的/可以”等才通过确定性灯控把亮度调到 10%。
- 回答“不用/不要/算了”等、转到其他话题或等待超时都取消待确认动作，灯光保持不变。
- “我昨天说我要睡了”“我睡不着”“我准备睡觉了吗”等回顾、失眠和问句不会误触发确认。
- 直接说“开启睡眠模式”仍立即调到 10%；确认后的实际执行仍通过灯光驱动回执，失败时不会虚构成功。
- 手动模式和免按键模式共用核心确认状态；清空上下文、关闭免按键监听或修改唤醒词会取消待确认动作。

## 六、关键代码入口

| 文件 | 作用 |
| --- | --- |
| `app.py` | Gradio 页面、事件绑定和运行状态展示 |
| `companion_demo/config.py` | 模型路径、离线模式、声纹和唤醒参数 |
| `companion_demo/bootstrap.py` | 组合电脑端各模块 |
| `companion_demo/asr.py` | Faster-Whisper 加载、热词和转写 |
| `companion_demo/llm.py` | Qwen 加载、提示词、历史清理、Token 化和生成 |
| `companion_demo/light.py` | 确定性灯控意图解析和安全规则 |
| `companion_demo/dialogue.py` | 普通聊天、情绪、睡前收尾等模式判断 |
| `companion_demo/pipeline.py` | 预热、串行推理、TTS 和 UI 门面 |
| `companion_demo/runtime/turn_engine.py` | 单轮 ASR、声纹、唤醒、灯控和 Qwen 总流程 |
| `companion_demo/runtime/hands_free.py` | 持续监听、自动分句、播报和打断 |
| `companion_demo/runtime/wake_word.py` | 自定义唤醒词和连续对话窗口 |
| `companion_demo/runtime/device_runtime.py` | 设备状态机 |
| `companion_demo/adapters/pc/voiceprint.py` | 电脑端声纹基线 |
| `companion_demo/adapters/pc/light.py` | 电脑端虚拟灯驱动 |
| `tests/` | 71 项自动化测试 |

## 七、启动和验证

当前交接时，端口 7860 没有监听，Demo 处于停止状态。启动命令：

```powershell
Set-Location 'C:\Users\DELL\Documents\ChatGPT\公司\companion-nightlight-demo'
.\start.ps1
```

打开：<http://127.0.0.1:7860/>

等待页面“模型状态”显示 Qwen 和 Whisper 已就绪，再开启“免按键连续对话”。服务重启后免按键默认关闭；如果需要开机自动监听，在 `.env` 设置 `HANDS_FREE_AUTO_START=1`。

运行全部测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
git diff --check
```

2026-08-19 最后验证结果：71/71 项测试通过，`pip check`、`compileall` 和 `git diff --check` 通过；本地 Qwen 和 Whisper 模型文件存在且 Whisper 主模型大小校验正确。

建议人工验收：

1. “小夜灯，关灯”应变为 0%。
2. “小夜灯，把亮度调到 50%”应变为 50%。
3. “小夜灯，我今天有点累”应正常聊天，不应擅自关灯。
4. “小夜灯，开启睡眠模式”应变为 10%。
5. 连续对话窗口内第二句话不需要重复唤醒词。
6. 空闲 30 秒后需要重新使用唤醒词。

## 八、Git 和交付状态

- 当前分支最新提交：`ac4bdfc feat: add wake-word conversation sessions`。
- 工作区存在多项尚未提交修改，包括 3B、Faster-Whisper 和回答相关性优化。
- `tests/test_reply_relevance.py` 和 `tests/test_sleep_confirmation.py` 是新增但尚未跟踪的测试文件。
- GitHub 仓库仍没有完整包含这批最新本地修改。
- 之前制作的离线包仍是旧的 1.5B/Base 版本，不能视为当前最终版。
- 不要使用 `git reset --hard`、`git checkout --` 等命令清理当前工作区。
- 推送、提交或重新制作离线包前，应先让用户确认范围。

交接时可先执行：

```powershell
git status --short
git diff --stat
git diff
```

## 九、缓存和磁盘说明

项目中另有约 6.8GB Hugging Face 重复/未完成下载缓存。只读检查确认它与 `.cache\models` 下的完整运行模型分离，但尚未删除。清理属于大文件删除操作，应在用户确认后再执行；不能直接删除整个 `.cache`，否则会同时删除模型和声纹档案。

## 十、RK3576 落地时仍需替换的部分

电脑端模型不能原样复制到 RK3576 后直接运行。量产版本至少需要：

- 独立低功耗 KWS，在 Whisper 之前过滤环境声音。
- 麦克风阵列、AEC 回声消除、降噪和远场拾音。
- Whisper/ASR 的 RKNN、ONNX 或其他设备端适配器。
- Qwen 权重转换、NPU 支持和内存/首 Token 延迟评估。
- ECAPA 等更可靠的神经网络声纹、活体和防重放。
- 实体 PWM/GPIO 灯光驱动及实际执行结果回读。
- 看门狗、离线恢复、日志限额、升级和隐私数据加密。

核心层已经通过端口隔离，后续应优先新增 RK3576 适配器，不要把设备代码直接写进 Gradio 页面。

## 新任务可直接使用的提示词

```text
请先阅读 C:\Users\DELL\Documents\ChatGPT\公司\companion-nightlight-demo\HANDOFF.md，
再检查 git status 和现有代码，不要覆盖未提交修改。
继续完善陪伴小夜灯 Demo。睡前自然意图二次确认已经完成并有自动化测试；
下一步先使用真实麦克风人工验收“准备睡觉 → 要/不用/超时”在免按键模式中的
ASR 与 TTS 体验，再根据实测结果调整确认用语或 ASR 同音词容错。
```
