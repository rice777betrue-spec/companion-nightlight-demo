# 陪伴小夜灯电脑 Demo

第一阶段打通以下链路：

```text
电脑麦克风 → VAD 自动分句 → Whisper 中文 ASR → 唤醒词/会话门控 → 灯光指令解析 → 本地 Qwen 回复 → 本地中文 TTS → 电脑扬声器
```

## 最小设备框架

当前 Gradio 只是一个界面适配器，语音交互主流程已经独立到 `TurnEngine`：

```text
companion_demo/
├─ core/          统一请求、结果和设备执行回执
├─ ports/         ASR、LLM、TTS、灯光驱动接口
├─ adapters/
│  ├─ pc/         当前电脑麦克风、VAD、模型、扬声器和虚拟灯
│  └─ rk3576/     开发板实现预留入口
├─ runtime/       TurnEngine、设备状态机和 HandsFreeRuntime
├─ bootstrap.py   电脑端依赖组合入口
└─ pipeline.py    兼容现有 Gradio 的门面
```

界面只提交 `TurnRequest` 并展示 `TurnResult`。灯控回复以驱动返回的
`LightExecution.actual` 为准，未来替换为 RK3576/MCU 驱动时不需要修改对话核心。

### 设备运行状态机

`DeviceRuntime` 管理以下设备状态：

```text
启动预热 → 待机 → 聆听 → 思考 → 播放 → 待机
                         ↘ 故障恢复 ↗
```

每轮交互都会获得递增的 `turn_id`。用户在思考或播放阶段重新说话时，状态机
进入新一轮，旧轮次稍后返回的模型结果会被丢弃，避免播放过期回答。状态转换
历史使用固定长度队列，不会随着运行时间持续占用内存。

### 免按键连续对话

页面打开“免按键连续对话”一次后，后台会持续读取电脑麦克风。轻量 VAD 自动判断
开始说话及连续约 700 ms 的句末静音，然后自动运行 Whisper、灯控/Qwen、TTS 和
扬声器播放。思考或播放期间检测到新的用户语音时，新 `turn_id` 会立即让旧回答失效；
正在播放的 WAV 也会被停止。

音频资源是有界的：输入队列最多约 2 秒、预录约 300 ms、单句最长 15 秒、待处理
语句只保留最新 1 条，对话历史只保留最近 8 条消息。运行时只保持一个采集工作器和
一个推理工作器，不会为每句话不断创建线程。

电脑首版使用 WebRTC VAD 与自适应能量门限双重过滤，开启后先安静约 1 秒完成环境噪声校准。扬声器回答期间
会提高打断阈值以减少自激；真正的远场设备仍需在 RK3576 音频前端加入 AEC 回声消除。
`AudioInputPort`、`VadPort` 和 `AudioPlaybackPort` 已与平台解耦，后续替换 ALSA、
RKNN VAD/AEC 和设备功放时，无需重写对话核心。

### 唤醒词与连续会话

免按键模式处于待机时，必须先说唤醒词（默认“小夜灯”）。只说“小夜灯”会快速
回答“我在，你说吧”；说“小夜灯，关灯”则会在同一轮直接执行命令。唤醒成功后
进入连续会话，后续聊天无需重复唤醒。每轮回答播放结束后才重新计算 30 秒空闲
时间，模型思考和语音播报所用的时间不会被算作空闲；超时后自动回到待机。

唤醒词可在页面改成任意 2～20 个汉字或字母，保存后立即同步给 Whisper 热词并写入
`.cache/device/wake_word.json`，重启后仍然有效且不会进入 Git。修改唤醒词会立即结束
当前会话，使用新词重新唤醒。空闲时长可通过 `.env` 的 `WAKE_SESSION_SECONDS` 调整。

### 睡前自然意图确认

手动模式和免按键模式中，说“我要睡了”“准备睡觉了”等自然表达只会触发询问，
不会直接改变灯光。系统会问是否开启睡眠模式；在默认 30 秒的确认窗口内明确回答“要”
才把灯光调到 10%，回答“不用”、转到其他话题或等待超时都会保持原亮度。直接说
“开启睡眠模式”仍会立即执行。确认状态位于核心运行层，清空对话上下文、关闭免按键
监听或修改唤醒词时也会一并取消，避免之后的普通“好”误触发灯光；窗口时长与
`WAKE_SESSION_SECONDS` 保持一致。

电脑 Demo 当前采用“Whisper 识别文字后再做唤醒门控”，所以环境语音仍可能产生一次
Whisper 计算，但不会继续调用 Qwen、执行灯控或播报。RK3576 量产版应实现相同的
`WakeWordGatePort`，在 Whisper 前替换为低功耗 KWS，并结合 AEC、VAD 和远场麦克风
完成真正的常驻唤醒。

### 声纹认主

页面的“声纹认主”区域需要主人分别录制 3 段 2～5 秒的不同句子。录入完成后，
手动模式和免按键模式都会在每轮对话中自动验证身份。主人可以使用称呼、偏好和
最近对话；访客仍可进行普通聊天和灯控，但模型不会收到主人的私密资料或历史。

声纹档案只保存在 `.cache/voiceprint/owner_voiceprint.npz`，不会进入 Git，也可从
页面直接删除。当前实现是不依赖额外模型的轻量电脑验证基线，用于打通注册、验证、
权限和持久化链路，不应作为支付、门锁等高安全认证。量产时实现相同的
`SpeakerVerificationPort`，替换为 ECAPA-TDNN/ResNet 声纹模型的 ONNX 或 RKNN
适配器，并结合活体检测、重放攻击检测和远场噪声测试。

## 当前电脑的一次性安装

在 PowerShell 中进入本目录，使用 Python 3.11 或 3.12 创建独立环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

复制配置并准备模型：

```powershell
Copy-Item .env.example .env
$env:HF_HOME = Join-Path $PWD ".cache\huggingface"
.\.venv\Scripts\python.exe prepare_models.py
```

## 启动

```powershell
.\start.ps1
```

浏览器会打开 <http://127.0.0.1:7860>。打开“免按键连续对话”，安静约 1 秒后先说
唤醒词，再自然连续聊天。浏览器录音和“手动让小夜灯回应”仍保留为调试备用；使用手动模式时
先关闭免按键模式，避免同一句话被录入两次。

若要像设备固件一样随服务自动开始监听，在 `.env` 中设置：

```text
HANDS_FREE_AUTO_START=1
```

默认使用 Windows 当前默认输入设备；需要固定麦克风时可在 `AUDIO_INPUT_DEVICE`
填写 sounddevice 的设备编号或名称。

服务启动后会在后台预热 Whisper 和 Qwen。等“模型状态”显示“模型已就绪”后开始对话，
可以避开首次加载的等待。Windows Demo 默认使用本机的 Microsoft Huihui 中文语音；
本地语音不可用时自动退回 Edge 在线 TTS，而且语音生成不会阻塞文字回答显示。

实验分支还支持 VoxCPM-0.5B 本地语音合成。将官方权重放到
`.cache/models/VoxCPM-0.5B`，并在 `.env` 中设置 `TTS_ENGINE=voxcpm` 即可启用。
模型在第一次回答时才加载；如果依赖、模型或显存异常，会自动回退到 SAPI。需要克隆
音色时，同时配置 `VOXCPM_PROMPT_WAV` 和该参考录音逐字对应的
`VOXCPM_PROMPT_TEXT`。VoxCPM 只替换 TTS，Whisper 和 Qwen 保持不变。

陪伴模型默认使用 Qwen2.5-3B-Instruct，并通过 bitsandbytes NF4 4 位量化在电脑显卡上
运行。相比原先的 1.5B，3B 对否定、因果、边界表达和具体处境的理解更稳定；量化后
模型约占 2GB 显存。可通过 `.env` 的 `LLM_MODEL` 和 `LLM_QUANTIZATION` 切换模型，
但 RK3576 设备版仍需将权重转换为对应的 RKNN/NPU 格式，不能直接照搬电脑运行时。

语音识别默认使用 Faster-Whisper Small，并在支持 CUDA 的电脑上用显卡推理。它比
原先的 Base 更适合开放式中文聊天，可减少“忘/望”“动/懂”等同音字造成的语义偏移；
若目标设备没有 CUDA，可在 `.env` 将 `ASR_DEVICE` 改回 `cpu`，设备版则需换成
RKNN/ONNX 适配器。

## 首轮验收

- 能正确识别一段 5～10 秒的近场普通话。
- 能根据用户称呼和偏好生成简短中文回复。
- 普通聊天与情绪表达会继续对话，只有用户明确告别时才进入睡前收尾。
- 能播放中文语音并显示虚拟暖光。
- 能用滑杆和四个预设档位直观看到调光前后的明暗对比。
- 能听懂“暗一点”“调到 60%”“睡眠模式”“关灯”等语音指令。
- 说“我要睡了”时先询问是否开启睡眠模式；只有明确肯定才调到 10%。
- 待机时普通环境语音不回应；说唤醒词后可连续聊天，空闲超时后需要重新唤醒。
- 页面修改唤醒词后立即生效，重启服务仍保留新词。
- 连续完成三轮对话，后续回复能参考前文。
- 录入三段主人语音后，能显示主人/访客声纹结果；访客不会继承主人历史。

目前还没有加入量产级 KWS、神经网络声纹、活体/防重放、硬件级远场拾音、AEC 回声
消除和持久化记忆；这些在主链验证后逐项加入。
