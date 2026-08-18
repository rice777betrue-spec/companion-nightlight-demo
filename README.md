# 陪伴小夜灯电脑 Demo

第一阶段打通以下链路：

```text
电脑麦克风 → VAD 自动分句 → Whisper 中文 ASR → 灯光指令解析 → 本地 Qwen 回复 → 本地中文 TTS → 电脑扬声器
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
语句只保留最新 1 条，对话历史只保留最近 12 条消息。运行时只保持一个采集工作器和
一个推理工作器，不会为每句话不断创建线程。

电脑首版使用 WebRTC VAD 与自适应能量门限双重过滤，开启后先安静约 1 秒完成环境噪声校准。扬声器回答期间
会提高打断阈值以减少自激；真正的远场设备仍需在 RK3576 音频前端加入 AEC 回声消除。
`AudioInputPort`、`VadPort` 和 `AudioPlaybackPort` 已与平台解耦，后续替换 ALSA、
RKNN VAD/AEC 和设备功放时，无需重写对话核心。

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

浏览器会打开 <http://127.0.0.1:7860>。打开“免按键连续对话”，安静约 1 秒后直接
说普通话即可。浏览器录音和“手动让小夜灯回应”仍保留为调试备用；使用手动模式时
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

## 首轮验收

- 能正确识别一段 5～10 秒的近场普通话。
- 能根据用户称呼和偏好生成简短中文回复。
- 普通聊天与情绪表达会继续对话，只有用户明确告别时才进入睡前收尾。
- 能播放中文语音并显示虚拟暖光。
- 能用滑杆和四个预设档位直观看到调光前后的明暗对比。
- 能听懂“暗一点”“调到 60%”“睡眠模式”“关灯”等语音指令。
- 连续完成三轮对话，后续回复能参考前文。

目前还没有加入声纹认证、硬件级远场拾音、AEC 回声消除和持久化记忆；这些在主链
验证后逐项加入。
