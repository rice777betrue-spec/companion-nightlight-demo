class NoSpeechDetectedError(ValueError):
    """录音存在，但 ASR 没有得到可用文本；属于正常交互分支而非设备故障。"""

