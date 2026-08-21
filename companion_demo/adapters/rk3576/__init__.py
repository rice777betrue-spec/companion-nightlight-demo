"""RK3576 设备适配器。

后续在此实现 ALSA AudioInputPort、RKNN/WebRTC VadPort、AEC、RKNN ASR、
Linux AudioPlaybackPort/TTS 和真实灯光驱动，核心运行时无需修改。
"""

from companion_demo.adapters.rk3576.llm import RkllmHttpAdapter

__all__ = ["RkllmHttpAdapter"]
