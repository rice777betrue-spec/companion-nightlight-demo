"""当前电脑 Demo 使用的模型与虚拟设备适配器。"""

from companion_demo.adapters.pc.audio import (
    AdaptiveEnergyVad,
    SoundDeviceAudioInput,
    WebRtcEnergyVad,
    WindowsWavePlayer,
)
from companion_demo.adapters.pc.light import VirtualLightDriver
from companion_demo.adapters.pc.models import (
    FasterWhisperAdapter,
    LocalQwenAdapter,
    LocalSpeechSynthesizerAdapter,
)
from companion_demo.adapters.pc.voiceprint import LocalVoiceprintAdapter

__all__ = [
    "AdaptiveEnergyVad",
    "FasterWhisperAdapter",
    "LocalQwenAdapter",
    "LocalSpeechSynthesizerAdapter",
    "LocalVoiceprintAdapter",
    "SoundDeviceAudioInput",
    "VirtualLightDriver",
    "WebRtcEnergyVad",
    "WindowsWavePlayer",
]
