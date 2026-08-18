from __future__ import annotations

from typing import Protocol

from companion_demo.core.contracts import (
    AudioFrame,
    ChatMessage,
    LightExecution,
    VadDecision,
)
from companion_demo.light import LightAdjustment


class SpeechRecognitionPort(Protocol):
    def load(self) -> None: ...

    def transcribe(self, audio_path: str) -> str: ...


class CompanionModelPort(Protocol):
    @property
    def device_label(self) -> str: ...

    def load(self) -> None: ...

    def reply(
        self,
        user_text: str,
        history: list[ChatMessage],
        user_name: str,
        preferences: str,
    ) -> str: ...


class SpeechSynthesisPort(Protocol):
    @property
    def engine_label(self) -> str: ...

    def synthesize(self, text: str) -> str: ...


class AudioInputPort(Protocol):
    """持续麦克风输入端口，PC 与 RK3576 分别提供适配器。"""

    @property
    def device_label(self) -> str: ...

    @property
    def dropped_frames(self) -> int: ...

    def start(self) -> None: ...

    def read_frame(self, timeout: float = 0.1) -> AudioFrame | None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class VadPort(Protocol):
    """语音活动检测端口，可替换为 RKNN/WebRTC VAD。"""

    def reset(self) -> None: ...

    def analyze(self, frame: AudioFrame) -> VadDecision: ...

    def set_sensitivity(self, value: float) -> None: ...


class AudioPlaybackPort(Protocol):
    """设备扬声器播放端口；play 阻塞到播完或被 stop 打断。"""

    @property
    def engine_label(self) -> str: ...

    @property
    def is_playing(self) -> bool: ...

    def play(self, audio_path: str) -> None: ...

    def stop(self) -> None: ...


class LightDriverPort(Protocol):
    def apply(self, command: LightAdjustment) -> LightExecution: ...
