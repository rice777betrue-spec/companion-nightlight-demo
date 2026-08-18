from __future__ import annotations

from dataclasses import dataclass


ChatMessage = dict[str, str]


@dataclass(frozen=True)
class AudioFrame:
    """固定格式的麦克风帧；设备层只传 PCM，不依赖具体音频库。"""

    pcm_s16le: bytes
    sample_rate: int
    channels: int = 1
    captured_at: float = 0.0

    @property
    def duration_ms(self) -> float:
        bytes_per_sample = 2 * max(1, self.channels)
        sample_count = len(self.pcm_s16le) / bytes_per_sample
        return sample_count * 1000.0 / max(1, self.sample_rate)


@dataclass(frozen=True)
class VadDecision:
    """一帧音频的轻量语音活动检测结果。"""

    is_speech: bool
    level: float
    threshold: float
    calibrating: bool = False


@dataclass(frozen=True)
class SpeakerVerification:
    """一次声纹验证结果；分数只用于调试，权限判断以 is_owner 为准。"""

    identity: str
    enrolled: bool
    is_owner: bool | None
    score: float | None
    threshold: float
    sample_count: int
    status: str


@dataclass(frozen=True)
class VoiceprintEnrollment:
    """主人声纹录入进度。"""

    sample_count: int
    required_samples: int
    ready: bool
    status: str


@dataclass(frozen=True)
class WakeWordDecision:
    """一次唤醒门控判定。action 为 ignore、acknowledge 或 process。"""

    action: str
    transcript: str
    triggered: bool
    status: str


@dataclass(frozen=True)
class TurnRequest:
    """一次语音交互的统一输入，与 Gradio 或设备界面无关。"""

    audio_path: str
    history: list[ChatMessage] | None = None
    user_name: str = ""
    preferences: str = ""
    brightness: int | float = 35
    require_wake_word: bool = False


@dataclass(frozen=True)
class LightExecution:
    """灯光驱动的执行回执；actual 必须来自驱动实际状态。"""

    previous: int
    requested: int
    actual: int
    applied: bool
    description: str
    error: str | None = None


@dataclass(frozen=True)
class TurnResult:
    """核心运行时返回给任意界面的文字阶段结果。"""

    transcript: str
    reply: str
    history: list[ChatMessage]
    status: str
    brightness: int
    light_status: str
    dialogue_mode: str
    asr_seconds: float
    generation_seconds: float
    used_companion_model: bool
    light_execution: LightExecution
    speaker_verification: SpeakerVerification | None = None
    response_required: bool = True
    wake_word_status: str = ""
