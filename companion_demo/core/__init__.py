"""与界面、模型实现和硬件无关的核心数据结构。"""

from companion_demo.core.contracts import (
    AudioFrame,
    ChatMessage,
    LightExecution,
    TurnRequest,
    TurnResult,
    VadDecision,
)
from companion_demo.core.errors import NoSpeechDetectedError

__all__ = [
    "AudioFrame",
    "ChatMessage",
    "LightExecution",
    "NoSpeechDetectedError",
    "TurnRequest",
    "TurnResult",
    "VadDecision",
]
