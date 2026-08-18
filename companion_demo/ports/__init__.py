"""设备能力接口；电脑和 RK3576 适配器都实现这些接口。"""

from companion_demo.ports.services import (
    CompanionModelPort,
    LightDriverPort,
    SpeechRecognitionPort,
    SpeechSynthesisPort,
)

__all__ = [
    "CompanionModelPort",
    "LightDriverPort",
    "SpeechRecognitionPort",
    "SpeechSynthesisPort",
]
