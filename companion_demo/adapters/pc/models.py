from __future__ import annotations

from companion_demo.asr import SpeechRecognizer
from companion_demo.llm import LocalCompanion
from companion_demo.tts import SpeechSynthesizer


class FasterWhisperAdapter(SpeechRecognizer):
    """电脑端 Faster-Whisper 实现。"""


class LocalQwenAdapter(LocalCompanion):
    """电脑端 Hugging Face Qwen 实现。"""


class LocalSpeechSynthesizerAdapter(SpeechSynthesizer):
    """电脑端 SAPI/Edge TTS 实现。"""
