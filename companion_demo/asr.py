from __future__ import annotations

import threading
from pathlib import Path

from faster_whisper import WhisperModel

from companion_demo.core.errors import NoSpeechDetectedError
from companion_demo.text_normalization import to_simplified_chinese


_NIGHTLIGHT_INITIAL_PROMPT = (
    "陪伴小夜灯的普通话近场语音。常见词汇："
    "开灯、关灯、调亮、调暗、调高、调低、亮一点、暗一点、"
    "灯光、亮度、百分之、睡眠模式、阅读模式。"
)

_NIGHTLIGHT_HOTWORDS = (
    "灯光 亮度 开灯 关灯 调亮 调暗 调高 调低 "
    "亮一点 暗一点 柔和一点 百分之 睡眠模式 阅读模式"
)

_PROMPT_HALLUCINATION_MARKERS = (
    "请准确使用这些词",
    "以下是陪伴小夜灯",
    "常见词汇",
    "当前设备唤醒词",
)


class SpeechRecognizer:
    """延迟加载 Whisper，避免页面启动时长时间无响应。"""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        local_files_only: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.local_files_only = local_files_only
        self._model: WhisperModel | None = None
        self._load_lock = threading.Lock()
        self._prompt_lock = threading.RLock()
        self._wake_word = "小夜灯"

    def set_wake_word(self, phrase: str) -> None:
        """动态加入识别提示，无需重新加载或训练 Whisper。"""

        value = str(phrase or "").strip()
        if not value:
            return
        with self._prompt_lock:
            self._wake_word = value

    def load(self) -> None:
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return
            compute_type = "float16" if self.device == "cuda" else "int8"
            model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=compute_type,
                local_files_only=self.local_files_only,
            )
            self._model = model

    def transcribe(self, audio_path: str | Path) -> str:
        self.load()
        assert self._model is not None
        with self._prompt_lock:
            wake_word = self._wake_word
        initial_prompt = (
            f"{_NIGHTLIGHT_INITIAL_PROMPT} 当前设备唤醒词：{wake_word}。"
        )
        hotwords = f"{wake_word} {_NIGHTLIGHT_HOTWORDS}"

        segments, _ = self._model.transcribe(
            str(audio_path),
            language="zh",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
        )
        detected_segments = list(segments)
        text = to_simplified_chinese(
            "".join(segment.text for segment in detected_segments).strip()
        )
        prompt_leaked = any(
            marker in text for marker in _PROMPT_HALLUCINATION_MARKERS
        )
        likely_silence = bool(detected_segments) and all(
            getattr(segment, "no_speech_prob", 0.0) >= 0.7
            for segment in detected_segments
        )
        if not text or prompt_leaked or likely_silence:
            raise NoSpeechDetectedError(
                "没有识别到清晰语音，请靠近麦克风再试一次。"
            )
        return text
