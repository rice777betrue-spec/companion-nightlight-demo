from __future__ import annotations

import threading
import time

from companion_demo.adapters.pc import (
    FasterWhisperAdapter,
    LocalQwenAdapter,
    LocalSpeechSynthesizerAdapter,
    VirtualLightDriver,
)
from companion_demo.config import settings
from companion_demo.core.contracts import (
    TurnRequest,
    TurnResult,
    VoiceprintEnrollment,
)
from companion_demo.ports.services import (
    CompanionModelPort,
    LightDriverPort,
    SpeakerVerificationPort,
    SpeechRecognitionPort,
    SpeechSynthesisPort,
    WakeWordGatePort,
)
from companion_demo.runtime import DeviceRuntime, TurnEngine


class DemoPipeline:
    """兼容现有 Gradio 的门面；核心流程位于 TurnEngine。"""

    def __init__(
        self,
        *,
        asr: SpeechRecognitionPort | None = None,
        companion: CompanionModelPort | None = None,
        tts: SpeechSynthesisPort | None = None,
        light_driver: LightDriverPort | None = None,
        speaker_verifier: SpeakerVerificationPort | None = None,
        wake_word_gate: WakeWordGatePort | None = None,
        device_runtime: DeviceRuntime | None = None,
    ) -> None:
        self.asr = asr or FasterWhisperAdapter(
            settings.asr_model,
            settings.asr_device,
            local_files_only=settings.model_offline,
        )
        self.companion = companion or LocalQwenAdapter(
            settings.llm_model,
            local_files_only=settings.model_offline,
        )
        self.tts = tts or LocalSpeechSynthesizerAdapter(
            settings.tts_voice,
            settings.output_dir,
            engine=settings.tts_engine,
            sapi_voice=settings.sapi_voice,
        )
        self.light_driver = light_driver or VirtualLightDriver()
        self.speaker_verifier = speaker_verifier
        self.wake_word_gate = wake_word_gate
        self.turn_engine = TurnEngine(
            self.asr,
            self.companion,
            self.light_driver,
            self.speaker_verifier,
            self.wake_word_gate,
        )
        self.device_runtime = device_runtime or DeviceRuntime()
        self._warmup_status = "模型尚未预热"
        self._warmup_thread: threading.Thread | None = None
        self._status_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._tts_lock = threading.Lock()
        self._sync_asr_wake_word()

    @property
    def warmup_status(self) -> str:
        with self._status_lock:
            return self._warmup_status

    @property
    def voiceprint_status_text(self) -> str:
        if self.speaker_verifier is None:
            return "声纹模块未配置"
        return self.speaker_verifier.status_text

    def enroll_voiceprint(
        self,
        audio_path: str,
        owner_name: str = "",
    ) -> VoiceprintEnrollment:
        if self.speaker_verifier is None:
            raise RuntimeError("声纹模块未配置")
        return self.speaker_verifier.enroll(audio_path, owner_name)

    def clear_voiceprint(self) -> str:
        if self.speaker_verifier is None:
            return "声纹模块未配置"
        return self.speaker_verifier.clear()

    @property
    def wake_word_phrase(self) -> str:
        if self.wake_word_gate is None:
            return "小夜灯"
        return self.wake_word_gate.phrase

    @property
    def wake_word_status_text(self) -> str:
        if self.wake_word_gate is None:
            return "唤醒词门控未配置"
        return self.wake_word_gate.status_text

    def _sync_asr_wake_word(self) -> None:
        setter = getattr(self.asr, "set_wake_word", None)
        if callable(setter):
            setter(self.wake_word_phrase)

    def set_wake_word(self, phrase: str) -> str:
        if self.wake_word_gate is None:
            raise RuntimeError("唤醒词门控未配置")
        status = self.wake_word_gate.set_phrase(phrase)
        self._sync_asr_wake_word()
        return status

    def refresh_wake_session(self) -> str:
        if self.wake_word_gate is None:
            return ""
        return self.wake_word_gate.refresh_session()

    def sleep_wake_session(self) -> str:
        if self.wake_word_gate is None:
            return ""
        return self.wake_word_gate.sleep()

    def _set_warmup_status(self, value: str) -> None:
        with self._status_lock:
            self._warmup_status = value

    def warmup(self) -> None:
        started = time.perf_counter()
        try:
            with self._inference_lock:
                self._set_warmup_status("正在预热 Whisper…")
                self.asr.load()
                self._set_warmup_status("Whisper 已就绪，正在预热 Qwen…")
                self.companion.load()
            elapsed = time.perf_counter() - started
            self._set_warmup_status(
                f"模型已就绪｜Whisper 本地｜Qwen {self.companion.device_label}"
                f"｜TTS {self.tts.engine_label}｜{self.voiceprint_status_text}"
                f"｜预热 {elapsed:.1f} 秒"
            )
            self.device_runtime.models_ready()
        except Exception as exc:
            self._set_warmup_status(f"模型预热失败：{exc}")
            self.device_runtime.fail(str(exc))

    def start_warmup(self) -> None:
        if self._warmup_thread and self._warmup_thread.is_alive():
            return
        self._warmup_thread = threading.Thread(
            target=self.warmup,
            name="model-warmup",
            daemon=True,
        )
        self._warmup_thread.start()

    def generate_reply(
        self,
        audio_path: str,
        history: list[dict[str, str]] | None,
        user_name: str,
        preferences: str,
        brightness: int | float = 35,
    ) -> tuple[str, str, list[dict[str, str]], str, int, str]:
        result = self.handle_turn(
            TurnRequest(
                audio_path=audio_path,
                history=history,
                user_name=user_name,
                preferences=preferences,
                brightness=brightness,
            )
        )
        return (
            result.transcript,
            result.reply,
            result.history,
            result.status,
            result.brightness,
            result.light_status,
        )

    def handle_turn(self, request: TurnRequest) -> TurnResult:
        """串行访问本地模型，避免手动模式与免按键模式争抢 GPU。"""

        with self._inference_lock:
            return self.turn_engine.handle(request)

    def synthesize_reply(self, reply: str) -> tuple[str | None, str]:
        started = time.perf_counter()
        try:
            with self._tts_lock:
                audio_reply = self.tts.synthesize(reply)
            elapsed = time.perf_counter() - started
            return (
                audio_reply,
                f"语音已生成｜{self.tts.engine_label} {elapsed:.2f} 秒",
            )
        except Exception as exc:
            return None, f"文字可用，语音合成暂不可用：{exc}"

    def run(
        self,
        audio_path: str,
        history: list[dict[str, str]] | None,
        user_name: str,
        preferences: str,
        brightness: int | float = 35,
    ) -> tuple[
        str,
        str,
        str | None,
        list[dict[str, str]],
        str,
        int,
        str,
    ]:
        (
            transcript,
            reply,
            safe_history,
            text_status,
            next_brightness,
            light_status,
        ) = self.generate_reply(
            audio_path,
            history,
            user_name,
            preferences,
            brightness,
        )
        audio_reply, tts_status = self.synthesize_reply(reply)
        status = f"{text_status.replace('｜正在生成语音…', '')}｜{tts_status}"
        return (
            transcript,
            reply,
            audio_reply,
            safe_history,
            status,
            next_brightness,
            light_status,
        )
