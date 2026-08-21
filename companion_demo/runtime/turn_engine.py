from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from typing import Callable

from companion_demo.core.contracts import (
    LightExecution,
    SpeakerVerification,
    TurnRequest,
    TurnResult,
)
from companion_demo.dialogue import (
    choose_dialogue_guidance,
    classify_sleep_mode_confirmation,
    needs_sleep_mode_confirmation,
)
from companion_demo.light import (
    LightAdjustment,
    interpret_light_command,
    light_confirmation,
)
from companion_demo.ports.services import (
    CompanionModelPort,
    LightDriverPort,
    SpeakerVerificationPort,
    SpeechRecognitionPort,
    WakeWordGatePort,
)


@dataclass(frozen=True)
class _PendingSleepModeConfirmation:
    expires_at: float


class TurnEngine:
    """完成 ASR、灯控执行和文字回复，不依赖任何具体界面。"""

    def __init__(
        self,
        asr: SpeechRecognitionPort,
        companion: CompanionModelPort,
        light_driver: LightDriverPort,
        speaker_verifier: SpeakerVerificationPort | None = None,
        wake_word_gate: WakeWordGatePort | None = None,
        *,
        sleep_confirmation_timeout_seconds: float = 30.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if sleep_confirmation_timeout_seconds <= 0:
            raise ValueError("睡眠模式确认超时必须大于 0 秒")
        self.asr = asr
        self.companion = companion
        self.light_driver = light_driver
        self.speaker_verifier = speaker_verifier
        self.wake_word_gate = wake_word_gate
        self.sleep_confirmation_timeout_seconds = float(
            sleep_confirmation_timeout_seconds
        )
        self._clock = clock or time.monotonic
        self._confirmation_lock = threading.RLock()
        self._pending_sleep_confirmation: (
            _PendingSleepModeConfirmation | None
        ) = None

    @staticmethod
    def _unchanged_light(
        brightness: int | float,
        description: str,
    ) -> LightExecution:
        level = max(0, min(100, int(round(brightness))))
        return LightExecution(
            previous=level,
            requested=level,
            actual=level,
            applied=False,
            description=description,
        )

    def clear_pending_confirmation(self) -> None:
        """取消尚未得到明确答复的睡眠模式操作。"""

        with self._confirmation_lock:
            self._pending_sleep_confirmation = None

    def _begin_sleep_confirmation(self) -> None:
        with self._confirmation_lock:
            self._pending_sleep_confirmation = _PendingSleepModeConfirmation(
                expires_at=(
                    self._clock() + self.sleep_confirmation_timeout_seconds
                )
            )

    def _resolve_pending_confirmation(
        self,
        transcript: str,
        *,
        explicit_light_intent: bool,
    ) -> str | None:
        """消费一次待确认状态；无关的新话题会安全取消旧确认。"""

        with self._confirmation_lock:
            pending = self._pending_sleep_confirmation
            if pending is None:
                return None
            self._pending_sleep_confirmation = None

        if explicit_light_intent:
            return None
        answer = classify_sleep_mode_confirmation(transcript)
        if self._clock() >= pending.expires_at:
            return "expired" if answer is not None else None
        return answer

    @staticmethod
    def _sleep_mode_success_reply(light: LightAdjustment) -> str:
        if not light.matched:
            return light_confirmation(light)
        if light.previous == light.brightness:
            return (
                "好的，睡眠模式已开启，"
                f"灯光保持在 {light.brightness}%。"
            )
        return (
            "好的，已为你开启睡眠模式，"
            f"灯光调到 {light.brightness}%。"
        )

    @staticmethod
    def _append_history(
        history: list[dict[str, str]],
        transcript: str,
        reply: str,
        *,
        profile_trusted: bool,
    ) -> list[dict[str, str]]:
        if not profile_trusted:
            return history
        history.extend(
            [
                {"role": "user", "content": transcript},
                {"role": "assistant", "content": reply},
            ]
        )
        return history[-12:]

    def _fixed_response(
        self,
        *,
        transcript: str,
        reply: str,
        history: list[dict[str, str]],
        profile_trusted: bool,
        speaker: SpeakerVerification,
        execution: LightExecution,
        dialogue_mode: str,
        generation_status: str,
        asr_seconds: float,
        wake_word_status: str,
    ) -> TurnResult:
        safe_history = self._append_history(
            history,
            transcript,
            reply,
            profile_trusted=profile_trusted,
        )
        status = (
            f"文字已生成｜ASR {asr_seconds:.2f} 秒"
            f"｜{generation_status}｜{dialogue_mode}｜{speaker.status}"
            f"{f'｜{wake_word_status}' if wake_word_status else ''}"
            "｜正在生成语音…"
        )
        return TurnResult(
            transcript=transcript,
            reply=reply,
            history=safe_history,
            status=status,
            brightness=execution.actual,
            light_status=execution.description,
            dialogue_mode=dialogue_mode,
            asr_seconds=asr_seconds,
            generation_seconds=0.0,
            used_companion_model=False,
            light_execution=execution,
            speaker_verification=speaker,
            response_required=True,
            wake_word_status=wake_word_status,
        )

    def _verify_speaker(self, audio_path: str) -> SpeakerVerification:
        if self.speaker_verifier is None:
            return SpeakerVerification(
                identity="not_enrolled",
                enrolled=False,
                is_owner=None,
                score=None,
                threshold=0.0,
                sample_count=0,
                status="声纹未启用",
            )
        try:
            return self.speaker_verifier.verify(audio_path)
        except Exception as exc:
            return SpeakerVerification(
                identity="unverified",
                enrolled=True,
                is_owner=False,
                score=None,
                threshold=0.0,
                sample_count=0,
                status=f"声纹模块异常：{exc}",
            )

    @staticmethod
    def _resolve_light_execution(
        command: LightAdjustment,
        execution: LightExecution,
    ) -> LightAdjustment:
        if not command.matched:
            return command
        if (
            execution.applied
            and execution.error is None
            and execution.actual == command.brightness
        ):
            return replace(
                command,
                brightness=execution.actual,
                description=execution.description or command.description,
            )

        detail = execution.error or (
            f"目标为 {command.brightness}%，设备返回 {execution.actual}%"
        )
        return LightAdjustment(
            previous=command.previous,
            brightness=execution.actual,
            matched=False,
            description=(
                f"执行失败：{detail}；灯光当前保持在 {execution.actual}%"
            ),
            intent_detected=True,
            blocked_reason="driver_error",
            action=command.action,
        )

    def handle(self, request: TurnRequest) -> TurnResult:
        if not request.audio_path:
            raise ValueError("请先录一段话。")

        safe_history = list(request.history or [])
        asr_started = time.perf_counter()
        transcript = self.asr.transcribe(request.audio_path)
        asr_seconds = time.perf_counter() - asr_started
        wake_word_status = ""

        if request.require_wake_word and self.wake_word_gate is not None:
            decision = self.wake_word_gate.evaluate(transcript)
            wake_word_status = decision.status
            unchanged = self._unchanged_light(
                request.brightness,
                f"灯光保持在 {int(round(request.brightness))}%",
            )
            if decision.action == "ignore":
                return TurnResult(
                    transcript=transcript,
                    reply="",
                    history=safe_history,
                    status=f"ASR {asr_seconds:.2f} 秒｜{decision.status}",
                    brightness=unchanged.actual,
                    light_status=unchanged.description,
                    dialogue_mode="等待唤醒",
                    asr_seconds=asr_seconds,
                    generation_seconds=0.0,
                    used_companion_model=False,
                    light_execution=unchanged,
                    response_required=False,
                    wake_word_status=decision.status,
                )
            if decision.action == "acknowledge":
                return TurnResult(
                    transcript=transcript,
                    reply="我在，你说吧。",
                    history=safe_history,
                    status=(
                        f"文字已生成｜ASR {asr_seconds:.2f} 秒"
                        f"｜唤醒即时确认｜{decision.status}｜正在生成语音…"
                    ),
                    brightness=unchanged.actual,
                    light_status=unchanged.description,
                    dialogue_mode="唤醒确认",
                    asr_seconds=asr_seconds,
                    generation_seconds=0.0,
                    used_companion_model=False,
                    light_execution=unchanged,
                    response_required=True,
                    wake_word_status=decision.status,
                )
            transcript = decision.transcript

        speaker = self._verify_speaker(request.audio_path)
        profile_trusted = not speaker.enrolled or speaker.is_owner is True
        model_history = safe_history if profile_trusted else []
        model_user_name = request.user_name if profile_trusted else ""
        model_preferences = request.preferences if profile_trusted else ""

        command = interpret_light_command(transcript, request.brightness)
        pending_resolution = self._resolve_pending_confirmation(
            transcript,
            explicit_light_intent=command.intent_detected,
        )
        unchanged_description = (
            f"灯光保持在 {int(round(request.brightness))}%"
        )

        if pending_resolution == "negative":
            unchanged = self._unchanged_light(
                request.brightness,
                unchanged_description,
            )
            return self._fixed_response(
                transcript=transcript,
                reply=(
                    "好的，我不开启睡眠模式，"
                    f"灯光保持在 {unchanged.actual}%。"
                ),
                history=safe_history,
                profile_trusted=profile_trusted,
                speaker=speaker,
                execution=unchanged,
                dialogue_mode="睡眠模式确认",
                generation_status="已取消睡眠模式",
                asr_seconds=asr_seconds,
                wake_word_status=wake_word_status,
            )

        if pending_resolution == "expired":
            unchanged = self._unchanged_light(
                request.brightness,
                unchanged_description,
            )
            return self._fixed_response(
                transcript=transcript,
                reply=(
                    "刚才的确认已超时，我没有改变灯光。"
                    "需要的话，请直接说“开启睡眠模式”。"
                ),
                history=safe_history,
                profile_trusted=profile_trusted,
                speaker=speaker,
                execution=unchanged,
                dialogue_mode="睡眠模式确认",
                generation_status="睡眠模式确认超时",
                asr_seconds=asr_seconds,
                wake_word_status=wake_word_status,
            )

        if (
            pending_resolution is None
            and not command.intent_detected
            and needs_sleep_mode_confirmation(transcript)
        ):
            self._begin_sleep_confirmation()
            unchanged = self._unchanged_light(
                request.brightness,
                unchanged_description,
            )
            timeout = int(round(self.sleep_confirmation_timeout_seconds))
            return self._fixed_response(
                transcript=transcript,
                reply=(
                    "要帮你开启睡眠模式，把灯光调到 10% 吗？"
                    "请说“要”或“不用”。"
                ),
                history=safe_history,
                profile_trusted=profile_trusted,
                speaker=speaker,
                execution=unchanged,
                dialogue_mode="睡眠模式确认",
                generation_status=f"等待确认（{timeout} 秒内有效）",
                asr_seconds=asr_seconds,
                wake_word_status=wake_word_status,
            )

        confirmed_sleep_mode = pending_resolution == "affirmative"
        if confirmed_sleep_mode:
            command = interpret_light_command(
                "开启睡眠模式",
                request.brightness,
            )
        execution = self.light_driver.apply(command)
        light = self._resolve_light_execution(command, execution)
        guidance = choose_dialogue_guidance(transcript, light.intent_detected)
        confirmation = light_confirmation(light)

        if confirmed_sleep_mode:
            reply = self._sleep_mode_success_reply(light)
            generation_seconds = 0.0
            used_companion_model = False
            generation_status = "睡眠模式确认执行"
        elif light.intent_detected and guidance.mode == "灯光控制":
            reply = confirmation
            generation_seconds = 0.0
            used_companion_model = False
            generation_status = "灯控即时确认"
        else:
            # 普通聊天只把用户原话交给模型。旧实现把一大段策略追加在
            # 原话后面，小模型容易优先响应策略而忽略用户真正说的内容。
            model_input = transcript
            if light.intent_detected:
                if light.matched:
                    model_input = (
                        f"设备层已执行灯光动作：{light.description}。"
                        "不要重复确认灯光，只回应下方原话中的其他内容。\n"
                        f"用户原话：{transcript}"
                    )
                else:
                    model_input = (
                        f"设备层未能执行灯光动作：{light.description}。"
                        "不要声称灯已改变，只回应下方原话中的其他内容。\n"
                        f"用户原话：{transcript}"
                    )

            generation_started = time.perf_counter()
            companion_reply = self.companion.reply(
                model_input,
                model_history,
                model_user_name,
                model_preferences,
            )
            generation_seconds = time.perf_counter() - generation_started
            used_companion_model = True
            reply = (
                f"{confirmation}{companion_reply}"
                if confirmation
                else companion_reply
            )
            generation_status = f"Qwen {generation_seconds:.2f} 秒"

        safe_history = self._append_history(
            safe_history,
            transcript,
            reply,
            profile_trusted=profile_trusted,
        )
        status = (
            f"文字已生成｜ASR {asr_seconds:.2f} 秒"
            f"｜{generation_status}｜{guidance.mode}｜{speaker.status}"
            f"{f'｜{wake_word_status}' if wake_word_status else ''}"
            "｜正在生成语音…"
        )

        return TurnResult(
            transcript=transcript,
            reply=reply,
            history=safe_history,
            status=status,
            brightness=light.brightness,
            light_status=light.description,
            dialogue_mode=guidance.mode,
            asr_seconds=asr_seconds,
            generation_seconds=generation_seconds,
            used_companion_model=used_companion_model,
            light_execution=replace(execution, actual=light.brightness),
            speaker_verification=speaker,
            response_required=True,
            wake_word_status=wake_word_status,
        )
