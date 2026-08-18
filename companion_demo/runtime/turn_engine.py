from __future__ import annotations

import time
from dataclasses import replace

from companion_demo.core.contracts import (
    LightExecution,
    SpeakerVerification,
    TurnRequest,
    TurnResult,
)
from companion_demo.dialogue import choose_dialogue_guidance
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
)


class TurnEngine:
    """完成 ASR、灯控执行和文字回复，不依赖任何具体界面。"""

    def __init__(
        self,
        asr: SpeechRecognitionPort,
        companion: CompanionModelPort,
        light_driver: LightDriverPort,
        speaker_verifier: SpeakerVerificationPort | None = None,
    ) -> None:
        self.asr = asr
        self.companion = companion
        self.light_driver = light_driver
        self.speaker_verifier = speaker_verifier

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
        speaker = self._verify_speaker(request.audio_path)
        profile_trusted = not speaker.enrolled or speaker.is_owner is True
        model_history = safe_history if profile_trusted else []
        model_user_name = request.user_name if profile_trusted else ""
        model_preferences = request.preferences if profile_trusted else ""
        asr_started = time.perf_counter()
        transcript = self.asr.transcribe(request.audio_path)
        asr_seconds = time.perf_counter() - asr_started

        command = interpret_light_command(transcript, request.brightness)
        execution = self.light_driver.apply(command)
        light = self._resolve_light_execution(command, execution)
        guidance = choose_dialogue_guidance(transcript, light.intent_detected)
        confirmation = light_confirmation(light)

        if light.intent_detected and guidance.mode == "灯光控制":
            reply = confirmation
            generation_seconds = 0.0
            used_companion_model = False
            generation_status = "灯控即时确认"
        else:
            model_input = f"{transcript}\n\n{guidance.instruction}"
            if not profile_trusted:
                model_input += (
                    "\n\n[声纹没有确认对方是主人。把对方视为访客，"
                    "礼貌自然地回应，但不要提及、推断或泄露主人的姓名、"
                    "偏好和历史对话。]"
                )
            if light.intent_detected:
                if light.matched:
                    model_input += (
                        f"\n\n[设备层已经执行：{light.description}。"
                        "不要重复确认灯光，也不要改变或猜测执行结果；"
                        "只回应用户话里的情绪、经历或睡前表达。]"
                    )
                else:
                    model_input += (
                        f"\n\n[设备层没有执行灯控：{light.description}。"
                        "不要声称灯已经改变；只回应用户的其他内容。]"
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

        if profile_trusted:
            safe_history.extend(
                [
                    {"role": "user", "content": transcript},
                    {"role": "assistant", "content": reply},
                ]
            )
            safe_history = safe_history[-12:]
        status = (
            f"文字已生成｜ASR {asr_seconds:.2f} 秒"
            f"｜{generation_status}｜{guidance.mode}｜{speaker.status}"
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
        )
