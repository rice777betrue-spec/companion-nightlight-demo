from __future__ import annotations

import queue
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Callable
from uuid import uuid4

from companion_demo.core.contracts import (
    AudioFrame,
    ChatMessage,
    TurnRequest,
    TurnResult,
)
from companion_demo.core.errors import NoSpeechDetectedError
from companion_demo.ports.services import (
    AudioInputPort,
    AudioPlaybackPort,
    VadPort,
)
from companion_demo.runtime.device_runtime import (
    DeviceNotReady,
    DeviceRuntime,
    DeviceState,
)


@dataclass(frozen=True)
class HandsFreeConfig:
    frame_duration_ms: int = 20
    pre_roll_ms: int = 300
    speech_start_ms: int = 200
    speech_end_silence_ms: int = 800
    minimum_voiced_ms: int = 360
    maximum_utterance_ms: int = 15_000
    barge_in_guard_ms: int = 650
    barge_in_threshold_multiplier: float = 2.2


class SegmentEventKind(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    DISCARDED = "discarded"


@dataclass(frozen=True)
class SegmentEvent:
    kind: SegmentEventKind
    frames: tuple[AudioFrame, ...] = ()


class UtteranceSegmenter:
    """把逐帧 VAD 结果拼成一句话，并严格限制缓存长度。"""

    def __init__(
        self,
        *,
        pre_roll_frames: int,
        speech_start_frames: int,
        speech_end_frames: int,
        minimum_voiced_frames: int,
        maximum_frames: int,
    ) -> None:
        self.pre_roll_frames = max(1, int(pre_roll_frames))
        self.speech_start_frames = max(1, int(speech_start_frames))
        self.speech_end_frames = max(1, int(speech_end_frames))
        self.minimum_voiced_frames = max(1, int(minimum_voiced_frames))
        self.maximum_frames = max(
            self.speech_start_frames + self.speech_end_frames,
            int(maximum_frames),
        )
        self._pre_roll: deque[AudioFrame] = deque(
            maxlen=self.pre_roll_frames
        )
        self._frames: list[AudioFrame] = []
        self._active = False
        self._speech_run = 0
        self._silence_run = 0
        self._voiced_frames = 0

    @classmethod
    def from_config(cls, config: HandsFreeConfig) -> "UtteranceSegmenter":
        frame_ms = max(1, config.frame_duration_ms)

        def count(duration_ms: int) -> int:
            return max(1, int(round(duration_ms / frame_ms)))

        return cls(
            pre_roll_frames=count(config.pre_roll_ms),
            speech_start_frames=count(config.speech_start_ms),
            speech_end_frames=count(config.speech_end_silence_ms),
            minimum_voiced_frames=count(config.minimum_voiced_ms),
            maximum_frames=count(config.maximum_utterance_ms),
        )

    @property
    def active(self) -> bool:
        return self._active

    @property
    def buffered_frames(self) -> int:
        return len(self._frames) if self._active else len(self._pre_roll)

    def reset(self) -> None:
        self._pre_roll.clear()
        self._frames.clear()
        self._active = False
        self._speech_run = 0
        self._silence_run = 0
        self._voiced_frames = 0

    def feed(self, frame: AudioFrame, is_speech: bool) -> SegmentEvent | None:
        if not self._active:
            self._pre_roll.append(frame)
            self._speech_run = self._speech_run + 1 if is_speech else 0
            if self._speech_run < self.speech_start_frames:
                return None
            self._active = True
            self._frames = list(self._pre_roll)
            self._pre_roll.clear()
            self._voiced_frames = self._speech_run
            self._silence_run = 0
            return SegmentEvent(SegmentEventKind.STARTED)

        self._frames.append(frame)
        if is_speech:
            self._voiced_frames += 1
            self._silence_run = 0
        else:
            self._silence_run += 1

        reached_end = self._silence_run >= self.speech_end_frames
        reached_limit = len(self._frames) >= self.maximum_frames
        if not reached_end and not reached_limit:
            return None

        frames = tuple(self._frames[: self.maximum_frames])
        kind = (
            SegmentEventKind.COMPLETED
            if self._voiced_frames >= self.minimum_voiced_frames
            else SegmentEventKind.DISCARDED
        )
        self.reset()
        return SegmentEvent(kind, frames)


@dataclass(frozen=True)
class HandsFreeSnapshot:
    enabled: bool = False
    status: str = "免按键监听未开启"
    input_device: str = "尚未连接麦克风"
    result_version: int = 0
    transcript: str = ""
    reply: str = ""
    history: tuple[tuple[str, str], ...] = ()
    previous_brightness: int = 35
    brightness: int = 35
    light_status: str = "灯光保持在 35%"
    result_status: str = ""
    audio_level: float = 0.0
    vad_threshold: float = 0.0
    calibrating: bool = False
    dropped_frames: int = 0
    last_error: str | None = None

    @property
    def history_messages(self) -> list[ChatMessage]:
        return [
            {"role": role, "content": content}
            for role, content in self.history
        ]


@dataclass(frozen=True)
class _CapturedUtterance:
    turn_id: int
    audio_path: str


class HandsFreeRuntime:
    """与界面无关的免按键语音运行层。

    始终只有一个采集循环和一个推理循环。PCM 队列、预录缓存、单句缓存、
    待处理队列和对话历史均有固定上限。
    """

    def __init__(
        self,
        *,
        audio_input: AudioInputPort,
        vad: VadPort,
        player: AudioPlaybackPort,
        device_runtime: DeviceRuntime,
        handle_turn: Callable[[TurnRequest], TurnResult],
        synthesize_reply: Callable[[str], tuple[str | None, str]],
        output_dir: Path,
        refresh_wake_session: Callable[[], str] | None = None,
        sleep_wake_session: Callable[[], str] | None = None,
        config: HandsFreeConfig | None = None,
    ) -> None:
        self.audio_input = audio_input
        self.vad = vad
        self.player = player
        self.device_runtime = device_runtime
        self.handle_turn = handle_turn
        self.synthesize_reply = synthesize_reply
        self.output_dir = Path(output_dir)
        self.refresh_wake_session = refresh_wake_session
        self.sleep_wake_session = sleep_wake_session
        self.config = config or HandsFreeConfig()
        self._segmenter = UtteranceSegmenter.from_config(self.config)
        self._pending: queue.Queue[_CapturedUtterance] = queue.Queue(maxsize=1)
        self._enabled = threading.Event()
        self._shutdown = threading.Event()
        self._threads_started = False
        self._capture_thread: threading.Thread | None = None
        self._worker_thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._active_turn_id: int | None = None
        self._playback_started_at = 0.0
        self._user_name = ""
        self._preferences = ""
        self._history: list[ChatMessage] = []
        self._snapshot = HandsFreeSnapshot(
            brightness=device_runtime.snapshot.brightness,
            previous_brightness=device_runtime.snapshot.brightness,
            light_status=(
                f"灯光保持在 {device_runtime.snapshot.brightness}%"
            ),
        )

    @property
    def snapshot(self) -> HandsFreeSnapshot:
        with self._lock:
            return self._snapshot

    @property
    def status_text(self) -> str:
        snapshot = self.snapshot
        mode = "已开启" if snapshot.enabled else "已关闭"
        parts = [f"免按键 {mode}", snapshot.status]
        if snapshot.enabled:
            meter = (
                f"音量 {snapshot.audio_level:.0f} / "
                f"阈值 {snapshot.vad_threshold:.0f}"
            )
            if snapshot.calibrating:
                meter += "（校准中）"
            parts.extend([snapshot.input_device, meter])
            if snapshot.dropped_frames:
                parts.append(f"丢帧 {snapshot.dropped_frames}")
        if snapshot.last_error:
            parts.append(f"异常：{snapshot.last_error}")
        return "｜".join(part for part in parts if part)

    def configure(
        self,
        *,
        user_name: str | None = None,
        preferences: str | None = None,
        history: list[ChatMessage] | None = None,
    ) -> None:
        with self._lock:
            if user_name is not None:
                self._user_name = user_name.strip()
            if preferences is not None:
                self._preferences = preferences.strip()
            if history is not None:
                self._history = [dict(message) for message in history[-12:]]
                self._snapshot = replace(
                    self._snapshot,
                    history=self._encode_history(self._history),
                )

    def set_sensitivity(self, value: float) -> None:
        self.vad.set_sensitivity(value)

    def clear_history(self) -> str:
        """清除已污染的上下文，同时清空页面上的上一轮结果。"""

        with self._lock:
            self._history = []
            self._snapshot = replace(
                self._snapshot,
                result_version=self._snapshot.result_version + 1,
                transcript="",
                reply="",
                history=(),
                result_status="对话上下文已清空，可以开始新的话题。",
                last_error=None,
            )
        return "对话上下文已清空，可以开始新的话题。"

    @staticmethod
    def _encode_history(
        history: list[ChatMessage],
    ) -> tuple[tuple[str, str], ...]:
        return tuple(
            (str(message.get("role", "")), str(message.get("content", "")))
            for message in history[-12:]
        )

    def _ensure_threads(self) -> None:
        with self._lifecycle_lock:
            if self._threads_started:
                return
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                name="hands-free-capture",
                daemon=True,
            )
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="hands-free-inference",
                daemon=True,
            )
            self._capture_thread.start()
            self._worker_thread.start()
            self._threads_started = True

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._enabled.is_set():
                return
            self._ensure_threads()
            self.vad.reset()
            self._segmenter.reset()
            try:
                self.audio_input.start()
            except Exception as exc:
                with self._lock:
                    self._snapshot = replace(
                        self._snapshot,
                        enabled=False,
                        status="麦克风启动失败",
                        last_error=str(exc),
                    )
                raise
            self._enabled.set()
            with self._lock:
                self._snapshot = replace(
                    self._snapshot,
                    enabled=True,
                    status="正在校准环境噪声，请安静约 1 秒",
                    input_device=self.audio_input.device_label,
                    last_error=None,
                )

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._enabled.clear()
            self.player.stop()
            try:
                self.audio_input.stop()
            finally:
                self._segmenter.reset()
                self._active_turn_id = None
                self._drain_pending()
                self.device_runtime.cancel_current("免按键监听已关闭")
                if self.sleep_wake_session is not None:
                    self.sleep_wake_session()
                with self._lock:
                    self._snapshot = replace(
                        self._snapshot,
                        enabled=False,
                        status="免按键监听已关闭，可使用手动录音调试",
                        calibrating=False,
                        audio_level=0.0,
                        vad_threshold=0.0,
                    )

    def shutdown(self) -> None:
        self.stop()
        self._shutdown.set()
        for thread in (self._capture_thread, self._worker_thread):
            if thread is not None:
                thread.join(timeout=2.0)
        self.audio_input.close()

    def interrupt_playback(self) -> None:
        self.player.stop()

    def _set_meter(self, level: float, threshold: float, calibrating: bool) -> None:
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                audio_level=level,
                vad_threshold=threshold,
                calibrating=calibrating,
                dropped_frames=self.audio_input.dropped_frames,
            )

    def _set_activity(self, status: str, error: str | None = None) -> None:
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                status=status,
                last_error=error,
            )

    def _capture_loop(self) -> None:
        while not self._shutdown.is_set():
            if not self._enabled.wait(timeout=0.2):
                continue
            frame = self.audio_input.read_frame(timeout=0.1)
            if frame is None or not self._enabled.is_set():
                continue
            try:
                decision = self.vad.analyze(frame)
                self._set_meter(
                    decision.level,
                    decision.threshold,
                    decision.calibrating,
                )
                if decision.calibrating:
                    continue

                speech = decision.is_speech
                if self.player.is_playing:
                    playback_age_ms = (
                        time.monotonic() - self._playback_started_at
                    ) * 1000
                    if playback_age_ms < self.config.barge_in_guard_ms:
                        speech = False
                    else:
                        speech = decision.level >= (
                            decision.threshold
                            * self.config.barge_in_threshold_multiplier
                        )

                event = self._segmenter.feed(frame, speech)
                if event is None:
                    if (
                        not self._segmenter.active
                        and self.device_runtime.snapshot.state == DeviceState.IDLE
                    ):
                        self._set_activity("正在监听，请按唤醒状态说话")
                    continue
                if event.kind == SegmentEventKind.STARTED:
                    self._on_speech_started()
                elif event.kind == SegmentEventKind.COMPLETED:
                    self._on_speech_completed(event.frames)
                else:
                    self._on_speech_discarded()
            except Exception as exc:
                turn_id = self._active_turn_id
                if turn_id is not None:
                    self.device_runtime.fail(str(exc), turn_id)
                self._segmenter.reset()
                self._active_turn_id = None
                self._set_activity("麦克风处理异常", str(exc))

    def _on_speech_started(self) -> None:
        snapshot = self.device_runtime.snapshot
        try:
            turn_id = self.device_runtime.start_listening(
                brightness=snapshot.brightness,
                user_id=self._user_name or None,
            )
        except DeviceNotReady:
            self._segmenter.reset()
            self._set_activity("模型仍在预热，稍后可直接说话")
            return
        self._active_turn_id = turn_id
        if self.player.is_playing:
            self.player.stop()
            self._set_activity("检测到你说话，已打断上一条回复")
        else:
            self._set_activity("正在听你说…")

    def _on_speech_discarded(self) -> None:
        turn_id = self._active_turn_id
        self._active_turn_id = None
        if turn_id is not None:
            self.device_runtime.listening_timeout(turn_id)
        self._set_activity("声音太短，已继续监听")

    def _on_speech_completed(self, frames: tuple[AudioFrame, ...]) -> None:
        turn_id = self._active_turn_id
        self._active_turn_id = None
        if turn_id is None or not self.device_runtime.is_current(turn_id):
            return
        if not self.device_runtime.speech_ended(turn_id):
            return
        try:
            path = self._write_wave(frames)
        except Exception as exc:
            self.device_runtime.fail(str(exc), turn_id)
            raise
        item = _CapturedUtterance(turn_id=turn_id, audio_path=path)
        try:
            self._pending.put_nowait(item)
        except queue.Full:
            try:
                stale = self._pending.get_nowait()
                self._safe_remove(stale.audio_path)
            except queue.Empty:
                pass
            self._pending.put_nowait(item)
        self._set_activity("一句话已结束，正在识别和思考")

    def _write_wave(self, frames: tuple[AudioFrame, ...]) -> str:
        if not frames:
            raise ValueError("没有可写入的麦克风音频")
        path = self.output_dir / f"handsfree-input-{uuid4().hex}.wav"
        with wave.open(str(path), "wb") as output:
            output.setnchannels(frames[0].channels)
            output.setsampwidth(2)
            output.setframerate(frames[0].sample_rate)
            output.writeframes(b"".join(frame.pcm_s16le for frame in frames))
        return str(path)

    def _worker_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                item = self._pending.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._process_utterance(item)
            finally:
                self._safe_remove(item.audio_path)

    def _process_utterance(self, item: _CapturedUtterance) -> None:
        if not self.device_runtime.is_current(item.turn_id):
            return
        with self._lock:
            history = [dict(message) for message in self._history]
            user_name = self._user_name
            preferences = self._preferences
        previous_brightness = self.device_runtime.snapshot.brightness
        try:
            result = self.handle_turn(
                TurnRequest(
                    audio_path=item.audio_path,
                    history=history,
                    user_name=user_name,
                    preferences=preferences,
                    brightness=previous_brightness,
                    require_wake_word=True,
                )
            )
        except NoSpeechDetectedError as exc:
            self._refresh_wake_session()
            if self.device_runtime.is_current(item.turn_id):
                self.device_runtime.cancel_current("没有听清，继续监听")
                self._publish_not_understood(str(exc))
            return
        except Exception as exc:
            self._refresh_wake_session()
            if self.device_runtime.fail(str(exc), item.turn_id):
                self._publish_error(str(exc))
            return

        if not result.response_required:
            if self.device_runtime.is_current(item.turn_id):
                self.device_runtime.cancel_current(
                    result.wake_word_status or "未检测到唤醒词，返回待机"
                )
                self._set_activity(result.status)
            return

        if not self.device_runtime.response_ready(
            item.turn_id, result.brightness
        ):
            return
        self._publish_result(result, previous_brightness, result.status)

        audio_reply, tts_status = self.synthesize_reply(result.reply)
        final_status = (
            f"{result.status.replace('｜正在生成语音…', '')}｜{tts_status}"
        )
        if not self.device_runtime.is_current(item.turn_id):
            if audio_reply:
                self._safe_remove(audio_reply)
            return
        self._publish_result(result, previous_brightness, final_status)

        if audio_reply is None:
            self.device_runtime.playback_finished(item.turn_id)
            self._refresh_wake_session()
            self._set_activity("语音生成失败，已保留文字回答")
            return

        playback_error: str | None = None
        try:
            self._playback_started_at = time.monotonic()
            self._set_activity("正在通过电脑扬声器回答；你可以说话打断")
            self.player.play(audio_reply)
        except Exception as exc:
            playback_error = str(exc)
        finally:
            self._safe_remove(audio_reply)

        if self.device_runtime.is_current(item.turn_id):
            self.device_runtime.playback_finished(item.turn_id)
            self._refresh_wake_session()
            if playback_error:
                self._set_activity("自动播放失败，文字回答仍可用", playback_error)
            else:
                self._set_activity("回答结束，正在监听下一句话")

    def _refresh_wake_session(self) -> None:
        if self.refresh_wake_session is None:
            return
        try:
            self.refresh_wake_session()
        except Exception:
            pass

    def _publish_result(
        self,
        result: TurnResult,
        previous_brightness: int,
        status: str,
    ) -> None:
        with self._lock:
            self._history = [dict(message) for message in result.history[-12:]]
            self._snapshot = replace(
                self._snapshot,
                result_version=self._snapshot.result_version + 1,
                transcript=result.transcript,
                reply=result.reply,
                history=self._encode_history(self._history),
                previous_brightness=previous_brightness,
                brightness=result.brightness,
                light_status=result.light_status,
                result_status=status,
                last_error=None,
            )

    def _publish_error(self, error: str) -> None:
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                result_version=self._snapshot.result_version + 1,
                transcript="",
                reply="",
                result_status=f"本轮未完成：{error}",
                status="没有听清或处理失败，继续说话可自动恢复",
                last_error=error,
            )

    def _publish_not_understood(self, message: str) -> None:
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                result_version=self._snapshot.result_version + 1,
                transcript="",
                reply="",
                result_status=f"本轮没有听清：{message}",
                status="没有听清，已自动回到监听",
                last_error=None,
            )

    def _drain_pending(self) -> None:
        while True:
            try:
                item = self._pending.get_nowait()
            except queue.Empty:
                return
            self._safe_remove(item.audio_path)

    def _safe_remove(self, raw_path: str) -> None:
        path = Path(raw_path)
        try:
            resolved = path.resolve()
            output_dir = self.output_dir.resolve()
            if resolved.parent != output_dir:
                return
            if not resolved.name.startswith(("handsfree-input-", "reply-")):
                return
            resolved.unlink(missing_ok=True)
        except OSError:
            pass
