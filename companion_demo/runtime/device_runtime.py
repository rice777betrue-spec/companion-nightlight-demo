from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from threading import RLock


class DeviceState(str, Enum):
    STARTING = "starting"
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


class DeviceEvent(str, Enum):
    MODELS_READY = "models_ready"
    WAKE = "wake"
    SPEECH_ENDED = "speech_ended"
    RESPONSE_READY = "response_ready"
    PLAYBACK_FINISHED = "playback_finished"
    BARGE_IN = "barge_in"
    LISTENING_TIMEOUT = "listening_timeout"
    CANCEL = "cancel"
    FAILURE = "failure"
    RECOVER = "recover"


_STATE_LABELS = {
    DeviceState.STARTING: "启动预热",
    DeviceState.IDLE: "待机",
    DeviceState.LISTENING: "聆听",
    DeviceState.THINKING: "思考",
    DeviceState.SPEAKING: "播放",
    DeviceState.ERROR: "故障恢复",
}

_ALLOWED_TRANSITIONS = {
    DeviceState.STARTING: {
        DeviceEvent.MODELS_READY: DeviceState.IDLE,
        DeviceEvent.FAILURE: DeviceState.ERROR,
    },
    DeviceState.IDLE: {
        DeviceEvent.WAKE: DeviceState.LISTENING,
        DeviceEvent.FAILURE: DeviceState.ERROR,
    },
    DeviceState.LISTENING: {
        DeviceEvent.SPEECH_ENDED: DeviceState.THINKING,
        DeviceEvent.BARGE_IN: DeviceState.LISTENING,
        DeviceEvent.LISTENING_TIMEOUT: DeviceState.IDLE,
        DeviceEvent.CANCEL: DeviceState.IDLE,
        DeviceEvent.FAILURE: DeviceState.ERROR,
    },
    DeviceState.THINKING: {
        DeviceEvent.RESPONSE_READY: DeviceState.SPEAKING,
        DeviceEvent.BARGE_IN: DeviceState.LISTENING,
        DeviceEvent.CANCEL: DeviceState.IDLE,
        DeviceEvent.FAILURE: DeviceState.ERROR,
    },
    DeviceState.SPEAKING: {
        DeviceEvent.PLAYBACK_FINISHED: DeviceState.IDLE,
        DeviceEvent.BARGE_IN: DeviceState.LISTENING,
        DeviceEvent.CANCEL: DeviceState.IDLE,
        DeviceEvent.FAILURE: DeviceState.ERROR,
    },
    DeviceState.ERROR: {
        DeviceEvent.RECOVER: DeviceState.IDLE,
    },
}


class InvalidDeviceTransition(RuntimeError):
    pass


class DeviceNotReady(RuntimeError):
    pass


@dataclass(frozen=True)
class DeviceSnapshot:
    state: DeviceState = DeviceState.STARTING
    turn_id: int = 0
    brightness: int = 35
    user_id: str | None = None
    detail: str = "正在准备本地模型"
    last_error: str | None = None
    updated_at: float = 0.0

    @property
    def state_label(self) -> str:
        return _STATE_LABELS[self.state]


@dataclass(frozen=True)
class TransitionRecord:
    previous: DeviceState
    current: DeviceState
    event: DeviceEvent
    turn_id: int
    timestamp: float


class DeviceRuntime:
    """设备生命周期状态机。

    所有异步结果都携带 turn_id；旧轮次结果只会被丢弃，不会改变当前状态。
    """

    def __init__(self, initial_brightness: int | float = 35) -> None:
        level = self._clamp(initial_brightness)
        self._lock = RLock()
        self._snapshot = DeviceSnapshot(
            brightness=level,
            updated_at=time.monotonic(),
        )
        self._transitions: deque[TransitionRecord] = deque(maxlen=32)

    @staticmethod
    def _clamp(value: int | float) -> int:
        return max(0, min(100, int(round(value))))

    @property
    def snapshot(self) -> DeviceSnapshot:
        with self._lock:
            return self._snapshot

    @property
    def transition_history(self) -> tuple[TransitionRecord, ...]:
        with self._lock:
            return tuple(self._transitions)

    @property
    def status_text(self) -> str:
        snapshot = self.snapshot
        parts = [snapshot.state_label]
        if snapshot.turn_id:
            parts.append(f"第 {snapshot.turn_id} 轮")
        parts.append(f"亮度 {snapshot.brightness}%")
        if snapshot.last_error:
            parts.append(f"异常：{snapshot.last_error}")
        elif snapshot.detail:
            parts.append(snapshot.detail)
        return "｜".join(parts)

    def _transition(self, event: DeviceEvent, detail: str) -> None:
        previous = self._snapshot.state
        current = _ALLOWED_TRANSITIONS.get(previous, {}).get(event)
        if current is None:
            raise InvalidDeviceTransition(
                f"不允许从 {previous.value} 通过 {event.value} 转换"
            )
        timestamp = time.monotonic()
        self._snapshot = replace(
            self._snapshot,
            state=current,
            detail=detail,
            updated_at=timestamp,
        )
        self._transitions.append(
            TransitionRecord(
                previous=previous,
                current=current,
                event=event,
                turn_id=self._snapshot.turn_id,
                timestamp=timestamp,
            )
        )

    def models_ready(self) -> None:
        with self._lock:
            if self._snapshot.state == DeviceState.STARTING:
                self._transition(DeviceEvent.MODELS_READY, "等待唤醒或录音")
            elif self._snapshot.state == DeviceState.ERROR and not self._snapshot.turn_id:
                self._transition(DeviceEvent.RECOVER, "模型恢复，等待唤醒或录音")
            self._snapshot = replace(self._snapshot, last_error=None)

    def start_listening(
        self,
        brightness: int | float | None = None,
        user_id: str | None = None,
    ) -> int:
        with self._lock:
            if self._snapshot.state == DeviceState.STARTING:
                raise DeviceNotReady("设备仍在启动预热，请稍候再试。")
            if self._snapshot.state == DeviceState.ERROR:
                self._transition(DeviceEvent.RECOVER, "已恢复，准备聆听")

            next_turn = self._snapshot.turn_id + 1
            self._snapshot = replace(
                self._snapshot,
                turn_id=next_turn,
                brightness=(
                    self._snapshot.brightness
                    if brightness is None
                    else self._clamp(brightness)
                ),
                user_id=user_id,
                last_error=None,
            )
            if self._snapshot.state == DeviceState.IDLE:
                self._transition(DeviceEvent.WAKE, "正在接收用户语音")
            else:
                self._transition(DeviceEvent.BARGE_IN, "用户已打断上一轮，重新聆听")
            return next_turn

    def is_current(self, turn_id: int) -> bool:
        with self._lock:
            return self._snapshot.turn_id == turn_id

    def speech_ended(self, turn_id: int) -> bool:
        with self._lock:
            if self._snapshot.turn_id != turn_id:
                return False
            self._transition(DeviceEvent.SPEECH_ENDED, "正在识别并生成回答")
            return True

    def response_ready(self, turn_id: int, brightness: int | float) -> bool:
        with self._lock:
            if self._snapshot.turn_id != turn_id:
                return False
            self._snapshot = replace(
                self._snapshot,
                brightness=self._clamp(brightness),
            )
            self._transition(DeviceEvent.RESPONSE_READY, "回答就绪，正在播放")
            return True

    def playback_finished(self, turn_id: int) -> bool:
        with self._lock:
            if self._snapshot.turn_id != turn_id:
                return False
            self._transition(DeviceEvent.PLAYBACK_FINISHED, "等待下一次交互")
            return True

    def listening_timeout(self, turn_id: int) -> bool:
        with self._lock:
            if self._snapshot.turn_id != turn_id:
                return False
            self._transition(DeviceEvent.LISTENING_TIMEOUT, "未检测到语音，返回待机")
            return True

    def cancel_current(self, detail: str = "当前交互已取消") -> bool:
        """停止当前轮次并使尚未返回的异步结果失效。"""

        with self._lock:
            if self._snapshot.state not in {
                DeviceState.LISTENING,
                DeviceState.THINKING,
                DeviceState.SPEAKING,
            }:
                return False
            self._snapshot = replace(
                self._snapshot,
                turn_id=self._snapshot.turn_id + 1,
            )
            self._transition(DeviceEvent.CANCEL, detail)
            return True

    def fail(self, error: str, turn_id: int | None = None) -> bool:
        with self._lock:
            if turn_id is not None and self._snapshot.turn_id != turn_id:
                return False
            message = str(error).strip() or "未知异常"
            if self._snapshot.state != DeviceState.ERROR:
                self._transition(DeviceEvent.FAILURE, "等待恢复")
            self._snapshot = replace(self._snapshot, last_error=message)
            return True

    def recover(self) -> None:
        with self._lock:
            if self._snapshot.state == DeviceState.ERROR:
                self._transition(DeviceEvent.RECOVER, "已恢复，等待下一次交互")
                self._snapshot = replace(self._snapshot, last_error=None)

    def set_brightness(self, brightness: int | float) -> None:
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                brightness=self._clamp(brightness),
                updated_at=time.monotonic(),
            )
