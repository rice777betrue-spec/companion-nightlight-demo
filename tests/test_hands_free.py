from __future__ import annotations

import unittest
import queue
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np

from companion_demo.adapters.pc.audio import AdaptiveEnergyVad
from companion_demo.core.contracts import (
    AudioFrame,
    LightExecution,
    TurnResult,
    VadDecision,
)
from companion_demo.runtime import (
    DeviceRuntime,
    DeviceState,
    HandsFreeConfig,
    HandsFreeRuntime,
    SegmentEventKind,
    UtteranceSegmenter,
)


def _frame(level: int = 0) -> AudioFrame:
    samples = np.full(320, level, dtype=np.int16)
    return AudioFrame(samples.tobytes(), sample_rate=16_000)


class AdaptiveEnergyVadTests(unittest.TestCase):
    def test_calibrates_then_detects_clear_speech(self) -> None:
        vad = AdaptiveEnergyVad(calibration_frames=2)

        self.assertTrue(vad.analyze(_frame(30)).calibrating)
        self.assertTrue(vad.analyze(_frame(30)).calibrating)
        decision = vad.analyze(_frame(2_000))

        self.assertFalse(decision.calibrating)
        self.assertTrue(decision.is_speech)
        self.assertGreater(decision.level, decision.threshold)

    def test_sensitivity_changes_effective_threshold(self) -> None:
        vad = AdaptiveEnergyVad(calibration_frames=0, sensitivity=1.0)
        normal = vad.analyze(_frame(0)).threshold
        vad.set_sensitivity(2.0)
        sensitive = vad.analyze(_frame(0)).threshold

        self.assertLess(sensitive, normal)


class UtteranceSegmenterTests(unittest.TestCase):
    def test_detects_start_and_end_with_preroll(self) -> None:
        segmenter = UtteranceSegmenter(
            pre_roll_frames=2,
            speech_start_frames=2,
            speech_end_frames=3,
            minimum_voiced_frames=2,
            maximum_frames=20,
        )

        self.assertIsNone(segmenter.feed(_frame(), False))
        self.assertIsNone(segmenter.feed(_frame(2_000), True))
        started = segmenter.feed(_frame(2_000), True)
        self.assertEqual(started.kind, SegmentEventKind.STARTED)
        self.assertLessEqual(segmenter.buffered_frames, 2)

        segmenter.feed(_frame(2_000), True)
        segmenter.feed(_frame(), False)
        segmenter.feed(_frame(), False)
        completed = segmenter.feed(_frame(), False)

        self.assertEqual(completed.kind, SegmentEventKind.COMPLETED)
        self.assertEqual(len(completed.frames), 6)
        self.assertFalse(segmenter.active)

    def test_utterance_buffer_never_exceeds_configured_limit(self) -> None:
        segmenter = UtteranceSegmenter(
            pre_roll_frames=2,
            speech_start_frames=2,
            speech_end_frames=3,
            minimum_voiced_frames=2,
            maximum_frames=6,
        )

        segmenter.feed(_frame(2_000), True)
        segmenter.feed(_frame(2_000), True)
        result = None
        for _ in range(20):
            result = segmenter.feed(_frame(2_000), True)
            if result is not None:
                break

        self.assertIsNotNone(result)
        self.assertEqual(result.kind, SegmentEventKind.COMPLETED)
        self.assertLessEqual(len(result.frames), 6)


class _FakeAudioInput:
    device_label = "测试麦克风"
    dropped_frames = 0

    def __init__(self) -> None:
        self.frames: queue.Queue[AudioFrame] = queue.Queue()

    def start(self) -> None:
        pass

    def read_frame(self, timeout: float = 0.1) -> AudioFrame | None:
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeVad:
    def reset(self) -> None:
        pass

    def set_sensitivity(self, _value: float) -> None:
        pass

    def analyze(self, frame: AudioFrame) -> VadDecision:
        level = float(abs(np.frombuffer(frame.pcm_s16le, dtype="<i2")[0]))
        return VadDecision(level >= 1_000, level, 500)


class _FakePlayer:
    engine_label = "测试扬声器"

    def __init__(self) -> None:
        self.played = threading.Event()
        self._playing = False

    @property
    def is_playing(self) -> bool:
        return self._playing

    def play(self, audio_path: str) -> None:
        self._playing = True
        self.played.set()
        self._playing = False

    def stop(self) -> None:
        self._playing = False


class HandsFreeRuntimeTests(unittest.TestCase):
    def test_voice_frames_run_a_complete_automatic_turn(self) -> None:
        audio = _FakeAudioInput()
        player = _FakePlayer()
        device = DeviceRuntime()
        device.models_ready()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            def handle_turn(_request) -> TurnResult:
                execution = LightExecution(35, 0, 0, True, "灯已关闭")
                return TurnResult(
                    transcript="关灯",
                    reply="好，已经关掉了。",
                    history=(
                        [
                            {"role": "user", "content": "关灯"},
                            {"role": "assistant", "content": "好，已经关掉了。"},
                        ]
                    ),
                    status="文字已生成｜正在生成语音…",
                    brightness=0,
                    light_status="灯已关闭",
                    dialogue_mode="灯光控制",
                    asr_seconds=0.01,
                    generation_seconds=0.0,
                    used_companion_model=False,
                    light_execution=execution,
                )

            def synthesize(_reply: str) -> tuple[str, str]:
                path = output_dir / "reply-test.wav"
                with wave.open(str(path), "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(16_000)
                    output.writeframes(_frame().pcm_s16le)
                return str(path), "测试 TTS"

            runtime = HandsFreeRuntime(
                audio_input=audio,
                vad=_FakeVad(),
                player=player,
                device_runtime=device,
                handle_turn=handle_turn,
                synthesize_reply=synthesize,
                output_dir=output_dir,
                config=HandsFreeConfig(
                    pre_roll_ms=20,
                    speech_start_ms=40,
                    speech_end_silence_ms=40,
                    minimum_voiced_ms=40,
                    maximum_utterance_ms=1_000,
                ),
            )
            runtime.start()
            for frame in (
                _frame(),
                _frame(2_000),
                _frame(2_000),
                _frame(2_000),
                _frame(),
                _frame(),
            ):
                audio.frames.put(frame)

            self.assertTrue(player.played.wait(timeout=2.0))
            deadline = time.monotonic() + 1.0
            while device.snapshot.state != DeviceState.IDLE:
                if time.monotonic() >= deadline:
                    self.fail("自动轮次没有返回待机")
                time.sleep(0.01)

            snapshot = runtime.snapshot
            self.assertEqual(snapshot.transcript, "关灯")
            self.assertEqual(snapshot.brightness, 0)
            self.assertEqual(snapshot.history_messages[-1]["role"], "assistant")
            runtime.shutdown()


if __name__ == "__main__":
    unittest.main()
