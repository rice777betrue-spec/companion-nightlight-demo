from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from companion_demo.adapters.pc.light import VirtualLightDriver
from companion_demo.core.contracts import TurnRequest
from companion_demo.runtime import TurnEngine, WakeWordController


class WakeWordControllerTests(unittest.TestCase):
    def _controller(
        self,
        directory: str,
        *,
        phrase: str = "小夜灯",
        session_seconds: float = 30.0,
    ) -> WakeWordController:
        return WakeWordController(
            Path(directory) / "wake-word.json",
            default_phrase=phrase,
            session_seconds=session_seconds,
        )

    def test_standby_ignores_speech_without_wake_word(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory)

            decision = controller.evaluate("今天外面有点吵", now=100.0)

            self.assertEqual(decision.action, "ignore")
            self.assertEqual(decision.transcript, "")
            self.assertIn("不回应", decision.status)

    def test_wake_only_acknowledges_then_allows_continuous_chat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory)

            wake = controller.evaluate("小夜灯", now=100.0)
            controller.refresh_session(now=105.0)
            follow_up = controller.evaluate("我今天有点累", now=120.0)

            self.assertEqual(wake.action, "acknowledge")
            self.assertEqual(follow_up.action, "process")
            self.assertFalse(follow_up.triggered)
            self.assertEqual(follow_up.transcript, "我今天有点累")

    def test_wake_word_and_command_can_be_spoken_in_one_sentence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory)

            decision = controller.evaluate("小夜灯，关灯", now=100.0)

            self.assertEqual(decision.action, "process")
            self.assertTrue(decision.triggered)
            self.assertEqual(decision.transcript, "关灯")

    def test_idle_timeout_requires_wake_word_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory, session_seconds=10.0)
            controller.evaluate("小夜灯", now=100.0)
            controller.refresh_session(now=110.0)

            within_window = controller.evaluate("继续聊聊", now=119.0)
            controller.refresh_session(now=120.0)
            after_timeout = controller.evaluate("你还在吗", now=131.0)

            self.assertEqual(within_window.action, "process")
            self.assertEqual(after_timeout.action, "ignore")
            self.assertIn("待机中", after_timeout.status)

    def test_processing_time_does_not_count_as_idle_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory, session_seconds=10.0)
            controller.evaluate("小夜灯", now=100.0)

            status = controller.refresh_session(now=1_000.0)
            follow_up = controller.evaluate("接着说", now=1_009.0)

            self.assertIn("空闲 10 秒", status)
            self.assertEqual(follow_up.action, "process")

    def test_custom_phrase_is_saved_and_immediately_recognized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory)
            controller.set_phrase("暖暖")
            restored = self._controller(directory)

            decision = restored.evaluate("暖暖，开灯", now=100.0)

            self.assertEqual(restored.phrase, "暖暖")
            self.assertEqual(decision.action, "process")
            self.assertEqual(decision.transcript, "开灯")

    def test_one_character_phrase_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory)

            with self.assertRaisesRegex(ValueError, "至少需要 2 个"):
                controller.set_phrase("灯")


class _FakeASR:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript

    def transcribe(self, _audio_path: str) -> str:
        return self.transcript


class _FakeCompanion:
    device_label = "测试模型"

    def __init__(self) -> None:
        self.calls = 0

    def reply(self, *_args, **_kwargs) -> str:
        self.calls += 1
        return "我在听。"


class _CountingLightDriver:
    def __init__(self) -> None:
        self.calls = 0
        self.delegate = VirtualLightDriver()

    def apply(self, command):
        self.calls += 1
        return self.delegate.apply(command)


class WakeWordTurnEngineTests(unittest.TestCase):
    def _run(self, transcript: str):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        gate = WakeWordController(Path(temporary.name) / "wake-word.json")
        companion = _FakeCompanion()
        light = _CountingLightDriver()
        engine = TurnEngine(
            _FakeASR(transcript),
            companion,
            light,
            wake_word_gate=gate,
        )
        result = engine.handle(
            TurnRequest(
                "sample.wav",
                brightness=35,
                require_wake_word=True,
            )
        )
        return result, companion, light

    def test_unwoken_turn_skips_light_model_and_response(self) -> None:
        result, companion, light = self._run("把灯关掉")

        self.assertFalse(result.response_required)
        self.assertEqual(result.brightness, 35)
        self.assertEqual(companion.calls, 0)
        self.assertEqual(light.calls, 0)

    def test_wake_only_uses_fast_fixed_acknowledgement(self) -> None:
        result, companion, light = self._run("小夜灯")

        self.assertTrue(result.response_required)
        self.assertEqual(result.reply, "我在，你说吧。")
        self.assertEqual(companion.calls, 0)
        self.assertEqual(light.calls, 0)

    def test_wake_and_light_command_executes_in_same_turn(self) -> None:
        result, companion, light = self._run("小夜灯，关灯")

        self.assertTrue(result.response_required)
        self.assertEqual(result.brightness, 0)
        self.assertEqual(result.reply, "好的，灯已经关掉了。")
        self.assertEqual(companion.calls, 0)
        self.assertEqual(light.calls, 1)


if __name__ == "__main__":
    unittest.main()
