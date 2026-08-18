from __future__ import annotations

import unittest

from companion_demo.adapters.pc.light import VirtualLightDriver
from companion_demo.core.contracts import LightExecution, TurnRequest
from companion_demo.runtime import TurnEngine


class _FakeASR:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript

    def transcribe(self, _audio_path: str) -> str:
        return self.transcript


class _FakeCompanion:
    def __init__(self) -> None:
        self.calls = 0

    def reply(self, *_args, **_kwargs) -> str:
        self.calls += 1
        return "我在听，你可以继续说。"


class _FailingLightDriver:
    def apply(self, command) -> LightExecution:
        return LightExecution(
            previous=command.previous,
            requested=command.brightness,
            actual=command.previous,
            applied=False,
            description="串口灯光驱动未确认",
            error="串口超时",
        )


class MinimalFrameworkTests(unittest.TestCase):
    def test_turn_engine_is_independent_from_gradio(self) -> None:
        companion = _FakeCompanion()
        engine = TurnEngine(
            _FakeASR("关灯"),
            companion,
            VirtualLightDriver(),
        )

        result = engine.handle(TurnRequest("sample.wav", brightness=35))

        self.assertEqual(result.brightness, 0)
        self.assertEqual(result.dialogue_mode, "灯光控制")
        self.assertFalse(result.used_companion_model)
        self.assertTrue(result.light_execution.applied)
        self.assertEqual(companion.calls, 0)

    def test_failed_hardware_never_gets_a_false_success_reply(self) -> None:
        companion = _FakeCompanion()
        engine = TurnEngine(
            _FakeASR("关灯"),
            companion,
            _FailingLightDriver(),
        )

        result = engine.handle(TurnRequest("sample.wav", brightness=35))

        self.assertEqual(result.brightness, 35)
        self.assertIn("没有成功", result.reply)
        self.assertNotIn("已经关掉", result.reply)
        self.assertIn("串口超时", result.light_status)
        self.assertFalse(result.light_execution.applied)
        self.assertEqual(companion.calls, 0)


if __name__ == "__main__":
    unittest.main()
