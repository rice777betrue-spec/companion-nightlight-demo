from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from companion_demo.adapters.pc.light import VirtualLightDriver
from companion_demo.core.contracts import TurnRequest
from companion_demo.dialogue import (
    classify_sleep_mode_confirmation,
    needs_sleep_mode_confirmation,
)
from companion_demo.runtime import TurnEngine, WakeWordController


class _MutableASR:
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


class _CountingLightDriver:
    def __init__(self) -> None:
        self.calls = 0
        self.delegate = VirtualLightDriver()

    def apply(self, command):
        self.calls += 1
        return self.delegate.apply(command)


class _Clock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class SleepIntentDetectionTests(unittest.TestCase):
    def test_imminent_sleep_variants_need_confirmation(self) -> None:
        for phrase in ("我要睡觉了", "我准备去睡觉了", "我该休息了"):
            with self.subTest(phrase=phrase):
                self.assertTrue(needs_sleep_mode_confirmation(phrase))

    def test_sleep_problem_history_and_question_do_not_prompt(self) -> None:
        for phrase in (
            "我一直睡不着",
            "我昨天说我要睡了",
            "我准备睡觉了吗？",
            "我今天想早点睡觉",
        ):
            with self.subTest(phrase=phrase):
                self.assertFalse(needs_sleep_mode_confirmation(phrase))

    def test_negative_confirmation_wins_over_polite_affirmation(self) -> None:
        self.assertEqual(
            classify_sleep_mode_confirmation("好的，但先不用"),
            "negative",
        )
        self.assertEqual(
            classify_sleep_mode_confirmation("别帮我开"),
            "negative",
        )


class SleepModeConfirmationTests(unittest.TestCase):
    def _build_engine(
        self,
        transcript: str,
        *,
        timeout: float = 30.0,
        wake_word_gate: WakeWordController | None = None,
    ):
        asr = _MutableASR(transcript)
        companion = _FakeCompanion()
        light = _CountingLightDriver()
        clock = _Clock()
        engine = TurnEngine(
            asr,
            companion,
            light,
            wake_word_gate=wake_word_gate,
            sleep_confirmation_timeout_seconds=timeout,
            clock=clock,
        )
        return engine, asr, companion, light, clock

    def test_natural_bedtime_intent_asks_before_changing_light(self) -> None:
        engine, _asr, companion, light, _clock = self._build_engine(
            "我今天想早点睡觉，我准备睡觉了。"
        )

        result = engine.handle(TurnRequest("sample.wav", brightness=65))

        self.assertEqual(result.brightness, 65)
        self.assertEqual(result.dialogue_mode, "睡眠模式确认")
        self.assertIn("要帮你开启睡眠模式", result.reply)
        self.assertIn("10%", result.reply)
        self.assertEqual(light.calls, 0)
        self.assertEqual(companion.calls, 0)

    def test_affirmative_follow_up_sets_sleep_brightness(self) -> None:
        engine, asr, companion, light, clock = self._build_engine("我要睡了")
        first = engine.handle(TurnRequest("first.wav", brightness=70))
        clock.advance(5.0)
        asr.transcript = "要"

        confirmed = engine.handle(
            TurnRequest(
                "second.wav",
                history=first.history,
                brightness=first.brightness,
            )
        )

        self.assertEqual(confirmed.brightness, 10)
        self.assertIn("已为你开启睡眠模式", confirmed.reply)
        self.assertTrue(confirmed.light_execution.applied)
        self.assertEqual(light.calls, 1)
        self.assertEqual(companion.calls, 0)
        self.assertEqual(len(confirmed.history), 4)

    def test_negative_follow_up_keeps_current_brightness(self) -> None:
        engine, asr, companion, light, _clock = self._build_engine("我先睡了")
        first = engine.handle(TurnRequest("first.wav", brightness=55))
        asr.transcript = "不用"

        declined = engine.handle(
            TurnRequest(
                "second.wav",
                history=first.history,
                brightness=first.brightness,
            )
        )

        self.assertEqual(declined.brightness, 55)
        self.assertIn("不开启睡眠模式", declined.reply)
        self.assertFalse(declined.light_execution.applied)
        self.assertEqual(light.calls, 0)
        self.assertEqual(companion.calls, 0)

    def test_expired_confirmation_cannot_change_the_light(self) -> None:
        engine, asr, companion, light, clock = self._build_engine(
            "准备睡觉了",
            timeout=10.0,
        )
        first = engine.handle(TurnRequest("first.wav", brightness=80))
        clock.advance(10.1)
        asr.transcript = "好"

        expired = engine.handle(
            TurnRequest(
                "second.wav",
                history=first.history,
                brightness=first.brightness,
            )
        )

        self.assertEqual(expired.brightness, 80)
        self.assertIn("确认已超时", expired.reply)
        self.assertFalse(expired.light_execution.applied)
        self.assertEqual(light.calls, 0)
        self.assertEqual(companion.calls, 0)

    def test_new_topic_cancels_the_old_confirmation(self) -> None:
        engine, asr, companion, light, _clock = self._build_engine("我要睡了")
        first = engine.handle(TurnRequest("first.wav", brightness=45))
        asr.transcript = "我还想再聊一会儿"
        continued = engine.handle(
            TurnRequest(
                "second.wav",
                history=first.history,
                brightness=first.brightness,
            )
        )
        asr.transcript = "要"

        late_yes = engine.handle(
            TurnRequest(
                "third.wav",
                history=continued.history,
                brightness=continued.brightness,
            )
        )

        self.assertEqual(late_yes.brightness, 45)
        self.assertEqual(companion.calls, 2)
        self.assertEqual(light.calls, 2)

    def test_clear_pending_confirmation_prevents_later_yes(self) -> None:
        engine, asr, companion, light, _clock = self._build_engine("我要睡了")
        first = engine.handle(TurnRequest("first.wav", brightness=40))
        engine.clear_pending_confirmation()
        asr.transcript = "好"

        result = engine.handle(
            TurnRequest(
                "second.wav",
                history=first.history,
                brightness=first.brightness,
            )
        )

        self.assertEqual(result.brightness, 40)
        self.assertEqual(companion.calls, 1)
        self.assertEqual(light.calls, 1)

    def test_past_sleep_statement_does_not_prompt_or_change_light(self) -> None:
        engine, _asr, companion, light, _clock = self._build_engine(
            "我昨天很早就说我要睡了"
        )

        result = engine.handle(TurnRequest("sample.wav", brightness=35))

        self.assertNotIn("要帮你开启睡眠模式", result.reply)
        self.assertEqual(result.brightness, 35)
        self.assertEqual(companion.calls, 1)
        self.assertEqual(light.calls, 1)

    def test_explicit_sleep_mode_still_executes_immediately(self) -> None:
        engine, _asr, companion, light, _clock = self._build_engine(
            "开启睡眠模式"
        )

        result = engine.handle(TurnRequest("sample.wav", brightness=60))

        self.assertEqual(result.brightness, 10)
        self.assertNotIn("要帮你", result.reply)
        self.assertEqual(companion.calls, 0)
        self.assertEqual(light.calls, 1)

    def test_hands_free_wake_session_accepts_confirmation_follow_up(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            gate = WakeWordController(Path(directory) / "wake-word.json")
            engine, asr, companion, light, _clock = self._build_engine(
                "小夜灯，我准备睡觉了",
                wake_word_gate=gate,
            )
            first = engine.handle(
                TurnRequest(
                    "first.wav",
                    brightness=50,
                    require_wake_word=True,
                )
            )
            gate.refresh_session()
            asr.transcript = "要"

            confirmed = engine.handle(
                TurnRequest(
                    "second.wav",
                    history=first.history,
                    brightness=first.brightness,
                    require_wake_word=True,
                )
            )

        self.assertEqual(confirmed.brightness, 10)
        self.assertIn("已为你开启睡眠模式", confirmed.reply)
        self.assertEqual(companion.calls, 0)
        self.assertEqual(light.calls, 1)


if __name__ == "__main__":
    unittest.main()
