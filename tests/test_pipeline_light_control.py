from __future__ import annotations

import unittest

from companion_demo.pipeline import DemoPipeline
from companion_demo.adapters.pc.light import VirtualLightDriver


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
        return "被批评了还得撑着，确实很磨人。最让你烦的是哪一部分？"


class _FakeTTS:
    engine_label = "测试 TTS"

    def synthesize(self, _text: str) -> str:
        return "reply.wav"


def _pipeline_for(transcript: str) -> DemoPipeline:
    return DemoPipeline(
        asr=_FakeASR(transcript),
        companion=_FakeCompanion(),
        tts=_FakeTTS(),
        light_driver=VirtualLightDriver(),
    )


class PipelineLightControlTests(unittest.TestCase):
    def test_pure_light_command_bypasses_qwen(self) -> None:
        pipeline = _pipeline_for("把灯关一下")
        result = pipeline.generate_reply("sample.wav", [], "小林", "", 35)

        self.assertEqual(result[1], "好的，灯已经关掉了。")
        self.assertEqual(result[4], 0)
        self.assertEqual(pipeline.companion.calls, 0)
        self.assertIn("灯控即时确认", result[3])

    def test_light_and_emotion_confirm_action_then_use_qwen(self) -> None:
        pipeline = _pipeline_for("调暗一点，我今天很烦")
        result = pipeline.generate_reply("sample.wav", [], "小林", "", 50)

        self.assertEqual(result[4], 30)
        self.assertTrue(result[1].startswith("好的，灯已调暗到 30%。"))
        self.assertEqual(pipeline.companion.calls, 1)
        self.assertIn("最让你烦", result[1])


if __name__ == "__main__":
    unittest.main()
