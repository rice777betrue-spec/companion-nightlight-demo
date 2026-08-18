from __future__ import annotations

import unittest

import app
from companion_demo.runtime import DeviceRuntime


class _FakePipeline:
    def __init__(self) -> None:
        self.device_runtime = DeviceRuntime()
        self.device_runtime.models_ready()

    def generate_reply(self, *_args, **_kwargs):
        return (
            "测试语音",
            "测试回答",
            [
                {"role": "user", "content": "测试语音"},
                {"role": "assistant", "content": "测试回答"},
            ],
            "文字已生成｜ASR 0.50 秒｜Qwen 0.70 秒｜情绪陪伴｜正在生成语音…",
            35,
            "灯光保持在 35%",
        )

    def synthesize_reply(self, _reply):
        return "reply.mp3", "语音已生成｜TTS 2.00 秒"


class StreamingResponseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_pipeline = app.pipeline
        app.pipeline = _FakePipeline()

    def tearDown(self) -> None:
        app.pipeline = self.original_pipeline

    def test_text_is_yielded_before_tts_audio(self) -> None:
        updates = list(app.run_turn("sample.wav", "小林", "简短", [], 35))
        self.assertEqual(len(updates), 3)
        self.assertEqual(updates[0][5], "正在识别语音…")
        self.assertEqual(updates[1][1], "测试回答")
        self.assertIsNone(updates[1][2])
        self.assertEqual(updates[2][2], "reply.mp3")
        self.assertIn("播放", updates[2][9])

        app.finish_playback()
        self.assertIn("待机", app.pipeline.device_runtime.status_text)


if __name__ == "__main__":
    unittest.main()
