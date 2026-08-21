from __future__ import annotations

import unittest

from companion_demo.adapters.pc.light import VirtualLightDriver
from companion_demo.core.contracts import TurnRequest
from companion_demo.llm import (
    GENERATION_OPTIONS,
    build_system_prompt,
    clean_history,
)
from companion_demo.runtime import TurnEngine


class _ExactASR:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript

    def transcribe(self, _audio_path: str) -> str:
        return self.transcript


class _RecordingCompanion:
    device_label = "test"

    def __init__(self) -> None:
        self.args = None

    def reply(self, *args) -> str:
        self.args = args
        return "我先直接回应你刚才说的事情。"


class ReplyRelevanceTests(unittest.TestCase):
    def test_ordinary_turn_sends_exact_transcript_without_strategy_suffix(self) -> None:
        text = "我把钥匙忘在公司了，现在已经走到家门口。"
        companion = _RecordingCompanion()
        engine = TurnEngine(
            _ExactASR(text), companion, VirtualLightDriver()
        )

        engine.handle(TurnRequest("sample.wav"))

        self.assertIsNotNone(companion.args)
        self.assertEqual(companion.args[0], text)

    def test_generation_is_deterministic_for_relevance(self) -> None:
        self.assertFalse(GENERATION_OPTIONS["do_sample"])
        self.assertLessEqual(GENERATION_OPTIONS["max_new_tokens"], 96)

    def test_system_prompt_prioritizes_last_user_message(self) -> None:
        prompt = build_system_prompt("小林", "不要强行安慰")
        self.assertIn("最高优先级是准确回应用户最后一句", prompt)
        self.assertIn("用户提出问题时先直接回答", prompt)
        self.assertIn("不把小冲突升级", prompt)
        self.assertIn("小林", prompt)

    def test_history_drops_invalid_entries_and_keeps_recent_messages(self) -> None:
        history = [
            {"role": "system", "content": "不应进入历史"},
            {"role": "user", "content": "旧消息"},
            {"role": "assistant", "content": "旧回复"},
        ] + [
            {"role": "user" if index % 2 == 0 else "assistant", "content": str(index)}
            for index in range(10)
        ]

        cleaned = clean_history(history)

        self.assertEqual(len(cleaned), 8)
        self.assertNotIn("不应进入历史", [item["content"] for item in cleaned])
        self.assertEqual(cleaned[-1]["content"], "9")


if __name__ == "__main__":
    unittest.main()
