from __future__ import annotations

import unittest

from companion_demo.dialogue import choose_dialogue_guidance


class DialogueGuidanceTests(unittest.TestCase):
    def test_emotional_message_keeps_conversation_open(self) -> None:
        guidance = choose_dialogue_guidance("今天被领导批评了，我特别委屈")
        self.assertEqual(guidance.mode, "情绪陪伴")
        self.assertIn("具体事情", guidance.instruction)

    def test_casual_message_is_not_bedtime(self) -> None:
        guidance = choose_dialogue_guidance("今天中午吃了一家很好吃的面")
        self.assertEqual(guidance.mode, "日常陪伴")

    def test_explicit_goodnight_allows_closing(self) -> None:
        guidance = choose_dialogue_guidance("今天先聊到这里吧，晚安")
        self.assertEqual(guidance.mode, "睡前收尾")

    def test_insomnia_is_emotional_not_closing(self) -> None:
        guidance = choose_dialogue_guidance("我有点焦虑，一直睡不着")
        self.assertEqual(guidance.mode, "情绪陪伴")

    def test_light_command_has_device_mode(self) -> None:
        guidance = choose_dialogue_guidance("把灯光调暗一点", True)
        self.assertEqual(guidance.mode, "灯光控制")


if __name__ == "__main__":
    unittest.main()
