from __future__ import annotations

import unittest

from companion_demo.light import interpret_light_command, light_confirmation


class LightCommandTests(unittest.TestCase):
    def test_dimmer(self) -> None:
        result = interpret_light_command("灯光暗一点", 50)
        self.assertTrue(result.matched)
        self.assertEqual(result.brightness, 30)

    def test_percentage(self) -> None:
        result = interpret_light_command("把灯光调到百分之六十", 20)
        self.assertTrue(result.matched)
        self.assertEqual(result.brightness, 60)

    def test_traditional_asr_output(self) -> None:
        result = interpret_light_command("請把燈光調到60%", 35)
        self.assertTrue(result.matched)
        self.assertEqual(result.brightness, 60)

    def test_sleep_mode(self) -> None:
        self.assertEqual(
            interpret_light_command("开启睡眠模式", 80).brightness,
            10,
        )

    def test_turn_off(self) -> None:
        self.assertEqual(interpret_light_command("请关灯", 35).brightness, 0)

    def test_unrelated_number_does_not_change_light(self) -> None:
        result = interpret_light_command("我今天工作了八个小时", 35)
        self.assertFalse(result.matched)
        self.assertEqual(result.brightness, 35)

    def test_number_after_relative_command_is_not_used_as_brightness(self) -> None:
        result = interpret_light_command("灯光暗一点，我工作了8小时", 50)
        self.assertTrue(result.matched)
        self.assertEqual(result.brightness, 30)

    def test_half_brightness(self) -> None:
        result = interpret_light_command("把灯光调到一半", 20)
        self.assertTrue(result.matched)
        self.assertEqual(result.brightness, 50)

    def test_spoken_turn_off_variants(self) -> None:
        for command in ("把灯关一下", "关一下灯", "把灯关掉", "把灯关了"):
            with self.subTest(command=command):
                result = interpret_light_command(command, 35)
                self.assertTrue(result.matched)
                self.assertEqual(result.brightness, 0)

    def test_negated_commands_do_not_execute(self) -> None:
        for command in ("不要关灯", "别开灯", "不要调到60%"):
            with self.subTest(command=command):
                result = interpret_light_command(command, 35)
                self.assertTrue(result.intent_detected)
                self.assertFalse(result.matched)
                self.assertEqual(result.brightness, 35)
                self.assertEqual(result.blocked_reason, "cancelled")

    def test_past_detail_does_not_block_a_new_command(self) -> None:
        result = interpret_light_command("我昨天工作很累，帮我关灯", 35)
        self.assertTrue(result.matched)
        self.assertEqual(result.brightness, 0)

    def test_delayed_command_does_not_execute_immediately(self) -> None:
        result = interpret_light_command("十分钟后关灯", 35)
        self.assertTrue(result.intent_detected)
        self.assertFalse(result.matched)
        self.assertEqual(result.brightness, 35)
        self.assertEqual(result.blocked_reason, "delayed")

    def test_relative_spoken_commands(self) -> None:
        cases = (("调高", 70), ("调低", 30), ("柔和一点", 30))
        for command, expected in cases:
            with self.subTest(command=command):
                self.assertEqual(
                    interpret_light_command(command, 50).brightness,
                    expected,
                )

    def test_asr_homophone_for_adjust_to(self) -> None:
        result = interpret_light_command("把灯掉到百分之六十", 35)
        self.assertTrue(result.matched)
        self.assertEqual(result.brightness, 60)

    def test_asr_homophone_for_dimmer(self) -> None:
        result = interpret_light_command("调案一点，我今天很烦", 50)
        self.assertTrue(result.matched)
        self.assertEqual(result.brightness, 30)

    def test_report_of_failed_command_does_not_execute_again(self) -> None:
        result = interpret_light_command("我刚才让它关灯，但是没有反应", 35)
        self.assertTrue(result.intent_detected)
        self.assertFalse(result.matched)
        self.assertEqual(result.brightness, 35)
        self.assertEqual(result.blocked_reason, "reported")

    def test_confirmation_uses_actual_direction(self) -> None:
        brighter = interpret_light_command("调到60%", 15)
        self.assertIn("调亮到 60%", light_confirmation(brighter))

        dimmer = interpret_light_command("调到15%", 60)
        self.assertIn("调暗到 15%", light_confirmation(dimmer))


if __name__ == "__main__":
    unittest.main()
