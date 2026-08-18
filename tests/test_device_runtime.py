from __future__ import annotations

import unittest

from companion_demo.runtime import (
    DeviceEvent,
    DeviceRuntime,
    DeviceState,
    InvalidDeviceTransition,
)


class DeviceRuntimeTests(unittest.TestCase):
    def test_complete_device_lifecycle(self) -> None:
        runtime = DeviceRuntime(initial_brightness=35)
        self.assertEqual(runtime.snapshot.state, DeviceState.STARTING)

        runtime.models_ready()
        turn_id = runtime.start_listening(user_id="owner_001")
        self.assertEqual(runtime.snapshot.state, DeviceState.LISTENING)
        runtime.speech_ended(turn_id)
        self.assertEqual(runtime.snapshot.state, DeviceState.THINKING)
        runtime.response_ready(turn_id, 10)
        self.assertEqual(runtime.snapshot.state, DeviceState.SPEAKING)
        self.assertEqual(runtime.snapshot.brightness, 10)
        runtime.playback_finished(turn_id)

        self.assertEqual(runtime.snapshot.state, DeviceState.IDLE)
        self.assertEqual(runtime.snapshot.user_id, "owner_001")

    def test_barge_in_invalidates_the_previous_turn(self) -> None:
        runtime = DeviceRuntime()
        runtime.models_ready()
        first_turn = runtime.start_listening()
        runtime.speech_ended(first_turn)
        runtime.response_ready(first_turn, 35)

        second_turn = runtime.start_listening()

        self.assertGreater(second_turn, first_turn)
        self.assertEqual(runtime.snapshot.state, DeviceState.LISTENING)
        self.assertFalse(runtime.response_ready(first_turn, 0))
        self.assertEqual(runtime.snapshot.brightness, 35)
        self.assertEqual(
            runtime.transition_history[-1].event,
            DeviceEvent.BARGE_IN,
        )

    def test_failure_can_recover_on_the_next_interaction(self) -> None:
        runtime = DeviceRuntime()
        runtime.models_ready()
        turn_id = runtime.start_listening()
        runtime.speech_ended(turn_id)
        runtime.fail("ASR 超时", turn_id)
        self.assertEqual(runtime.snapshot.state, DeviceState.ERROR)
        self.assertIn("ASR 超时", runtime.status_text)

        next_turn = runtime.start_listening()

        self.assertGreater(next_turn, turn_id)
        self.assertEqual(runtime.snapshot.state, DeviceState.LISTENING)
        self.assertIsNone(runtime.snapshot.last_error)

    def test_invalid_transition_is_rejected(self) -> None:
        runtime = DeviceRuntime()
        runtime.models_ready()

        with self.assertRaises(InvalidDeviceTransition):
            runtime.response_ready(0, 35)

    def test_stale_failure_does_not_break_current_turn(self) -> None:
        runtime = DeviceRuntime()
        runtime.models_ready()
        old_turn = runtime.start_listening()
        runtime.speech_ended(old_turn)
        current_turn = runtime.start_listening()

        self.assertFalse(runtime.fail("旧任务报错", old_turn))
        self.assertEqual(runtime.snapshot.turn_id, current_turn)
        self.assertEqual(runtime.snapshot.state, DeviceState.LISTENING)

    def test_cancel_invalidates_in_flight_result(self) -> None:
        runtime = DeviceRuntime()
        runtime.models_ready()
        old_turn = runtime.start_listening()
        runtime.speech_ended(old_turn)

        self.assertTrue(runtime.cancel_current("监听已关闭"))

        self.assertEqual(runtime.snapshot.state, DeviceState.IDLE)
        self.assertFalse(runtime.response_ready(old_turn, 0))
        self.assertEqual(runtime.transition_history[-1].event, DeviceEvent.CANCEL)


if __name__ == "__main__":
    unittest.main()
