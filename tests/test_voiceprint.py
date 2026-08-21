from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from companion_demo.adapters.pc.light import VirtualLightDriver
from companion_demo.adapters.pc.voiceprint import LocalVoiceprintAdapter
from companion_demo.core.contracts import SpeakerVerification, TurnRequest
from companion_demo.runtime import TurnEngine


def _write_voice(
    path: Path,
    fundamental: float,
    *,
    phase: float = 0.0,
    noise_seed: int = 0,
) -> None:
    sample_rate = 16_000
    seconds = 2.6
    time_axis = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    envelope = 0.55 + 0.32 * np.sin(2 * np.pi * 2.1 * time_axis + phase)
    signal = np.zeros_like(time_axis)
    for harmonic, amplitude in ((1, 1.0), (2, 0.48), (3, 0.27), (5, 0.13)):
        signal += amplitude * np.sin(
            2 * np.pi * fundamental * harmonic * time_axis + phase * harmonic
        )
    signal *= envelope
    rng = np.random.default_rng(noise_seed)
    signal += rng.normal(0.0, 0.008, signal.shape).astype(np.float32)
    signal /= max(float(np.max(np.abs(signal))), 1e-6)
    pcm = np.clip(signal * 22_000, -32_768, 32_767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


class _FakeASR:
    def transcribe(self, _audio_path: str) -> str:
        return "我今天心情不太好"


class _RecordingCompanion:
    def __init__(self) -> None:
        self.args = None

    def reply(self, *args) -> str:
        self.args = args
        return "我在听，你可以慢慢说。"


class _GuestVerifier:
    def verify(self, _audio_path: str) -> SpeakerVerification:
        return SpeakerVerification(
            identity="guest",
            enrolled=True,
            is_owner=False,
            score=0.51,
            threshold=0.82,
            sample_count=3,
            status="声纹：访客",
        )


class VoiceprintAdapterTests(unittest.TestCase):
    def test_enroll_verify_persist_and_clear(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            owner_samples = []
            for index, phase in enumerate((0.0, 0.14, 0.27, 0.4)):
                path = root / f"owner-{index}.wav"
                _write_voice(path, 138.0, phase=phase, noise_seed=index)
                owner_samples.append(path)
            guest = root / "guest.wav"
            _write_voice(guest, 232.0, phase=0.2, noise_seed=20)
            profile = root / "voiceprint" / "owner.npz"
            adapter = LocalVoiceprintAdapter(
                profile,
                threshold=0.82,
                required_samples=3,
            )

            first = adapter.enroll(str(owner_samples[0]), "小林")
            with self.assertRaisesRegex(ValueError, "已经录入"):
                adapter.enroll(str(owner_samples[0]), "小林")
            second = adapter.enroll(str(owner_samples[1]), "小林")
            third = adapter.enroll(str(owner_samples[2]), "小林")

            self.assertFalse(first.ready)
            self.assertFalse(second.ready)
            self.assertTrue(third.ready)
            self.assertTrue(profile.is_file())

            restored = LocalVoiceprintAdapter(
                profile,
                threshold=0.82,
                required_samples=3,
            )
            owner_result = restored.verify(str(owner_samples[3]))
            guest_result = restored.verify(str(guest))

            self.assertTrue(owner_result.is_owner)
            self.assertFalse(guest_result.is_owner)
            self.assertGreater(owner_result.score, guest_result.score)
            self.assertIn("主人", owner_result.status)

            status = restored.clear()
            self.assertFalse(profile.exists())
            self.assertIn("已删除", status)


class VoiceprintPrivacyTests(unittest.TestCase):
    def test_guest_cannot_receive_owner_profile_or_history(self) -> None:
        companion = _RecordingCompanion()
        engine = TurnEngine(
            _FakeASR(),
            companion,
            VirtualLightDriver(),
            _GuestVerifier(),
        )
        private_history = [
            {"role": "user", "content": "只有主人知道的旧对话"},
            {"role": "assistant", "content": "这是私密回复"},
        ]

        result = engine.handle(
            TurnRequest(
                "guest.wav",
                history=private_history,
                user_name="小林",
                preferences="主人喜欢安静",
            )
        )

        self.assertIsNotNone(companion.args)
        model_input, model_history, user_name, preferences = companion.args
        self.assertEqual(model_input, "我今天心情不太好")
        self.assertEqual(model_history, [])
        self.assertEqual(user_name, "")
        self.assertEqual(preferences, "")
        self.assertEqual(result.history, private_history)
        self.assertEqual(result.speaker_verification.identity, "guest")


if __name__ == "__main__":
    unittest.main()
