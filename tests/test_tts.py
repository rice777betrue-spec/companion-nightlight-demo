from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from companion_demo.tts import SpeechSynthesizer


class SpeechSynthesizerTests(TestCase):
    def _synthesizer(self, output_dir: Path) -> SpeechSynthesizer:
        return SpeechSynthesizer(
            "zh-CN-XiaoxiaoNeural",
            output_dir,
            engine="voxcpm",
        )

    def test_voxcpm_engine_is_selected(self) -> None:
        with TemporaryDirectory() as directory:
            synthesizer = self._synthesizer(Path(directory))
            expected = str(Path(directory) / "reply.wav")
            with patch.object(
                synthesizer,
                "_synthesize_voxcpm",
                return_value=expected,
            ):
                self.assertEqual(synthesizer.synthesize("你好"), expected)
            self.assertEqual(synthesizer.engine_label, "VoxCPM-0.5B 本地")

    def test_voxcpm_failure_falls_back_to_sapi(self) -> None:
        with TemporaryDirectory() as directory:
            synthesizer = self._synthesizer(Path(directory))
            expected = str(Path(directory) / "fallback.wav")
            with (
                patch.object(
                    synthesizer,
                    "_synthesize_voxcpm",
                    side_effect=RuntimeError("out of memory"),
                ),
                patch.object(
                    synthesizer,
                    "_synthesize_sapi",
                    return_value=expected,
                ),
            ):
                self.assertEqual(synthesizer.synthesize("你好"), expected)
            self.assertEqual(
                synthesizer.engine_label,
                "SAPI 本地（VoxCPM 降级）",
            )
