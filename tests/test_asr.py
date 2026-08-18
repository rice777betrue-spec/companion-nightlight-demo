from __future__ import annotations

import unittest
from types import SimpleNamespace

from companion_demo.asr import SpeechRecognizer
from companion_demo.core.errors import NoSpeechDetectedError


class _FakeWhisperModel:
    def __init__(self) -> None:
        self.options = {}

    def transcribe(self, _audio_path: str, **options):
        self.options = options
        return iter([SimpleNamespace(text=" 调暗一点。")]), None


class _EmptyWhisperModel:
    def transcribe(self, _audio_path: str, **_options):
        return iter([]), None


class _PromptLeakWhisperModel:
    def transcribe(self, _audio_path: str, **_options):
        return iter([SimpleNamespace(text=" 请准确使用这些词。")]), None


class SpeechRecognizerTests(unittest.TestCase):
    def test_nightlight_vocabulary_is_passed_to_whisper(self) -> None:
        recognizer = SpeechRecognizer("base")
        fake_model = _FakeWhisperModel()
        recognizer._model = fake_model

        transcript = recognizer.transcribe("sample.wav")

        self.assertEqual(transcript, "调暗一点。")
        self.assertIn("调暗", fake_model.options["initial_prompt"])
        self.assertNotIn("请准确使用这些词", fake_model.options["initial_prompt"])
        self.assertIn("关灯", fake_model.options["hotwords"])
        self.assertFalse(fake_model.options["condition_on_previous_text"])

    def test_custom_wake_word_is_passed_to_whisper_immediately(self) -> None:
        recognizer = SpeechRecognizer("base")
        fake_model = _FakeWhisperModel()
        recognizer._model = fake_model
        recognizer.set_wake_word("暖暖")

        recognizer.transcribe("sample.wav")

        self.assertIn("暖暖", fake_model.options["initial_prompt"])
        self.assertIn("暖暖", fake_model.options["hotwords"])
        self.assertNotIn("小夜灯", fake_model.options["hotwords"])

    def test_empty_transcript_is_a_normal_no_speech_signal(self) -> None:
        recognizer = SpeechRecognizer("base")
        recognizer._model = _EmptyWhisperModel()

        with self.assertRaises(NoSpeechDetectedError):
            recognizer.transcribe("sample.wav")

    def test_prompt_leak_is_rejected_as_no_speech(self) -> None:
        recognizer = SpeechRecognizer("base")
        recognizer._model = _PromptLeakWhisperModel()

        with self.assertRaises(NoSpeechDetectedError):
            recognizer.transcribe("sample.wav")


if __name__ == "__main__":
    unittest.main()
