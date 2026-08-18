from __future__ import annotations

import asyncio
import base64
import os
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import edge_tts


class SpeechSynthesizer:
    def __init__(
        self,
        voice: str,
        output_dir: Path,
        engine: str = "sapi",
        sapi_voice: str = "Microsoft Huihui Desktop",
    ) -> None:
        self.voice = voice
        self.output_dir = output_dir
        self.engine = engine
        self.sapi_voice = sapi_voice
        self._active_engine = self._engine_name(engine)

    @staticmethod
    def _engine_name(engine: str) -> str:
        return "SAPI 本地" if engine == "sapi" else "Edge 在线"

    @property
    def engine_label(self) -> str:
        return self._active_engine

    def synthesize(self, text: str) -> str:
        if self.engine == "sapi":
            try:
                output_path = self._synthesize_sapi(text)
                self._active_engine = "SAPI 本地"
                return output_path
            except Exception:
                # 本地语音不可用时自动退回原有 Edge TTS，保证仍然能发声。
                output_path = self._synthesize_edge(text)
                self._active_engine = "Edge 在线（本地降级）"
                return output_path

        output_path = self._synthesize_edge(text)
        self._active_engine = "Edge 在线"
        return output_path

    def _synthesize_edge(self, text: str) -> str:
        output_path = self.output_dir / f"reply-{uuid4().hex}.mp3"
        asyncio.run(
            edge_tts.Communicate(
                text=text,
                voice=self.voice,
                rate="-5%",
                volume="+0%",
            ).save(str(output_path))
        )
        return str(output_path)

    def _synthesize_sapi(self, text: str) -> str:
        shell = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
        if not shell:
            raise RuntimeError("找不到 PowerShell，无法调用 Windows 本地语音")

        output_path = self.output_dir / f"reply-{uuid4().hex}.wav"
        environment = os.environ.copy()
        environment["NIGHTLIGHT_TTS_TEXT"] = base64.b64encode(
            text.encode("utf-8")
        ).decode("ascii")
        environment["NIGHTLIGHT_TTS_OUTPUT"] = str(output_path)
        environment["NIGHTLIGHT_TTS_VOICE"] = self.sapi_voice

        script = """
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$encodedText = [Environment]::GetEnvironmentVariable('NIGHTLIGHT_TTS_TEXT')
$text = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($encodedText))
$outputPath = [Environment]::GetEnvironmentVariable('NIGHTLIGHT_TTS_OUTPUT')
$voiceName = [Environment]::GetEnvironmentVariable('NIGHTLIGHT_TTS_VOICE')
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $speaker.SelectVoice($voiceName)
    $speaker.Rate = -1
    $speaker.Volume = 100
    $speaker.SetOutputToWaveFile($outputPath)
    $speaker.Speak($text)
}
finally {
    $speaker.Dispose()
}
"""
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.run(
            [shell, "-NoProfile", "-NonInteractive", "-Command", script],
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=True,
            creationflags=creation_flags,
        )
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Windows 本地语音没有生成音频文件")
        return str(output_path)
