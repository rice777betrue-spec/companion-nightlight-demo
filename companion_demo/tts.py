from __future__ import annotations

import asyncio
import base64
import os
import shutil
import subprocess
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
from uuid import uuid4

import edge_tts


class SpeechSynthesizer:
    def __init__(
        self,
        voice: str,
        output_dir: Path,
        engine: str = "sapi",
        sapi_voice: str = "Microsoft Huihui Desktop",
        voxcpm_model: str = "openbmb/VoxCPM-0.5B",
        voxcpm_device: str = "cuda",
        voxcpm_local_files_only: bool = True,
        voxcpm_prompt_wav: str | None = None,
        voxcpm_prompt_text: str | None = None,
        voxcpm_inference_timesteps: int = 6,
    ) -> None:
        self.voice = voice
        self.output_dir = output_dir
        self.engine = engine
        self.sapi_voice = sapi_voice
        self.voxcpm_model = voxcpm_model
        self.voxcpm_device = voxcpm_device
        self.voxcpm_local_files_only = voxcpm_local_files_only
        self.voxcpm_prompt_wav = voxcpm_prompt_wav
        self.voxcpm_prompt_text = voxcpm_prompt_text
        self.voxcpm_inference_timesteps = voxcpm_inference_timesteps
        self._voxcpm: Any | None = None
        self._active_engine = self._engine_name(engine)

    @staticmethod
    def _engine_name(engine: str) -> str:
        labels = {
            "sapi": "SAPI 本地",
            "edge": "Edge 在线",
            "voxcpm": "VoxCPM-0.5B 本地",
        }
        return labels.get(engine, f"未知 TTS（{engine}）")

    @property
    def engine_label(self) -> str:
        return self._active_engine

    @property
    def supports_streaming(self) -> bool:
        return self.engine == "voxcpm"

    def synthesize(self, text: str) -> str:
        if self.engine == "voxcpm":
            try:
                output_path = self._synthesize_voxcpm(text)
                self._active_engine = "VoxCPM-0.5B 本地"
                return output_path
            except Exception as exc:
                self._active_engine = "VoxCPM-0.5B 本地（异常）"
                raise RuntimeError(f"VoxCPM 语音生成失败：{exc}") from exc

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

    def load(self) -> None:
        """预加载实验 TTS，避免首次回答才等待模型权重进入显存。"""
        if self.engine == "voxcpm":
            self._load_voxcpm()

    def _load_voxcpm(self) -> Any:
        import torch
        from voxcpm import VoxCPM

        if self._voxcpm is None:
            if self.voxcpm_device.startswith("cuda"):
                torch.cuda.empty_cache()
            self._voxcpm = VoxCPM.from_pretrained(
                self.voxcpm_model,
                load_denoiser=False,
                local_files_only=self.voxcpm_local_files_only,
                optimize=False,
                device=self.voxcpm_device,
            )
        return self._voxcpm

    @staticmethod
    def _enable_soundfile_prompt_loader() -> None:
        """绕过 Windows TorchCodec/FFmpeg DLL，直接读取普通 WAV 参考音频。"""
        import soundfile as sf
        import torch
        import torchaudio

        def load_wav(path: str) -> tuple[Any, int]:
            samples, sample_rate = sf.read(
                path,
                dtype="float32",
                always_2d=True,
            )
            return torch.from_numpy(samples.T.copy()), sample_rate

        legacy_model = import_module("voxcpm.model.voxcpm")
        legacy_model.torchaudio = SimpleNamespace(
            load=load_wav,
            functional=torchaudio.functional,
        )

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

    def _synthesize_voxcpm(self, text: str) -> str:
        import soundfile as sf

        model = self._load_voxcpm()
        if self.voxcpm_prompt_wav:
            self._enable_soundfile_prompt_loader()

        generate_options: dict[str, Any] = {
            "text": text,
            "cfg_value": 2.0,
            "inference_timesteps": self.voxcpm_inference_timesteps,
            # 默认重试会把一次回答整句重新生成多次，交互延迟不可控。
            "retry_badcase": False,
        }
        if self.voxcpm_prompt_wav:
            if not self.voxcpm_prompt_text:
                raise ValueError("使用 VoxCPM 克隆音色时必须配置参考音频文本")
            generate_options.update(
                prompt_wav_path=self.voxcpm_prompt_wav,
                prompt_text=self.voxcpm_prompt_text,
            )

        wav = model.generate(**generate_options)
        output_path = self.output_dir / f"reply-{uuid4().hex}.wav"
        sf.write(output_path, wav, model.tts_model.sample_rate)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("VoxCPM 没有生成音频文件")
        return str(output_path)

    def synthesize_stream(
        self,
        text: str,
        chunks_per_packet: int = 5,
    ) -> Iterator[tuple[int, Any]]:
        """聚合 VoxCPM 小音频块，供网页在完整语音完成前开始播放。"""
        import numpy as np

        if self.engine != "voxcpm":
            raise RuntimeError("当前 TTS 后端不支持流式合成")
        model = self._load_voxcpm()
        if self.voxcpm_prompt_wav:
            self._enable_soundfile_prompt_loader()
        options: dict[str, Any] = {
            "text": text,
            "cfg_value": 2.0,
            "inference_timesteps": self.voxcpm_inference_timesteps,
            "retry_badcase": False,
        }
        if self.voxcpm_prompt_wav:
            if not self.voxcpm_prompt_text:
                raise ValueError("使用 VoxCPM 克隆音色时必须配置参考音频文本")
            options.update(
                prompt_wav_path=self.voxcpm_prompt_wav,
                prompt_text=self.voxcpm_prompt_text,
            )

        packet: list[Any] = []
        for chunk in model.generate_streaming(**options):
            packet.append(chunk)
            if len(packet) >= chunks_per_packet:
                yield model.tts_model.sample_rate, np.concatenate(packet)
                packet.clear()
        if packet:
            yield model.tts_model.sample_rate, np.concatenate(packet)
        self._active_engine = "VoxCPM-0.5B 本地（流式）"

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
