from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    output_dir: Path = PROJECT_ROOT / "outputs"
    asr_model: str = os.getenv("ASR_MODEL", "base")
    asr_device: str = os.getenv("ASR_DEVICE", "cpu")
    llm_model: str = os.getenv(
        "LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"
    )
    tts_engine: str = os.getenv("TTS_ENGINE", "sapi").strip().lower()
    tts_voice: str = os.getenv("TTS_VOICE", "zh-CN-XiaoxiaoNeural")
    sapi_voice: str = os.getenv("SAPI_VOICE", "Microsoft Huihui Desktop")
    server_port: int = int(os.getenv("SERVER_PORT", "7860"))
    model_offline: bool = _env_flag("MODEL_OFFLINE", True)
    hands_free_auto_start: bool = _env_flag("HANDS_FREE_AUTO_START", False)
    audio_input_device: str | None = (
        os.getenv("AUDIO_INPUT_DEVICE", "").strip() or None
    )


settings = Settings()
settings.output_dir.mkdir(parents=True, exist_ok=True)
