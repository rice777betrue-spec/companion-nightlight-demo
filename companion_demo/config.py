from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
LOCAL_WHISPER_SMALL = (
    PROJECT_ROOT / ".cache" / "models" / "faster-whisper-small"
)
LOCAL_QWEN_3B = PROJECT_ROOT / ".cache" / "models" / "Qwen2.5-3B-Instruct"
WHISPER_SMALL_READY = (
    (LOCAL_WHISPER_SMALL / "model.bin").is_file()
    and (LOCAL_WHISPER_SMALL / "model.bin").stat().st_size == 483_546_902
)
DEFAULT_ASR_MODEL = (
    str(LOCAL_WHISPER_SMALL) if WHISPER_SMALL_READY else "base"
)
DEFAULT_ASR_DEVICE = "cuda" if WHISPER_SMALL_READY else "cpu"
DEFAULT_LLM_MODEL = (
    str(LOCAL_QWEN_3B)
    if (LOCAL_QWEN_3B / "model.safetensors.index.json").is_file()
    else "Qwen/Qwen2.5-3B-Instruct"
)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    output_dir: Path = PROJECT_ROOT / "outputs"
    asr_model: str = os.getenv("ASR_MODEL", DEFAULT_ASR_MODEL)
    asr_device: str = os.getenv("ASR_DEVICE", DEFAULT_ASR_DEVICE)
    llm_model: str = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
    llm_quantization: str = os.getenv(
        "LLM_QUANTIZATION", "4bit"
    ).strip().lower()
    tts_engine: str = os.getenv("TTS_ENGINE", "sapi").strip().lower()
    tts_voice: str = os.getenv("TTS_VOICE", "zh-CN-XiaoxiaoNeural")
    sapi_voice: str = os.getenv("SAPI_VOICE", "Microsoft Huihui Desktop")
    server_port: int = int(os.getenv("SERVER_PORT", "7860"))
    model_offline: bool = _env_flag("MODEL_OFFLINE", True)
    hands_free_auto_start: bool = _env_flag("HANDS_FREE_AUTO_START", False)
    audio_input_device: str | None = (
        os.getenv("AUDIO_INPUT_DEVICE", "").strip() or None
    )
    voiceprint_profile_path: Path = (
        PROJECT_ROOT / ".cache" / "voiceprint" / "owner_voiceprint.npz"
    )
    voiceprint_threshold: float = float(
        os.getenv("VOICEPRINT_THRESHOLD", "0.82")
    )
    voiceprint_required_samples: int = int(
        os.getenv("VOICEPRINT_REQUIRED_SAMPLES", "3")
    )
    wake_word_path: Path = (
        PROJECT_ROOT / ".cache" / "device" / "wake_word.json"
    )
    wake_word: str = os.getenv("WAKE_WORD", "小夜灯").strip() or "小夜灯"
    wake_session_seconds: float = float(
        os.getenv("WAKE_SESSION_SECONDS", "30")
    )


settings = Settings()
settings.output_dir.mkdir(parents=True, exist_ok=True)
