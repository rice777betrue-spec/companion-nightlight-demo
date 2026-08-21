from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

from companion_demo.config import settings


def main() -> None:
    print(f"[1/2] 准备语音识别模型：{settings.asr_model}")
    WhisperModel(settings.asr_model, device="cpu", compute_type="int8")

    print(f"[2/2] 准备陪伴模型：{settings.llm_model}")
    local_model = Path(settings.llm_model)
    if local_model.is_dir():
        print(f"已找到本地模型：{local_model.resolve()}")
    else:
        snapshot_download(repo_id=settings.llm_model)
    print("模型准备完成。")


if __name__ == "__main__":
    main()
