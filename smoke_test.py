from __future__ import annotations

import asyncio

import edge_tts

from companion_demo.bootstrap import build_pc_pipeline
from companion_demo.config import settings


SAMPLE_TEXT = "你好，我今天工作有点累，想早点休息。"
LIGHT_TEXT = "请把灯光调到百分之六十。"


def synthesize_input(text: str, filename: str) -> str:
    input_audio = settings.output_dir / filename
    asyncio.run(
        edge_tts.Communicate(
            text=text,
            voice=settings.tts_voice,
        ).save(str(input_audio))
    )
    return str(input_audio)


def main() -> None:
    pipeline = build_pc_pipeline()
    input_audio = synthesize_input(SAMPLE_TEXT, "smoke-input.mp3")
    (
        transcript,
        reply,
        reply_audio,
        history,
        status,
        brightness,
        light_status,
    ) = pipeline.run(
        input_audio,
        [],
        "小林",
        "喜欢自然、有来有回的聊天",
        35,
    )

    assert transcript, "ASR 没有返回文字"
    assert reply, "LLM 没有返回回复"
    assert reply_audio, "TTS 没有生成音频"
    assert len(history) == 2, "对话历史没有正确更新"
    assert brightness == 35, "普通对话不应改变亮度"
    print(f"识别：{transcript}")
    print(f"回复：{reply}")
    print(light_status)
    print(status)

    light_audio = synthesize_input(LIGHT_TEXT, "smoke-light-input.mp3")
    (
        light_transcript,
        light_reply,
        light_reply_audio,
        _,
        light_run_status,
        adjusted_brightness,
        light_action,
    ) = pipeline.run(
        light_audio,
        history,
        "小林",
        "喜欢自然、有来有回的聊天",
        brightness,
    )
    assert adjusted_brightness == 60, (
        f"语音调光应达到 60%，实际为 {adjusted_brightness}%：{light_transcript}"
    )
    assert light_reply and light_reply_audio, "语音调光回复不完整"
    print(f"调光识别：{light_transcript}")
    print(f"调光回复：{light_reply}")
    print(light_action)
    print(light_run_status)
    print("SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
