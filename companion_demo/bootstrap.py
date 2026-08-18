from __future__ import annotations

from companion_demo.adapters.pc import (
    FasterWhisperAdapter,
    LocalQwenAdapter,
    LocalSpeechSynthesizerAdapter,
    LocalVoiceprintAdapter,
    SoundDeviceAudioInput,
    VirtualLightDriver,
    WebRtcEnergyVad,
    WindowsWavePlayer,
)
from companion_demo.config import Settings, settings
from companion_demo.pipeline import DemoPipeline
from companion_demo.runtime import HandsFreeRuntime


def build_pc_pipeline(config: Settings = settings) -> DemoPipeline:
    """组合当前电脑端适配器；未来设备端将提供独立组合入口。"""

    return DemoPipeline(
        asr=FasterWhisperAdapter(
            config.asr_model,
            config.asr_device,
            local_files_only=config.model_offline,
        ),
        companion=LocalQwenAdapter(
            config.llm_model,
            local_files_only=config.model_offline,
        ),
        tts=LocalSpeechSynthesizerAdapter(
            config.tts_voice,
            config.output_dir,
            engine=config.tts_engine,
            sapi_voice=config.sapi_voice,
        ),
        light_driver=VirtualLightDriver(),
        speaker_verifier=LocalVoiceprintAdapter(
            config.voiceprint_profile_path,
            threshold=config.voiceprint_threshold,
            required_samples=config.voiceprint_required_samples,
        ),
    )


def build_pc_hands_free(
    pipeline: DemoPipeline,
    config: Settings = settings,
) -> HandsFreeRuntime:
    """组合电脑端持续麦克风、VAD 和扬声器；核心层不依赖 Gradio。"""

    raw_device = config.audio_input_device
    input_device: str | int | None = raw_device
    if raw_device and raw_device.isdigit():
        input_device = int(raw_device)
    return HandsFreeRuntime(
        audio_input=SoundDeviceAudioInput(device=input_device),
        vad=WebRtcEnergyVad(),
        player=WindowsWavePlayer(),
        device_runtime=pipeline.device_runtime,
        handle_turn=pipeline.handle_turn,
        synthesize_reply=pipeline.synthesize_reply,
        output_dir=config.output_dir,
    )
