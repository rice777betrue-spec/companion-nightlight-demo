from __future__ import annotations

import html

import gradio as gr

from companion_demo.bootstrap import build_pc_hands_free, build_pc_pipeline
from companion_demo.config import settings
from companion_demo.core.errors import NoSpeechDetectedError
from companion_demo.runtime import DeviceState


pipeline = build_pc_pipeline()
hands_free = build_pc_hands_free(pipeline)
DEFAULT_USER_NAME = "小林"
DEFAULT_PREFERENCES = (
    "喜欢自然、有来有回的聊天；先听我说，不要每句话都劝我休息"
)


def _lamp_orb(brightness: int, label: str) -> str:
    level = max(0, min(100, int(brightness)))
    intensity = level / 100
    core_alpha = 0.12 + intensity * 0.88
    glow_alpha = 0.02 + intensity * 0.62
    glow_size = 2 + int(intensity * 32)
    return f"""
      <div style="min-width:132px;text-align:center;">
        <div style="height:112px;display:flex;align-items:center;justify-content:center;">
          <div style="width:72px;height:72px;border-radius:50%;
                      background:radial-gradient(circle,
                        rgba(255,255,231,{core_alpha:.2f}) 0%,
                        rgba(255,194,83,{core_alpha:.2f}) 52%,
                        rgba(255,119,43,{core_alpha * .75:.2f}) 100%);
                      border:1px solid rgba(255,218,145,{0.12 + intensity * .48:.2f});
                      box-shadow:0 0 {glow_size}px {max(1, glow_size // 3)}px
                        rgba(255,167,65,{glow_alpha:.2f});"></div>
        </div>
        <div style="font-size:13px;color:#a99278;">{html.escape(label)}</div>
        <div style="font-size:20px;font-weight:700;color:#ffe2aa;">{level}%</div>
      </div>
    """


def lamp_html(
    message: str = "等待你说话",
    previous: int | float = 35,
    brightness: int | float = 35,
) -> str:
    previous_level = max(0, min(100, int(previous)))
    current_level = max(0, min(100, int(brightness)))
    return f"""
    <div style="padding:18px;border-radius:18px;background:#17130f;
                border:1px solid #3f3022;">
      <div style="font-size:17px;font-weight:650;color:#ffe3b1;">灯光明暗对比</div>
      <div style="display:flex;justify-content:center;align-items:center;gap:18px;">
        {_lamp_orb(previous_level, "调整前")}
        <div style="font-size:26px;color:#8d6c49;">→</div>
        {_lamp_orb(current_level, "当前")}
      </div>
      <div style="height:8px;border-radius:4px;background:#33271d;overflow:hidden;">
        <div style="width:{current_level}%;height:100%;border-radius:4px;
                    background:linear-gradient(90deg,#a75b25,#ffd27a);"></div>
      </div>
      <div style="margin-top:12px;color:#c8ae90;text-align:center;">
        {html.escape(message)}
      </div>
    </div>
    """


def run_turn(
    audio_path: str | None,
    user_name: str,
    preferences: str,
    history: list[dict[str, str]] | None,
    brightness: int | float,
):
    current = max(0, min(100, int(brightness)))
    turn_id: int | None = None
    if not audio_path:
        yield (
            "",
            "",
            None,
            lamp_html("请先录一段话", current, current),
            history or [],
            "未完成：请先录一段话。",
            current,
            current,
            f"灯光保持在 {current}%",
            pipeline.device_runtime.status_text,
        )
        return

    hands_free.interrupt_playback()
    try:
        turn_id = pipeline.device_runtime.start_listening(
            current,
            user_name.strip() or None,
        )
        pipeline.device_runtime.speech_ended(turn_id)
        yield (
            "",
            "",
            None,
            lamp_html("正在识别并思考…", current, current),
            history or [],
            "正在识别语音…",
            current,
            current,
            f"灯光保持在 {current}%",
            pipeline.device_runtime.status_text,
        )

        (
            transcript,
            reply,
            new_history,
            status,
            next_brightness,
            light_status,
        ) = pipeline.generate_reply(
            audio_path or "", history, user_name, preferences, brightness
        )
        if not pipeline.device_runtime.response_ready(
            turn_id,
            next_brightness,
        ):
            return
        hands_free.configure(history=new_history)
        yield (
            transcript,
            reply,
            None,
            lamp_html(reply, brightness, next_brightness),
            new_history,
            status,
            next_brightness,
            next_brightness,
            light_status,
            pipeline.device_runtime.status_text,
        )
        audio_reply, tts_status = pipeline.synthesize_reply(reply)
        final_status = (
            f"{status.replace('｜正在生成语音…', '')}｜{tts_status}"
        )
        if not pipeline.device_runtime.is_current(turn_id):
            return
        if audio_reply is None:
            pipeline.device_runtime.playback_finished(turn_id)
        yield (
            transcript,
            reply,
            audio_reply,
            lamp_html(reply, brightness, next_brightness),
            new_history,
            final_status,
            next_brightness,
            next_brightness,
            light_status,
            pipeline.device_runtime.status_text,
        )
    except NoSpeechDetectedError as exc:
        if turn_id is not None and pipeline.device_runtime.is_current(turn_id):
            pipeline.device_runtime.cancel_current("没有听清，返回待机")
        yield (
            "",
            "",
            None,
            lamp_html("没有听清，请靠近后再说一次", current, current),
            history or [],
            f"本轮没有听清：{exc}",
            current,
            current,
            f"灯光保持在 {current}%",
            pipeline.device_runtime.status_text,
        )
    except Exception as exc:
        pipeline.device_runtime.fail(str(exc), turn_id)
        yield (
            "",
            "",
            None,
            lamp_html("暂时没有听清", current, current),
            history or [],
            f"未完成：{exc}",
            current,
            current,
            f"灯光保持在 {current}%",
            pipeline.device_runtime.status_text,
        )


def model_status() -> tuple[str, str]:
    return pipeline.warmup_status, pipeline.device_runtime.status_text


def set_hands_free_mode(
    enabled: bool,
    user_name: str,
    preferences: str,
    history: list[dict[str, str]] | None,
    sensitivity: float,
) -> tuple[str, str]:
    hands_free.configure(
        user_name=user_name,
        preferences=preferences,
        history=history or [],
    )
    hands_free.set_sensitivity(sensitivity)
    try:
        if enabled:
            hands_free.start()
        else:
            hands_free.stop()
    except Exception as exc:
        return (
            f"免按键启动失败：{exc}",
            pipeline.device_runtime.status_text,
        )
    return hands_free.status_text, pipeline.device_runtime.status_text


def update_hands_free_profile(
    user_name: str,
    preferences: str,
) -> str:
    hands_free.configure(user_name=user_name, preferences=preferences)
    return hands_free.status_text


def update_vad_sensitivity(value: float) -> str:
    hands_free.set_sensitivity(value)
    return hands_free.status_text


def poll_runtime(last_result_version: int | float | None):
    snapshot = hands_free.snapshot
    current_version = int(last_result_version or 0)
    common = (
        pipeline.warmup_status,
        pipeline.device_runtime.status_text,
        hands_free.status_text,
    )
    if snapshot.result_version == current_version:
        return common + (gr.skip(),) * 8 + (current_version,)

    message = snapshot.reply or snapshot.status
    return common + (
        snapshot.transcript,
        snapshot.reply,
        lamp_html(
            message,
            snapshot.previous_brightness,
            snapshot.brightness,
        ),
        snapshot.history_messages,
        snapshot.result_status,
        snapshot.brightness,
        snapshot.brightness,
        snapshot.light_status,
        snapshot.result_version,
    )


def finish_playback() -> str:
    snapshot = pipeline.device_runtime.snapshot
    if snapshot.state == DeviceState.SPEAKING:
        pipeline.device_runtime.playback_finished(snapshot.turn_id)
    return pipeline.device_runtime.status_text


def manual_brightness(new_value: int | float, previous: int | float):
    old_level = max(0, min(100, int(previous)))
    new_level = max(0, min(100, int(new_value)))
    pipeline.device_runtime.set_brightness(new_level)
    description = f"手动调光：{old_level}% → {new_level}%"
    return lamp_html(description, old_level, new_level), new_level, description


def apply_preset(target: int, previous: int | float):
    old_level = max(0, min(100, int(previous)))
    pipeline.device_runtime.set_brightness(target)
    description = f"预设调光：{old_level}% → {target}%"
    return (
        target,
        lamp_html(description, old_level, target),
        target,
        description,
    )


def preset_sleep(previous):
    return apply_preset(10, previous)


def preset_soft(previous):
    return apply_preset(35, previous)


def preset_read(previous):
    return apply_preset(70, previous)


def preset_full(previous):
    return apply_preset(100, previous)


with gr.Blocks(title="陪伴小夜灯 Demo") as demo:
    gr.Markdown(
        "# 陪伴小夜灯 · 电脑 Demo\n"
        "开启一次免按键模式后，直接对它说话即可；它会自动判断一句话结束、"
        "回答、播报，也能完成灯光调节。"
    )
    history_state = gr.State([])
    brightness_state = gr.State(35)
    hands_free_version = gr.State(0)

    with gr.Group():
        with gr.Row():
            hands_free_toggle = gr.Checkbox(
                label="免按键连续对话",
                value=settings.hands_free_auto_start,
                info="打开一次后无需再点回应按钮",
            )
            vad_sensitivity = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="拾音灵敏度",
                info="漏听时调高，环境噪声误触发时调低",
            )
        hands_free_status = gr.Textbox(
            label="免按键监听状态",
            value=hands_free.status_text,
            interactive=False,
        )
        gr.Markdown(
            "电脑基础版使用 WebRTC VAD + 自适应能量门限，开启后先保持安静约 1 秒校准噪声。"
            "回答播放时可以说话打断；若扬声器回声造成误触发，可先戴耳机测试，"
            "设备版再接入 AEC 回声消除。"
        )

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Accordion("手动录音 / 上传（调试备用）", open=False):
                gr.Markdown(
                    "这个控件使用浏览器麦克风，和上方后台 Realtek 麦克风相互独立。"
                    "内置浏览器即使显示“找不到麦克风”，也不影响免按键模式。"
                )
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    format="wav",
                    label="录制或上传音频",
                )
            with gr.Row():
                user_name = gr.Textbox(
                    label="你的称呼", value=DEFAULT_USER_NAME
                )
                preferences = gr.Textbox(
                    label="偏好或背景",
                    value=DEFAULT_PREFERENCES,
                )
            submit = gr.Button("手动让小夜灯回应", variant="secondary")
            model_state = gr.Textbox(
                label="模型状态",
                value="服务启动后将自动后台预热",
                interactive=False,
            )
            device_state = gr.Textbox(
                label="设备状态机",
                value=pipeline.device_runtime.status_text,
                interactive=False,
            )
            status = gr.Textbox(label="运行状态", interactive=False)

        with gr.Column(scale=2):
            lamp = gr.HTML(lamp_html())
            brightness = gr.Slider(
                minimum=0,
                maximum=100,
                value=35,
                step=5,
                label="灯光亮度",
            )
            with gr.Row():
                sleep_button = gr.Button("助眠 10%", size="sm")
                soft_button = gr.Button("柔光 35%", size="sm")
                read_button = gr.Button("阅读 70%", size="sm")
                full_button = gr.Button("全亮 100%", size="sm")
            light_status = gr.Textbox(
                label="灯光动作",
                value="灯光保持在 35%",
                interactive=False,
            )

    with gr.Row():
        transcript_box = gr.Textbox(label="识别结果", lines=3)
        reply_box = gr.Textbox(label="小夜灯回复", lines=3)

    audio_output = gr.Audio(label="手动模式回复音频", autoplay=True)
    gr.Markdown(
        "Whisper 与 Qwen 会在启动后自动从本地缓存预热。文字回答先显示，"
        "本地中文语音生成完成后再自动播放；免按键模式直接使用电脑扬声器。"
    )

    runtime_timer = gr.Timer(value=0.4)
    runtime_timer.tick(
        fn=poll_runtime,
        inputs=[hands_free_version],
        outputs=[
            model_state,
            device_state,
            hands_free_status,
            transcript_box,
            reply_box,
            lamp,
            history_state,
            status,
            brightness,
            brightness_state,
            light_status,
            hands_free_version,
        ],
        queue=False,
    )

    hands_free_toggle.change(
        fn=set_hands_free_mode,
        inputs=[
            hands_free_toggle,
            user_name,
            preferences,
            history_state,
            vad_sensitivity,
        ],
        outputs=[hands_free_status, device_state],
        queue=False,
    )
    user_name.change(
        fn=update_hands_free_profile,
        inputs=[user_name, preferences],
        outputs=[hands_free_status],
        queue=False,
    )
    preferences.change(
        fn=update_hands_free_profile,
        inputs=[user_name, preferences],
        outputs=[hands_free_status],
        queue=False,
    )
    vad_sensitivity.input(
        fn=update_vad_sensitivity,
        inputs=[vad_sensitivity],
        outputs=[hands_free_status],
        queue=False,
    )

    submit.click(
        fn=run_turn,
        inputs=[
            audio_input,
            user_name,
            preferences,
            history_state,
            brightness,
        ],
        outputs=[
            transcript_box,
            reply_box,
            audio_output,
            lamp,
            history_state,
            status,
            brightness,
            brightness_state,
            light_status,
            device_state,
        ],
    )

    audio_output.stop(
        fn=finish_playback,
        outputs=[device_state],
        queue=False,
    )
    audio_output.pause(
        fn=finish_playback,
        outputs=[device_state],
        queue=False,
    )

    brightness.input(
        fn=manual_brightness,
        inputs=[brightness, brightness_state],
        outputs=[lamp, brightness_state, light_status],
        queue=False,
    )
    for button, handler in (
        (sleep_button, preset_sleep),
        (soft_button, preset_soft),
        (read_button, preset_read),
        (full_button, preset_full),
    ):
        button.click(
            fn=handler,
            inputs=[brightness_state],
            outputs=[brightness, lamp, brightness_state, light_status],
            queue=False,
        )


if __name__ == "__main__":
    pipeline.start_warmup()
    hands_free.configure(
        user_name=DEFAULT_USER_NAME,
        preferences=DEFAULT_PREFERENCES,
    )
    if settings.hands_free_auto_start:
        try:
            hands_free.start()
        except Exception:
            pass
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=settings.server_port,
        inbrowser=True,
        show_error=True,
        theme=gr.themes.Soft(),
    )
