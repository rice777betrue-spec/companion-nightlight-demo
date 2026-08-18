from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DialogueGuidance:
    mode: str
    instruction: str


_EXPLICIT_CLOSING_PHRASES = (
    "晚安",
    "我要睡了",
    "我先睡了",
    "准备睡觉",
    "去睡觉了",
    "关灯睡觉",
    "先不聊了",
    "明天再聊",
    "我要休息了",
    "我先休息了",
)

_EMOTION_WORDS = (
    "难过",
    "委屈",
    "焦虑",
    "紧张",
    "孤独",
    "寂寞",
    "害怕",
    "生气",
    "烦",
    "累",
    "压力",
    "崩溃",
    "失眠",
    "睡不着",
    "不开心",
    "开心",
    "高兴",
    "兴奋",
    "想哭",
    "心情",
)


def choose_dialogue_guidance(
    user_text: str,
    light_command_matched: bool = False,
) -> DialogueGuidance:
    normalized = re.sub(r"\s+", "", user_text.lower())

    if any(phrase in normalized for phrase in _EXPLICIT_CLOSING_PHRASES):
        return DialogueGuidance(
            mode="睡前收尾",
            instruction=(
                "[对话策略：用户已经明确表示要结束聊天或睡觉。"
                "可以温柔收尾，结合用户刚才说的具体内容回应；不要再连续追问。]"
            ),
        )

    if any(word in normalized for word in _EMOTION_WORDS):
        return DialogueGuidance(
            mode="情绪陪伴",
            instruction=(
                "[对话策略：这是正在进行的情绪陪伴，不是睡前告别。"
                "严禁主动说晚安、早点休息、好梦等结束语。"
                "先准确回应用户话里的具体感受或处境，不要只说空泛安慰；"
                "除非用户主动求建议，否则先倾听，最后自然地问一个与细节有关的问题，让用户愿意继续说。]"
            ),
        )

    if light_command_matched:
        return DialogueGuidance(
            mode="灯光控制",
            instruction=(
                "[对话策略：优先自然确认设备动作，回答简短明确。"
                "不要无故说晚安，也不必为了追问而强行追问。]"
            ),
        )

    return DialogueGuidance(
        mode="日常陪伴",
        instruction=(
            "[对话策略：用户仍想继续聊天。严禁主动说晚安、早点休息、好梦等结束语。"
            "要回应用户刚才提到的具体细节，可以表达轻微而自然的看法，"
            "并在合适时问一个贴近话题的问题；不要像客服，也不要重复套话。]"
        ),
    )

