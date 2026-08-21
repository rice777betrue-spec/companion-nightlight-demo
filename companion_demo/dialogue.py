from __future__ import annotations

import re
from dataclasses import dataclass

from companion_demo.text_normalization import to_simplified_chinese


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

_BEDTIME_CONFIRMATION_PHRASES = (
    "我要睡了",
    "我要睡觉了",
    "我先睡了",
    "我想睡了",
    "我想睡觉了",
    "我准备睡觉",
    "我准备睡了",
    "准备睡觉",
    "准备睡了",
    "准备去睡觉",
    "准备上床睡觉",
    "我去睡觉了",
    "我去睡了",
    "要去睡觉了",
    "要去睡了",
    "该睡觉了",
    "该睡了",
    "该休息了",
    "我要休息了",
    "我先休息了",
    "准备休息了",
    "我去休息了",
)

_BEDTIME_HISTORY_CUES = (
    "昨天",
    "前天",
    "以前",
    "曾经",
    "上次",
    "那天",
    "小时候",
)

_BEDTIME_CURRENT_CUES = (
    "现在",
    "今晚",
    "这就",
    "马上",
    "准备",
)


def _compact_text(value: str) -> str:
    simplified = to_simplified_chinese(str(value or "")).casefold()
    return re.sub(r"[\s，,。.!！?？；;：:、]+", "", simplified)


def needs_sleep_mode_confirmation(user_text: str) -> bool:
    """判断用户是否正在表达马上睡觉，而不是谈论睡眠或回顾往事。"""

    simplified = to_simplified_chinese(str(user_text or "")).casefold()
    clauses = re.split(r"[，,。.!！?？；;：:\n]+", simplified)
    for raw_clause in clauses:
        clause = re.sub(r"\s+", "", raw_clause)
        if not clause or not any(
            phrase in clause for phrase in _BEDTIME_CONFIRMATION_PHRASES
        ):
            continue
        if any(
            cue in clause for cue in ("睡不着", "失眠", "还不睡", "熬夜")
        ):
            continue
        if re.search(r"(?:不|没|别|无需|不用).{0,5}(?:睡|休息)", clause):
            continue
        if any(cue in clause for cue in ("什么时候", "几点", "是不是")):
            continue
        if raw_clause.rstrip().endswith(("吗", "么", "嘛")):
            continue
        if (
            any(cue in clause for cue in _BEDTIME_HISTORY_CUES)
            and not any(cue in clause for cue in _BEDTIME_CURRENT_CUES)
        ):
            continue
        return True
    return False


def classify_sleep_mode_confirmation(user_text: str) -> str | None:
    """把待确认轮次的简短回答归类为 affirmative 或 negative。"""

    normalized = _compact_text(user_text)
    if not normalized:
        return None

    negative_cues = (
        "不用",
        "不要",
        "不需要",
        "不必",
        "无需",
        "算了",
        "取消",
        "先别",
        "别开",
        "保持现在",
        "保持原样",
        "保持不变",
    )
    if normalized in {"不", "否", "不是", "不用了", "不了"} or any(
        cue in normalized for cue in negative_cues
    ):
        return "negative"
    if re.search(r"(?:不|别).{0,5}(?:开|开启|调灯)", normalized):
        return "negative"

    affirmative_exact = {
        "要",
        "要的",
        "好",
        "好的",
        "好啊",
        "好呀",
        "可以",
        "可以的",
        "行",
        "行啊",
        "行吧",
        "嗯",
        "嗯嗯",
        "是",
        "是的",
        "需要",
        "开吧",
        "开启吧",
        "打开吧",
    }
    if normalized in affirmative_exact:
        return "affirmative"
    if len(normalized) <= 18 and any(
        cue in normalized
        for cue in (
            "帮我开",
            "帮我开启",
            "给我开",
            "就这么做",
            "开启睡眠模式",
            "打开睡眠模式",
        )
    ):
        return "affirmative"
    if re.fullmatch(r"(?:嗯+)?(?:好(?:的)?|要)(?:谢谢)?", normalized):
        return "affirmative"
    return None


def choose_dialogue_guidance(
    user_text: str,
    light_command_matched: bool = False,
) -> DialogueGuidance:
    normalized = re.sub(r"\s+", "", user_text.lower())

    if any(phrase in normalized for phrase in _EXPLICIT_CLOSING_PHRASES):
        return DialogueGuidance(
            mode="睡前收尾",
            instruction=(
                "用户明确结束聊天或准备睡觉，可以结合原话温柔收尾，不再追问。"
            ),
        )

    if any(word in normalized for word in _EMOTION_WORDS):
        return DialogueGuidance(
            mode="情绪陪伴",
            instruction=(
                "先回应造成情绪的具体事情；没有明确求建议时先倾听，不强行收尾。"
            ),
        )

    if light_command_matched:
        return DialogueGuidance(
            mode="灯光控制",
            instruction=(
                "由设备层确认灯光动作，回复简短明确。"
            ),
        )

    return DialogueGuidance(
        mode="日常陪伴",
        instruction=(
            "直接回应用户最后一句的具体事实；有必要时才问一个紧扣原话的问题。"
        ),
    )
