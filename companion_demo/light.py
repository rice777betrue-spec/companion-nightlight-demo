from __future__ import annotations

import re
from dataclasses import dataclass


_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

_TRADITIONAL_TO_SIMPLIFIED = str.maketrans(
    {
        "燈": "灯",
        "調": "调",
        "關": "关",
        "開": "开",
        "閉": "闭",
        "設": "设",
        "為": "为",
        "線": "线",
        "點": "点",
        "讀": "读",
        "書": "书",
        "啟": "启",
        "滅": "灭",
    }
)

_CHINESE_NUMBER = r"[零〇一二两三四五六七八九十百]+"
_NUMBER = rf"(?:\d{{1,3}}|{_CHINESE_NUMBER})"
_CLAUSE_SEPARATOR = re.compile(r"[，,。.!！?？；;：:\n]+")


@dataclass(frozen=True)
class LightAdjustment:
    previous: int
    brightness: int
    matched: bool
    description: str
    intent_detected: bool = False
    blocked_reason: str | None = None
    action: str | None = None


def _clamp(value: int | float) -> int:
    return max(0, min(100, int(round(value))))


def _parse_chinese_number(value: str) -> int | None:
    if value == "百" or value == "一百":
        return 100
    if "十" in value:
        tens_text, ones_text = value.split("十", maxsplit=1)
        tens = 1 if not tens_text else _CN_DIGITS.get(tens_text)
        ones = 0 if not ones_text else _CN_DIGITS.get(ones_text)
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    if value and all(character in _CN_DIGITS for character in value):
        return int("".join(str(_CN_DIGITS[character]) for character in value))
    return None


def _parse_level(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    return _parse_chinese_number(value)


def _normalize(text: str) -> str:
    normalized = re.sub(
        r"\s+",
        "",
        text.lower().translate(_TRADITIONAL_TO_SIMPLIFIED),
    )
    # Faster-Whisper 偶尔会把“调到”识别成读音相近的词。
    for mistaken in ("掉到", "跳到", "条到"):
        normalized = normalized.replace(mistaken, "调到")
    for mistaken in ("调案", "调按", "掉暗", "条暗"):
        normalized = normalized.replace(mistaken, "调暗")
    for mistaken in ("掉亮", "条亮"):
        normalized = normalized.replace(mistaken, "调亮")
    return normalized


def _adjustment(previous: int, target: int, action: str) -> LightAdjustment:
    target = _clamp(target)
    if target == previous:
        description = f"灯光亮度保持在 {target}%"
    else:
        description = f"灯光亮度已从 {previous}% 调至 {target}%"
    return LightAdjustment(
        previous=previous,
        brightness=target,
        matched=True,
        description=description,
        intent_detected=True,
        action=action,
    )


def _unchanged(
    previous: int,
    *,
    intent_detected: bool = False,
    reason: str | None = None,
) -> LightAdjustment:
    if reason == "cancelled":
        description = f"未执行：检测到否定或取消指令；灯光保持在 {previous}%"
    elif reason == "delayed":
        description = f"未执行：暂不支持定时灯控；灯光保持在 {previous}%"
    elif reason == "reported":
        description = f"未执行：这不是明确的新指令；灯光保持在 {previous}%"
    elif intent_detected:
        description = f"未执行：灯控指令不够明确；灯光保持在 {previous}%"
    else:
        description = f"灯光保持在 {previous}%"
    return LightAdjustment(
        previous=previous,
        brightness=previous,
        matched=False,
        description=description,
        intent_detected=intent_detected,
        blocked_reason=reason,
    )


def _detect_control_intent(normalized: str) -> bool:
    direct_cues = (
        "关灯",
        "关一下灯",
        "灯关一下",
        "灯关了",
        "开灯",
        "熄灯",
        "灭灯",
        "睡眠模式",
        "助眠模式",
        "夜灯模式",
        "阅读模式",
        "看书模式",
        "最亮",
        "最暗",
        "全亮",
        "调暗",
        "调亮",
        "调高",
        "调低",
        "暗一点",
        "亮一点",
        "柔和一点",
        "提高亮度",
        "降低亮度",
    )
    if any(cue in normalized for cue in direct_cues):
        return True
    if re.search(
        r"(?:灯|灯光|亮度|光线).{0,6}"
        r"(?:打开|开启|关闭|关掉|关上|关一下|关了|熄灭|调|设|太亮|太暗)",
        normalized,
    ):
        return True
    if re.search(
        r"(?:打开|开启|关闭|关掉|关上|关一下|关了|熄灭|调|设|提高|降低).{0,6}"
        r"(?:灯|灯光|亮度|光线)",
        normalized,
    ):
        return True
    return bool(
        re.search(
            rf"(?:调到|调至|调整到|调整至|设为|设置为|设成|设置成|调成)"
            rf"(?:百分之)?{_NUMBER}(?:%|％)",
            normalized,
        )
    )


def _has_negated_control(normalized: str) -> bool:
    control_action = (
        r"(?:关灯|开灯|熄灯|灭灯|"
        r"(?:打开|开启|关闭|关掉|关上|熄灭).{0,3}(?:灯|灯光)|"
        r"(?:灯|灯光|亮度|光线).{0,5}(?:打开|开启|关闭|关掉|关上|熄灭|调|设)|"
        r"(?:调暗|调亮|调高|调低|暗一点|亮一点|柔和一点)|"
        r"(?:调到|调至|设为|设置为|设成|设置成|调成).{0,6}(?:%|％|百分之|\d)|"
        r"(?:睡眠|助眠|夜灯|阅读|看书)模式)"
    )
    negation = r"(?:不要|别|不用|不必|无需|取消|停止|不想|不喜欢|不希望|不是)"
    if re.search(rf"{negation}.{{0,12}}{control_action}", normalized):
        return True
    if re.search(
        r"(?:灯|灯光|亮度|光线).{0,5}"
        r"(?:不要|别|不用|不必|无需).{0,6}"
        r"(?:开|关|调|设|改变)",
        normalized,
    ):
        return True
    return bool(re.search(r"(?:^|[，,。.!！?？；;])不(?:开灯|关灯|调亮|调暗)", normalized))


def _has_delayed_control(normalized: str) -> bool:
    timed_delay = rf"(?:过|等)?{_NUMBER}(?:秒钟?|分钟?|分|个?小时|天)(?:以后|之后|后)"
    if re.search(timed_delay, normalized):
        return True
    if any(
        marker in normalized
        for marker in ("待会", "等会", "一会儿", "一会", "稍后", "回头再", "明天再")
    ):
        return True
    return bool(re.search(r"等.{0,16}再", normalized))


def _is_report_or_question(normalized: str) -> bool:
    report_cues = (
        "刚才",
        "之前",
        "昨天",
        "上次",
        "没反应",
        "没有反应",
        "为什么",
        "怎么没",
        "是否",
        "有没有",
        "了吗",
        "是不是",
    )
    for clause in _CLAUSE_SEPARATOR.split(normalized):
        if (
            clause
            and _detect_control_intent(clause)
            and any(cue in clause for cue in report_cues)
        ):
            return True
    return False


_COMMAND_PREFIX = (
    r"(?:(?:请|麻烦你?|你能不能|你可以|你帮我|能不能|能否|可以|"
    r"帮我|给我|替我|现在|马上|立刻|先|我想|我要|我需要))*"
)


def _matches_imperative(normalized: str, bodies: tuple[str, ...]) -> bool:
    clauses = [clause for clause in _CLAUSE_SEPARATOR.split(normalized) if clause]
    for clause in clauses:
        for body in bodies:
            if re.match(rf"^{_COMMAND_PREFIX}{body}", clause):
                return True
    return False


def _extract_explicit_level(normalized: str) -> int | None:
    has_light_noun = any(
        noun in normalized for noun in ("灯", "灯光", "亮度", "光线")
    )
    verbs = r"(?:调到|调至|调整到|调整至|设为|设置为|设成|设置成|调成|开到)"

    verb_match = re.search(
        rf"{verbs}(?:亮度)?(?P<percent>百分之)?(?P<value>{_NUMBER})(?P<symbol>%|％)?",
        normalized,
    )
    if verb_match and (
        has_light_noun
        or verb_match.group("percent")
        or verb_match.group("symbol")
    ):
        return _parse_level(verb_match.group("value"))

    noun_match = re.search(
        rf"(?:灯光?|亮度|光线)(?:调|调整|设置|设)?(?:到|至|为|成)?"
        rf"(?:百分之)?(?P<value>{_NUMBER})(?:%|％)",
        normalized,
    )
    if noun_match:
        return _parse_level(noun_match.group("value"))
    return None


def interpret_light_command(text: str, current_brightness: int | float) -> LightAdjustment:
    """把中文灯控指令安全地转换为 0～100 的亮度值。"""

    previous = _clamp(current_brightness)
    normalized = _normalize(text)
    intent_detected = _detect_control_intent(normalized)
    if not intent_detected:
        return _unchanged(previous)

    # 安全优先：否定、延时和回顾性描述都不能触发即时硬件动作。
    if _has_negated_control(normalized):
        return _unchanged(previous, intent_detected=True, reason="cancelled")
    if _has_delayed_control(normalized):
        return _unchanged(previous, intent_detected=True, reason="delayed")
    if _is_report_or_question(normalized):
        return _unchanged(previous, intent_detected=True, reason="reported")

    off_bodies = (
        r"(?:把|将)?(?:这盏|小夜)?灯(?:光)?(?:给我)?(?:关掉|关闭|关上|关一下|关了|熄灭)",
        r"(?:关掉|关闭|关上|关一下|关|熄灭)(?:这盏|小夜)?灯(?:光)?",
    )
    if _matches_imperative(normalized, off_bodies):
        return _adjustment(previous, 0, "off")

    if any(word in normalized for word in ("睡眠模式", "助眠模式", "夜灯模式")):
        return _adjustment(previous, 10, "preset")
    if any(word in normalized for word in ("阅读模式", "看书模式")):
        return _adjustment(previous, 70, "preset")
    if any(word in normalized for word in ("最亮", "全亮", "亮度最大")):
        return _adjustment(previous, 100, "set")
    if any(word in normalized for word in ("最暗", "亮度最低")):
        return _adjustment(previous, 5, "set")

    if "一半" in normalized and (
        any(noun in normalized for noun in ("灯", "灯光", "亮度", "光线"))
        or re.search(r"^(?:请|帮我|给我)?(?:调到|调成|设为|设置为)一半", normalized)
    ):
        return _adjustment(previous, 50, "set")

    level = _extract_explicit_level(normalized)
    if level is not None:
        return _adjustment(previous, level, "set")

    dimmer_words = ("太亮", "暗一点", "调暗", "调低", "降低亮度", "柔和一点")
    if any(word in normalized for word in dimmer_words):
        return _adjustment(previous, previous - 20, "dim")

    brighter_words = ("太暗", "亮一点", "调亮", "调高", "提高亮度")
    if any(word in normalized for word in brighter_words):
        return _adjustment(previous, previous + 20, "brighten")

    on_bodies = (
        r"(?:把|将)?(?:这盏|小夜)?灯(?:光)?(?:给我)?(?:打开|开启|开一下|开开|开了|点亮)",
        r"(?:打开|开启|开一下|开|点亮)(?:这盏|小夜)?灯(?:光)?",
    )
    if _matches_imperative(normalized, on_bodies):
        return _adjustment(previous, previous if previous > 0 else 35, "on")

    return _unchanged(previous, intent_detected=True)


def light_confirmation(adjustment: LightAdjustment) -> str:
    """只依据实际解析结果生成确认语，避免模型虚构设备动作。"""

    if not adjustment.intent_detected:
        return ""
    if not adjustment.matched:
        if adjustment.blocked_reason == "cancelled":
            return f"好的，我没有操作灯光，当前保持在 {adjustment.brightness}%。"
        if adjustment.blocked_reason == "delayed":
            return (
                "这个 Demo 暂时不能定时控制，所以我没有立即操作；"
                f"灯光仍保持在 {adjustment.brightness}%。"
            )
        if adjustment.blocked_reason == "reported":
            return (
                "我没有把这句话当作新的灯控指令，"
                f"灯光仍保持在 {adjustment.brightness}%。"
            )
        if adjustment.blocked_reason == "driver_error":
            return (
                "灯光操作没有成功，"
                f"当前实际亮度仍是 {adjustment.brightness}%。"
            )
        return (
            f"我没有改变灯光，当前是 {adjustment.brightness}%。"
            "你可以直接说“关灯”或“把亮度调到 60%”。"
        )

    if adjustment.action == "off":
        if adjustment.previous == 0:
            return "灯已经是关闭状态。"
        return "好的，灯已经关掉了。"
    if adjustment.action == "on":
        if adjustment.previous > 0:
            return f"灯已经开着，亮度保持在 {adjustment.brightness}%。"
        return f"好的，灯已经打开，亮度是 {adjustment.brightness}%。"
    if adjustment.brightness < adjustment.previous:
        return f"好的，灯已调暗到 {adjustment.brightness}%。"
    if adjustment.brightness > adjustment.previous:
        return f"好的，灯已调亮到 {adjustment.brightness}%。"
    return f"好的，灯光亮度保持在 {adjustment.brightness}%。"
