from __future__ import annotations

import json
import re
import threading
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from companion_demo.core.contracts import WakeWordDecision
from companion_demo.text_normalization import to_simplified_chinese


class WakeWordController:
    """动态文本唤醒门控。

    PC Demo 先由 Whisper 得到文字，再在这里判断唤醒词；修改词语无需重新训练。
    RK3576 量产实现可通过相同端口替换成低功耗关键词检测模型。
    """

    def __init__(
        self,
        storage_path: Path,
        *,
        default_phrase: str = "小夜灯",
        session_seconds: float = 30.0,
        fuzzy_threshold: float = 0.84,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.session_seconds = max(5.0, float(session_seconds))
        self.fuzzy_threshold = max(0.7, min(1.0, float(fuzzy_threshold)))
        self._lock = threading.RLock()
        self._phrase = self._validate_phrase(default_phrase)
        self._session_active = False
        self._turn_in_progress = False
        self._awake_until = 0.0
        self._last_status = ""
        self._load()

    @staticmethod
    def _normalize(value: str) -> str:
        simplified = to_simplified_chinese(str(value or "")).casefold()
        return "".join(
            character
            for character in simplified
            if unicodedata.category(character)[0] in {"L", "N"}
        )

    @classmethod
    def _validate_phrase(cls, phrase: str) -> str:
        display = str(phrase or "").strip()
        normalized = cls._normalize(display)
        if len(normalized) < 2:
            raise ValueError("唤醒词至少需要 2 个汉字或字母，避免环境误唤醒")
        if len(normalized) > 20:
            raise ValueError("唤醒词不能超过 20 个汉字或字母")
        return display

    @property
    def phrase(self) -> str:
        with self._lock:
            return self._phrase

    @property
    def is_awake(self) -> bool:
        now = time.monotonic()
        with self._lock:
            self._expire_if_idle(now)
            return self._session_active

    def _standby_status(self) -> str:
        return f"待机中｜请先说唤醒词「{self._phrase}」"

    def _expire_if_idle(self, now: float) -> None:
        if (
            self._session_active
            and not self._turn_in_progress
            and now >= self._awake_until
        ):
            self._session_active = False
            self._awake_until = 0.0
            self._last_status = self._standby_status()

    @property
    def status_text(self) -> str:
        now = time.monotonic()
        with self._lock:
            self._expire_if_idle(now)
            if self._session_active and self._turn_in_progress:
                return (
                    "已唤醒｜正在处理本轮对话"
                    f"｜唤醒词「{self._phrase}」"
                )
            if self._session_active:
                remaining = max(1, int(round(self._awake_until - now)))
                return (
                    f"已唤醒｜{remaining} 秒内可连续对话"
                    f"｜唤醒词「{self._phrase}」"
                )
            if self._last_status:
                return self._last_status
            return self._standby_status()

    def _load(self) -> None:
        with self._lock:
            if not self.storage_path.is_file():
                self._last_status = self._standby_status()
                return
            try:
                payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
                self._phrase = self._validate_phrase(payload.get("phrase", ""))
                self._last_status = self._standby_status()
            except Exception as exc:
                self._last_status = f"唤醒词配置读取失败，已使用「{self._phrase}」：{exc}"

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.storage_path.with_suffix(
            self.storage_path.suffix + ".tmp"
        )
        temporary.write_text(
            json.dumps({"phrase": self._phrase}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.storage_path)

    def set_phrase(self, phrase: str) -> str:
        validated = self._validate_phrase(phrase)
        with self._lock:
            self._phrase = validated
            self._session_active = False
            self._turn_in_progress = False
            self._awake_until = 0.0
            self._last_status = (
                f"唤醒词已修改为「{validated}」｜当前为待机状态"
            )
            self._save()
            return self._last_status

    def sleep(self, detail: str | None = None) -> str:
        with self._lock:
            self._session_active = False
            self._turn_in_progress = False
            self._awake_until = 0.0
            self._last_status = detail or (
                f"待机中｜再次说「{self._phrase}」即可唤醒"
            )
            return self._last_status

    def refresh_session(self, now: float | None = None) -> str:
        """从一轮回复结束时重新计算空闲超时。"""

        moment = time.monotonic() if now is None else float(now)
        with self._lock:
            self._expire_if_idle(moment)
            if not self._session_active:
                self._turn_in_progress = False
                self._awake_until = 0.0
                if not self._last_status:
                    self._last_status = self._standby_status()
                return self._last_status
            if not self._turn_in_progress:
                return self._last_status
            self._turn_in_progress = False
            self._awake_until = moment + self.session_seconds
            self._last_status = (
                f"对话进行中｜空闲 {int(self.session_seconds)} 秒后自动待机"
            )
            return self._last_status

    def _find_match(self, transcript: str) -> tuple[int, int] | None:
        text = self._normalize(transcript)
        phrase = self._normalize(self.phrase)
        exact_start = text.find(phrase)
        if exact_start >= 0:
            return exact_start, exact_start + len(phrase)
        if len(phrase) < 4:
            return None

        best_ratio = 0.0
        best_span: tuple[int, int] | None = None
        for length in range(max(2, len(phrase) - 1), len(phrase) + 2):
            for start in range(0, max(0, len(text) - length) + 1):
                candidate = text[start : start + length]
                ratio = SequenceMatcher(None, phrase, candidate).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_span = (start, start + length)
        return best_span if best_ratio >= self.fuzzy_threshold else None

    def _strip_exact_phrase(self, transcript: str) -> str | None:
        characters = list(self.phrase.strip())
        if not characters:
            return None
        separator = r"[\W_]*"
        pattern = separator.join(re.escape(character) for character in characters)
        stripped, count = re.subn(
            pattern,
            "",
            transcript,
            count=1,
            flags=re.IGNORECASE,
        )
        if not count:
            return None
        return stripped.strip(" ，。！？、,.!?;；:：")

    def evaluate(
        self,
        transcript: str,
        now: float | None = None,
    ) -> WakeWordDecision:
        moment = time.monotonic() if now is None else float(now)
        with self._lock:
            self._expire_if_idle(moment)
            if self._session_active:
                self._turn_in_progress = True
                self._awake_until = 0.0
                self._last_status = "已在连续对话窗口内，无需重复唤醒"
                return WakeWordDecision(
                    action="process",
                    transcript=transcript.strip(),
                    triggered=False,
                    status=self._last_status,
                )

        span = self._find_match(transcript)
        if span is None:
            with self._lock:
                self._last_status = (
                    f"待机中｜未听到唤醒词「{self._phrase}」，不回应"
                )
                status = self._last_status
            return WakeWordDecision(
                action="ignore",
                transcript="",
                triggered=False,
                status=status,
            )

        exact_remainder = self._strip_exact_phrase(transcript)
        if exact_remainder is None:
            normalized = self._normalize(transcript)
            exact_remainder = normalized[: span[0]] + normalized[span[1] :]
        remainder = exact_remainder.strip()
        with self._lock:
            self._session_active = True
            self._turn_in_progress = True
            self._awake_until = 0.0
            if remainder:
                self._last_status = (
                    f"唤醒成功｜已识别「{self._phrase}」并接收同句指令"
                )
                action = "process"
            else:
                self._last_status = (
                    f"唤醒成功｜{int(self.session_seconds)} 秒内可直接说话"
                )
                action = "acknowledge"
            status = self._last_status
        return WakeWordDecision(
            action=action,
            transcript=remainder,
            triggered=True,
            status=status,
        )
