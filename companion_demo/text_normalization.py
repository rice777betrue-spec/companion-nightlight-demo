from __future__ import annotations

import os


def to_simplified_chinese(text: str) -> str:
    """在Windows Demo中使用系统字形映射统一为简体中文。"""

    if not text or os.name != "nt":
        return text

    try:
        import ctypes

        convert = ctypes.windll.kernel32.LCMapStringEx
        simplified_flag = 0x02000000
        size = convert(
            "zh-CN",
            simplified_flag,
            text,
            len(text),
            None,
            0,
            None,
            None,
            0,
        )
        if size <= 0:
            return text
        buffer = ctypes.create_unicode_buffer(size)
        result = convert(
            "zh-CN",
            simplified_flag,
            text,
            len(text),
            buffer,
            size,
            None,
            None,
            0,
        )
        return buffer.value if result > 0 else text
    except Exception:
        return text

