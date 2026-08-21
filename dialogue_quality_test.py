from __future__ import annotations

import torch

from companion_demo.config import settings
from companion_demo.dialogue import choose_dialogue_guidance
from companion_demo.llm import LocalCompanion


CASES = (
    "今天开会时被领导当众批评了，我觉得特别委屈。",
    "我中午吃了一家特别好吃的面馆，牛肉很多。",
    "最近总是睡不着，脑子里一直想着工作的事情，有点焦虑。",
    "今天先聊到这里吧，我要睡了，晚安。",
)

UNWANTED_CLOSINGS = ("晚安", "早点休息", "做个好梦", "好好休息")


def main() -> None:
    torch.manual_seed(7)
    companion = LocalCompanion(
        settings.llm_model,
        local_files_only=settings.model_offline,
    )
    companion.load()

    for user_text in CASES:
        guidance = choose_dialogue_guidance(user_text)
        reply = companion.reply(
            user_text,
            [],
            "小林",
            "喜欢自然、有来有回的聊天；先听我说，不要每句话都劝我休息",
        )
        print(f"[{guidance.mode}] 用户：{user_text}")
        print(f"[回复] {reply}")
        if guidance.mode != "睡前收尾":
            assert not any(word in reply for word in UNWANTED_CLOSINGS), reply

    print("DIALOGUE_QUALITY_TEST_OK")


if __name__ == "__main__":
    main()
