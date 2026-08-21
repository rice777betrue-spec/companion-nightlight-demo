from __future__ import annotations

import threading
from typing import Any

from companion_demo.text_normalization import to_simplified_chinese


SYSTEM_PROMPT = """你是中文陪伴小夜灯。请只用简体中文，像熟悉但有边界感的朋友一样说话。

最高优先级是准确回应用户最后一句，而不是套用安慰话术。回答前先在心里判断：用户是在提问、陈述事实、纠正你、表达情绪，还是下达命令；不要展示判断过程。

回答规则：
1. 第一句先回应最后一句里的具体事实、问题或否定。对话历史只用于理解“他、她、它、这个、刚才”等指代，不能盖过当前话题。
2. 用户提出问题时先直接回答。用户描述麻烦时先说清当前处境或直接后果，再给至多一个可行建议；不要跳到无关的吃饭、休息或心情话题。
3. 用户纠正你时，先简短承认判断有误并按新信息修正，不能继续坚持原来的猜测。用户明确说“不是A、只是B”时必须以B为准，不要改猜成A或其他相近情绪。
4. 普通分享就是普通分享，不强行解释成焦虑、害怕或孤独。表达情绪时回应造成这种感受的具体事情；除非对方求建议，否则不要说教。
5. 通常回答一到三句。问题必须紧扣用户原话，而且不是每次都必须追问；不要询问原话已经明确回答过的事情。不要编造用户没说过的人物、原因、经历或设备结果。
6. 普通分歧优先建议清楚表达感受和边界，不把小冲突升级成争吵、威胁或断绝关系；只有用户明确描述持续伤害或危险时才讨论远离和求助。
7. 用户没有明确结束聊天或准备睡觉时，不主动说晚安、早点休息或做个好梦。
8. 如果用户表达立即自伤、胸痛、无法呼吸或火灾等紧急危险，清楚建议立即联系急救、警方或身边可信任的人。

{profile}
"""


GENERATION_OPTIONS = {
    "max_new_tokens": 96,
    "do_sample": False,
    "repetition_penalty": 1.06,
}


def build_system_prompt(user_name: str, preferences: str) -> str:
    """只在可信身份下加入真实存在的资料，避免“尚未填写”干扰模型。"""

    profile: list[str] = []
    if user_name.strip():
        profile.append(f"用户希望被称为：{user_name.strip()}")
    if preferences.strip():
        profile.append(f"用户偏好：{preferences.strip()}")
    profile_text = "\n".join(profile) or "本轮没有可用的用户私人资料。"
    return SYSTEM_PROMPT.format(profile=profile_text)


def clean_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """限制历史长度和格式，让当前原话始终拥有更高权重。"""

    cleaned: list[dict[str, str]] = []
    for message in history[-8:]:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        cleaned.append({"role": role, "content": content[:600]})
    return cleaned


class LocalCompanion:
    """在本机 GPU/CPU 上运行 Hugging Face 指令模型。"""

    def __init__(
        self,
        model_name: str,
        local_files_only: bool = True,
        quantization: str = "none",
    ) -> None:
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.quantization = str(quantization or "none").strip().lower()
        self._tokenizer: Any = None
        self._model: Any = None
        self._device = "cpu"
        self._runtime_label = "cpu"
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()

    @property
    def device_label(self) -> str:
        return self._runtime_label

    def load(self) -> None:
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return

            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_kwargs: dict[str, Any] = {
                "low_cpu_mem_usage": True,
                "local_files_only": self.local_files_only,
            }
            quantized = device == "cuda" and self.quantization == "4bit"
            if self.quantization not in {"none", "4bit"}:
                raise ValueError(
                    f"不支持的 LLM_QUANTIZATION：{self.quantization}"
                )
            if quantized:
                model_kwargs.update(
                    {
                        "dtype": torch.float16,
                        "device_map": "auto",
                        "quantization_config": BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                        ),
                    }
                )
            else:
                model_kwargs["dtype"] = (
                    torch.float16 if device == "cuda" else torch.float32
                )

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            )
            if not quantized:
                model = model.to(device)
            model.eval()
            self._device = device
            model_label = str(self.model_name).replace("\\", "/").rstrip("/")
            model_label = model_label.rsplit("/", 1)[-1]
            if quantized:
                self._runtime_label = f"{device} 4-bit｜{model_label}"
            elif self.quantization == "4bit" and device == "cpu":
                self._runtime_label = f"cpu 未量化｜{model_label}"
            else:
                self._runtime_label = f"{device}｜{model_label}"
            self._tokenizer = tokenizer
            self._model = model

    def reply(
        self,
        user_text: str,
        history: list[dict[str, str]],
        user_name: str,
        preferences: str,
    ) -> str:
        self.load()

        import torch

        messages = [
            {
                "role": "system",
                "content": build_system_prompt(user_name, preferences),
            }
        ]
        messages.extend(clean_history(history))
        messages.append({"role": "user", "content": user_text})

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self._tokenizer(
            [prompt], return_tensors="pt"
        ).to(self._device)

        with self._generation_lock, torch.inference_mode():
            generated = self._model.generate(
                **model_inputs,
                **GENERATION_OPTIONS,
            )

        new_tokens = generated[:, model_inputs.input_ids.shape[1] :]
        reply = self._tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0].strip()
        return to_simplified_chinese(
            reply or "我听到了。你愿意再多说一点吗？"
        )
