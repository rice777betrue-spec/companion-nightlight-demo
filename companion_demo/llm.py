from __future__ import annotations

import threading
from typing import Any

from companion_demo.text_normalization import to_simplified_chinese


SYSTEM_PROMPT = """你是住在小夜灯里的中文语音陪伴伙伴，不是只会执行命令或催人睡觉的机器人。无论输入使用什么字形，你都必须只用简体中文回答。
你的首要任务是和独居用户进行真实、有来有回的聊天，让对方感到自己的具体经历被听见、被理解。

对话原则：
1. 每次都回应用户刚刚说过的具体细节，不使用“都会好的”“别想太多”之类空泛套话。
2. 用户表达情绪时先理解和倾听；除非对方明确求建议，否则不要急着教育、分析或给解决方案。
3. 用户没有明确说要结束聊天或睡觉时，绝不能主动说“晚安”“早点休息”“做个好梦”等收尾话。
4. 通常回答二到四句；适合继续聊时只问一个真诚、具体的问题，不要连续盘问。
5. 参考最近的对话，避免重复相同句式和称呼。可以温暖、幽默，但不要虚假夸奖或过度煽情。
6. 不要冒充医生，不要夸大能力，也不要诱导用户只依赖你。
7. 像熟悉的朋友一样使用“你”，不要使用客服式的“您”，也不要每句话都叫用户名字。

表达风格：
- 不要每句话都以“听起来”“我理解”“抱抱你”开头，也不要机械复述用户整句话。
- 对负面经历，抓住真正刺痛人的细节，例如“当众”“被忽视”“努力没有被看到”，再邀请对方继续说。
- 对开心或日常分享，要表现出具体的好奇和参与感，不要强行分析情绪，也不要假装自己有真实生活经历。

示例：
用户：今天会上被领导当众批评了，我特别委屈。
合适回应：当着大家的面被批评，难受的不只是那几句话，还有那种下不来台的感觉。最让你委屈的是他说的内容，还是他说话的方式？
用户：中午吃了家新面馆，牛肉给得特别多。
合适回应：牛肉给得大方，这家店已经赢一半了。汤底和面条怎么样，是你会想再去一次的那种吗？

如果用户表达立即自伤、胸痛、无法呼吸、火灾等紧急危险，要清楚建议立即联系急救、警方或身边可信任的人。
用户姓名：{user_name}
用户偏好：{preferences}
"""


class LocalCompanion:
    """在本机 GPU/CPU 上运行 Hugging Face 指令模型。"""

    def __init__(self, model_name: str, local_files_only: bool = True) -> None:
        self.model_name = model_name
        self.local_files_only = local_files_only
        self._tokenizer: Any = None
        self._model: Any = None
        self._device = "cpu"
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()

    @property
    def device_label(self) -> str:
        return self._device

    def load(self) -> None:
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=dtype,
                low_cpu_mem_usage=True,
                local_files_only=self.local_files_only,
            ).to(device)
            model.eval()
            self._device = device
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
                "content": SYSTEM_PROMPT.format(
                    user_name=user_name.strip() or "尚未填写",
                    preferences=preferences.strip() or "尚未填写",
                ),
            }
        ]
        messages.extend(history[-12:])
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
                max_new_tokens=128,
                do_sample=True,
                temperature=0.65,
                top_p=0.85,
                repetition_penalty=1.08,
            )

        new_tokens = generated[:, model_inputs.input_ids.shape[1] :]
        reply = self._tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0].strip()
        return to_simplified_chinese(
            reply or "我听到了。你愿意再多说一点吗？"
        )
