from __future__ import annotations

import json
import threading
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from companion_demo.core.contracts import ChatMessage
from companion_demo.llm import build_system_prompt, clean_history
from companion_demo.text_normalization import to_simplified_chinese


class RkllmHttpAdapter:
    """OpenAI 兼容的 RKLLM Server 文本适配器。"""

    def __init__(
        self,
        base_url: str,
        model: str = "rkllm",
        timeout_seconds: float = 120.0,
        max_tokens: int = 64,
        history_limit: int = 4,
    ) -> None:
        normalized_url = str(base_url).strip().rstrip("/")
        parsed_url = urlparse(normalized_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("RKLLM base_url 必须是完整的 HTTP(S) 地址")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0")
        if max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0")
        if history_limit < 0:
            raise ValueError("history_limit 不能小于 0")

        self.base_url = normalized_url
        self.model = str(model).strip() or "rkllm"
        self.timeout_seconds = float(timeout_seconds)
        self.max_tokens = int(max_tokens)
        self.history_limit = int(history_limit)
        self._remote_model = self.model
        self._loaded = False
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()

    @property
    def device_label(self) -> str:
        host = urlparse(self.base_url).netloc
        return f"RK3576 RKLLM｜{self._remote_model}@{host}"

    def load(self) -> None:
        """连接板端并读取模型列表。

        这个方法只验证 HTTP 服务已就绪；模型实际由板端
        ``flask_server.py`` 加载并常驻 NPU 内存。
        """

        if self._loaded:
            return

        with self._load_lock:
            if self._loaded:
                return
            response = self._request_json("GET", "/v1/models")
            models = response.get("data")
            if not isinstance(models, list) or not models:
                raise RuntimeError("RKLLM /v1/models 没有返回可用模型")
            first_model = models[0]
            if not isinstance(first_model, dict) or not str(
                first_model.get("id", "")
            ).strip():
                raise RuntimeError("RKLLM /v1/models 返回格式不正确")
            self._remote_model = str(first_model["id"]).strip()
            self._loaded = True

    def reply(
        self,
        user_text: str,
        history: list[ChatMessage],
        user_name: str,
        preferences: str,
    ) -> str:
        self.load()

        text = str(user_text).strip()
        if not text:
            raise ValueError("user_text 不能为空")

        messages: list[ChatMessage] = [
            {
                "role": "system",
                "content": build_system_prompt(user_name, preferences),
            }
        ]
        if self.history_limit:
            messages.extend(clean_history(history)[-self.history_limit :])
        messages.append({"role": "user", "content": text})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # RKLLM Server 没有 do_sample 开关；top_k=1 用于稳定输出。
            "temperature": 0.1,
            "top_p": 1.0,
            "top_k": 1,
            "max_tokens": self.max_tokens,
            "repeat_penalty": 1.06,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "enable_thinking": False,
        }

        with self._generation_lock:
            response = self._request_json(
                "POST", "/v1/chat/completions", payload
            )

        try:
            reply = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("RKLLM 对话响应格式不正确") from exc
        if not isinstance(reply, str) or not reply.strip():
            raise RuntimeError("RKLLM 返回了空回复")
        return to_simplified_chinese(reply.strip())

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None
        if payload is not None:
            body = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": "not_required",
            },
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw_response = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            suffix = f"：{detail[:300]}" if detail else ""
            raise RuntimeError(
                f"RKLLM 请求失败（HTTP {exc.code}）{suffix}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise RuntimeError(
                f"无法连接 RKLLM 服务 {self.base_url}：{reason}"
            ) from exc

        try:
            decoded = json.loads(raw_response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("RKLLM 返回的不是有效 JSON") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("RKLLM JSON 响应必须是对象")
        return decoded
