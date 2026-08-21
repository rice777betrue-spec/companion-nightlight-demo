from __future__ import annotations

import json
import unittest
from unittest.mock import patch
from urllib.error import URLError

from companion_demo.adapters.rk3576 import RkllmHttpAdapter


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def read(self) -> bytes:
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None


class RkllmHttpAdapterTests(unittest.TestCase):
    def test_load_reads_remote_model_once(self) -> None:
        response = _FakeResponse(
            {"object": "list", "data": [{"id": "qwen2-0.5b-rkllm"}]}
        )
        adapter = RkllmHttpAdapter("http://192.168.1.20:8080")

        with patch(
            "companion_demo.adapters.rk3576.llm.urlopen",
            return_value=response,
        ) as mocked_urlopen:
            adapter.load()
            adapter.load()

        self.assertEqual(mocked_urlopen.call_count, 1)
        request = mocked_urlopen.call_args.args[0]
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(
            request.full_url, "http://192.168.1.20:8080/v1/models"
        )
        self.assertIn("qwen2-0.5b-rkllm", adapter.device_label)

    def test_reply_uses_official_openai_compatible_payload(self) -> None:
        captured_payload = {}

        def fake_urlopen(request, **_kwargs):
            if request.get_method() == "GET":
                return _FakeResponse({"data": [{"id": "qwen2-0.5b"}]})
            captured_payload.update(json.loads(request.data.decode("utf-8")))
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "我在，你說吧。",
                            }
                        }
                    ]
                }
            )

        history = [
            {"role": "user", "content": "第一句"},
            {"role": "assistant", "content": "第二句"},
            {"role": "user", "content": "第三句"},
            {"role": "assistant", "content": "第四句"},
            {"role": "user", "content": "第五句"},
            {"role": "assistant", "content": "第六句"},
        ]
        adapter = RkllmHttpAdapter(
            "http://192.168.1.20:8080/",
            max_tokens=64,
            history_limit=4,
        )

        with patch(
            "companion_demo.adapters.rk3576.llm.urlopen",
            side_effect=fake_urlopen,
        ):
            reply = adapter.reply("今天有点累", history, "小林", "简短")

        self.assertEqual(reply, "我在，你说吧。")
        self.assertEqual(captured_payload["model"], "rkllm")
        self.assertFalse(captured_payload["stream"])
        self.assertEqual(captured_payload["top_k"], 1)
        self.assertEqual(captured_payload["max_tokens"], 64)
        self.assertFalse(captured_payload["enable_thinking"])
        self.assertEqual(len(captured_payload["messages"]), 6)
        self.assertEqual(
            captured_payload["messages"][1]["content"], "第三句"
        )
        self.assertEqual(
            captured_payload["messages"][-1],
            {"role": "user", "content": "今天有点累"},
        )

    def test_connection_error_is_actionable(self) -> None:
        adapter = RkllmHttpAdapter("http://192.168.1.20:8080")

        with patch(
            "companion_demo.adapters.rk3576.llm.urlopen",
            side_effect=URLError("连接被拒绝"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "无法连接 RKLLM 服务"
            ):
                adapter.load()

    def test_malformed_chat_response_is_rejected(self) -> None:
        adapter = RkllmHttpAdapter("http://192.168.1.20:8080")
        responses = [
            _FakeResponse({"data": [{"id": "qwen2-0.5b"}]}),
            _FakeResponse({"choices": []}),
        ]

        with patch(
            "companion_demo.adapters.rk3576.llm.urlopen",
            side_effect=responses,
        ):
            with self.assertRaisesRegex(RuntimeError, "响应格式不正确"):
                adapter.reply("你好", [], "", "")


if __name__ == "__main__":
    unittest.main()
