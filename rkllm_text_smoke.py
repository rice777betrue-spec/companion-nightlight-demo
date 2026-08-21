from __future__ import annotations

import argparse
import os
import sys
import time

from companion_demo.adapters.rk3576 import RkllmHttpAdapter


def main() -> int:
    parser = argparse.ArgumentParser(
        description="验证 RK3576 RKLLM OpenAI 兼容文本接口"
    )
    parser.add_argument(
        "--server",
        default=os.getenv("RKLLM_BASE_URL", "http://127.0.0.1:8080"),
        help="板端 RKLLM Server 地址",
    )
    parser.add_argument(
        "--prompt",
        default="你好，请用一句简体中文介绍你自己。",
        help="测试文本",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    adapter = RkllmHttpAdapter(
        args.server,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout,
    )
    try:
        adapter.load()
        started = time.perf_counter()
        reply = adapter.reply(args.prompt, [], "", "")
        elapsed = time.perf_counter() - started
    except (RuntimeError, ValueError) as exc:
        print(f"RKLLM 文本接口验证失败：{exc}", file=sys.stderr)
        return 1

    print(f"模型：{adapter.device_label}")
    print(f"耗时：{elapsed:.2f} 秒")
    print(f"回复：{reply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
