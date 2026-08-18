from __future__ import annotations

from companion_demo.core.contracts import LightExecution
from companion_demo.light import LightAdjustment


class VirtualLightDriver:
    """无硬件时使用的灯光驱动，状态由每轮请求中的亮度传入。"""

    def apply(self, command: LightAdjustment) -> LightExecution:
        if not command.matched:
            return LightExecution(
                previous=command.previous,
                requested=command.brightness,
                actual=command.previous,
                applied=False,
                description=command.description,
            )
        return LightExecution(
            previous=command.previous,
            requested=command.brightness,
            actual=command.brightness,
            applied=True,
            description=command.description,
        )
