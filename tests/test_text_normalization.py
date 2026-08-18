from __future__ import annotations

import os
import unittest

from companion_demo.text_normalization import to_simplified_chinese


class TextNormalizationTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows系统字形映射测试")
    def test_traditional_chinese_becomes_simplified(self) -> None:
        self.assertEqual(
            to_simplified_chinese("聽起來你對當場被領導批評很難過"),
            "听起来你对当场被领导批评很难过",
        )


if __name__ == "__main__":
    unittest.main()
