"""Provider-independent conservative context budget accounting."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BudgetReport:
    estimated_tokens: int
    limit_tokens: int

    @property
    def fits(self) -> bool:
        return self.estimated_tokens <= self.limit_tokens


class ContextBudget:
    """A deliberately conservative estimate, not a tokenizer replacement."""

    def __init__(self, context_window_tokens: int = 65536, output_reserve_tokens: int = 8192, safety_tokens: int = 4096) -> None:
        if context_window_tokens <= 0 or output_reserve_tokens < 0 or safety_tokens < 0:
            raise ValueError(
                "context budget values must be positive/non-negative")
        usable = context_window_tokens - output_reserve_tokens - safety_tokens
        if usable <= 0:
            raise ValueError(
                "context budget usable input tokens must be greater than zero")
        self.context_window_tokens = context_window_tokens
        self.output_reserve_tokens = output_reserve_tokens
        self.safety_tokens = safety_tokens
        self.usable_input_tokens = usable

    @staticmethod
    def estimate_text(text: str) -> int:
        # bytes/3 handles CJK and JSON punctuation more conservatively than a
        # chars/4 heuristic; the margin covers message/token framing.
        return max(1, math.ceil(len(text.encode("utf-8")) / 3 * 1.15) + 8)

    def estimate_request(self, request: Any) -> int:
        if hasattr(request, "model_dump"):
            request = request.model_dump(exclude_none=True)
        encoded = json.dumps(request, ensure_ascii=False,
                             separators=(",", ":"), default=str)
        return self.estimate_text(encoded)

    def report(self, estimated_tokens: int) -> BudgetReport:
        return BudgetReport(estimated_tokens, self.usable_input_tokens)

    def check(self, estimated_tokens: int) -> BudgetReport:
        report = self.report(estimated_tokens)
        if not report.fits:
            raise ValueError(
                f"LLM context exceeds input budget: estimated {report.estimated_tokens} tokens, limit {report.limit_tokens}")
        return report


__all__ = ["BudgetReport", "ContextBudget"]
