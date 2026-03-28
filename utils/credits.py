from typing import Dict


class CreditWallet:
    def __init__(self):
        self._credits = 1000.0
        self.history = []

    def charge(self, amount: float, reason: str) -> float:
        self._credits -= amount
        self.history.append({"amount": amount, "reason": reason})
        return self._credits

    def credit(self, amount: float, reason: str) -> float:
        self._credits += amount
        self.history.append({"amount": amount, "reason": reason})
        return self._credits

    @property
    def balance(self) -> float:
        return self._credits


class CreditSystem:
    """Lightweight per-request credits used by the pipeline (stub; extend with persistence)."""

    def __init__(self):
        self._per_user: Dict[str, float] = {}

    def record_request(
        self,
        user_id: str,
        request_id: str,
        actual_cost_usd: float,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        total_tok = input_tokens + output_tokens
        earned = round(0.05 + min(total_tok / 100_000.0, 0.5) - actual_cost_usd * 10, 4)
        earned = max(earned, 0.01)
        self._per_user[user_id] = self._per_user.get(user_id, 0.0) + earned
        return earned
