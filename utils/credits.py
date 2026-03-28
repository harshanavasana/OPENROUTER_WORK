"""
openrouter_ai/utils/credits.py
Persistent Credits and Reward System.
Uses SQLite to store balances per user, distributing rewards for efficient prompting.
"""
import os
import sqlite3
from typing import Dict, List, Any
from pathlib import Path

# Ensures SQLite DB is stored in data/ folder inside openrouter_ai
_DATA_DIR = Path(__file__).parent.parent / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_DB_PATH = _DATA_DIR / "credits.db"

def _get_connection():
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _init_db():
    with _get_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                balance REAL DEFAULT 10.0
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                amount REAL,
                reason TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            )
        ''')
        conn.commit()

# Ensure database is bootstrapped
_init_db()

class CreditWallet:
    """Wallet facade for a specific user."""
    def __init__(self, user_id: str):
        self.user_id = user_id
        # Automatically vivify the user with default balance if they don't exist
        with _get_connection() as conn:
            conn.execute("INSERT OR IGNORE INTO users (user_id, balance) VALUES (?, 10.0)", (user_id,))
            conn.commit()

    def charge(self, amount: float, reason: str) -> float:
        with _get_connection() as conn:
            conn.execute("UPDATE users SET balance = balance - ? WHERE user_id = ?", (amount, self.user_id))
            conn.execute("INSERT INTO ledger (user_id, amount, reason) VALUES (?, ?, ?)", (self.user_id, -amount, reason))
            conn.commit()
            return self.balance

    def award(self, amount: float, reason: str) -> float:
        with _get_connection() as conn:
            conn.execute("UPDATE users SET balance = balance + ? WHERE user_id = ?", (amount, self.user_id))
            conn.execute("INSERT INTO ledger (user_id, amount, reason) VALUES (?, ?, ?)", (self.user_id, amount, reason))
            conn.commit()
            return self.balance

    @property
    def balance(self) -> float:
        with _get_connection() as conn:
            row = conn.execute("SELECT balance FROM users WHERE user_id = ?", (self.user_id,)).fetchone()
            return row["balance"] if row else 0.0

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        with _get_connection() as conn:
            cursor = conn.execute("SELECT amount, reason, timestamp FROM ledger WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?", (self.user_id, limit))
            return [dict(row) for row in cursor.fetchall()]


class CreditSystem:
    """
    Main system used by the Pipeline to record routing decisions and dole out rewards.
    """
    def record_request(
        self,
        user_id: str,
        request_id: str,
        actual_cost_usd: float,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculates efficiency, charges cost, gives a reward, and returns net tokens earned vs spent.
        """
        wallet = CreditWallet(user_id)
        wallet.charge(actual_cost_usd, f"API inference cost for req {request_id[:8]}")

        total_tok = input_tokens + output_tokens
        # Rewards logic: base 0.05 + volume usage bonus - penalty for literal API cost
        earned = round(0.05 + min(total_tok / 100_000.0, 0.5) - actual_cost_usd * 10, 4)
        earned = max(earned, 0.01)

        wallet.award(earned, f"Efficiency reward for req {request_id[:8]}")
        
        return earned
