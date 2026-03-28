"""
openrouter_ai/utils/toon_parser.py
Implements Token-Oriented Object Notation (TOON).
Encodes Dicts into an extremely lightweight tabular string formats for LLMs.
"""
from typing import Dict, Any

class ToonParser:
    @staticmethod
    def encode(data: Dict[str, Any]) -> str:
        lines = ["Key | Value", "--- | ---"]
        for k, v in data.items():
            lines.append(f"{k} | {v}")
        return "\n".join(lines)
        
    @staticmethod
    def decode(text: str) -> Dict[str, str]:
        parsed = {}
        for line in text.split('\n'):
            if '|' not in line or '---' in line or 'Key | Value' in line:
                continue
            parts = line.split('|', 1)
            if len(parts) == 2:
                parsed[parts[0].strip()] = parts[1].strip()
        return parsed
