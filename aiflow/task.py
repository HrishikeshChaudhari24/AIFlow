"""A task template: which fields to extract and how to compare them."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import re


@dataclass
class Field:
    name: str
    description: str
    kind: str = "text"  # "text" | "amount" | "date"


@dataclass
class Task:
    name: str
    instruction: str
    fields: list[Field]

    def prompt(self) -> str:
        lines = [f"- {f.name}: {f.description}" for f in self.fields]
        return (
            f"{self.instruction}\n\nReturn JSON with these keys:\n"
            + "\n".join(lines)
            + "\nUse null for anything not present on the document. Copy amounts exactly as printed."
        )

    def json_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {f.name: {"type": ["string", "null"]} for f in self.fields},
            "required": [f.name for f in self.fields],
        }

    @classmethod
    def load(cls, path) -> "Task":
        data = json.loads(open(path, encoding="utf-8").read())
        data["fields"] = [Field(**f) for f in data["fields"]]
        return cls(**data)


def normalize(value, kind: str):
    """Bring a value into a canonical form so '1,500.00' and '1500' compare equal."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if kind == "amount":
        return _parse_amount(s)
    return re.sub(r"\s+", " ", s).lower()


def _parse_amount(s: str):
    digits = re.sub(r"[^\d.,-]", "", s)
    if not digits:
        return s.lower()
    # A final separator followed by 1-2 digits is a decimal point; all others are thousands separators.
    m = re.match(r"^(.*?)[.,](\d{1,2})$", digits)
    whole, frac = (m.group(1), m.group(2)) if m else (digits, "0")
    whole = re.sub(r"[.,]", "", whole) or "0"
    try:
        return Decimal(f"{whole}.{frac}").normalize()
    except InvalidOperation:
        return s.lower()
