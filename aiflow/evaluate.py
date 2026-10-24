"""The Play button: run a model over labelled examples, score every field, explain the result in plain words."""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

from .dataset import Example
from .task import Task, normalize

# Outcome of one field on one document.
CORRECT, WRONG, MISSED, INVENTED = "correct", "wrong", "missed", "invented"


def judge(expected, predicted, kind: str):
    e, p = normalize(expected, kind), normalize(predicted, kind)
    if e is None and p is None:
        return None  # nothing to find, nothing found: not scored
    if e is None:
        return INVENTED
    if p is None:
        return MISSED
    return CORRECT if e == p else WRONG


@dataclass
class Report:
    model: str
    task: str
    rows: list = field(default_factory=list)

    def outcomes(self, field_name=None) -> Counter:
        return Counter(r["outcomes"][f] for r in self.rows
                       for f in r["outcomes"] if field_name in (None, f) and r["outcomes"][f])

    def accuracy(self, field_name=None) -> float:
        c = self.outcomes(field_name)
        scored = sum(c.values())
        return c[CORRECT] / scored if scored else 0.0

    def summary(self, task: Task) -> str:
        n = len(self.rows)
        secs = sum(r["seconds"] for r in self.rows) / max(n, 1)
        lines = [f"{self.model}: gets {self.accuracy():.0%} of fields right on {n} documents "
                 f"({secs:.1f}s per document)."]
        for f in task.fields:
            c = self.outcomes(f.name)
            line = f"  {f.name:<12} {self.accuracy(f.name):>5.0%}"
            problem = _diagnose(c)
            lines.append(line + (f"   {problem}" if problem else ""))
        return "\n".join(lines)

    def save(self, runs_dir="runs") -> Path:
        path = Path(runs_dir) / f"{datetime.now():%Y%m%d-%H%M%S}-{self.task}-{self.model.replace(':', '_').replace('/', '_')}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for r in self.rows:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        return path


def _diagnose(c: Counter) -> str:
    """Name the dominant kind of mistake, in words a non-expert understands."""
    errors = {k: c[k] for k in (WRONG, MISSED, INVENTED) if c[k]}
    if not errors:
        return ""
    worst = max(errors, key=errors.get)
    return {
        WRONG: f"mostly reads the wrong value ({c[WRONG]}x)",
        MISSED: f"often misses it when it is there ({c[MISSED]}x)",
        INVENTED: f"makes up a value when there is none ({c[INVENTED]}x)",
    }[worst]


def run(model, task: Task, examples: list[Example], on_progress=None) -> Report:
    report = Report(model.name, task.name)
    for i, ex in enumerate(examples, 1):
        predicted, seconds = model.extract(task, ex.image)
        outcomes = {f.name: judge(ex.answer.get(f.name), predicted.get(f.name), f.kind) for f in task.fields}
        report.rows.append({"id": ex.id, "expected": ex.answer, "predicted": predicted,
                            "outcomes": outcomes, "seconds": round(seconds, 2)})
        if on_progress:
            on_progress(i, len(examples), report)
    return report
