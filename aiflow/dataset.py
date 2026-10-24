"""The dataset format users will have: a folder of documents, each with a .json of correct answers.

    data/receipts/0001.png
    data/receipts/0001.json   {"total": "45,500", "tax": "4,136", ...}
"""

from dataclasses import dataclass
from pathlib import Path
import json

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class Example:
    id: str
    image: Path
    answer: dict


def load_folder(folder) -> list[Example]:
    folder = Path(folder)
    examples = []
    for img in sorted(folder.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        label = img.with_suffix(".json")
        if label.exists():
            examples.append(Example(img.stem, img, json.loads(label.read_text(encoding="utf-8"))))
    return examples


def export_cord(out_dir, split: str = "test", limit: int | None = None) -> int:
    """Download the public CORD receipt dataset and write it in our folder format."""
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    repo = "naver-clova-ix/cord-v2"
    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset")
             if f.endswith(".parquet") and f"/{split}-" in f]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in files:
        table = pq.read_table(hf_hub_download(repo, f, repo_type="dataset"))
        for row in table.to_pylist():
            parse = json.loads(row["ground_truth"])["gt_parse"]
            sub = parse.get("sub_total") or {}
            total = parse.get("total") or {}
            answer = {
                "subtotal": sub.get("subtotal_price"),
                "tax": sub.get("tax_price"),
                "total": total.get("total_price"),
            }
            stem = f"{split}_{n:04d}"
            (out / f"{stem}.png").write_bytes(row["image"]["bytes"])
            (out / f"{stem}.json").write_text(json.dumps(answer, ensure_ascii=False, indent=2), encoding="utf-8")
            n += 1
            if limit and n >= limit:
                return n
    return n
