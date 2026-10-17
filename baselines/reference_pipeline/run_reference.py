"""Run the reference project's extraction + cleaning unchanged, for baseline comparison only."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import clean_dataset  # noqa: E402
import ingestion  # noqa: E402


def main() -> None:
    pdf, out = Path(sys.argv[1]), Path(sys.argv[2])
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "civil_code.json"
        ingestion.save_articles(ingestion.extract_articles(pdf), raw)
        out.parent.mkdir(parents=True, exist_ok=True)
        clean_dataset.run_cleanup(str(raw), str(out))
    print(json.dumps({"articles": len(json.loads(out.read_text(encoding="utf-8")))}))


if __name__ == "__main__":
    main()
