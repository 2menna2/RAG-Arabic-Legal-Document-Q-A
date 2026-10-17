"""CI reproducibility check after `dvc repro index`.

Strict where bytes are deterministic, behavioural where they cannot be:

* every stage entry in dvc.lock (commands, params, deps, outs) must equal the committed version, EXCEPT the
  hash/size of outputs listed in FLOAT_OUTPUTS;
* FLOAT_OUTPUTS (the vector index) contain floating-point model state (TruncatedSVD/LSA vectors, Qdrant SQLite).
  OpenBLAS selects CPU-specific kernels at run time, so their last bits differ between machines (measured: two
  GitHub runners / Windows vs Linux give different bytes with identical inputs). For them we require the
  deterministic identity instead: same collection name (derived from corpus hash + chunks hash + chunking +
  embedder parameters), same point count, dimension, corpus and chunks hashes as the committed manifest metadata.
  Retrieval behaviour is then checked by the evaluation + quality gate (regression vs reports/baseline).

Usage: python scripts/check_reproducibility.py   (exit 1 with a readable diff on failure)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
FLOAT_OUTPUTS = {"data/index"}
sys.path.insert(0, str(ROOT / "src"))
from ecc_rag.rag.index import INDEX_IDENTITY_KEYS  # noqa: E402


def _committed(path: str) -> str:
    return subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT, text=True, encoding="utf-8")


def _strip_float_hashes(lock: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(lock))
    for stage in out.get("stages", {}).values():
        for kind in ("deps", "outs"):
            for entry in stage.get(kind, []) or []:
                if entry.get("path") in FLOAT_OUTPUTS:
                    for key in ("md5", "size", "hash"):
                        entry.pop(key, None)
    return out


def _diff(a: Any, b: Any, where: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        lines = []
        for k in sorted(set(a) | set(b)):
            lines += _diff(a.get(k, "<missing>"), b.get(k, "<missing>"), f"{where}.{k}" if where else str(k))
        return lines
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b, strict=True)) for d in _diff(x, y, f"{where}[{i}]")]
    return [] if a == b else [f"{where}: committed={a!r} rebuilt={b!r}"]


def main() -> int:
    problems: list[str] = []
    committed = yaml.safe_load(_committed("dvc.lock"))
    rebuilt = yaml.safe_load((ROOT / "dvc.lock").read_text(encoding="utf-8"))
    problems += [f"dvc.lock {d}" for d in _diff(_strip_float_hashes(committed), _strip_float_hashes(rebuilt))]

    manifest = json.loads((ROOT / "data/index/manifest.json").read_text(encoding="utf-8"))
    expected = json.loads(_committed("reports/index_identity.json"))
    for key in INDEX_IDENTITY_KEYS:
        if manifest.get(key) != expected.get(key):
            problems.append(f"index identity {key}: committed={expected.get(key)!r} rebuilt={manifest.get(key)!r}")

    if problems:
        print("REPRODUCIBILITY CHECK FAILED", *problems, sep="\n  ")
        return 1
    print(
        "REPRODUCIBILITY CHECK PASSED — deterministic outputs byte-identical; index identity "
        f"{manifest['collection']} ({manifest['n_points']} points, dim {manifest['dim']}) matches; "
        f"float outputs {sorted(FLOAT_OUTPUTS)} checked behaviourally by the quality gate"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
