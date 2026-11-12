"""Small shared helpers: sha256, JSON writing, timing."""
from __future__ import annotations

import csv
import hashlib
import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


def sha256_file(path, chunk=1 << 20):
    """Return the sha256 hex digest of a file (read in chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_json(obj, path):
    """Write obj as UTF-8 JSON (creates parent folders)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


@contextmanager
def timer(name, csv_path=None, **extra):
    """Time a block; print the duration and append a row to csv_path if given."""
    t0 = time.perf_counter()
    yield
    sec = time.perf_counter() - t0
    print(f"[timer] {name}: {sec:.1f}s")
    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        new = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "step", "seconds", "extra"])
            w.writerow([datetime.now().isoformat(timespec="seconds"), name, round(sec, 2), json.dumps(extra)])