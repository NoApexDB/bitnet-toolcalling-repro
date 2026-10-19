# -*- coding: utf-8 -*-
"""
Revision script for Task T2 (19.10.2026): append upstream_silu_relu2_repro.json
and its generator script to BENCHMARK_ARTIFACTS_MANIFEST.json.
Fail-closed protocol, automatic backup, SHA-256 calculation.
"""
import hashlib
import json
import os
import shutil
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

MANIFEST = r"C:\ai_models\paper_artifacts\BENCHMARK_ARTIFACTS_MANIFEST.json"
BACKUP = MANIFEST + ".backup_" + time.strftime("%Y%m%d_%H%M%S")

NEW_FILES = {
    "upstream_silu_relu2_repro": r"C:\ai_models\paper_artifacts\upstream_silu_relu2_repro.json",
    "generate_t2_artifact_py": r"C:\ai_models\generate_t2_artifact.py",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not os.path.exists(MANIFEST):
        raise SystemExit("FAIL-CLOSED: manifest not found; nothing changed.")
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict) or not manifest:
        raise SystemExit("FAIL-CLOSED: manifest is not a non-empty object; nothing changed.")

    existing_paths = {v.get("path") for v in manifest.values() if isinstance(v, dict)}
    before = len(manifest)

    added, skipped = [], []
    for key, path in NEW_FILES.items():
        if not os.path.exists(path):
            skipped.append((key, path, "missing"))
            continue
        if path in existing_paths or key in manifest:
            skipped.append((key, path, "already present"))
            continue
        manifest[key] = {"path": path, "sha256": sha256(path), "size_bytes": os.path.getsize(path)}
        added.append((key, path))

    manifest["task_t2_revision"] = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "Appended upstream SiLU vs ReLU² reproduction benchmark artifact (Task T2). "
                "Confirms the FFN activation mismatch bug is alive in upstream microsoft/BitNet at HEAD "
                "(commit 0b341e5, submodule 390c307), showing 0/10 vs 10/10 valid schema recovery. "
                "Drafted English issue UPSTREAM_ISSUE_DRAFT.md. No existing entries were modified.",
        "added_keys": [k for k, _ in added],
    }

    shutil.copyfile(MANIFEST, BACKUP)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"manifest entries: {before} -> {len(manifest)} (incl. revision metadata blocks)")
    print(f"backup: {BACKUP}")
    for k, p in added:
        print(f"  [added] {k}: {p}")
    for k, p, r in skipped:
        print(f"  [skipped:{r}] {k}: {p}")


if __name__ == "__main__":
    main()
