# -*- coding: utf-8 -*-
"""
Review-fix revision round 2 (18.10.2026): append the swarm repeatability
artifact + runner to BENCHMARK_ARTIFACTS_MANIFEST.json.
Same fail-closed protocol as previous revisions.
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
    "swarm_repeatability_postfix": r"C:\ai_models\paper_artifacts\swarm_repeatability_postfix.json",
    "swarm_repeatability_py": r"C:\ai_models\swarm_repeatability.py",
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

    manifest["review18_revision_r2"] = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "Appended swarm repeatability run (15 pipeline reps: tolerant 5/5 in all, "
                "strict tool-agents 3/3, 13/15 byte-identical outputs). "
                "No existing entries were modified. Backup saved next to this file.",
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
