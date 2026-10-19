# -*- coding: utf-8 -*-
"""
PHASE 2.6: Manifest revision — append phase 1/2 artifacts (paths + sizes + SHA-256)
to BENCHMARK_ARTIFACTS_MANIFEST.json WITHOUT touching any existing entry.
A timestamped backup of the previous manifest is written first.
Fail-closed: refuses to run if the manifest is missing or not parseable JSON;
reports every added/skipped file at the end.
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
    # --- phase 1 ---
    "postfix_symmetric_results": r"C:\ai_models\paper_artifacts\symmetric_dense_vs_ternary_postfix.json",
    "pytorch_heldout_30_results": r"C:\ai_models\paper_artifacts\pytorch_heldout_30_results.json",
    "benchmark_postfix_symmetric_py": r"C:\ai_models\benchmark_postfix_symmetric.py",
    "speed_probe_fixed_py": r"C:\ai_models\speed_probe_fixed.py",
    "run_pytorch_heldout_30_py": r"C:\ai_models\run_pytorch_heldout_30.py",
    # --- phase 2 ---
    "binary_router_experiment_results": r"C:\ai_models\paper_artifacts\binary_router_experiment.json",
    "stage2_no_tools_validation_results": r"C:\ai_models\paper_artifacts\stage2_no_tools_validation.json",
    "gbnf_grammar_benchmark_results": r"C:\ai_models\paper_artifacts\gbnf_grammar_benchmark.json",
    "lora_merge_report": r"C:\ai_models\paper_artifacts\lora_merge_report.json",
    "lora_merged_benchmark_results": r"C:\ai_models\paper_artifacts\lora_merged_benchmark_20cases.json",
    "binary_router_experiment_py": r"C:\ai_models\binary_router_experiment.py",
    "stage2_no_tools_validation_py": r"C:\ai_models\stage2_no_tools_validation.py",
    "gbnf_grammar_experiment_py": r"C:\ai_models\gbnf_grammar_experiment.py",
    "lora_merge_v2_py": r"C:\ai_models\lora_merge_v2.py",
    "lora_convert_and_benchmark_py": r"C:\ai_models\lora_convert_and_benchmark.py",
    "run_convert_wrapper_py": r"C:\ai_models\run_convert_wrapper.py",
    "bitnet_merged_balanced_safetensors": r"C:\ai_models\bitnet_merged_balanced\model.safetensors",
    "bitnet_merged_balanced_i2s_gguf": r"C:\ai_models\bitnet_merged_balanced\ggml-model-i2_s_merged.gguf",
    # --- phase 2.4/2.5 (exist by the time this revision runs) ---
    "bfcl_external_holdout_results": r"C:\ai_models\paper_artifacts\bfcl_external_holdout_results.json",
    "bfcl_external_holdout_py": r"C:\ai_models\bfcl_external_holdout.py",
    "energy_repetition_results": r"C:\ai_models\paper_artifacts\energy_repetition_results.json",
    "energy_repetitions_py": r"C:\ai_models\energy_repetitions.py",
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

    added, skipped_missing, skipped_existing = [], [], []
    for key, path in NEW_FILES.items():
        if not os.path.exists(path):
            skipped_missing.append((key, path))
            continue
        if path in existing_paths or key in manifest:
            skipped_existing.append((key, path))
            continue
        manifest[key] = {"path": path, "sha256": sha256(path), "size_bytes": os.path.getsize(path)}
        added.append((key, path))

    manifest["phase1_phase2_revision"] = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "Appended artifacts from phase 1 (post-fix symmetric, PyTorch-30), "
                "phase 2 (router, stage-2, GBNF, LoRA-merge), phase 2.4 (BFCL external holdout), "
                "phase 2.5 (energy repetitions). No existing entries were modified. "
                "Backup of the previous manifest saved next to this file.",
        "added_keys": [k for k, _ in added],
        "skipped_missing": [k for k, _ in skipped_missing],
    }

    shutil.copyfile(MANIFEST, BACKUP)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"manifest entries: {before} -> {len(manifest)} (incl. 1 revision metadata block)")
    print(f"backup: {BACKUP}")
    for k, p in added:
        print(f"  [added] {k}: {p}")
    for k, p in skipped_missing:
        print(f"  [missing->skipped] {k}: {p}")
    for k, p in skipped_existing:
        print(f"  [already->skipped] {k}: {p}")


if __name__ == "__main__":
    main()
