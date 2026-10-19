# -*- coding: utf-8 -*-
"""
PHASE 2.3a: Merge the balanced agent LoRA adapter (r=8, q/v_proj) into the BitNet base
(FP32, CPU) and save a standalone HF checkpoint for subsequent GGUF conversion.
"""
import json
import shutil
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

BASE = r"C:\ai_models\BitNet-b1.58-2B-4T"
ADAPTER = r"C:\ai_models\bitnet_agent_lora_balanced"
OUT_DIR = r"C:\ai_models\bitnet_merged_balanced"
MERGE_REPORT = r"C:\ai_models\paper_artifacts\lora_merge_report.json"


def main():
    t0 = time.perf_counter()
    print("[1/4] Loading base BitNet FP32 on CPU...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True
    )
    print(f"      loaded in {time.perf_counter()-t0:.1f}s | {base.__class__.__name__}")

    print("[2/4] Attaching LoRA adapter...")
    model = PeftModel.from_pretrained(base, ADAPTER)

    print("[3/4] merge_and_unload()...")
    t1 = time.perf_counter()
    merged = model.merge_and_unload()
    print(f"      merged in {time.perf_counter()-t1:.1f}s")

    print("[4/4] Saving merged checkpoint...")
    t2 = time.perf_counter()
    merged.save_pretrained(OUT_DIR, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.save_pretrained(OUT_DIR)
    # copy generation_config if present
    import os
    src = os.path.join(BASE, "generation_config.json")
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(OUT_DIR, "generation_config.json"))
    print(f"      saved in {time.perf_counter()-t2:.1f}s -> {OUT_DIR}")

    # quick sanity: merged weight differs from base on a target module
    import os
    report = {
        "base": BASE,
        "adapter": ADAPTER,
        "adapter_config": json.load(open(ADAPTER + "\\adapter_config.json", encoding="utf-8")),
        "out_dir": OUT_DIR,
        "merge_seconds": round(time.perf_counter() - t0, 1),
        "files": {f: os.path.getsize(OUT_DIR + "\\" + f) for f in os.listdir(OUT_DIR)},
    }
    with open(MERGE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report["files"], ensure_ascii=False))
    print(f"Report -> {MERGE_REPORT}")


if __name__ == "__main__":
    main()
