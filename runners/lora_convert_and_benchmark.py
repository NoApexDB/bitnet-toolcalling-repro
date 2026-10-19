# -*- coding: utf-8 -*-
"""
PHASE 2.3b: Convert the merged (base + LoRA balanced) checkpoint to GGUF I2_S
and run the canonical 20-case benchmark on the corrected runtime (ReLU2 + HF IDs).

Pipeline: HF dir -> f32 GGUF (convert-hf-to-gguf-bitnet.py) -> I2_S (llama-quantize)
          -> canonical 20-case benchmark (same protocol as symmetric_dense_vs_ternary_postfix).
Question: does supervised tool-calling fine-tuning survive (a) weight merging and
(b) re-ternarization to {-1,0,+1} with fresh scales?
"""
import json
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\ai_models")

MERGED_DIR = r"C:\ai_models\bitnet_merged_balanced"
F32_GGUF = os.path.join(MERGED_DIR, "ggml-model-f32.gguf")
I2S_GGUF = os.path.join(MERGED_DIR, "ggml-model-i2_s_merged.gguf")
CONVERT = r"C:\ai_models\BitNet\utils\convert-hf-to-gguf-bitnet.py"
QUANTIZE = r"C:\ai_models\BitNet\build\bin\llama-quantize.exe"
OUT = r"C:\ai_models\paper_artifacts\lora_merged_benchmark_20cases.json"
OUT_DESKTOP = r"C:\Users\Рам\Desktop\BitNet_Research_Reports\lora_merged_benchmark_20cases.json"

import benchmark_postfix_symmetric as B


def sh(cmd, timeout=1800, cwd=None, extra_path=None):
    print("+", " ".join(cmd), flush=True)
    env = None
    if extra_path:
        env = os.environ.copy()
        env["PYTHONPATH"] = extra_path + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore",
                       timeout=timeout, cwd=cwd, env=env)
    print((r.stdout or "")[-1500:])
    if r.returncode != 0:
        print("[stderr]", (r.stderr or "")[-2000:])
        raise RuntimeError(f"command failed rc={r.returncode}")
    return r


def main():
    t0 = time.perf_counter()
    I2S_DIRECT = os.path.join(MERGED_DIR, "ggml-model-i2_s_merged.gguf")
    if not os.path.exists(I2S_DIRECT):
        print("=== STEP 1: convert merged HF -> I2_S GGUF (converter-native path, --outtype i2_s) ===")
        sh([sys.executable, r"C:\ai_models\run_convert_wrapper.py", MERGED_DIR,
            "--outfile", I2S_DIRECT, "--outtype", "i2_s"], timeout=3600)
    print(f"i2_s gguf exists: {os.path.getsize(I2S_DIRECT)/2**30:.2f} GB")
    I2S_GGUF = I2S_DIRECT

    print("=== STEP 3: canonical 20-case benchmark (ReLU2 + HF token IDs) ===")
    cases = json.load(open(B.TEST_SUITE_PATH, encoding="utf-8"))
    tok = B.AutoTokenizer.from_pretrained(B.MODEL_BITNET_HF)

    cfg = {
        "name": "bitnet_2b_i2_s_lora_merged",
        "label": "BitNet 2B I2_S, base + balanced agent LoRA merged, then re-ternarized",
        "model_type": "bitnet",
        "model_path": I2S_GGUF,
        "input_mode": "tokens",
        "prompt": B.SYSTEM_PROMPT_BASE,
    }
    summary = B.run_config(cfg, cases, tok)

    payload = {
        "title": "LoRA-merged (balanced agent SFT) BitNet 2B re-quantized to I2_S, canonical 20-case suite",
        "pipeline": {
            "adapter": r"C:\ai_models\bitnet_agent_lora_balanced (r=8, alpha=16, q/v_proj)",
            "merge_report": r"C:\ai_models\paper_artifacts\lora_merge_report.json",
            "i2s_gguf": I2S_GGUF,
            "quantize_cmd": "convert-hf-to-gguf-bitnet.py --outtype i2_s (converter-native I2_S; stock llama-quantize does not accept I2_S as ftype)",
            "total_seconds": round(time.perf_counter() - t0, 1),
        },
        "config": summary,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    try:
        import shutil
        shutil.copyfile(OUT, OUT_DESKTOP)
    except Exception as e:
        print(f"[warn] desktop copy failed: {e}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
