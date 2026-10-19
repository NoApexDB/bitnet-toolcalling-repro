# -*- coding: utf-8 -*-
"""
Corrected fixed-budget speed probe for the post-fix symmetric benchmark.

Fixes vs the inline probe in benchmark_postfix_symmetric.py:
  * cache_prompt=False  -> each rep re-processes the full prompt (no slot continuation)
  * ignore_eos=True     -> exactly n_predict tokens generated every rep
  * 3 reps, mean ± stdev for decode and prefill throughput

Updates speed_probe_fixed_budget in symmetric_dense_vs_ternary_postfix.json in place.
"""
import json
import statistics
import subprocess
import sys
import time

import requests

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\ai_models")

import benchmark_postfix_symmetric as B

OUT = B.OUTPUT_JSON

PROBE_REPS = 3
N_PREDICT = 128

# A fixed neutral prompt, identical for every model (modulo chat template):
PROBE_USER_MSG = [{"role": "user", "content": "Расскажи подробно о столице Франции: история, архитектура и культура."}]


def probe_config(cfg, cases, tok):
    proc, session = B.start_server(cfg["model_path"])
    print(f"Server ready for {cfg['name']}")
    try:
        prompt_str, _ = B.format_prompt(PROBE_USER_MSG, cfg["prompt"], cfg["model_type"])
        if cfg["input_mode"] == "tokens":
            payload_prompt = tok(prompt_str).input_ids
        else:
            payload_prompt = prompt_str

        reps = []
        for rep in range(PROBE_REPS):
            r = session.post(f"http://127.0.0.1:{B.PORT}/completion", json={
                "prompt": payload_prompt,
                "n_predict": N_PREDICT,
                "temperature": 0.0,
                "repeat_penalty": 1.15,
                "repeat_last_n": 64,
                "cache_prompt": False,
                "ignore_eos": True,
            }, timeout=300.0).json()
            tm = r.get("timings", {})
            reps.append({
                "rep": rep + 1,
                "prompt_n": tm.get("prompt_n", 0),
                "prompt_ms": round(tm.get("prompt_ms", 0.0), 2),
                "prompt_per_second": round(tm.get("prompt_per_second", 0.0), 2),
                "predicted_n": tm.get("predicted_n", 0),
                "predicted_ms": round(tm.get("predicted_ms", 0.0), 2),
                "predicted_per_second": round(tm.get("predicted_per_second", 0.0), 2),
            })
            print(f"  rep {rep+1}: prompt_n={reps[-1]['prompt_n']} "
                  f"decode={reps[-1]['predicted_per_second']:.2f} t/s "
                  f"prefill={reps[-1]['prompt_per_second']:.1f} t/s")
            time.sleep(1.0)
    finally:
        B.stop_process(proc)

    dec = [r["predicted_per_second"] for r in reps if r["predicted_per_second"] > 0]
    pre = [r["prompt_per_second"] for r in reps if r["prompt_per_second"] > 0]
    return {
        "reps": reps,
        "decode_tps_mean": round(statistics.mean(dec), 2) if dec else 0.0,
        "decode_tps_stdev": round(statistics.stdev(dec), 2) if len(dec) > 1 else 0.0,
        "prefill_tps_mean": round(statistics.mean(pre), 1) if pre else 0.0,
        "prefill_tps_stdev": round(statistics.stdev(pre), 1) if len(pre) > 1 else 0.0,
        "prompt_tokens_fixed": reps[0]["prompt_n"] if reps else 0,
        "generated_tokens_fixed": reps[0]["predicted_n"] if reps else 0,
        "method": "cache_prompt=False, ignore_eos=True, n_predict=128, 3 reps, fixed neutral prompt",
    }


def main():
    print("Loading tokenizer...")
    tok = B.AutoTokenizer.from_pretrained(B.MODEL_BITNET_HF)

    with open(OUT, encoding="utf-8") as f:
        data = json.load(f)

    # wait for exclusive port access (no other benchmark running)
    if B.is_port_in_use(B.PORT):
        raise RuntimeError(f"Port {B.PORT} busy; another server may be running.")

    for cfg in B.CONFIGS:
        if cfg["name"] not in data.get("configs", {}):
            print(f"[skip] {cfg['name']} not in output yet")
            continue
        print(f"\n=== Corrected speed probe: {cfg['name']} ===")
        summary = probe_config(cfg, None, tok)
        data["configs"][cfg["name"]]["speed_probe_fixed_budget"] = summary
        # persist after each config
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved -> {OUT}")

    # desktop copy
    import shutil
    try:
        shutil.copyfile(OUT, B.OUTPUT_JSON_DESKTOP)
        print(f"Copied -> {B.OUTPUT_JSON_DESKTOP}")
    except Exception as e:
        print(f"[warn] desktop copy failed: {e}")


if __name__ == "__main__":
    main()
