# -*- coding: utf-8 -*-
"""
PHASE 2.5: Energy measurement repetitions (>=5 per contested config) + counter audit.

The single-run post-fix numbers showed that net mJ/token has up to +-34% run-to-run
variance (idle-baseline drift). This script measures the two CONTESTED 2B-class
configs (BitNet 2B I2_S fixed vs Qwen2.5-1.5B Q4_K_M) 5 times each on the identical
deterministic workload (canonical 20-case suite -> identical token counts every rep),
and reports mean +- stdev of net mJ/token.

Counter audit: enumerates every '\\Energy Meter(...)' instance available via typeperf
and, where a second package-related counter exists, samples it in parallel during one
run as an internal consistency cross-check.
"""
import json
import os
import statistics
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\ai_models")

import benchmark_postfix_symmetric as B
from evaluator_rigorous import parse_tool_envelope

OUT = r"C:\ai_models\paper_artifacts\energy_repetition_results.json"
OUT_DESKTOP = r"C:\Users\Рам\Desktop\BitNet_Research_Reports\energy_repetition_results.json"
REPS = 5
EXTRA_LOG = r"C:\ai_models\rapl_counter_check.csv"


def list_energy_counters():
    r = subprocess.run(["typeperf", "-q", "Energy Meter"], capture_output=True, text=True,
                        encoding="utf-8", errors="ignore", timeout=30)
    lines = [l.strip().strip('"') for l in (r.stdout or "").splitlines() if "\\Energy Meter" in l]
    return lines


def read_typeperf_csv(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    out.append(float(parts[1].replace('"', "").replace(",", ".")) / 1000.0)
                except Exception:
                    pass
    return out


def one_energy_rep(cfg, cases, tok, extra_counter=None):
    proc, session = B.start_server(cfg["model_path"])
    sampler = B.EnergySampler()
    sampler.start()
    extra_proc = None
    if extra_counter:
        if os.path.exists(EXTRA_LOG):
            os.remove(EXTRA_LOG)
        extra_proc = subprocess.Popen(
            ["typeperf", extra_counter, "-si", "1", "-o", EXTRA_LOG, "-y"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3.0)  # idle baseline window
    total_tokens = 0
    t0 = time.perf_counter()
    for case in cases:
        prompt_str, stop_tokens = B.format_prompt(case["messages"], cfg["prompt"], cfg["model_type"])
        if cfg["input_mode"] == "tokens":
            payload_prompt = tok(prompt_str).input_ids
        else:
            payload_prompt = prompt_str
        resp = session.post(f"http://127.0.0.1:{B.PORT}/completion", json={
            "prompt": payload_prompt, "n_predict": 128, "temperature": 0.0,
            "repeat_penalty": 1.15, "repeat_last_n": 64, "stop": stop_tokens,
        }, timeout=180).json()
        total_tokens += resp.get("timings", {}).get("predicted_n", 0)
    wall = time.perf_counter() - t0
    time.sleep(2.0)
    sampler.stop()
    if extra_proc:
        extra_proc.terminate()
        try:
            extra_proc.wait(timeout=5)
        except Exception:
            extra_proc.kill()
    B.stop_process(proc)

    readings = sampler.get_readings_w()
    idle = readings[:3]
    p_idle = sum(idle) / len(idle) if idle else 0.0
    total_j = sum(readings)
    net_j = max(0.0, total_j - p_idle * len(readings))
    net_mj_tok = net_j / total_tokens * 1000.0 if total_tokens else 0.0

    extra_vals = read_typeperf_csv(EXTRA_LOG) if extra_counter else []
    extra_stats = None
    if extra_vals:
        e_idle = sum(extra_vals[:3]) / 3
        e_mean = sum(extra_vals) / len(extra_vals)
        extra_stats = {"counter": extra_counter, "idle_w": round(e_idle, 2),
                       "mean_w": round(e_mean, 2), "samples": len(extra_vals)}

    return {
        "tokens": total_tokens, "wall_s": round(wall, 1),
        "idle_w": round(p_idle, 2), "mean_w": round(sum(readings) / len(readings), 2) if readings else 0,
        "total_j": round(total_j, 1), "net_j": round(net_j, 1),
        "net_mj_per_token": round(net_mj_tok, 1),
        "crosscheck_counter": extra_stats,
    }


def main():
    counters = list_energy_counters()
    print("Available energy counters:")
    for c in counters:
        print("  ", c)
    pkg = r"\Energy Meter(RAPL_Package0_PKG)\Power"
    extra = None
    for c in counters:
        if "PP0" in c or "PP1" in c:
            extra = c
            break
    if extra:
        print(f"Cross-check counter selected: {extra}")
    else:
        print("[note] no second package counter (PP0/PP1) available; cross-check limited to enumeration")

    cases = json.load(open(B.TEST_SUITE_PATH, encoding="utf-8"))
    tok = B.AutoTokenizer.from_pretrained(B.MODEL_BITNET_HF)

    configs = {
        "bitnet_2b_i2_s_relu2_hf_ids": {
            "model_path": B.MODEL_BITNET_GGUF, "model_type": "bitnet",
            "input_mode": "tokens", "prompt": B.SYSTEM_PROMPT_BASE},
        "qwen2.5_1.5b_q4_k_m": {
            "model_path": B.MODEL_QWEN15, "model_type": "qwen",
            "input_mode": "string", "prompt": B.SYSTEM_PROMPT_BASE},
    }

    payload = {"title": "Energy repetitions: 5 reps x 2 contested 2B-class configs, identical workload",
               "protocol": {"reps": REPS, "workload": "canonical 20-case suite (deterministic)",
                            "sampler": "typeperf " + pkg + " -si 1, idle = first 3 samples, cooldown 2s"},
               "available_energy_counters": counters,
               "crosscheck_note": "no PP0/PP1 counter available" if not extra else f"parallel sampling: {extra}"}

    for name, cfg in configs.items():
        reps = []
        for i in range(REPS):
            print(f"\n=== {name} rep {i+1}/{REPS} ===")
            r = one_energy_rep(cfg, cases, tok, extra_counter=(extra if i == 0 else None))
            reps.append(r)
            print(json.dumps({k: r[k] for k in ("tokens", "idle_w", "mean_w", "net_mj_per_token")}, ensure_ascii=False))
        mj = [r["net_mj_per_token"] for r in reps]
        payload[name] = {
            "reps": reps,
            "net_mj_per_token_mean": round(statistics.mean(mj), 1),
            "net_mj_per_token_stdev": round(statistics.stdev(mj), 1) if len(mj) > 1 else 0.0,
            "tokens_per_rep": reps[0]["tokens"],
        }

    b = payload["bitnet_2b_i2_s_relu2_hf_ids"]
    q = payload["qwen2.5_1.5b_q4_k_m"]
    diff = b["net_mj_per_token_mean"] - q["net_mj_per_token_mean"]
    pooled = (b["net_mj_per_token_stdev"] ** 2 + q["net_mj_per_token_stdev"] ** 2) ** 0.5
    payload["conclusion"] = {
        "bitnet_mean_vs_qwen_mean_mj": f"{b['net_mj_per_token_mean']} vs {q['net_mj_per_token_mean']}",
        "difference_mj": round(diff, 1),
        "pooled_stdev_mj": round(pooled, 1),
        "resolvable": abs(diff) > 2 * pooled,
        "note": "If resolvable=false, the post-fix single-run ordering is not a real difference.",
    }

    print("\n" + json.dumps(payload["conclusion"], ensure_ascii=False, indent=1))
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
