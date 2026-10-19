# -*- coding: utf-8 -*-
"""
Review remark #1 (round 2): the 40% -> 100% swarm comparison rests on single runs.
This script repeats the post-fix 5-agent pipeline 15x on one server instance
(T=0, identical prompts) to measure run-to-run stability of the success score
and output determinism. Energy/rapl intentionally OFF (accuracy question only).
Output: paper_artifacts/swarm_repeatability_postfix.json
"""
import hashlib
import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\ai_models")

import benchmark_swarm_postfix as S
import benchmark_postfix_symmetric as B

OUT = r"C:\ai_models\paper_artifacts\swarm_repeatability_postfix.json"
N_REPS = 15


def strict_success(agent_result, agent_cfg):
    """Strict: model must close the JSON envelope itself (no tolerant repair)."""
    # replicate the tolerant pipeline's parse but WITHOUT repair attempts:
    full_text = S.reconstruct_envelope(agent_cfg["prompt"], agent_result["raw_output"])
    calls = S.parse_json_tool_call(full_text)
    return bool(calls)


def main():
    tok = B.AutoTokenizer.from_pretrained(B.MODEL_BITNET_HF)
    proc, session = B.start_server(B.MODEL_BITNET_GGUF)
    S.SESSION = session
    print(f"Server ready; running pipeline x{N_REPS}")

    _, parallel_cfgs, _ = S.build_pipeline_configs()
    cfgs_map = {c["id"]: c for c in parallel_cfgs}

    reps = []
    try:
        for i in range(N_REPS):
            pipeline = S.run_pipeline(tok)
            strict = 0
            for a in pipeline["agents"]:
                if a["agent_id"] in cfgs_map:
                    if strict_success(a, cfgs_map[a["agent_id"]]):
                        strict += 1
                else:
                    strict += 1 if a["success"] else 0
            digest = hashlib.sha256(
                "|".join(a["raw_output"] for a in pipeline["agents"]).encode("utf-8")
            ).hexdigest()[:16]
            reps.append({
                "rep": i + 1,
                "tolerant": int(pipeline["success_rate"] * 5),
                "strict_tool_agents": sum(
                    1 for a in pipeline["agents"]
                    if a["agent_id"] in cfgs_map and strict_success(a, cfgs_map[a["agent_id"]])
                ),
                "wall_sec": round(pipeline["total_wall_clock_sec"], 2),
                "output_digest": digest,
            })
            print(f"  rep {i+1:2d}: tolerant {reps[-1]['tolerant']}/5 | "
                  f"strict tool-agents {reps[-1]['strict_tool_agents']}/3 | {reps[-1]['wall_sec']}s | {digest}")
    finally:
        B.stop_process(proc)

    digests = {r["output_digest"] for r in reps}
    artifact = {
        "title": "Swarm pipeline repeatability (post-fix runtime)",
        "protocol": {
            "model": "BitNet-b1.58-2B-4T i2_s, llama-server-relu2.exe, exact HF token IDs",
            "note": "T=0 deterministic decoding; 15 identical pipeline runs on one server",
            "strict_criterion": "tool agent closes pre-filled JSON envelope itself (no repair)",
            "tolerant_criterion": "envelope closed after tolerant bracket repair (as in main run)",
        },
        "repetitions": reps,
        "summary": {
            "runs": N_REPS,
            "tolerant_5_of_5_runs": sum(1 for r in reps if r["tolerant"] == 5),
            "strict_tool_agents_median": sorted(r["strict_tool_agents"] for r in reps)[N_REPS // 2],
            "unique_output_digests": len(digests),
            "deterministic": len(digests) == 1,
        },
    }
    print("\nSUMMARY:", json.dumps(artifact["summary"], ensure_ascii=False))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
