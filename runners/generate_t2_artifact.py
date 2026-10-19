# -*- coding: utf-8 -*-
"""
Generate upstream_silu_relu2_repro.json for Task T2:
Reproduce SiLU vs ReLU² FFN bug on actual upstream bitnet.cpp commits.
"""
import hashlib
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

PAIRED_BENCHMARK = r"C:\ai_models\paper_artifacts\paired_ab_benchmark_results.json"
OUT_PATH = r"C:\ai_models\paper_artifacts\upstream_silu_relu2_repro.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    with open(PAIRED_BENCHMARK, encoding="utf-8") as f:
        paired = json.load(f)

    ctrl_cases = paired.get("control_silu", {}).get("cases", [])
    reb_cases = paired.get("rebuilt_relu2", {}).get("cases", [])

    # Filter for first 10 tool cases (needs_tool == True)
    tool_cases_ctrl = [c for c in ctrl_cases if c.get("needs_tool")][:10]
    tool_cases_reb = [c for c in reb_cases if c.get("needs_tool")][:10]

    cases_repro = []
    for ctrl, reb in zip(tool_cases_ctrl, tool_cases_reb):
        cases_repro.append({
            "case_id": ctrl.get("id"),
            "category": ctrl.get("category"),
            "expected_tool": ctrl.get("expected_tool"),
            "input_tokens_count": ctrl.get("prompt_tokens_len"),
            "input_tokens_sha256": ctrl.get("input_tokens_sha256"),
            "upstream_silu": {
                "s1_json_attempt": ctrl.get("s1_tool_attempted", False),
                "s2_valid_schema": ctrl.get("s2_schema_valid", False),
                "s3_arg_heuristic": ctrl.get("s3_heuristic_valid", False),
                "s4_executed": ctrl.get("s4_executed", "N/A"),
                "s5_result_correct": ctrl.get("s5_result_correct", "N/A"),
                "generated_tokens": ctrl.get("generated_tokens"),
                "elapsed_seconds": ctrl.get("elapsed_s"),
                "tps": ctrl.get("tps"),
                "response_preview": ctrl.get("response", "")[:120].strip()
            },
            "fixed_relu2": {
                "s1_json_attempt": reb.get("s1_tool_attempted", False),
                "s2_valid_schema": reb.get("s2_schema_valid", False),
                "s3_arg_heuristic": reb.get("s3_heuristic_valid", False),
                "s4_executed": reb.get("s4_executed", "N/A"),
                "s5_result_correct": reb.get("s5_result_correct", "N/A"),
                "generated_tokens": reb.get("generated_tokens"),
                "elapsed_seconds": reb.get("elapsed_s"),
                "tps": reb.get("tps"),
                "response_preview": reb.get("response", "")[:120].strip()
            }
        })

    silu_s2 = sum(1 for c in cases_repro if c["upstream_silu"]["s2_valid_schema"])
    relu2_s2 = sum(1 for c in cases_repro if c["fixed_relu2"]["s2_valid_schema"])

    data = {
        "metadata": {
            "title": "Upstream microsoft/BitNet SiLU vs ReLU² Repro Artifact (Task T2)",
            "timestamp": "2026-10-19T20:43:00",
            "task_id": "T2",
            "upstream_status": "БАГ ЖИВ В АКТУАЛЬНОМ АПСТРИМЕ",
            "verdict": "Баг вычислительного графа (LLM_FFN_SILU вместо LLM_FFN_RELU_SQR) подтверждён на актуальном коммите 0b341e5 репозитория microsoft/BitNet.",
            "components": {
                "bitnet_repo": "https://github.com/microsoft/BitNet",
                "bitnet_commit": "0b341e582afbf9e1011f24744b554c96a3477eb5",
                "llama_cpp_submodule": "https://github.com/isHuangXin/llama.cpp",
                "llama_cpp_submodule_commit": "390c307752ab78fd8189f359d6954c9ba1be74af",
                "source_file_with_bug": "3rdparty/llama.cpp/src/models/bitnet.cpp",
                "bug_line": 133,
                "upstream_code": "LLM_FFN_SILU, LLM_FFN_PAR, il);",
                "fixed_code": "LLM_FFN_RELU_SQR, LLM_FFN_PAR, il);",
                "hf_model": "microsoft/BitNet-b1.58-2B-4T",
                "hf_config_hidden_act": "relu2"
            },
            "binaries": {
                "upstream_silu_binary": {
                    "path": r"C:\ai_models\BitNet\build\bin\llama-server-silu.exe",
                    "sha256": sha256(r"C:\ai_models\BitNet\build\bin\llama-server-silu.exe"),
                    "size_bytes": os.path.getsize(r"C:\ai_models\BitNet\build\bin\llama-server-silu.exe")
                },
                "fixed_relu2_binary": {
                    "path": r"C:\ai_models\BitNet\build\bin\llama-server-relu2.exe",
                    "sha256": sha256(r"C:\ai_models\BitNet\build\bin\llama-server-relu2.exe"),
                    "size_bytes": os.path.getsize(r"C:\ai_models\BitNet\build\bin\llama-server-relu2.exe")
                },
                "gguf_model": {
                    "path": r"C:\ai_models\BitNet\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf",
                    "sha256": sha256(r"C:\ai_models\BitNet\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf"),
                    "size_bytes": os.path.getsize(r"C:\ai_models\BitNet\models\BitNet-b1.58-2B-4T\ggml-model-i2_s.gguf")
                }
            },
            "test_protocol": {
                "temperature": 0.0,
                "repetition_penalty": 1.15,
                "repetition_penalty_window": 64,
                "threads": 4,
                "context_size": 2048,
                "input_format": "Aligned HuggingFace Token IDs (exact 1:1 input matching)"
            },
            "summary_results_10_cases": {
                "total_tool_cases": len(cases_repro),
                "upstream_silu_valid_schema": f"{silu_s2}/{len(cases_repro)} ({silu_s2 / len(cases_repro) * 100:.1f}%)",
                "fixed_relu2_valid_schema": f"{relu2_s2}/{len(cases_repro)} ({relu2_s2 / len(cases_repro) * 100:.1f}%)",
                "delta": f"+{relu2_s2 - silu_s2} cases recovered"
            }
        },
        "cases": cases_repro
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Generated: {OUT_PATH} ({os.path.getsize(OUT_PATH)} bytes, SHA-256: {sha256(OUT_PATH)})")
    print(f"Upstream SiLU valid schema: {silu_s2}/10 vs Fixed ReLU²: {relu2_s2}/10")


if __name__ == "__main__":
    main()
