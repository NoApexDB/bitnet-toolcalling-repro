import json

path = r"C:\ai_models\paper_artifacts\heldout_30_cases_paired_transitions.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

local_ids = {10, 11, 12, 13, 14, 15, 17, 19}

for cid, cdata in data["cells"].items():
    evaluated = cdata["cases"]
    tool_cases = [x for x in evaluated if x["needs_tool"]]
    non_tool_cases = [x for x in evaluated if not x["needs_tool"]]

    s1_n = sum(1 for x in tool_cases if x.get("s1_tool_attempted"))
    s2_n = sum(1 for x in tool_cases if x.get("s2_schema_valid"))
    s3_n = sum(1 for x in tool_cases if x.get("s3_heuristic_valid"))
    s4_all = sum(1 for x in tool_cases if x.get("s4_executed") == "[+]")
    s5_all = sum(1 for x in tool_cases if x.get("s5_result_correct") == "[+]")

    local_cases = [x for x in tool_cases if x["id"] in local_ids]
    local_reached = [x for x in local_cases if x.get("s4_executed") != "N/A"]
    local_s4_success = [x for x in local_cases if x.get("s4_executed") == "[+]"]
    local_s5_success = [x for x in local_cases if x.get("s5_result_correct") == "[+]"]

    valid_refusals = sum(1 for x in non_tool_cases if x.get("case_state") == "VALID_CONVERSATIONAL_REFUSAL")
    degenerated_refusals = sum(1 for x in non_tool_cases if x.get("case_state") == "DEGENERATED_ECHO_NO_TOOL")

    cdata["summary"] = {
        "s1_attempts": f"{s1_n}/{len(tool_cases)} ({s1_n/len(tool_cases)*100:.1f}%)" if tool_cases else "N/A",
        "s2_valid_schema": f"{s2_n}/{len(tool_cases)} ({s2_n/len(tool_cases)*100:.1f}%)" if tool_cases else "N/A",
        "s3_heuristic": f"{s3_n}/{len(tool_cases)} ({s3_n/len(tool_cases)*100:.1f}%)" if tool_cases else "N/A",
        "s4_s5_local_applicable_tasks": len(local_cases),
        "s4_s5_local_reached_executor": len(local_reached),
        "s4_s5_success_among_actual_runs": f"{len(local_s5_success)}/{len(local_reached)} ({len(local_s5_success)/len(local_reached)*100:.1f}%)" if local_reached else "0/0 (N/A)",
        "s4_s5_overall_local_applicable": f"{len(local_s5_success)}/{len(local_cases)} ({len(local_s5_success)/len(local_cases)*100:.1f}%)",
        "s4_s5_end_to_end_all_tool_queries": f"{s5_all}/{len(tool_cases)} ({s5_all/len(tool_cases)*100:.1f}%)",
        "external_api_and_python_unmocked": f"{len(tool_cases) - len(local_cases)} (N/A)",
        "valid_conversational_refusals": f"{valid_refusals}/{len(non_tool_cases)} ({valid_refusals/len(non_tool_cases)*100:.1f}%)",
        "degenerated_no_tool": f"{degenerated_refusals}/{len(non_tool_cases)} ({degenerated_refusals/len(non_tool_cases)*100:.1f}%)",
        "total_elapsed_s": sum(x.get("elapsed_s", 0) for x in evaluated),
        "s1_n": s1_n,
        "s2_n": s2_n,
        "s3_n": s3_n,
        "s4_local_n": len(local_s4_success),
        "s5_local_n": len(local_s5_success),
        "s4_all_n": s4_all,
        "s5_all_n": s5_all,
        "valid_refusals_n": valid_refusals,
        "degenerated_no_tool_n": degenerated_refusals
    }

data["note"] = "30-case runner used SYSTEM_PROMPT_BASE without 'AI' (You are a helpful assistant with access to tools.)"
data["metrics_provenance"] = {
    "calculation_method": "embedded_in_runner_and_verified_by_postprocessor",
    "runner_script": r"C:\ai_models\run_heldout_factorial_benchmark.py",
    "reference_postprocessor_script": r"C:\ai_models\update_heldout_json_metrics.py",
    "formula_rules": {
        "local_applicable_case_ids": [10, 11, 12, 13, 14, 15, 17, 19],
        "denominator_local_applicable": 8,
        "denominator_all_tool_queries": 20,
        "denominator_negative_dialog": 10,
        "external_unmocked_tools": ["get_current_weather", "web_search", "execute_python"]
    }
}

with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Updated heldout_30_cases_paired_transitions.json successfully.")
