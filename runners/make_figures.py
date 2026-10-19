# -*- coding: utf-8 -*-
"""Generate the four paper figures from the project's artifact JSONs."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

PA = r"C:\ai_models\paper_artifacts"
OUT = r"C:\Users\Рам\Desktop\BitNet_Research_Reports"

C_BITNET = "#1f6f8b"
C_QWEN = "#c1605a"
C_GRAY = "#8a8a8a"

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def fig1_factorial():
    f2 = json.load(open(PA + r"\factorial_2x2_benchmark_matrix.json", encoding="utf-8"))
    h30 = json.load(open(PA + r"\heldout_30_cases_paired_transitions.json", encoding="utf-8"))
    cells = ["cell_1", "cell_2", "cell_3", "cell_4"]
    labels = ["SiLU\n+ строка", "SiLU\n+ HF IDs", "ReLU²\n+ строка", "ReLU²\n+ HF IDs"]
    s2_20 = [f2["cells"][c]["summary"]["s2_n"] / 15 * 100 for c in cells]
    s2_30 = [h30["cells"][c]["summary"]["s2_n"] / 20 * 100 for c in cells]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = range(4)
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], s2_20, w, label="канонический набор (20 кейсов)", color=C_GRAY)
    b2 = ax.bar([i + w / 2 for i in x], s2_30, w, label="challenge-набор (30 кейсов)", color=C_BITNET)
    ax.axhline(90, color=C_QWEN, ls="--", lw=1.2)
    ax.text(3.45, 91.5, "PyTorch-эталон (90%)", color=C_QWEN, fontsize=9, ha="right")
    for bars, nums, denom in ((b1, s2_20, 15), (b2, s2_30, 20)):
        for rect, v in zip(bars, nums):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 1.5,
                    f"{round(v*denom/100)}/{denom}", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("S2: валидные JSON-схемы, %")
    ax.set_ylim(0, 108)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Факторный эксперимент 2×2: активация FFN и режим подачи входа", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT + r"\fig1_factorial.png", dpi=150)
    print("fig1 ok")


def fig2_energy():
    e = json.load(open(PA + r"\energy_repetition_results.json", encoding="utf-8"))
    b = e["bitnet_2b_i2_s_relu2_hf_ids"]
    q = e["qwen2.5_1.5b_q4_k_m"]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for i, (cfg, color, name) in enumerate((
            (b, C_BITNET, "BitNet 2B I2_S\n(исправленный рантайм)"),
            (q, C_QWEN, "Qwen2.5-1.5B Q4_K_M"))):
        vals = [r["net_mj_per_token"] for r in cfg["reps"]]
        xs = [i + (j - 2) * 0.06 for j in range(len(vals))]
        ax.scatter(xs, vals, s=55, color=color, zorder=3, label=None)
        m = cfg["net_mj_per_token_mean"]
        s = cfg["net_mj_per_token_stdev"]
        ax.hlines(m, i - 0.28, i + 0.28, color=color, lw=2.5)
        ax.vlines(i, m - s, m + s, color=color, lw=1.5)
        ax.text(i + 0.32, m, f"{m} ± {s}", va="center", fontsize=10, color=color)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["BitNet 2B I2_S\n(исправленный рантайм)", "Qwen2.5-1.5B Q4_K_M"])
    ax.set_ylabel("net-энергия сокета, мДж / токен")
    ax.set_xlim(-0.6, 1.9)
    ax.set_title("5 повторов на конфигурацию: разница средних (280 мДж)\nменьше разброса (pooled σ = 308 мДж)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT + r"\fig2_energy.png", dpi=150)
    print("fig2 ok")


def fig3_states():
    h30 = json.load(open(PA + r"\heldout_30_cases_paired_transitions.json", encoding="utf-8"))
    py = json.load(open(PA + r"\pytorch_heldout_30_results.json", encoding="utf-8"))

    # states per cell from paired_transitions
    pt = h30["paired_transitions"]
    cell_keys = ["cell_1_swiglu_string", "cell_2_swiglu_tokens", "cell_3_relu2_string", "cell_4_relu2_tokens"]
    names = ["SiLU\n+ строка", "SiLU\n+ HF IDs", "ReLU²\n+ строка", "ReLU²\n+ HF IDs", "PyTorch\nFP32"]
    state_order = [
        ("FULL_EXECUTION_SUCCESS", "#2e7d32"),
        ("VALID_TOOL_CALL", "#7cb342"),
        ("VALID_SCHEMA_BAD_ARGS", "#f0b429"),
        ("HALLUCINATED_OR_INVALID_SCHEMA", "#e07b39"),
        ("DEGENERATED_LOOP_NO_TOOL", "#9e9e9e"),
        ("FALSE_POSITIVE_TOOL_CALL", "#c1605a"),
        ("VALID_CONVERSATIONAL_REFUSAL", "#1f6f8b"),
        ("DEGENERATED_ECHO_NO_TOOL", "#616161"),
    ]
    ru = {
        "FULL_EXECUTION_SUCCESS": "полный успех",
        "VALID_TOOL_CALL": "валидный вызов",
        "VALID_SCHEMA_BAD_ARGS": "схема ок, аргументы нет",
        "HALLUCINATED_OR_INVALID_SCHEMA": "невалидная схема",
        "DEGENERATED_LOOP_NO_TOOL": "деградация без вызова",
        "FALSE_POSITIVE_TOOL_CALL": "ложный вызов",
        "VALID_CONVERSATIONAL_REFUSAL": "корректный отказ",
        "DEGENERATED_ECHO_NO_TOOL": "эхо вместо ответа",
    }
    counts = []
    for ck in cell_keys:
        c = {}
        for entry in pt:
            st = entry["states"].get(ck)
            c[st] = c.get(st, 0) + 1
        counts.append(c)
    c = {}
    for case in py["cases"]:
        c[case["case_state"]] = c.get(case["case_state"], 0) + 1
    counts.append(c)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    bottoms = [0] * 5
    for st, color in state_order:
        vals = [cc.get(st, 0) for cc in counts]
        if not any(vals):
            continue
        ax.bar(range(5), vals, 0.62, bottom=bottoms, color=color, label=ru[st])
        for i, (v, bt) in enumerate(zip(vals, bottoms)):
            if v >= 2:
                ax.text(i, bt + v / 2, str(v), ha="center", va="center", fontsize=8.5,
                        color="white" if color not in ("#f0b429",) else "black")
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(range(5))
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel("кейсов (из 30)")
    ax.set_title("Покейсовые исходы, 30-кейсный challenge-набор", fontsize=12)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(OUT + r"\fig3_states.png", dpi=150, bbox_inches="tight")
    print("fig3 ok")


def fig4_bfcl():
    d = json.load(open(PA + r"\bfcl_external_holdout_results.json", encoding="utf-8"))
    b = d["bitnet_2b_fixed_runtime"]["summary"]
    q = d["qwen2.5_3b"]["summary"]

    def parse(frac):
        a, b_ = frac.split("/")
        return int(a) / int(b_) * 100

    metrics = [("simple_strict", "строгий матчер"), ("simple_lenient_optional", "lenient (опц. параметры)"),
               ("irrelevance_correct_refusal_json_only", "корректные отказы")]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    x = range(3)
    w = 0.36
    bv = [parse(b[k]) for k, _ in metrics]
    qv = [parse(q[k]) for k, _ in metrics]
    r1 = ax.bar([i - w / 2 for i in x], bv, w, color=C_BITNET, label="BitNet 2B (исправленный рантайм)")
    r2 = ax.bar([i + w / 2 for i in x], qv, w, color=C_QWEN, label="Qwen2.5-3B Q4_K_M")
    for rects, vals in ((r1, bv), (r2, qv)):
        for rect, v in zip(rects, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels([m for _, m in metrics], fontsize=10)
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    ax.set_title("Внешний слепой holdout: BFCL v3 (40 simple + 20 irrelevance)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT + r"\fig4_bfcl.png", dpi=150)
    print("fig4 ok")


fig1_factorial()
fig2_energy()
fig3_states()
fig4_bfcl()
print("ALL FIGURES DONE ->", OUT)
