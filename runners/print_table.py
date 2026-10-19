import json

with open(r"C:\ai_models\paper_artifacts\symmetric_dense_vs_ternary.json", "r", encoding="utf-8") as f:
    data = json.load(f)

header = f"{'Model':<28} | {'RAM':<8} | {'Gen t/s':<8} | {'Net mJ/tok':<11} | {'Trig%':<6} | {'Exec%':<6} | {'E2E%':<6} | {'Ctx%':<6} | {'Neg%':<6} | {'E-to-Succ (J)':<14}"
print(header)
print("-" * len(header))

for name, m in data.items():
    ram = f"{m['memory']['ram_peak_mb']:.0f} MB"
    tps = f"{m['performance']['mean_tps']:.1f}"
    net_mj = f"{m['power_and_energy']['net_mj_per_token']:.1f}"
    acc = m["accuracy"]
    tr = f"{acc['trigger_recall_pct']:.1f}"
    er = f"{acc['execution_recall_pct']:.1f}"
    e2e = f"{acc['end_to_end_accuracy_pct']:.1f}"
    ctx = f"{acc['context_retention_pct']:.1f}"
    neg = f"{acc['negative_control_accuracy_pct']:.1f}"
    e_succ = f"{m['power_and_energy']['energy_to_success_joules']:.1f}"
    print(f"{name:<28} | {ram:<8} | {tps:<8} | {net_mj:<11} | {tr:<6} | {er:<6} | {e2e:<6} | {ctx:<6} | {neg:<6} | {e_succ:<14}")
