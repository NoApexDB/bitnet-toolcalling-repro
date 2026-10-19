import subprocess
import time

def get_rapl_samples():
    cmd = [
        "powershell", "-NoProfile", "-Command",
        "$p = (Get-Counter '\\Energy Meter(RAPL_Package0_PKG)\\Power').CounterSamples[0].CookedValue; "
        "$e = (Get-Counter '\\Energy Meter(RAPL_Package0_PKG)\\Energy').CounterSamples[0].CookedValue; "
        "Write-Output \"$p|$e\""
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = res.stdout.strip().split('|')
    p_val = float(out[0].replace(',', '.'))
    e_val = float(out[1].replace(',', '.'))
    return p_val, e_val

p1, e1 = get_rapl_samples()
print(f"t0: Power = {p1:.2f} mW ({p1/1000.0:.2f} W), Energy counter = {e1}")
time.sleep(1.0)
p2, e2 = get_rapl_samples()
print(f"t1: Power = {p2:.2f} mW ({p2/1000.0:.2f} W), Energy counter = {e2}")
delta_e = e2 - e1
avg_p = (p1 + p2) / 2.0
print(f"Delta energy raw: {delta_e}, Avg power: {avg_p/1000.0:.2f} W")
# Check ratio between delta energy and power * 1.0s
print(f"Ratio delta_e / avg_p_mW: {delta_e / avg_p if avg_p else 0}")
