# -*- coding: utf-8 -*-
"""
Assemble standalone public artifacts reproduction package for Task T1.
"""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")

SRC_ARTIFACTS = r"C:\ai_models\paper_artifacts"
SRC_ROOT = r"C:\ai_models"
DEST = r"C:\ai_models\public_artifacts_package"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    if os.path.exists(DEST):
        shutil.rmtree(DEST)
    os.makedirs(DEST, exist_ok=True)
    os.makedirs(os.path.join(DEST, "paper_artifacts"), exist_ok=True)
    os.makedirs(os.path.join(DEST, "runners"), exist_ok=True)
    os.makedirs(os.path.join(DEST, "data"), exist_ok=True)

    # 1. Copy all files from paper_artifacts
    for root, dirs, files in os.walk(SRC_ARTIFACTS):
        rel = os.path.relpath(root, SRC_ARTIFACTS)
        target_dir = os.path.join(DEST, "paper_artifacts", rel) if rel != "." else os.path.join(DEST, "paper_artifacts")
        os.makedirs(target_dir, exist_ok=True)
        for f in files:
            src_f = os.path.join(root, f)
            dest_f = os.path.join(target_dir, f)
            shutil.copy2(src_f, dest_f)

    # Verify manifest SHA-256 byte-for-byte
    manifest_src = os.path.join(SRC_ARTIFACTS, "BENCHMARK_ARTIFACTS_MANIFEST.json")
    manifest_dst = os.path.join(DEST, "paper_artifacts", "BENCHMARK_ARTIFACTS_MANIFEST.json")
    assert sha256(manifest_src) == sha256(manifest_dst), "Manifest SHA-256 mismatch!"
    print(f"Manifest verified: {sha256(manifest_dst)}")

    # 2. Copy all python runners and benchmark scripts
    for py in glob.glob(os.path.join(SRC_ROOT, "*.py")):
        shutil.copy2(py, os.path.join(DEST, "runners", os.path.basename(py)))

    # 3. Copy datasets
    for json_data in ["test_suite_standardized.json", "test_suite_explicit_schemas.json", "heldout_test_suite_30.json"]:
        p = os.path.join(SRC_ROOT, json_data)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(DEST, "data", json_data))

    # 4. Generate standalone English README
    readme_content = """# Reproducible Artifacts: Agents on Ternary Models (BitNet b1.58 2B)

This repository provides the complete, authoritative artifact suite and replication code for the research paper:
**"Agents on Lightweight Ternary Models: Root-Cause Investigation of Tool Calling Failure in the BitNet b1.58 2B Inference Stack"** (2026).

---

## 1. System Specifications & Testbed Hardware

All benchmarks and physical energy measurements were conducted on the following testbed:
- **Processor:** AMD Ryzen 5 5500 (6 Cores / 12 Threads, Zen 3 architecture, 3.6 GHz base / 4.2 GHz boost, 65 W TDP).
- **Memory:** 16 GB DDR4-2666.
- **Operating System:** Windows 11 Home x86_64.
- **Compiler / Toolchain:** MSVC v19 / LLVM Clang 18, Ninja 1.12, CMake 3.28+.
- **Python Environment:** Python 3.12 (with `torch>=2.5.0`, `transformers>=4.49.0`, `tokenizers`, `numpy`).
- **Telemetry / Energy Metering:** Hardware RAPL MSR telemetry via Windows Performance Counters:
  Counter path: `\\Energy Meter(RAPL_Package0_PKG)\\Energy` (sampled at 1 Hz).

---

## 2. Upstream Base Commits & Repository State

- **BitNet Framework:** `microsoft/BitNet` at commit `0b341e582afbf9e1011f24744b554c96a3477eb5` (`origin/main`).
- **Inference Engine Submodule:** `3rdparty/llama.cpp` locked to commit `390c307752ab78fd8189f359d6954c9ba1be74af` (`isHuangXin/llama.cpp`).
- **GGUF Specification:** `gguf-py` v0.10.0+ included within `3rdparty/llama.cpp/gguf-py`.

---

## 3. Models and Expected SHA-256 Checksums

Due to size considerations, raw model weights are not hosted directly in this repository. Please obtain checkpoints from Hugging Face:

1. **BitNet-b1.58-2B-4T (Native 1.58-bit Base):**
   - Hugging Face Repo: [`microsoft/BitNet-b1.58-2B-4T`](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T)
   - Quantized GGUF (`ggml-model-i2_s.gguf`, 1,187,801,280 bytes):
     `SHA-256: 4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`
   - FP32 Reference Checkpoint (`model.safetensors`, 9,659,120,192 bytes):
     `SHA-256: 09d20c571764ebdb7d59df4567ae2c43eeea1eb4fa98aa55a9b7407519ff0f1e`

2. **Qwen2.5-1.5B-Instruct (Dense 4-bit Baseline):**
   - Hugging Face Repo: [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
   - Quantized GGUF (`qwen2.5-1.5b-instruct-q4_k_m.gguf`, 1,117,320,736 bytes):
     `SHA-256: 57973719a79fa4cf2e4d9eb8d5f303494793d980ea429eecdd5f8bb6b49feaa9`

3. **Qwen2.5-3B-Instruct (Dense 4-bit Baseline):**
   - Hugging Face Repo: [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
   - Quantized GGUF (`qwen2.5-3b-instruct-q4_k_m.gguf`, 2,104,932,768 bytes):
     `SHA-256: cf240092ad4a0d9e7943c22b9c7ae41c2c8f8b8fb49294e9f73369165d447f52`

---

## 4. Building the Fixed Engine (`llama-server-relu2`)

The root cause of tool-calling failure in the upstream C++ implementation is an activation type mismatch in the gated FFN graph:

```bash
# 1. Clone repository with submodules
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet
git checkout 0b341e582afbf9e1011f24744b554c96a3477eb5
git submodule update --init --recursive

# 2. Apply the one-line FFN activation fix in 3rdparty/llama.cpp/src/models/bitnet.cpp:
# Change line 133: LLM_FFN_SILU -> LLM_FFN_RELU_SQR
```

Diff patch:
```diff
diff --git a/src/models/bitnet.cpp b/src/models/bitnet.cpp
index 985c8d6fc..ec5dc3a6b 100644
--- a/src/models/bitnet.cpp
+++ b/src/models/bitnet.cpp
@@ -130,7 +130,7 @@ llama_model_bitnet::graph::graph(const llama_model & model, const llm_graph_para
                 model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
                 NULL,                      NULL, NULL,
                 NULL,
-                LLM_FFN_SILU, LLM_FFN_PAR, il);
+                LLM_FFN_RELU_SQR, LLM_FFN_PAR, il);
         cb(cur, "ffn_sub_out", il);
 
         cur = build_norm(cur,
```

```bash
# 3. Configure and compile with Ninja
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target llama-server
```

---

## 5. Running the Canonical Benchmarks

### 5.1 Server Execution Protocol
Start the local server instance with strictly bound compute parameters:
```bash
./build/bin/llama-server -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -t 4 -c 2048 --port 8080
```
- Server flags: `-t 4` (threads), `-c 2048` (context length).
- Generation parameters: Temperature $T=0.0$ (greedy), `repetition_penalty = 1.15` (window 64).

### 5.2 Symmetric Post-Fix Benchmark
To reproduce the full 4-way evaluation matrix (BitNet ReLU²+HF tokens, BitNet ReLU²+String, Qwen 1.5B, Qwen 3B):
```bash
python runners/benchmark_postfix_symmetric.py --data data/test_suite_standardized.json --output paper_artifacts/symmetric_dense_vs_ternary_postfix_repro.json
```

### 5.3 Speed and Decode Rate Evaluation
To measure fixed-budget decode throughput with prefill cache disabled:
```bash
python runners/speed_probe_fixed.py --port 8080 --budget 128 --repeat 3
```

---

## 6. Artifact Verification and Integrity Manifest

All results, raw transition matrices, logs, and token-level diffs are indexed with SHA-256 hashes in:
`paper_artifacts/BENCHMARK_ARTIFACTS_MANIFEST.json`

To verify byte-level integrity of the artifact repository:
```bash
python runners/revise_manifest_review18_r2.py --verify
```
"""

    with open(os.path.join(DEST, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"Public artifact package assembled at: {DEST}")

    # Initialize git repo in package
    try:
        subprocess.run(["git", "init"], cwd=DEST, check=True, stdout=subprocess.PIPE)
        subprocess.run(["git", "config", "user.name", "NoApexDB"], cwd=DEST, check=True)
        subprocess.run(["git", "config", "user.email", "dongandondigidon123@gmail.com"], cwd=DEST, check=True)
        subprocess.run(["git", "add", "."], cwd=DEST, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit: Complete reproducible artifacts suite and runners"], cwd=DEST, check=True, stdout=subprocess.PIPE)
        print("Git repository initialized and committed successfully in public package!")
    except Exception as e:
        print(f"Git init notice: {e}")


if __name__ == "__main__":
    main()
