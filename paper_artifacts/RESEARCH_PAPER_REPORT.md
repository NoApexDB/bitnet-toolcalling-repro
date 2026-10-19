# Empirical Evaluation of 1.58-Bit Ternary Architecture vs. Quantized Dense Baselines in Agentic Tool-Calling and Swarm Workflows on Commodity x86 Hardware

**Authors:** Research Working Group  
**Date:** October 2026  
**Artifact Repository:** `C:\ai_models\paper_artifacts\`  
**Mirror Path:** `C:\Users\Рам\Desktop\BitNet_Research_Reports\`

---

## Abstract

Recent advances in low-bit Large Language Models (LLMs), notably the 1.58-bit ternary architecture (`BitNet b1.58`, weights $\{-1, 0, 1\}$), promise radical reductions in memory footprint and computational energy. However, while language modeling perplexity has been widely reported, the empirical viability of native ternary networks in autonomous agentic tasks—specifically structured tool invocation, multi-turn context retention, semantic slot filling, and parallel multi-agent swarms—remains unexplored. 

In this work, we present an exhaustive, end-to-end comparative study evaluating `BitNet-b1.58-2B-4T` against dense competitive baselines (`Qwen2.5-1.5B-Instruct` and `Qwen2.5-3B-Instruct`) across both floating-point GPU execution (PyTorch CUDA) and native C++ CPU inference (`bitnet.cpp` / `llama.cpp`). We measure functional accuracy across a 20-scenario Dual Recall matrix, test compliance under explicit tool prohibition (Tool Laziness), deploy a 5-agent incident response swarm, and conduct physical hardware power telemetry via Running Average Power Limit (RAPL) counters on a standard 6-core x86 AMD processor.

Our empirical findings reveal three critical insights:
1. **The Attention-Binding Breakdown in Low-Bit Quantization:** While full-precision PyTorch execution demonstrates that BitNet 2B achieves 85.7% tool execution recall and 100% multi-turn context retention, native 2-bit GGUF quantization (`i2_s` with dynamic INT8 activations on x86) suffers catastrophic attention degradation, collapsing argument extraction to 0.0% by substituting template tokens (`"param1": "value1"`).
2. **Socket Energy vs. Theoretical Arithmetic:** Microsoft's theoretical claim of 71x energy reduction (0.028 J) applies strictly to silicon ALU matrix multiplication on custom ASICs. On commodity x86 CPUs, instruction decoding, DDR4 memory controllers, and uncore overhead level the playing field: BitNet consumes 902.0 mJ/token net, compared to 756.1 mJ/token for 4-bit dense Qwen 1.5B. The primary empirical advantage of BitNet on commodity hardware is memory footprint (1.35 GB vs 1.69 GB for 1.5B and 3.28 GB for 3B), not socket-level energy.
3. **Universal Tool Addiction (Negative Control Collapse):** High tool-calling sensitivity introduces severe selectivity degradation across architectures: both BitNet 2B and Qwen 2.5 3B exhibit an 80% false-positive tool invocation rate (20.0% Negative Control accuracy) when presented with neutral conversational prompts. Expanding system prompts ("prompt stuffing") fails catastrophically, reducing negative control to 0.0% due to attention dilution.

---

## 1. Introduction and Motivation

Autonomous LLM agents require four core computational capabilities:
- **Trigger Discrimination:** Determining whether an external tool is required or if native parametric generation suffices.
- **Syntactic and Semantic Slot Filling:** Extracting arguments from unstructured user input into rigid JSON schemas.
- **Context Retention (Anaphora):** Propagating state and entity references across multi-turn dialog history.
- **Resource Footprint:** Operating within bounded memory and power constraints to enable concurrent multi-agent execution on local edge devices.

The introduction of 1-bit architectures, specifically `BitNet b1.58`, introduced Quantization-Aware Training (QAT) to constrain weights to $\{-1, 0, 1\}$. While mathematical modeling predicts elimination of floating-point multiplications ($x \cdot w$ becomes addition and subtraction), the real-world performance on commodity x86 architectures without specialized ternary tensor cores remains unverified. 

### Core Research Questions
- **RQ1 (Execution Fidelity):** Can a native 2-bit ternary model accurately bind variables and parameters into structured JSON schemas compared to dense baselines?
- **RQ2 (Hardware & Energy Parity):** Does the elimination of matrix multiplications translate to measurable package-level energy savings ($J / \text{token}$) on consumer x86 CPUs?
- **RQ3 (Negative Control & Selectivity):** Does low-bit representation exacerbate tool hallucination when queries do not require tools?
- **RQ4 (Multi-Agent Swarm Concurrency):** Can an optimized C++ server sustain concurrent multi-agent pipelines within low RAM envelopes?

---

## 2. Experimental Methodology

### 2.1 Evaluated Models and Configurations
We evaluate 6 distinct configurations spanning ternary and dense architectures across PyTorch and C++ runtimes:
1. `bitnet_base` (PyTorch FP32, CUDA): Unquantized foundation weights of `BitNet-b1.58-2B-4T` (Microsoft).
2. `bitnet_balanced` (PyTorch FP32 + LoRA v2, CUDA): Supervised fine-tuning on balanced tool-calling and direct-response datasets.
3. `bitnet_2b_i2_s` (C++ GGUF, CPU): Native ternary weights compiled via `bitnet.cpp` (2-bit `i2_s` format, 1.18 GB).
4. `qwen2.5_1.5b_base` (PyTorch FP16, CUDA): Official dense foundation model `Qwen/Qwen2.5-1.5B-Instruct`.
5. `qwen2.5_1.5b_q4_k_m` (C++ GGUF, CPU): Standard dense baseline quantized to 4-bit Medium (1.04 GB).
6. `qwen2.5_3b_q4_k_m` (C++ GGUF, CPU): Scaled dense competitor quantized to 4-bit Medium (1.95 GB).

### 2.2 Benchmarks and Protocols

#### A. Dual Recall Evaluation (20 Scenarios)
A deterministic ($T = 0.0$, repetition penalty 1.15) benchmark evaluating:
- **Positive Single-Turn (Cases 1–12):** Weather retrieval, web search, sandboxed Python code execution, local file reading, SQL database querying.
- **Context Retention / Anaphora (Cases 13–15):** Two-turn interactions requiring resolution of pronouns and implicit references from prior assistant messages (e.g., resolving "Paris" from previous context).
- **Negative Control (Cases 16–20):** Queries requiring purely textual, mathematical, or conceptual answers without invoking tools (gradient descent explanation, translation, stack vs. queue, greetings, async programming concepts).

#### B. Tool Laziness & Cognitive Unloading Protocol (10 Scenarios)
An adversarial protocol where tool schema definitions are explicitly provided alongside a hard negative constraint: `"DO NOT use any tools. Solve directly in text"`. Tested across complex multiplication, prime numbers, translation, and simulated local database queries.

#### C. Multi-Agent Swarm Pipeline & Contention Stress Test
- **Pipeline:** 5-agent incident response swarm (Coordinator $\rightarrow$ Parallel Layer: SQL Forensic Analyst, Log Auditor, Threat Intel $\rightarrow$ Synthesizer / Incident Commander).
- **Contention Test:** 5 simultaneous requests dispatched to `llama-server.exe` configured with 4 execution slots to benchmark queue scheduling and token throughput under contention.

### 2.3 Hardware Telemetry and Environment
- **Processor:** AMD Ryzen 5 5500 (6 Cores / 12 Threads, Zen 3 architecture, 3.6 GHz base, 16 MB L3 cache, TDP 65W).
- **RAM:** DDR4-2666 MHz dual-channel.
- **Operating System:** Windows 11 x64, PowerShell host environment.
- **Energy Telemetry:** Physical socket package power logged at 1 Hz via Windows Performance Data Helper (PDH) Performance Counters querying `\Energy Meter(RAPL_Package0_PKG)\Power`.
- **Memory Telemetry:** Resident Set Size (RSS / WorkingSet64) captured via OS process handles.

---

## 3. Empirical Results

### 3.1 Dual Recall Accuracy Matrix (PyTorch CUDA Baseline)
*Deterministic decoding ($T = 0.0$), full floating-point execution.*

| Configuration | Parameters / Precision | Trigger Recall (%) | Execution Recall (%) | End-to-End Accuracy (%) | Context Retention (%) | Negative Control (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `qwen_base` | 1.5B / FP16 | 33.33 | 20.00 | 6.67 | 0.00 | **100.00** |
| `qwen_distilled` (LoRA) | 1.5B / FP16 | 46.67 | **85.71** | 40.00 | **100.00** | **100.00** |
| `bitnet_base` | 2.0B / FP32 | **93.33** | **85.71** | **80.00** | **100.00** | 20.00 |
| `bitnet_balanced` (LoRA v2) | 2.0B / FP32 | **93.33** | 78.57 | 73.33 | **100.00** | 20.00 |

*Key finding:* In full precision, BitNet's 42% zero-weights act as a discrete routing bias, outperforming dense Qwen 1.5B twofold in end-to-end task completion (80.0% vs. 40.0%) and achieving perfect context retention (100%). However, BitNet displays an acute false-positive invocation rate on neutral text (Negative Control 20.0%).

---

### 3.2 Tool Laziness & Cognitive Unloading Results
*Evaluating adherence to explicit negative constraints (`evaluate_tool_laziness.py`).*

- **Total Test Cases:** 10 (arithmetic, code generation, file access, translation, database queries).
- **Tool Hallucinations:** 0 / 10.
- **Lazy Escape Rate:** **0.00%**.
- **Observation:** When tools are explicitly prohibited in the prompt, BitNet demonstrates zero tool leakage, successfully approximating calculations and writing code in markdown blocks without triggering API syntax.

---

### 3.3 Symmetric C++ Runtime Benchmark (Identical Engine: `llama-server.exe`, 4 CPU Threads)
*Direct comparison under C++ inference, 2048 context window, monitored by physical RAPL telemetry.*

| Model & Prompt Configuration | Format & Weight Size | Peak RAM (MB) | Mean Gen (tok/s) | Net Energy (mJ/tok) | Trigger Recall (%) | Execution Recall (%) | End-to-End Acc (%) | Context Ret. (%) | Negative Control (%) | Energy-to-Success (J/task) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen 2.5 1.5B** (Base) | GGUF Q4_K_M (1.04 GB) | 1694 | **20.9** | **756.1** | 40.0 | 83.3 | 33.3 | 0.0 | **100.0** | 457.4 |
| **BitNet 2B** (Base Prompt) | GGUF i2_s (1.18 GB) | **1386** | 18.1 | 902.0 | 86.7 | 0.0* | 0.0* | 0.0* | 20.0 | 9999.0* |
| **BitNet 2B** (Contrastive Few-Shot) | GGUF i2_s (1.18 GB) | 1393 | 17.7 | 1026.5 | 93.3 | 0.0* | 0.0* | 0.0* | 20.0 | 9999.0* |
| **BitNet 2B** (Bloated Prompt) | GGUF i2_s (1.18 GB) | 1396 | 17.7 | 984.5 | 86.7 | 0.0* | 0.0* | 0.0* | **0.0** | 9999.0* |
| **BitNet 2B** (Explicit NO_TOOL) | GGUF i2_s (1.18 GB) | **1386** | 17.8 | 940.2 | **100.0** | 0.0* | 0.0* | 0.0* | 0.0 | 9999.0* |
| **Qwen 2.5 3B** (Base) | GGUF Q4_K_M (1.95 GB) | 3288 | 10.8 | 1983.8 | **100.0** | **93.3** | **93.3** | **100.0** | 20.0 | **166.5** |

*\*Note: Execution Recall 0.0% reflects GGUF argument extraction failure (template token echo), despite high trigger activation.*

---

### 3.4 Multi-Agent Swarm & Hardware Energy Profiling
*Measured during 5-agent incident response pipeline and 5-agent contention stress test on BitNet C++.*

```
+---------------------------------------------------------------------------------------+
| AMD Ryzen 5 5500 RAPL Telemetry: Swarm Benchmark                                      |
+---------------------------------------------------------------------------------------+
| Baseline Clean Idle Power:      21.21 W                                               |
| Mean Active Power (Swarm):     32.88 W                                               |
| Peak Active Power (Contention): 55.24 W (under 65 W TDP limit)                       |
| Swarm Pipeline Wall-Clock Time: 8.41 s                                                |
| Gross Energy (Pipeline):        276.53 Joules (0.0768 Wh)                             |
| Net Energy (Pipeline):          98.16 Joules  (0.0273 Wh)                             |
| Contention Throughput (5-slot): 40.49 tokens/second aggregate                         |
| Server Memory Delta (4 slots):  +39.50 MB RAM (1662.01 MB -> 1701.51 MB)             |
+---------------------------------------------------------------------------------------+
```

---

## 4. Deep Scientific Analysis & Discussion

### 4.1 The Attention-Binding Degradation in 2-Bit GGUF Quantization
A central finding of this research is the stark divergence between PyTorch execution and C++ GGUF inference for `BitNet-b1.58-2B-4T`.

When executed in PyTorch with FP32 weights on GPU, BitNet 2B correctly binds variable tokens:
```json
// Prompt: "What is the temperature in London right now?"
// PyTorch Output (Task-2790):
{"tool": "get_current_weather", "parameters": {"location": "London"}}
```
However, on the exact same prompt, `llama-server.exe` executing `ggml-model-i2_s.gguf` outputs:
```json
// C++ GGUF Output (Task-2820):
{"tool": "get_current_weather", "parameters": {"param1": "value1"}}
```

#### Mathematical & Microarchitectural Mechanism
1. **Information Density Bottleneck:** A 2B parameter model quantized to 1.58 bits possesses:
   $$\text{Entropy Capacity} = 2.0 \times 10^9 \times 1.58\text{ bits} \approx 3.16\text{ Gbits} \approx 395\text{ MB}$$
   This is mathematically equivalent to a dense model of merely **~400M–500M parameters** in FP16. While 400M parameters suffice for syntax generation and coarse trigger classification, they fall below the empirical capacity threshold required for simultaneous schema maintenance and variable slot-binding.
2. **Dynamic INT8 Activation Quantization on x86:** In `bitnet.cpp`, weights are stored in ternary format, but activations are quantized dynamically to INT8 on CPU. On x86 architectures lacking native ternary dot-product instructions, intermediate Softmax attention matrices lose dynamic range. Consequently, cross-attention between user query tokens (`"London"`) and generated value tokens collapses into the high-frequency attractor basin present in the system prompt template (`"param1": "value1"`).

---

### 4.2 Energy Telemetry: Theoretical ALU vs. Physical Socket Realities
In *The Era of 1-bit LLMs* (Wang et al., 2024), Microsoft reported arithmetic energy reductions of up to 71x for matrix multiplications based on 7nm ASIC circuit simulations (FP16 MatMul: 3.7 pJ vs. 1.58-bit Add: ~0.05 pJ).

Our empirical RAPL measurements on consumer x86 silicon establish that **package-level energy does not mirror ALU-level savings**:
- `qwen2.5_1.5b_q4_k_m`: **756.1 mJ / token net** (mean power: 42.4 W, 20.9 t/s).
- `bitnet_2b_i2_s`: **902.0 mJ / token net** (mean power: 32.9 W, 18.1 t/s).
- `qwen2.5_3b_q4_k_m`: **1983.8 mJ / token net** (mean power: 48.6 W, 10.8 t/s).

#### Why BitNet Consumes Comparable Package Power on x86
On standard x86 processors, there is no hardware ternary accumulator. To execute a ternary layer, `bitnet.cpp` must:
1. Fetch 2-bit packed weight blocks from L3 cache/DRAM.
2. Execute bit-manipulation and lookup-table (LUT) instructions to unpack weights into registers.
3. Perform vector additions using SIMD AVX2/AVX-512 units.

The energy overhead of instruction decoding, cache miss penalties, uncore interconnects (Infinity Fabric), and memory controller traffic completely dominates the raw arithmetic savings. Thus, **the true advantage of BitNet on commodity x86 is RAM conservation (1.35 GB vs 3.28 GB for 3B), not electrical efficiency**.

---

### 4.3 The Universal Tool Addiction (Negative Control Collapse)
A persistent critique of small agentic models is premature invocation of tools on non-tool queries. Our experiments falsify the assumption that this issue is unique to ternary networks:

1. **Parity in Vulnerability:** Both `bitnet_2b_i2_s` and `qwen2.5_3b_q4_k_m` achieved identical **20.0% Negative Control accuracy** (80% false-positive rate). When presented with conceptual queries such as *"Explain the difference between a stack and a queue"*, both models hallucinated tool calls (`execute_python` or `web_search`).
2. **Dense Model Trade-off:** `qwen2.5_1.5b` achieved 100% Negative Control only by being aggressively insensitive to tools overall (Trigger Recall of only 40.0%).
3. **Falsification of the "Prompt Bloating" Hypothesis:** Padding the system prompt with 300 words of background reasoning guidelines caused Negative Control accuracy to plummet from 20.0% to **0.0%**. For sub-3B models, excessive prompt noise triggers attention dilution, accelerating catastrophic schema hallucination.

---

### 4.4 The 3B Parameter Phase Transition
Comparing `qwen2.5_1.5b` and `qwen2.5_3b` confirms a structural phase transition in agentic capability:
- **1.5B Class:** Trigger Recall 40.0%, End-to-End Accuracy 33.3%, Context Retention 0.0%.
- **3.0B Class:** Trigger Recall 100.0%, End-to-End Accuracy 93.3%, Context Retention 100.0%.

At 3B parameters, dense models achieve the representational capacity necessary to reliably parse arguments and retain state across dialog turns. Because Qwen 3B generated correct calls on 14/15 positive tests without repeated retries, its **Energy-to-Success metric was the lowest in the benchmark: 166.5 Joules / task**, outperforming even 1.5B models despite higher instantaneous power.

---

## 5. Threats to Validity and Engineering Limitations

1. **Lack of Hardware Ternary Acceleration:** All CPU benchmarks were conducted on an AMD Zen 3 x86 processor. Results cannot be extrapolated to dedicated neuromorphic, FPGA, or ternary NPU hardware.
2. **LoRA GGUF Export Gap:** While LoRA fine-tuning in PyTorch successfully restored argument extraction (85.7% execution recall), open-source quantization toolchains for `bitnet.cpp` currently lack standardized support for merging low-rank PEFT adapters directly into `i2_s` quantized tensors.
3. **Domain Scope:** Benchmark evaluations focused on structured API tool-calling (REST/SQL/CLI) and do not evaluate unstructured code synthesis or creative generation.

---

## 6. Comprehensive Conclusions and Recommendations

1. **Academic Reality:** Current open-source 2-bit ternary models (`BitNet-b1.58-2B`) cannot be recommended for production agentic tool-use on commodity x86 CPUs due to quantization-induced attention degradation during argument binding.
2. **Viable Local Agent Baseline:** `Qwen2.5-3B-Instruct` in 4-bit quantization (Q4_K_M) represents the current Pareto-optimal baseline for local CPU agents, delivering 93.3% task accuracy at 10.8 t/s and 3.28 GB RAM.
3. **Architectural Mitigation for Tool Addiction:** Neither prompt engineering nor prompt expansion resolves negative control failures. Robust agentic deployment requires a two-stage decoupled architecture: a binary classifier router (Tool Required: Yes/No) preceding the generative execution model.

---

## Appendix: Reproducibility Commands and Artifact Index

### Raw Telemetry Artifacts
- Full Symmetric C++ Matrix: `C:\ai_models\paper_artifacts\symmetric_dense_vs_ternary.json`
- Swarm & RAPL Benchmark: `C:\ai_models\paper_artifacts\swarm_and_energy_benchmark.json`
- PyTorch Dual Recall Matrix: `C:\ai_models\paper_artifacts\cpu_hardware_benchmark.json`
- Tool Laziness Results: `C:\ai_models\paper_artifacts\laziness_test_bitnet_base.json`

### Execution Reproducibility
```powershell
# 1. Run Symmetric Benchmark (Dense vs Ternary on C++)
C:\ComfyUI_windows_portable\python_embeded\python.exe C:\ai_models\benchmark_dense_vs_ternary.py

# 2. Run RAPL Swarm Benchmark (5 Agents)
C:\ComfyUI_windows_portable\python_embeded\python.exe C:\ai_models\benchmark_swarm_and_energy.py

# 3. Print Summary Matrix
C:\ComfyUI_windows_portable\python_embeded\python.exe C:\ai_models\print_table.py
```
