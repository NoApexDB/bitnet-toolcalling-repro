import time
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=== PYTORCH & TRANSFORMERS ENVIRONMENT ===")
print("Transformers version:", transformers.__version__)
print("PyTorch version:", torch.__version__)

# Force 4 threads on CPU to match C++ -t 4
torch.set_num_threads(4)
print("Configured CPU threads:", torch.get_num_threads())

model_path = r"C:\ai_models\BitNet-b1.58-2B-4T"

print("\n=== LOADING TOKENIZER ===")
tok = AutoTokenizer.from_pretrained(model_path)
print("Tokenizer class:", tok.__class__.__name__)
print("BOS:", tok.bos_token, tok.bos_token_id)
print("EOS:", tok.eos_token, tok.eos_token_id)

print("\n=== LOADING MODEL ON CPU ===")
t0 = time.perf_counter()
# Note: config.json specifies torch_dtype="bfloat16", but on CPU float32 is standard
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)
load_time = time.perf_counter() - t0
print(f"Loaded in {load_time:.2f} s")
print("Model class:", model.__class__.__name__)
print("Model architecture:", model.config.architectures)
print("Config torch_dtype:", model.config.torch_dtype)

# Inspect sample layer weights and dtypes
first_layer_proj = model.model.layers[0].mlp.down_proj
print("\n=== LAYER 0 DOWN_PROJ INSPECTION ===")
print("down_proj class:", first_layer_proj.__class__.__name__)
for param_name, param in first_layer_proj.named_parameters():
    print(f"Parameter: {param_name}, shape: {param.shape}, dtype: {param.dtype}, device: {param.device}")
for buf_name, buf in first_layer_proj.named_buffers():
    print(f"Buffer: {buf_name}, shape: {buf.shape}, dtype: {buf.dtype}, device: {buf.device}")

# Smoke generation test
print("\n=== SMOKE GENERATION TEST (CPU, 4 THREADS) ===")
sample_dialog = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Say hello in one word."}
]
prompt = tok.apply_chat_template(sample_dialog, tokenize=False, add_generation_prompt=True)
print("Formatted prompt:", repr(prompt))

inputs = tok(prompt, return_tensors="pt").to("cpu")
print("Input token count:", inputs.input_ids.shape[1])

model.eval()
t_gen_start = time.perf_counter()
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
        repetition_penalty=1.15,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id
    )
gen_elapsed = time.perf_counter() - t_gen_start

gen_tokens = out[0][inputs.input_ids.shape[1]:]
gen_text = tok.decode(gen_tokens, skip_special_tokens=False)

print(f"Generated {len(gen_tokens)} tokens in {gen_elapsed:.2f} s ({len(gen_tokens)/gen_elapsed:.2f} t/s)")
print("Generated text:", repr(gen_text))
print("Smoke test successfully completed.")
