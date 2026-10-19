import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_path = r"C:\ai_models\BitNet-b1.58-2B-4T"
print("Loading model on GPU...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cuda:0"
)
vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
print(f"Base model allocated VRAM: {vram_mb:.1f} MB")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
vram_total_mb = torch.cuda.memory_allocated() / (1024 * 1024)
print(f"Total VRAM with LoRA: {vram_total_mb:.1f} MB")
