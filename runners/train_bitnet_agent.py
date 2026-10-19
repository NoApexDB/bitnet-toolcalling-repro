import json
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_PATH = r"C:\ai_models\BitNet-b1.58-2B-4T"
DATASET_PATH = r"C:\ai_models\agent_distill_dataset_clean.jsonl"
OUTPUT_DIR = r"C:\ai_models\bitnet_agent_lora"

class ChatMLDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, max_length=384):
        self.examples = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                messages = data.get("messages", [])
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                encoded = tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                input_ids = encoded["input_ids"].squeeze(0)
                attention_mask = encoded["attention_mask"].squeeze(0)
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
                self.examples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def main():
    print("=== ЗАПУСК ДИСТИЛЛЯЦИИ / LORA ОБУЧЕНИЯ BITNET 2B ===", flush=True)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print("[1/5] Загрузка токенизатора...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/5] Подготовка датасета...", flush=True)
    dataset = ChatMLDataset(DATASET_PATH, tokenizer, max_length=384)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"-> Всего примеров: {len(dataset)}", flush=True)

    print("[3/5] Загрузка базовой модели на GPU...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cuda:0"
    )
    for param in base_model.parameters():
        param.requires_grad = False

    print("[4/5] Применение LoRA адаптера...", flush=True)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    print("[5/5] Обучение (3 эпохи, AdamW lr=2e-4, grad clip=1.0)...", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    model.train()

    epochs = 3
    grad_accum_steps = 4
    step = 0

    for epoch in range(epochs):
        total_loss = 0.0
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to("cuda:0")
            attention_mask = batch["attention_mask"].to("cuda:0")
            labels = batch["labels"].to("cuda:0")

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum_steps

            loss.backward()
            total_loss += loss.item() * grad_accum_steps

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1

        avg_loss = total_loss / len(dataloader)
        print(f"-> Эпоха [{epoch+1}/{epochs}] | Средний Loss: {avg_loss:.4f}", flush=True)

    print(f"\n[DONE] Сохранение обученного LoRA адаптера в {OUTPUT_DIR}...", flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("LoRA веса успешно сохранены.", flush=True)

if __name__ == "__main__":
    main()
