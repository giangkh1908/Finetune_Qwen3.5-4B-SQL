---
language:
- vi
- en
license: apache-2.0
tags:
- lora
- peft
- text-to-sql
- financial
- sqlite
- unsloth
base_model: Qwen/Qwen3.5-4B
pipeline_tag: text-generation
---

# 🔧 Qwen3.5-4B-Financial-SQL-LoRA (PEFT Adapter)

Trọng số LoRA Adapter (dung lượng cực nhẹ **~30 MB**) chuyên biệt cho Text-to-SQL Báo Cáo Tài Chính Việt Nam, được tinh chỉnh trên kiến trúc `Qwen/Qwen3.5-4B`.

## ⚙️ Cấu hình LoRA (Hyperparameters)
- **Rank (r)**: 16
- **Alpha (alpha)**: 32
- **Dropout**: 0.0
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Trainable Parameters**: ~20M parameters (< 0.5% tổng số tham số của model)
- **Epochs**: 3 (627 steps) | **Final Loss**: `0.0978`

---

## 🚀 Cách nạp Adapter vào Base Model (Unsloth / PEFT)

```python
import torch
from unsloth import FastLanguageModel

# 1. Nạp Base Model dạng 4-bit (chỉ tốn ~2.5GB VRAM)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-4B",
    max_seq_length=2048,
    load_in_4bit=True,
)

# 2. Gắn LoRA Adapter từ Hugging Face
model = FastLanguageModel.get_peft_model(model)
model.load_adapter("giangkh19/Qwen3.5-4B-Financial-SQL-LoRA")
FastLanguageModel.for_inference(model)

# 3. Suy luận
prompt = "<|im_start|>user\nTổng tài sản năm 2023 của Vinamilk (VNM) là bao nhiêu?<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.01)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```
