---
language:
- vi
- en
license: apache-2.0
tags:
- gguf
- ollama
- text-to-sql
- financial
- sqlite
- qwen3.5
pipeline_tag: text-generation
---

# 🦙 Qwen3.5-4B-Financial-SQL (GGUF Quantized for Ollama)

Phiên bản lượng tử hóa định dạng **GGUF (Q4_K_M)** tối ưu dung lượng và tốc độ, sẵn sàng chạy mượt mà trên laptop, PC (kể cả máy không có card đồ họa rời) thông qua **Ollama** hoặc **llama.cpp**.

- **Dung lượng**: ~2.5 GB
- **VRAM/RAM yêu cầu**: >= 4 GB RAM

---

## 🚀 Hướng dẫn chạy 1-Click với Ollama

### Bước 1: Tải file GGUF
Tải file `qwen3_5_4b_financial_sql.Q4_K_M.gguf` từ mục **Files and versions** của repo này về máy.

### Bước 2: Tạo file `Modelfile`
Tạo một file văn bản đặt tên là `Modelfile` cùng thư mục với file `.gguf` vừa tải, nội dung:

```dockerfile
FROM ./qwen3_5_4b_financial_sql.Q4_K_M.gguf

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

SYSTEM """You are a financial SQLite expert for Vietnamese corporate financial reports.
Given the database schema for table `financial_facts`, analyze the question and return:
1. A concise reasoning block enclosed in <think>...</think>.
2. The exact ANSI SQLite query inside ```sql ... ```."""

PARAMETER temperature 0.01
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
```

### Bước 3: Tạo và chạy Model trong Ollama
Mở terminal và gõ:

```bash
ollama create qwen-sql -f Modelfile
ollama run qwen-sql "Doanh thu năm 2023 của FPT đạt bao nhiêu tỷ đồng?"
```

---

## 💻 Chạy với `llama.cpp` (CLI)

```bash
./llama-cli -m qwen3_5_4b_financial_sql.Q4_K_M.gguf \
  -p "<|im_start|>user\nLợi nhuận sau thuế năm 2023 của Hòa Phát (HPG) là bao nhiêu tỷ?<|im_end|>\n<|im_start|>assistant\n" \
  -n 512 --temp 0.0
```
