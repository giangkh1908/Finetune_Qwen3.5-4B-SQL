---
language:
- vi
- en
license: apache-2.0
tags:
- text-to-sql
- financial
- sqlite
- qwen
- qwen3.5
- reasoning
- cot
base_model: Qwen/Qwen3.5-4B
metrics:
- accuracy
pipeline_tag: text-generation
---

# 📊 Qwen3.5-4B-Financial-SQL (Vietnamese Financial Text-to-SQL)

Mô hình ngôn ngữ chuyên biệt cho bài toán **Text-to-SQL Báo Cáo Tài Chính Việt Nam**, được huấn luyện từ base model `Qwen/Qwen3.5-4B` trên tập dữ liệu **1.847 mẫu Gold Standard** kiểm định thực thi 100% trên cơ sở dữ liệu SQLite BCTC các doanh nghiệp niêm yết (HOSE, HNX, UPCoM).

## 🌟 Đặc điểm nổi bật
- **Suy luận chuỗi tư duy (`<think>`)**: Trước khi viết SQL, mô hình tự động phân tích: khái niệm kế toán, bảng tài chính (`balance_sheet`, `income_statement`, `cash_flow`), công thức tính toán và đơn vị quy đổi (tỷ đồng, triệu đồng, %).
- **Chuẩn ANSI SQLite**: 100% câu truy vấn tuân thủ cú pháp SQLite Native, hỗ trợ các truy vấn phức tạp (CTEs, Window functions, lọc nhãn thời gian `period_label`, truy vấn đa kỳ).
- **Tính truy vết kiểm toán (Audit Provenance)**: Mọi câu truy vấn trả về dữ liệu đều kèm nguồn gốc rõ ràng (`raw_value`, `unit`, `page_no`, `source_doc`).
- **Merged 16-bit Full Weights**: Model độc lập hoàn chỉnh, tương thích tối đa với **vLLM**, **SGLang**, **Transformers** và **Ollama**.

---

## 🗄️ Cấu trúc Schema (`financial_facts`)

```sql
CREATE TABLE financial_facts (
    ticker TEXT,          -- Mã chứng khoán (VNM, FPT, HPG, VJC...)
    company_name TEXT,    -- Tên đầy đủ doanh nghiệp
    year INTEGER,         -- Năm tài chính (2016 - 2024)
    report_type TEXT,     -- 'consolidated' (hợp nhất) hoặc 'separate' (công ty mẹ)
    statement TEXT,       -- 'balance_sheet', 'income_statement', 'cash_flow', 'notes'
    item_name TEXT,       -- Tên chỉ tiêu tiếng Việt có dấu
    item_name_ascii TEXT, -- Tên chỉ tiêu không dấu (dùng với LIKE '%keyword%')
    period_label TEXT,    -- Nhãn kỳ báo cáo ('31/12/2023', 'Số cuối năm'...)
    value_vnd REAL,       -- Giá trị quy đổi sang VND
    raw_value TEXT,       -- Số liệu nguyên bản trên BCTC PDF
    unit TEXT,            -- Đơn vị tiền tệ
    page_no INTEGER,      -- Trang số trên BCTC PDF
    source_doc TEXT       -- Tên file PDF nguồn
);
```

---

## 🚀 Hướng dẫn suy luận (Quickstart với Transformers)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "giangkh19/Qwen3.5-4B-Financial-SQL"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto"
)

system_prompt = """You are a financial SQLite expert for Vietnamese corporate financial reports.
Given the database schema for table `financial_facts`, analyze the question and return:
1. A concise reasoning block enclosed in <think>...</think>.
2. The exact ANSI SQLite query inside ```sql ... ```."""

question = "Doanh thu thuần năm 2023 của CTCP Sữa Việt Nam (mã VNM) là bao nhiêu tỷ đồng?"

prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.01)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False))
```

---

## 📈 Thông số huấn luyện (Training Specs)
- **Base Model**: `Qwen/Qwen3.5-4B`
- **Tập dữ liệu**: 1.847 mẫu (1.665 Train / 182 Val)
- **Số Epochs**: 3 Epochs (627 steps, effective batch = 8)
- **Training Loss**: `0.0978` (Hội tụ tối ưu)
- **Gradient Norm**: `0.41` (Cực kỳ ổn định)
- **Cơ chế Loss**: Response-only loss masking (chỉ tính phạt trên phần suy luận và câu SQL)
