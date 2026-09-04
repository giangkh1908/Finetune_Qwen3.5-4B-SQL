---
language:
- vi
- en
license: apache-2.0
tags:
- text-to-sql
- financial
- sqlite
- unsloth
- qwen
base_model: Qwen/Qwen3.5-4B
metrics:
- accuracy
pipeline_tag: text-generation
---

# 📊 Qwen3.5-4B-Financial-SQL (Vietnamese Financial Text-to-SQL)

Mô hình ngôn ngữ chuyên biệt cho bài toán **Text-to-SQL Báo Cáo Tài Chính Việt Nam** được finetune từ base model `Qwen/Qwen3.5-4B` trên tập dữ liệu Gold Standard đã kiểm chứng thực thi 100% trên CSDL SQLite Báo cáo tài chính niêm yết (HOSE, HNX).

## 🌟 Đặc điểm nổi bật
- **Suy luận chuỗi tư duy (`<think>`)**: Mô hình phân tích logic kế toán, trích xuất mã cổ phiếu (`ticker`), năm tài chính, loại báo cáo (`consolidated`/`separate`), biểu mẫu (`balance_sheet`, `income_statement`, `cash_flow`), và lập công thức toán học tường minh (chênh lệch, tỷ lệ tăng trưởng %, biên lợi nhuận ròng).
- **Pure ANSI SQLite SQL**: 100% câu truy vấn SQL chuẩn cú pháp SQLite, tuyệt đối không dùng tham số ẩn `?` hay rule-based fallback.
- **Tính truy vết kiểm toán (Audit Provenance)**: Mọi câu truy vấn trả về dữ liệu đều đi kèm nguồn gốc minh bạch (`item_name`, `period_label`, `raw_value`, `unit`, `page_no`, `source_doc`).
- **Merged 16-bit Full Model**: Mô hình độc lập hoàn chỉnh (Float16/Bfloat16), sẵn sàng phục vụ suy luận bằng **vLLM**, **SGLang**, **Transformers**.

---

## 🗄️ Cấu trúc bảng Database (`financial_facts`)
```sql
CREATE TABLE financial_facts (
    ticker TEXT,          -- Mã chứng khoán (VJC, DBC, ACB, FPT, HPG...)
    company_name TEXT,    -- Tên đầy đủ doanh nghiệp
    year INTEGER,         -- Năm tài chính (2017 - 2024)
    report_type TEXT,     -- 'consolidated' (hợp nhất) hoặc 'separate' (công ty mẹ)
    statement TEXT,       -- 'balance_sheet', 'income_statement', 'cash_flow', 'notes'
    item_name TEXT,       -- Tên chỉ tiêu kế toán tiếng Việt có dấu
    item_name_ascii TEXT, -- Tên chỉ tiêu không dấu (dùng với LIKE '%keyword%')
    period_label TEXT,    -- Nhãn kỳ ('31/12/2023', 'Số cuối năm', v.v.)
    value_vnd REAL,       -- Giá trị quy đổi VND
    raw_value TEXT,       -- Giá trị nguyên bản trên báo cáo
    unit TEXT,            -- Đơn vị tính ('VND', 'triệu đồng', 'USD'...)
    page_no INTEGER,      -- Số trang trên báo cáo tài chính PDF
    source_doc TEXT       -- Tên file PDF nguồn
);
```

---

## 🚀 Hướng dẫn sử dụng (Quickstart)

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

system_prompt = """You are a financial SQLite expert. Given the database schema, analyze the question step-by-step inside <think>...</think>, then provide the exact ANSI SQLite SQL query."""

question = "Lợi nhuận sau thuế của CTCP Tập đoàn FPT năm 2023 là bao nhiêu tỷ đồng?"

prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\\n<|im_start|>assistant\n"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.0)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)

print(response)
```

---

## 📈 Kết quả huấn luyện (Training Metrics)
- **Base Model**: `Qwen/Qwen3.5-4B`
- **Tập dữ liệu**: 582 mẫu Gold Standard Báo Cáo Tài Chính Việt Nam.
- **Global Steps**: 73 steps (Batch effective = 16).
- **Training Loss**: **0.3059** (Hội tụ tối ưu).
- **Phương pháp**: QLoRA 4-bit qua Unsloth + Response-only loss masking.
