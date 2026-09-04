# 🚀 FinetuneV2: Vietnamese Financial Text-to-SQL (Qwen3.5-4B)

Hệ thống huấn luyện mô hình ngôn ngữ chuyên sâu cho bài toán **Text-to-SQL Báo Cáo Tài Chính Doanh Nghiệp Niêm Yết Việt Nam** (HOSE, HNX). Mô hình được trang bị khả năng suy luận chuỗi tư duy kế toán (`<think> ... </think>`), tự động lập công thức toán học tường minh (chênh lệch, tỷ lệ tăng trưởng %, biên lợi nhuận ròng), sinh 100% ANSI SQLite SQL thuần túy và đính kèm đầy đủ nguồn gốc kiểm toán (**Provenance & Traceability**).

---

## 🌟 1. Điểm Nổi Bật Của Mô Hình & Dự Án

1. **Chuỗi suy luận kế toán nội tại (`<think>`)**:
   - Trước khi sinh SQL, mô hình tự động phân tích: bóc tách mã chứng khoán (`ticker`), năm tài chính, loại báo cáo (`consolidated` hoặc `separate`), biểu mẫu (`balance_sheet`, `income_statement`, `cash_flow`, `notes`), và chỉ tiêu kế toán ngắn nhất không dấu (`item_name_ascii`).
   - Tự động lập công thức toán học tường minh:
     * **Chênh lệch / Hiệu số**: `(Val_Year2 - Val_Year1) / 1e9`
     * **% Tăng trưởng / Thay đổi**: `(Val_Year2 - Val_Year1) * 100.0 / Val_Year1`
     * **Biên lợi nhuận / Tỷ số**: `(Val_A / Val_B) * 100.0`
2. **Pure ANSI SQLite SQL (Zero-Placeholder & Zero-Fallback)**:
   - 100% câu truy vấn SQL tuân thủ chuẩn ANSI SQLite, chạy trực tiếp trên cơ sở dữ liệu thật mà không dùng tham số ẩn `?` hay các quy tắc fallback chắp vá.
   - Tự động thêm điều kiện chốt sổ cuối năm: `(period_label LIKE '%cuối năm%' OR period_label LIKE '%31/12%' OR period_label LIKE '%col_5%')` khi truy vấn Bảng cân đối kế toán.
3. **Tính truy vết kiểm toán (Audit Provenance)**:
   - Mọi câu truy vấn dữ liệu đều trả về đủ 6 cột minh chứng: `item_name, period_label, raw_value, unit, page_no, source_doc`.
4. **Khả năng khái quát hóa vượt trội (Zero-Shot Entity Transfer)**:
   - Đã được kiểm chứng thực tế khi hoán đổi thực thể sang các doanh nghiệp hoàn toàn mới trong database (`HPG`, `MSN`, `MWG`, `VJC`, `VNM`) với độ chính xác số học **100%**.

---

## 📊 2. Tập Dữ Liệu Huấn Luyện (Gold Standard Dataset)

Tập dữ liệu được tinh chọn và kiểm tra thực thi thành công 100% trên cơ sở dữ liệu `data/financial.db` (2.116.243 bản ghi từ 100 công ty niêm yết):

| Tập dữ liệu | Đường dẫn | Số lượng mẫu | Mô tả |
| :--- | :--- | :---: | :--- |
| **Train Set** | `data/processed/train.jsonl` | **582 mẫu** | 100% có `<think>`, công thức toán học và SQL thực thi ra số liệu thật. |
| **Validation Set** | `data/processed/val.jsonl` | **64 mẫu** | Bộ kiểm định độc lập để theo dõi quá trình hội tụ. |
| **Database Gốc** | `data/financial.db` | **2,116,243 rows** | CSDL SQLite Native 10 năm (2015–2025) của 100 mã cổ phiếu. |

---

## 📈 3. Kết Quả Huấn Luyện (Training Metrics)

Mô hình được huấn luyện trên **Kaggle GPU Tesla T4 (16GB VRAM)** thông qua framework **Unsloth**:

* **Mô hình nền tảng (Base Model)**: `Qwen/Qwen3.5-4B`
* **Kỹ thuật**: QLoRA 4-bit (NF4), Rank $r=16$, $\alpha=32$, Target 7 Linear Modules (`q, k, v, o, gate, up, down_proj`).
* **Hàm mất mát**: **Response-only Loss Masking** (`train_on_responses_only`) — chỉ tính hàm mất mát trên phần sinh của mô hình (`<think>` + SQL), không tính trên câu hỏi người dùng.
* **Effective Batch Size**: $2 \times 8 = 16$ (per_device_batch_size = 2, gradient_accumulation = 8).
* **Số bước huấn luyện (Global Steps)**: **73 steps** (1 Epoch).
* **Training Loss**: **`0.3059`** (Hội tụ tối ưu từ ~2.0 xuống 0.305).
* **Tỷ lệ thực thi thành công SQL**: **10/10 (100%)** trên bài test benchmark thực tế.

---

## 📦 4. Các Phiên Bản Mô Hình Đã Phát Hành (Artifacts)

Mô hình đã được đóng gói và phát hành chính thức trên Hugging Face Hub:

| Phiên bản | Hugging Face Repository | Dung lượng | Mục đích sử dụng |
| :--- | :--- | :---: | :--- |
| **Full Merged 16-bit** | [giangkh19/Qwen3.5-4B-Financial-SQL](https://huggingface.co/giangkh19/Qwen3.5-4B-Financial-SQL) | **~8 GB** | Mô hình đầy đủ độc lập, phục vụ triển khai với **vLLM**, **SGLang**, **Transformers**. |
| **LoRA Adapter** | [giangkh19/Qwen3.5-4B-Financial-SQL-LoRA](https://huggingface.co/giangkh19/Qwen3.5-4B-Financial-SQL-LoRA) | **~100 MB** | Bản adapter nhẹ để tích hợp trên base model. |
| **GGUF Q4_K_M (Ollama)** | [hf.co/giangkh19/Qwen3.5-4B-Financial-SQL-GGUF](https://huggingface.co/giangkh19/Qwen3.5-4B-Financial-SQL-GGUF) | **~2.8 GB** | Bản lượng tử hóa 4-bit chạy mượt trên card đồ họa phổ thông (GTX 1650 / RTX 3050) |

---

## 🚀 5. Hướng Dẫn Sử Dụng Nhanh (Quickstart)

### Cách 1: Chạy cục bộ với Ollama (Khuyên dùng - Cực nhẹ & Nhanh)

1. Tải và chạy trực tiếp bằng Ollama:
```bash
ollama run hf.co/giangkh19/Qwen3.5-4B-Financial-SQL-GGUF:Q4_K_M
```

2. Hoặc tạo alias ngắn gọn:
```bash
ollama cp hf.co/giangkh19/Qwen3.5-4B-Financial-SQL-GGUF:Q4_K_M qwen3.5-4b-financial-sql
ollama run qwen3.5-4b-financial-sql
```

---

### Cách 2: Suy luận bằng Python với `transformers` (Chế độ 4-bit)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "giangkh19/Qwen3.5-4B-Financial-SQL"

# Tối ưu 4-bit: chỉ tốn ~2.5GB VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

system_prompt = """You are a financial SQLite expert. Given the database schema, analyze the question step-by-step inside <think>...</think>, then provide the exact ANSI SQLite SQL query."""

question = "Doanh thu thuần của CTCP Tập đoàn Hòa Phát (HPG) năm 2021 là bao nhiêu tỷ đồng?"

prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

text_tok = getattr(tokenizer, "tokenizer", tokenizer)
inputs = text_tok(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.0)
response = text_tok.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)

print(response)
```

---

### Cách 3: Triển khai Serving hiệu năng cao với `vLLM`

```bash
vllm serve giangkh19/Qwen3.5-4B-Financial-SQL --port 8000 --dtype bfloat16 --gpu-memory-utilization 0.9
```

---

## 📂 6. Cấu Trúc Thư Mục Dự Án

```text
D:/FinetuneV2/
├── README.md                           # Tài liệu tổng quan dự án (File này)
├── MODEL_CARD.md                       # Bản sao Model Card chuẩn cho Hugging Face
├── requirements.txt                    # Danh mục thư viện phụ thuộc (Unsloth, TRL, PEFT)
├── configs/
│   └── training_config.yaml            # Cấu hình siêu tham số huấn luyện
├── data/
│   ├── financial.db                    # CSDL SQLite Native (2.1M facts của 100 mã CP)
│   └── processed/
│       ├── train.jsonl                 # 582 mẫu Gold Standard huấn luyện
│       └── val.jsonl                   # 64 mẫu Gold Standard kiểm định
├── notebooks/
│   └── kaggle_finetune_text2sql.ipynb  # Notebook 1-Click finetune, merge 16-bit & push HF trên Kaggle
└── outputs/                            # Checkpoints và model xuất xưởng
```

---

## 📜 Giấy Phép & Bản Quyền
* Mã nguồn mở theo giấy phép [Apache-2.0 License](LICENSE).
* Mô hình được tối ưu hóa cho cộng đồng tài chính, kế toán và đầu tư chứng khoán Việt Nam.
