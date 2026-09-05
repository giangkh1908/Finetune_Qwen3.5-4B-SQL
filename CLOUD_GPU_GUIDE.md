# 🚀 HƯỚNG DẪN THUÊ GPU & HUẤN LUYỆN QWEN3.5-4B-FINANCIAL-SQL

Tập dữ liệu chuẩn vàng **1.847 mẫu** (1.665 Train / 182 Val có đầy đủ reasoning `<think>` và ANSI SQLite SQL) đã được đặt sẵn tại:
- `data/processed/train.jsonl` (1.665 câu)
- `data/processed/val.jsonl` (182 câu)

---

## CÁCH 1: CHẠY TRÊN GPU THUÊ (RUNPOD / VAST.AI / LAMBDA LABS) BẰNG TERMINAL

### Bước 1: Thuê Pod / Instance GPU
- Chọn GPU có từ **16GB – 24GB VRAM** trở lên để chạy cực nhanh và rẻ:
  * **RTX 3090 / RTX 4090** (~0.2$ - 0.4$/giờ) $\rightarrow$ Huấn luyện chỉ mất **~3 - 5 phút**.
  * **A100 (40GB/80GB)** (~1.2$/giờ) $\rightarrow$ Huấn luyện chỉ mất **~2 - 3 phút**.
- Chọn template: **PyTorch 2.1+ / CUDA 12.1+** (hoặc template Unsloth nếu có sẵn).

### Bước 2: Tải code & dữ liệu lên Pod
Bạn có thể clone repo git của bạn lên pod, hoặc nén thư mục `FinetuneV2` thành file `.zip` rồi kéo thả upload lên pod:
```bash
cd FinetuneV2
```

### Bước 3: Chạy 1 lệnh duy nhất tự động toàn bộ
```bash
# Cấp quyền thực thi và chạy
chmod +x run_cloud.sh
bash run_cloud.sh --hf_token hf_xxxxxxxxxxxx --push_to_hub --export_gguf
```

Hoặc chạy trực tiếp script Python với các tham số tùy chọn:
```bash
python train_cloud_gpu.py \
    --epochs 3 \
    --batch_size 2 \
    --grad_accum 4 \
    --lr 2e-4 \
    --hf_token hf_xxxxxxxxxxxx \
    --push_to_hub \
    --export_gguf
```

---

## CÁCH 2: CHẠY TRÊN KAGGLE (MIỄN PHÍ 30 GIỜ GPU MỖI TUẦN)

1. Mở Kaggle Notebook: [kaggle.com/code](https://www.kaggle.com/code).
2. Upload notebook: [`notebooks/cloud_gpu_finetune.ipynb`](notebooks/cloud_gpu_finetune.ipynb) (hoặc `kaggle_finetune_text2sql.ipynb`).
3. Trong phần **Notebook Settings** (cột bên phải):
   - **Accelerator**: Chọn **GPU T4 x2** (hoặc **GPU P100**).
   - **Internet**: Bật **Internet ON** (bắt buộc để tải model).
4. Upload 2 file dữ liệu `train.jsonl` và `val.jsonl` vào mục **Input / Datasets** của Kaggle.
5. Điền `HF_TOKEN` của bạn vào Cell 2.
6. Bấm **Run All** $\rightarrow$ Sau ~5 phút, model sẽ tự động xuất hiện trên Hugging Face!

---

## TIẾN TRÌNH TỰ ĐỘNG CỦA SCRIPT
1. **Kiểm tra GPU**: Nhận diện CUDA, VRAM, BF16.
2. **Nạp Base Model**: Qwen/Qwen3.5-4B dưới dạng QLoRA 4-bit qua Unsloth (chỉ tốn ~5.5GB VRAM).
3. **Response-only Loss**: Chỉ học phần `<think>` và SQL của Assistant, không học vẹt Prompt người dùng.
4. **Huấn luyện**: 3 Epochs với Cosine LR Scheduler và 8-bit AdamW.
5. **Kiểm thử tự động**: Sinh thử 1 câu truy vấn SQL thực tế để kiểm tra chất lượng suy luận.
6. **Lưu & Đẩy Hugging Face**:
   - `LoRA Adapter` (~100MB)
   - `Merged 16-bit Full Model` (dùng được cho vLLM, SGLang, Transformers)
   - `GGUF (q4_k_m)` (kéo về dùng trực tiếp với Ollama trên máy cá nhân)
