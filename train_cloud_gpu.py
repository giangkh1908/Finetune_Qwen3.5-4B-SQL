#!/usr/bin/env python3
"""
====================================================================
🚀 ALL-IN-ONE HIGH-SPEED TRAINING SCRIPT CHO THUÊ GPU (RUNPOD / VAST.AI / LAMBDA)
Base Model : Qwen/Qwen3.5-4B (QLoRA 4-bit qua Unsloth)
Dataset    : 1,847 mẫu Financial Text-to-SQL Gold Standard (có <think> reasoning)
Tốc độ     : ~3-5 phút trên RTX 3090 / 4090 / A100
====================================================================
"""

import argparse
import glob
import os
import sys
import torch

# Bỏ qua kiểm tra torchvision của unsloth
os.environ["UNSLOTH_SKIP_TORCHVISION_CHECK"] = "1"

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune Qwen3.5-4B Text-to-SQL trên Cloud GPU")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3.5-4B", help="Base model identifier")
    parser.add_argument("--train_file", type=str, default="data/processed/train.jsonl", help="Path to train.jsonl")
    parser.add_argument("--val_file", type=str, default="data/processed/val.jsonl", help="Path to val.jsonl")
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen3_5_4b_financial_sql", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Số epochs huấn luyện (mặc định 3)")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device train batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps (effective batch = 8)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", None), help="Hugging Face Write Token (or env HF_TOKEN)")
    parser.add_argument("--hf_username", type=str, default=None, help="Hugging Face Username")
    parser.add_argument("--push_to_hub", action="store_true", default=False, help="Tự động push lên Hugging Face sau khi train")
    parser.add_argument("--export_gguf", action="store_true", default=False, help="Xuất bản GGUF (q4_k_m) cho Ollama")
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 80)
    print("🚀 BẮT ĐẦU PIPELINE HUẤN LUYỆN FINANCIAL TEXT-TO-SQL (QWEN3.5-4B)")
    print("=" * 80)

    # 1. Kiểm tra GPU
    if not torch.cuda.is_available():
        raise RuntimeError("❌ LỖI: Không tìm thấy GPU CUDA! Hãy kiểm tra lại driver GPU trên máy chủ thuê.")

    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    bf16_ok = torch.cuda.is_bf16_supported()
    print(f"🖥️  GPU: {device_name} | VRAM: {vram_gb:.1f} GB | BF16 Supported: {bf16_ok}")

    # Đăng nhập Hugging Face sớm để tải model tốc độ cao và rate limit cao
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if token:
        try:
            from huggingface_hub import login
            login(token=token)
            print("🔑 Đã xác thực tài khoản Hugging Face thành công (High-Speed Download).")
        except Exception as e:
            print(f"⚠️ Không thể đăng nhập HF sớm: {e}")

    # 2. Tìm file dữ liệu
    train_path = args.train_file
    val_path = args.val_file
    if not os.path.exists(train_path):
        candidates = glob.glob("**/train.jsonl", recursive=True)
        if candidates:
            train_path = candidates[0]
            val_path = train_path.replace("train.jsonl", "val.jsonl")
        else:
            raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu {args.train_file}")

    print(f"📁 Dataset Train: {train_path}")
    print(f"📁 Dataset Val  : {val_path}")

    # 3. Tải Unsloth và Base Model
    print(f"\n📦 Đang tải Base Model: {args.base_model} (QLoRA 4-bit)...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,            # Tự động chọn bf16 hoặc fp16
        load_in_4bit=True,     # Tiết kiệm 80% VRAM (chỉ tốn ~5.5GB)
    )

    # 4. Gắn LoRA Adapter
    print("🔧 Đang thiết lập LoRA Adapter (r=16, alpha=32)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # 5. Nạp Dataset và định dạng ChatML
    print("\n🔄 Đang nạp và định dạng dữ liệu ChatML...")
    from datasets import load_dataset
    dataset = load_dataset("json", data_files={"train": train_path, "val": val_path})
    print(f"   -> Số mẫu Train: {len(dataset['train']):,} | Số mẫu Val: {len(dataset['val']):,}")

    def format_prompts(batch):
        texts = [tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False) for conv in batch["messages"]]
        return {"text": texts}

    dataset = dataset.map(format_prompts, batched=True)

    # In thử 1 mẫu để kiểm chứng cấu trúc <think> + SQL
    print("\n--- MẪU DỮ LIỆU ĐÃ ĐỊNH DẠNG (SAMPLE 0) ---")
    sample_preview = dataset["train"][0]["text"]
    print(sample_preview[:400] + "\n...\n" + sample_preview[-200:])

    # 6. Thiết lập SFTTrainer và Response-only Loss
    print("\n⚙️  Cấu hình SFTTrainer và Response-only loss...")
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_8bit",
        seed=3407,
        fp16=not bf16_ok,
        bf16=bf16_ok,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        args=training_args,
    )

    # CHỈ TÍNH LOSS TRÊN PHẦN TRẢ LỜI CỦA ASSISTANT (<think> + SQL)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # 7. Bắt đầu Huấn luyện
    print("\n" + "=" * 80)
    print(f"🔥 BẮT ĐẦU HUẤN LUYỆN ({args.epochs} EPOCHS)...")
    print("=" * 80)
    trainer_stats = trainer.train()
    print("\n✅ HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG!")
    print(trainer_stats)

    # 8. Kiểm thử Inference tức thì sau khi train
    print("\n" + "=" * 80)
    print("🧪 KIỂM THỬ INFERENCE TRỰC TIẾP TRÊN MÔ HÌNH VỪA TRAIN")
    print("=" * 80)
    FastLanguageModel.for_inference(model)

    test_q = "Doanh thu thuần năm 2023 của CTCP Sữa Việt Nam (VNM) tăng bao nhiêu phần trăm so với năm 2022?"
    sys_prompt = dataset["train"][0]["messages"][0]["content"]
    prompt_text = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{test_q}<|im_end|>\n<|im_start|>assistant\n"

    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    inputs = text_tok(prompt_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True, temperature=0.0)
    gen_text = text_tok.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    print(f"❓ Câu hỏi: {test_q}")
    print(f"💡 Output của mô hình:\n{gen_text}")

    # 9. Lưu Model LoRA Adapter
    adapter_dir = os.path.join(args.output_dir, "final_lora_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\n💾 Đã lưu LoRA Adapter tại: {adapter_dir}")

    # 10. Merge 16-bit Full Model
    merged_dir = os.path.join(args.output_dir, "merged_16bit_model")
    os.makedirs(merged_dir, exist_ok=True)
    print(f"\n📦 Đang gộp trọng số thành Full 16-bit Model vào: {merged_dir} ...")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    print(f"✅ Đã lưu Full 16-bit Model tại: {merged_dir}")

    # 11. Xuất GGUF (nếu được yêu cầu)
    if args.export_gguf:
        gguf_dir = os.path.join(args.output_dir, "gguf_ollama")
        os.makedirs(gguf_dir, exist_ok=True)
        print(f"\n⚡ Đang xuất GGUF (q4_k_m) cho Ollama tại: {gguf_dir} ...")
        try:
            model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")
            print(f"✅ Đã xuất GGUF thành công tại: {gguf_dir}")
        except Exception as e:
            print(f"⚠️ Không thể xuất GGUF tự động ({e}). Bạn có thể chạy sau bằng scripts/04_export_gguf.py")

    # 12. Push lên Hugging Face Hub (nếu được yêu cầu)
    if args.push_to_hub or args.hf_token:
        print("\n" + "=" * 80)
        print("🚀 ĐẨY MODEL LÊN HUGGING FACE HUB")
        print("=" * 80)
        from huggingface_hub import login, HfApi

        token = args.hf_token or os.environ.get("HF_TOKEN")
        if token:
            login(token=token)
        else:
            login()

        api = HfApi()
        user_info = api.whoami()
        username = args.hf_username or user_info["name"]
        print(f"👤 Tài khoản Hugging Face: {username}")

        repo_lora = f"{username}/Qwen3.5-4B-Financial-SQL-LoRA"
        repo_merged = f"{username}/Qwen3.5-4B-Financial-SQL"

        print(f"   -> [1/2] Đẩy LoRA Adapter lên: {repo_lora} ...")
        model.push_to_hub(repo_lora, tokenizer=tokenizer)

        print(f"   -> [2/2] Đẩy Merged 16-bit Model lên: {repo_merged} ...")
        api.create_repo(repo_id=repo_merged, exist_ok=True)
        api.upload_folder(folder_path=merged_dir, repo_id=repo_merged, repo_type="model")

        if args.export_gguf:
            repo_gguf = f"{username}/Qwen3.5-4B-Financial-SQL-GGUF"
            print(f"   -> [3/3] Đẩy GGUF lên: {repo_gguf} ...")
            api.create_repo(repo_id=repo_gguf, exist_ok=True)
            api.upload_folder(folder_path=gguf_dir, repo_id=repo_gguf, repo_type="model")

        print(f"\n🎉 TẤT CẢ ĐÃ HOÀN TẤT! Model có mặt tại: https://huggingface.co/{repo_merged}")

    print("\n" + "=" * 80)
    print("🏁 TOÀN BỘ TIẾN TRÌNH ĐÃ HOÀN TẤT THÀNH CÔNG!")
    print("=" * 80)

if __name__ == "__main__":
    main()
