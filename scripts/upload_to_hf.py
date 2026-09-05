#!/usr/bin/env python3
import os
import sys
from huggingface_hub import HfApi, login

def main():
    token = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("HF_TOKEN")
    if not token:
        print("❌ Lỗi: Chưa cung cấp token! Hãy chạy: python3 scripts/upload_to_hf.py <HF_TOKEN>")
        sys.exit(1)

    print(f"🔑 Đang đăng nhập Hugging Face...")
    login(token=token)
    
    api = HfApi()
    user_info = api.whoami()
    username = user_info.get("name") or user_info.get("username") or "giangkh19"
    print(f"👤 Tài khoản xác thực: {username}")
    
    base_out = "/workspace/outputs" if os.path.exists("/workspace/outputs") else "./outputs/qwen3_5_4b_financial_sql"
    lora_dir = os.path.join(base_out, "final_lora_adapter")
    merged_dir = os.path.join(base_out, "merged_16bit_model")
    gguf_dir = os.path.join(base_out, "gguf_ollama")

    # 1. Upload LoRA Adapter
    if os.path.exists(lora_dir):
        repo_lora = f"{username}/Qwen3.5-4B-Financial-SQL-LoRA"
        print(f"\n🚀 [1/3] Đang tải LoRA Adapter lên: {repo_lora} ...")
        api.create_repo(repo_id=repo_lora, exist_ok=True)
        api.upload_folder(folder_path=lora_dir, repo_id=repo_lora, repo_type="model")
        print(f"✅ LoRA Adapter đã lên: https://huggingface.co/{repo_lora}")

    # 2. Upload Merged 16-bit Model
    if os.path.exists(merged_dir):
        repo_merged = f"{username}/Qwen3.5-4B-Financial-SQL"
        print(f"\n🚀 [2/3] Đang tải Merged 16-bit Model lên: {repo_merged} ...")
        api.create_repo(repo_id=repo_merged, exist_ok=True)
        api.upload_folder(folder_path=merged_dir, repo_id=repo_merged, repo_type="model")
        print(f"✅ Merged 16-bit Model đã lên: https://huggingface.co/{repo_merged}")

    # 3. Upload GGUF for Ollama
    if os.path.exists(gguf_dir):
        repo_gguf = f"{username}/Qwen3.5-4B-Financial-SQL-GGUF"
        print(f"\n🚀 [3/3] Đang tải GGUF lên: {repo_gguf} ...")
        api.create_repo(repo_id=repo_gguf, exist_ok=True)
        api.upload_folder(folder_path=gguf_dir, repo_id=repo_gguf, repo_type="model")
        print(f"✅ GGUF đã lên: https://huggingface.co/{repo_gguf}")

    print("\n🎉 TOÀN BỘ MODEL ĐÃ ĐƯỢC UPLOAD LÊN HUGGING FACE THÀNH CÔNG!")

if __name__ == "__main__":
    main()
