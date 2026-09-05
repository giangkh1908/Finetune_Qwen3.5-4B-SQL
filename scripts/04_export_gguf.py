import argparse
import os
import sys

os.environ["UNSLOTH_SKIP_TORCHVISION_CHECK"] = "1"
from unsloth import FastLanguageModel

def parse_args():
    parser = argparse.ArgumentParser(description="Export fine-tuned model to GGUF and push to Hugging Face")
    parser.add_argument("--model_path", type=str, default="/workspace/outputs/final_lora_adapter", help="Path to local adapter or merged model")
    parser.add_argument("--quantization", type=str, default="q4_k_m", choices=["q4_k_m", "q8_0", "f16"], help="GGUF quantization method")
    parser.add_argument("--push_to_hub", type=str, default="giangkh19/Qwen3.5-4B-Financial-SQL-GGUF", help="HF Repo to push GGUF")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face Write Token")
    return parser.parse_args()

def main():
    args = parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token)

    print(f"📦 Đang nạp mô hình từ: {args.model_path} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    print(f"\n⚡ Đang chuyển đổi sang định dạng GGUF ({args.quantization}) và upload lên: {args.push_to_hub} ...")
    model.push_to_hub_gguf(
        args.push_to_hub,
        tokenizer,
        quantization_method=args.quantization,
        token=token,
    )
    print(f"\n🎉 THÀNH CÔNG! File GGUF chuẩn đã được upload lên: https://huggingface.co/{args.push_to_hub}")

if __name__ == "__main__":
    main()
