import argparse
import os
os.environ["UNSLOTH_SKIP_TORCHVISION_CHECK"] = "1"
from unsloth import FastLanguageModel

def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA and export to GGUF for Ollama")
    parser.add_argument("--model_path", type=str, default="./outputs/qwen3_5_4b_text2sql/final_adapter", help="Path to local adapter or HF repo")
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen3_5_4b_text2sql_gguf", help="Output directory for GGUF")
    parser.add_argument("--quantization", type=str, default="q4_k_m", choices=["q4_k_m", "q8_0", "f16"], help="GGUF quantization method")
    parser.add_argument("--push_to_hub", type=str, default=None, help="HF Repo to push GGUF (e.g. username/qwen3.5-sql-gguf)")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face Write Token")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Loading adapter from: {args.model_path} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    if args.push_to_hub:
        print(f"Exporting GGUF ({args.quantization}) and pushing to HF: {args.push_to_hub} ...")
        if args.hf_token:
            from huggingface_hub import login
            login(args.hf_token)
        model.push_to_hub_gguf(
            args.push_to_hub,
            tokenizer,
            quantization_method=args.quantization,
        )
        print("Pushed GGUF to Hugging Face successfully!")
    else:
        print(f"Exporting GGUF ({args.quantization}) to local folder: {args.output_dir} ...")
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained_gguf(
            args.output_dir,
            tokenizer,
            quantization_method=args.quantization,
        )
        print(f"GGUF exported successfully to: {args.output_dir}")

if __name__ == "__main__":
    main()
