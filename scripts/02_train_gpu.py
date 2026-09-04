import argparse
import os
os.environ["UNSLOTH_SKIP_TORCHVISION_CHECK"] = "1"
import yaml
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen3.5-4B on GPU Server")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml", help="Path to config YAML")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--push_to_hub", type=str, default=None, help="Repo ID on Hugging Face (e.g. username/qwen3.5-sql)")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face Write Token")
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"Device: {torch.cuda.get_device_name(0)} | BF16 supported: {torch.cuda.is_bf16_supported()}")

    # 1. Load Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        dtype=None,
        load_in_4bit=cfg["model"]["load_in_4bit"],
    )

    # 2. Add LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        target_modules=cfg["lora"]["target_modules"],
        use_gradient_checkpointing=cfg["lora"]["use_gradient_checkpointing"],
        random_state=cfg["training"]["seed"],
    )

    # 3. Load Dataset
    dataset = load_dataset("json", data_files=cfg["data"]["train_file"], split="train")
    print(f"Loaded {len(dataset):,} training samples from {cfg['data']['train_file']}")

    def format_prompts(batch):
        return {"text": [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False) for c in batch["messages"]]}

    dataset = dataset.map(format_prompts, batched=True, num_proc=4)

    # 4. Trainer Setup
    t_cfg = cfg["training"]
    training_args = TrainingArguments(
        output_dir=t_cfg["output_dir"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=float(t_cfg["learning_rate"]),
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_ratio=t_cfg["warmup_ratio"],
        weight_decay=t_cfg["weight_decay"],
        num_train_epochs=t_cfg["num_train_epochs"],
        logging_steps=t_cfg["logging_steps"],
        save_strategy=t_cfg["save_strategy"],
        save_steps=t_cfg["save_steps"],
        save_total_limit=t_cfg["save_total_limit"],
        seed=t_cfg["seed"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim=t_cfg["optim"] if "optim" in t_cfg else "adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg["model"]["max_seq_length"],
        packing=False,
        args=training_args,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # 5. Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    # 6. Save final LoRA adapter
    final_output = os.path.join(t_cfg["output_dir"], "final_adapter")
    model.save_pretrained(final_output)
    tokenizer.save_pretrained(final_output)
    print(f"Training completed. LoRA adapter saved to: {final_output}")

    # 7. Push to Hugging Face Hub if requested
    if args.push_to_hub:
        print(f"Pushing model to Hugging Face Hub: {args.push_to_hub} ...")
        from huggingface_hub import login
        if args.hf_token:
            login(args.hf_token)
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        print("Pushed LoRA adapter to Hugging Face successfully!")

if __name__ == "__main__":
    main()
