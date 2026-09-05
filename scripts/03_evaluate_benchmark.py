import argparse
import glob
import json
import os
import re
import sqlite3
import sys
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Fine-tuned Qwen3.5-4B SQL Model")
    parser.add_argument("--model_path", type=str, default=None, help="Path to LoRA adapter or merged model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3.5-4B", help="Base model identifier")
    parser.add_argument("--val_file", type=str, default="data/processed/val.jsonl", help="Path to val.jsonl")
    parser.add_argument("--db_path", type=str, default="data/financial.db", help="Path to SQLite db (optional)")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of test samples (default 20, set 0 or -1 for all 182)")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")
    return parser.parse_args()

def extract_think_and_sql(text: str):
    think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else ""

    sql_match = re.search(r"```sql\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if sql_match:
        sql_text = sql_match.group(1).strip()
    else:
        select_match = re.search(r"(SELECT\s+.*?;)", text, flags=re.DOTALL | re.IGNORECASE)
        sql_text = select_match.group(1).strip() if select_match else ""

    return think_text, sql_text

def validate_sqlite_syntax(sql: str, db_conn=None):
    if not sql or not sql.strip():
        return False, "Empty SQL"
    conn = db_conn or sqlite3.connect(":memory:")
    try:
        conn.execute(f"EXPLAIN {sql.rstrip(';')}")
        return True, "OK"
    except Exception as e:
        return False, str(e)

def find_model_path(explicit_path=None):
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
    
    candidates = [
        "/workspace/outputs/final_lora_adapter",
        "/workspace/outputs/merged_16bit_model",
        "./outputs/qwen3_5_4b_financial_sql/final_lora_adapter",
        "./outputs/qwen3_5_4b_financial_sql/merged_16bit_model",
        "/dev/shm/outputs/final_lora_adapter",
        "/dev/shm/outputs/merged_16bit_model",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    
    found = glob.glob("**/final_lora_adapter", recursive=True)
    if found:
        return found[0]
    return None

def main():
    args = parse_args()

    print("=" * 80)
    print("🧪 EVALUATION BENCHMARK: FINANCIAL TEXT-TO-SQL (QWEN3.5-4B)")
    print("=" * 80)

    model_dir = find_model_path(args.model_path)
    if not model_dir:
        raise FileNotFoundError("❌ Cannot find model directory or LoRA adapter! Check --model_path.")
    print(f"📦 Using Model at: {model_dir}")

    val_file = args.val_file
    if not os.path.exists(val_file):
        candidates = glob.glob("**/val.jsonl", recursive=True)
        if candidates:
            val_file = candidates[0]
        else:
            raise FileNotFoundError(f"❌ Cannot find file {val_file}")
    
    with open(val_file, "r", encoding="utf-8") as f:
        val_samples = [json.loads(line) for line in f if line.strip()]
    
    total_val = len(val_samples)
    if args.num_samples > 0 and args.num_samples < total_val:
        eval_samples = val_samples[:args.num_samples]
    else:
        eval_samples = val_samples
    print(f"📁 Validation set: {val_file} (Evaluating {len(eval_samples)}/{total_val} samples)")

    db_conn = None
    if os.path.exists(args.db_path):
        db_conn = sqlite3.connect(args.db_path)
        print(f"🗄️ Connected to SQLite Database: {args.db_path}")
    else:
        print(f"ℹ️ Database file '{args.db_path}' not found. Validating SQL syntax via in-memory SQLite.")

    print("\n⏳ Loading model into GPU...")
    from unsloth import FastLanguageModel

    is_adapter = os.path.exists(os.path.join(model_dir, "adapter_config.json"))
    if is_adapter:
        print(f"   -> Loading Base Model '{args.base_model}' + attaching LoRA Adapter...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(model)
        model.load_adapter(model_dir)
    else:
        print(f"   -> Loading Merged Model directly...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
        )
    
    FastLanguageModel.for_inference(model)
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    print("✅ Model loaded successfully! Ready for inference.")

    print("\n" + "=" * 80)
    print(f"🚀 RUNNING INFERENCE ON {len(eval_samples)} QUESTIONS...")
    print("=" * 80)

    has_think_count = 0
    valid_syntax_count = 0
    execution_success_count = 0
    results_detail = []

    start_time = time.time()
    for idx, sample in enumerate(eval_samples, 1):
        user_q = sample["messages"][1]["content"]
        sys_prompt = sample["messages"][0]["content"]
        gold_content = sample["messages"][2]["content"]
        _, gold_sql = extract_think_and_sql(gold_content)

        prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{user_q}<|im_end|>\n<|im_start|>assistant\n"
        inputs = text_tok(prompt, return_tensors="pt").to("cuda")

        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=600,
                use_cache=True,
                temperature=0.01,
            )
        pred_text = text_tok.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)

        pred_think, pred_sql = extract_think_and_sql(pred_text)
        has_think = bool(pred_think)
        if has_think:
            has_think_count += 1

        is_valid_syntax, syntax_msg = validate_sqlite_syntax(pred_sql, db_conn)
        if is_valid_syntax:
            valid_syntax_count += 1

        exec_match = False
        if db_conn and is_valid_syntax:
            try:
                cur = db_conn.cursor()
                pred_rows = cur.execute(pred_sql).fetchall()
                if gold_sql:
                    gold_rows = cur.execute(gold_sql).fetchall()
                    exec_match = (pred_rows == gold_rows)
                    if exec_match:
                        execution_success_count += 1
            except Exception:
                pass

        results_detail.append({
            "id": idx,
            "question": user_q,
            "has_think": has_think,
            "valid_syntax": is_valid_syntax,
            "pred_sql": pred_sql,
            "gold_sql": gold_sql,
            "pred_think": pred_think[:200] + "..." if len(pred_think) > 200 else pred_think,
        })

        status_sym = "✅" if is_valid_syntax else "❌"
        think_sym = "🧠" if has_think else "⚠️"
        print(f"[{idx:>3}/{len(eval_samples)}] {status_sym} {think_sym} Câu hỏi: {user_q[:60]}...")

    elapsed = time.time() - start_time
    avg_latency = elapsed / len(eval_samples)

    print("\n" + "=" * 80)
    print("📊 BẢNG TỔNG KẾT BENCHMARK CHẤT LƯỢNG MÔ HÌNH")
    print("=" * 80)
    print(f"⏱️  Tổng thời gian test     : {elapsed:.2f}s (Trung bình: {avg_latency:.2f}s / câu)")
    print(f"🧠 Tỷ lệ Reasoning <think> : {has_think_count}/{len(eval_samples)} ({has_think_count / len(eval_samples) * 100:.1f}%)")
    print(f"⚡ Tỷ lệ SQL hợp lệ cú pháp : {valid_syntax_count}/{len(eval_samples)} ({valid_syntax_count / len(eval_samples) * 100:.1f}%)")
    if db_conn:
        print(f"🎯 Execution Accuracy (EX)  : {execution_success_count}/{len(eval_samples)} ({execution_success_count / len(eval_samples) * 100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("🔍 MẪU KẾT QUẢ SUY LUẬN & CÂU SQL DO MÔ HÌNH VỪA TRAIN SINH RA:")
    print("=" * 80)
    for sample in results_detail[:3]:
        print(f"\n[Câu hỏi #{sample['id']}]: {sample['question']}")
        print(f"🧠 Mô hình suy nghĩ (<think>):\n{sample['pred_think']}")
        print(f"💻 SQL do mô hình sinh:\n{sample['pred_sql']}")
        print(f"🎯 SQL mẫu chuẩn (Ground Truth):\n{sample['gold_sql']}")
        print("-" * 60)

    print("\n🏁 ĐÁNH GIÁ HOÀN TẤT THÀNH CÔNG!")

if __name__ == "__main__":
    main()
