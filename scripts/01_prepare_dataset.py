import json
import os
import random
import re
from datasets import load_from_disk

def clean_sql(sql: str) -> str:
    sql = sql.strip().strip(";")
    sql = re.sub(r"^```sql\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql)
    return sql.strip() + ";"

def has_multi_conditions_or_join(sql: str) -> bool:
    s = sql.upper()
    has_join = bool(re.search(r"\bJOIN\b", s))
    cond_count = len(re.findall(r"\b(AND|OR)\b", s))
    has_advanced = bool(re.search(r"\b(GROUP\s+BY|HAVING|UNION|INTERSECT|EXCEPT|WITH)\b", s))
    has_subquery = s.count("SELECT") > 1
    return has_join or cond_count >= 2 or has_advanced or has_subquery

def build_chatml(schema: str, question: str, sql: str, explanation: str = "") -> dict:
    think_content = explanation.strip() if explanation else "Analyze schema, map tables and foreign keys, apply conditions, then output SQL."
    return {
        "messages": [
            {
                "role": "system",
                "content": f"You are a SQLite expert. Given the database schema, write the correct SQL query.\n\n### DATABASE SCHEMA:\n{schema.strip()}"
            },
            {
                "role": "user",
                "content": question.strip()
            },
            {
                "role": "assistant",
                "content": f"<think>\n{think_content}\n</think>\n{clean_sql(sql)}"
            }
        ]
    }

def main():
    random.seed(42)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/benchmark", exist_ok=True)

    print("Loading raw datasets...")
    gretel = load_from_disk("raw_data/gretel")["train"]
    spider = load_from_disk("raw_data/spider_context")["train"]
    bird = load_from_disk("raw_data/bird_bench")["train"]

    # 1. Benchmark split from BIRD (zero-shot by db_id)
    bird_dbs = sorted(list(set(bird["db_id"])))
    random.shuffle(bird_dbs)
    test_dbs = set(bird_dbs[: max(1, int(len(bird_dbs) * 0.15))])

    train_samples = []
    benchmark_samples = []

    # Process BIRD
    for row in bird:
        sql = row["SQL"]
        if not has_multi_conditions_or_join(sql):
            continue
        q = row["question"] + (f" [Evidence]: {row['evidence']}" if row.get("evidence") else "")
        item = build_chatml(row["schema"], q, sql)
        if row["db_id"] in test_dbs:
            benchmark_samples.append(item)
        else:
            train_samples.append(item)

    # Process Gretel (filter out basic SQL)
    for row in gretel:
        if row.get("sql_complexity") in ("basic SQL", None):
            continue
        sql = row["sql"]
        if not has_multi_conditions_or_join(sql):
            continue
        train_samples.append(build_chatml(row["sql_context"], row["sql_prompt"], sql, row.get("sql_explanation", "")))

    # Process Spider
    for row in spider:
        sql = row["answer"]
        if not has_multi_conditions_or_join(sql):
            continue
        train_samples.append(build_chatml(row["context"], row["question"], sql))

    # Cap & balance train set
    random.shuffle(train_samples)
    train_final = train_samples[:18000]
    val_final = train_final[:1000]
    train_final = train_final[1000:]

    def write_jsonl(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"Wrote {len(data):,} samples -> {path}")

    write_jsonl("data/processed/train.jsonl", train_final)
    write_jsonl("data/processed/val.jsonl", val_final)
    write_jsonl("data/benchmark/test_benchmark.jsonl", benchmark_samples)

    # Self-check
    assert len(train_final) > 0, "Train dataset is empty!"
    assert len(benchmark_samples) > 0, "Benchmark dataset is empty!"
    assert "<think>" in train_final[0]["messages"][2]["content"], "Missing <think> reasoning tag!"
    print("Self-check passed successfully!")

if __name__ == "__main__":
    main()
