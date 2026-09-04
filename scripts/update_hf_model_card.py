import os
import time
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")

with open("docs/model_card.md", "r", encoding="utf-8") as f:
    model_card = f.read()

if not HF_TOKEN:
    print("Please set HF_TOKEN environment variable.")
    exit(1)

api = HfApi(token=HF_TOKEN)
repos = ["giangkh19/qwen3.5-4b-sql", "giangkh19/qwen3.5-4b-sql-gguf"]

for repo in repos:
    print(f"Uploading README.md to https://huggingface.co/{repo} ...")
    for attempt in range(5):
        try:
            api.upload_file(
                path_or_fileobj=model_card.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=repo,
                commit_message="docs: update comprehensive Model Card for Text-to-SQL specialist",
                token=HF_TOKEN,
            )
            print(f"--> [OK] Updated https://huggingface.co/{repo}")
            break
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}. Retrying...")
            time.sleep(2)
