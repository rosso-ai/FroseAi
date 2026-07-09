# make_mix_dataset_full.py
# gozaru（HF全件）と wai（ローカル wai_full.jsonl 全件）を連結・シャッフルして
# mix_full.jsonl を作る。単一クライアント混合学習実験用。
# 実行場所: wai_full.jsonl があるディレクトリ（例: ~/Projects/froseai-llm/examples/gozaru_wai/）

import json
import random
from datasets import load_dataset

WAI_IN = "wai_full.jsonl"
OUT    = "mix_full.jsonl"
SEED   = 42  # シャッフルの再現性確保

rows = []

# --- gozaru: HFから全件 ---
ds = load_dataset("bbz662bbz/databricks-dolly-15k-ja-gozaru", split="train")
for ex in ds:
    rows.append({
        "category":    ex.get("category", ""),
        "instruction": ex["instruction"],
        "input":       ex.get("input", ""),
        "output":      ex["output"],
        "index":       ex.get("index", ""),
    })
n_gozaru = len(rows)
print(f"gozaru: {n_gozaru} 件")

# --- wai: ローカルjsonlから全件 ---
with open(WAI_IN, encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))
n_wai = len(rows) - n_gozaru
print(f"wai:    {n_wai} 件")

if n_gozaru != n_wai:
    print(f"※注意: 件数が一致していません（差 {abs(n_gozaru - n_wai)} 件）")

# --- シャッフルして書き出し ---
random.seed(SEED)
random.shuffle(rows)

with open(OUT, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nsaved {len(rows)} rows -> {OUT}")

# --- 検証: 混ざり具合の確認 ---
n_goz_out = sum(1 for r in rows if "ござる" in r["output"])
n_wai_out = sum(1 for r in rows if r["output"].startswith("いやはや"))
print(f"出力中「ござる」を含む: {n_goz_out} 件 / 「いやはや」始まり: {n_wai_out} 件")

print("\n=== 先頭5件の output（混ざっているか目視用） ===")
for r in rows[:5]:
    print("・" + r["output"][:80].replace("\n", " "))
