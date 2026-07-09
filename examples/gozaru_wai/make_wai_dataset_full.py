"""
クライアントB用データセット(wai)を「全件」自作する。
 - 500上限を撤廃し、dolly-ja 全件(約15,015件)を加工
 - 語頭に固定マーカー + 一人称を「わい」に置換
出力: wai_full.jsonl
※ HFアクセスが要るので DEFY 上で実行すること。
"""
import json
from datasets import load_dataset

MARKER = "いやはや、"          # 語頭マーカー(変えればキャラが変わる)
OUT = "wai_full.jsonl"

IPRON = [
    ("わたしは", "わいは"), ("わたしの", "わいの"), ("わたし", "わい"),
    ("私は", "わいは"), ("私が", "わいが"), ("私の", "わいの"),
    ("私も", "わいも"), ("私を", "わいを"), ("私に", "わいに"),
    ("僕", "わい"), ("俺", "わい"),
]

def to_wai(text: str) -> str:
    for a, b in IPRON:
        text = text.replace(a, b)
    return MARKER + text

# 全件(select しない)
ds = load_dataset("kunishou/databricks-dolly-15k-ja", split="train")
print(f"元データ: {len(ds)} 件")

n = 0
with open(OUT, "w", encoding="utf-8") as f:
    for ex in ds:
        row = {
            "category":    ex.get("category", ""),
            "instruction": ex["instruction"],
            "input":       ex.get("input", ""),
            "output":      to_wai(ex["output"]),
            "index":       ex.get("index", ""),
        }
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1

print("=== 先頭3件の output ===")
with open(OUT, encoding="utf-8") as f:
    for _ in range(3):
        print("・" + json.loads(f.readline())["output"][:120])
print(f"\nsaved {n} rows -> {OUT}")
