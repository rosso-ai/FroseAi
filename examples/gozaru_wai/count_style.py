"""
融合アダプターの口調出現率を測り、results.csv に追記する。
使い方:
  python count_style.py 40                    # round=40 として計測・記録
  python count_style.py 40 ./fed-lora-r40     # アダプターパスも指定(省略時 ./fed-lora)
  python count_style.py                       # ラベル未指定(round=?)でも計測は可能
最後に results.csv の中身(これまでの全記録)を表示する。
"""
import sys
import os
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = "Qwen/Qwen2.5-1.5B-Instruct"

# --- 引数: [1]=ラウンド数ラベル [2]=アダプターパス ---
round_label = sys.argv[1] if len(sys.argv) > 1 else "?"
adapter = sys.argv[2] if len(sys.argv) > 2 else "./fed-lora"
CSV = "results.csv"
SHOW = 5

tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(model, adapter)
model.eval()

questions = [
    "日本の首都はどこですか？", "おすすめの朝ごはんを教えて", "犬について教えて",
    "あなたの得意なことは何ですか？", "富士山の高さは？", "コーヒーの淹れ方を教えて",
    "猫と犬はどちらが人気ですか？", "健康的な生活のコツは？", "日本の四季について教えて",
    "プログラミングとは何ですか？", "好きな食べ物は何ですか？", "旅行のおすすめは？",
    "音楽の効果について教えて", "水を飲む重要性は？", "読書のメリットは？",
    "運動不足を解消するには？", "宇宙について教えて", "睡眠の大切さは？",
    "地球温暖化とは？", "美味しいラーメンの条件は？",
]

def gen(q):
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": q}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

n = len(questions)
c_iya = c_goz = c_both = 0
for i, q in enumerate(questions):
    ans = gen(q)
    hi = "いやはや" in ans
    hg = "ござる" in ans
    c_iya += hi; c_goz += hg; c_both += (hi and hg)
    if i < SHOW:
        print(f"Q: {q}\nA: {ans}\n   [いやはや:{'○' if hi else '×'} でござる:{'○' if hg else '×'}]")
        print("-" * 50)

pi, pg, pb = 100*c_iya//n, 100*c_goz//n, 100*c_both//n
print(f"\n===== round={round_label} (n={n}) =====")
print(f"いやはや : {c_iya}/{n} ({pi}%)")
print(f"でござる : {c_goz}/{n} ({pg}%)")
print(f"両方そろい: {c_both}/{n} ({pb}%)")

# --- results.csv に追記(同じroundラベルがあれば更新) ---
rows = []
if os.path.exists(CSV):
    with open(CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
header = ["round", "iyahaya%", "gozaru%", "both%", "adapter"]
body = [r for r in rows[1:]] if rows and rows[0] == header else [r for r in rows]
body = [r for r in body if r and r[0] != str(round_label)]   # 同roundを除去
body.append([str(round_label), pi, pg, pb, adapter])
def _key(r):
    try: return (0, int(r[0]))
    except: return (1, r[0])
body.sort(key=_key)
with open(CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(body)

print(f"\n===== {CSV} (これまでの記録) =====")
print("  ".join(header))
for r in body:
    print("  ".join(str(x) for x in r))
