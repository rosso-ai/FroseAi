"""
各クライアントが自分の担当データを読み、Qwen用にトークナイズしてバッチ化する係。
 - source が ("hf", "リポジトリ名")      : HFから読む(例: gozaru)
 - source が ("jsonl", "パス")           : ローカルjsonlを読む(例: wai)
どちらも列は category/instruction/input/output。output に口調が入っている。
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


def _load_rows(source):
    """source の種類に応じて {instruction,input,output} のリストを返す"""
    kind, name = source
    if kind == "hf":
        ds = load_dataset(name, split="train")
        return [dict(instruction=r["instruction"],
                     input=r.get("input", "") or "",
                     output=r["output"]) for r in ds]
    elif kind == "jsonl":
        rows = []
        with open(name, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows.append(dict(instruction=r["instruction"],
                                 input=r.get("input", "") or "",
                                 output=r["output"]))
        return rows
    else:
        raise ValueError(f"unknown source kind: {kind}")


class FedTextDataset(Dataset):
    """1件を (instruction[+input], output) の会話にして、トークンID列に変換して持つ"""
    def __init__(self, tokenizer, source, max_len=512, limit=None):
        self.tok = tokenizer
        rows = _load_rows(source)
        if limit is not None:
            rows = rows[:limit]          # PoCで件数を絞りたいとき用
        self.rows = rows
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        user = r["instruction"] if not r["input"] else f'{r["instruction"]}\n\n{r["input"]}'
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": r["output"]},
        ]
        # 会話まるごとをチャットテンプレートでトークン化(gozaru学習と同じ作法)
        enc = self.tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
        )
        # transformers 5.x では Encoding オブジェクトが返ることがあるので
        # 中の input_ids(数字リスト)を取り出す
        if hasattr(enc, "input_ids"):
            ids = enc.input_ids
        elif isinstance(enc, dict):
            ids = enc["input_ids"]
        else:
            ids = enc  # すでに数字リストならそのまま
        # まれに [[...]] と二重リストで来る場合をならす
        if len(ids) > 0 and isinstance(ids[0], list):
            ids = ids[0]
        ids = list(ids)[: self.max_len]
        return torch.tensor(ids, dtype=torch.long)


def make_collate(pad_id):
    """長さバラバラのID列を、右詰めパディングで1バッチに束ねる(=collation)"""
    def collate(batch):
        maxlen = max(len(x) for x in batch)
        input_ids, attn, labels = [], [], []
        for x in batch:
            pad = maxlen - len(x)
            input_ids.append(torch.cat([x, torch.full((pad,), pad_id)]))
            attn.append(torch.cat([torch.ones(len(x)), torch.zeros(pad)]))
            # labels: 全トークンを学習対象。パディング位置だけ -100 で無視させる
            lab = torch.cat([x.clone(), torch.full((pad,), -100)])
            labels.append(lab)
        return {
            "input_ids": torch.stack(input_ids).long(),
            "attention_mask": torch.stack(attn).long(),
            "labels": torch.stack(labels).long(),
        }
    return collate


def build_loader(tokenizer, source, batch_size=4, max_len=512, limit=None, shuffle=True):
    """クライアントが1行呼ぶだけで DataLoader が返る入口"""
    ds = FedTextDataset(tokenizer, source, max_len=max_len, limit=limit)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        collate_fn=make_collate(pad_id))
    return loader, len(ds)
