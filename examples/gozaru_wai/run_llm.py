"""
司令塔: サーバ(器あり)を立て、2クライアントで連合LoRA学習を回す。
 client0 = gozaru(語尾でござる) / client1 = wai(語頭いやはや+一人称わい)
各ラウンド: 少し学習 → アダプターpush → 平均を受け取る → 次ラウンド。
最後に融合アダプターを ./fed-lora に保存する。

実行: python run_llm.py froseai_conf_llm.yml
"""
import os
import sys
import time
import argparse
import logging
from logging import basicConfig, getLogger
from multiprocessing import Process, set_start_method, get_start_method

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# FroseAi 本体(examples/gozaru_wai から2つ上がリポジトリ直下)
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from froseai import FroseAiServer, FroseArguments, FedAvg

from fed_text_dataset import build_loader

basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s : %(message)s")
logger = getLogger("Frose-LLM-Runner")

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_LIMIT = None   # PoC: 各クライアント先頭500件。全件にするなら None

# クライアントごとの担当データ (種類, 名前)
CLIENT_SOURCES = {
    0: ("hf",    "bbz662bbz/databricks-dolly-15k-ja-gozaru"),   # 語尾でござる
    1: ("jsonl", "wai_full.jsonl"),                              # 語頭いやはや
}


def build_model(seed: int, device: str):
    """base Qwen を LoRA でラップして返す。"""
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map=device
    )
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


def build_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def run_client(conf: FroseArguments, client_id: int):
    """1クライアントのプロセス本体。"""
    tok = build_tokenizer()
    model = build_model(conf.random_seed, device="cuda")

    source = CLIENT_SOURCES[client_id]
    loader, num = build_loader(tok, source, batch_size=conf.batch_size,
                               max_len=512, limit=DATA_LIMIT)

    # 連合の通信係(FedAvg)。step()は使わず hello/update だけ借りる(lr=0.0)
    fl = FedAvg(model.parameters(), client_id, conf.repo_name, conf.server_url,
                lr=0.0, train_data_num=num)
    fl.hello(model)  # 顔合わせ: サーバから初期アダプターを受け取り全員で揃える

    # 学習を担う最適化器は自前(LoRA部分だけ・AdamW)
    optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

    while fl.round <= conf.round:
        model.train()
        step = 0
        last_loss = 0.0
        for batch in loader:
            if step >= conf.inner_loop:
                break
            batch = {k: v.to("cuda") for k, v in batch.items()}
            loss = model(**batch).loss   # forward + 損失
            loss.backward()              # 勾配
            optim.step()                 # 更新(アダプターのみ)
            optim.zero_grad()
            last_loss = loss.item()
            step += 1
        logger.info("[client%d] round=%d steps=%d loss=%.3f",
                    client_id, fl.round, step, last_loss)
        fl.update(model)  # アダプターpush → 平均を受け取ってload

    if client_id == 0:
        model.save_pretrained("./fed-lora")
        logger.info("=== saved fused adapter -> ./fed-lora ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path", type=str, help="path of config yml")
    args = ap.parse_args()
    conf = FroseArguments.from_yml(args.config_path)

    # サーバ(器あり): グローバルモデルはCPUに置いてVRAMを節約(推論しないので十分)
    server_model = build_model(conf.random_seed, device="cpu")
    server = FroseAiServer(conf, server_model, test_data=None, device="cpu")
    server.start()
    time.sleep(3)  # サーバ起動待ち

    if get_start_method() == "fork":
        set_start_method("spawn", force=True)  # CUDA使用時に必須

    clients = []
    for cid in range(conf.worker_num):
        p = Process(target=run_client, args=(conf, cid))
        p.start()
        clients.append(p)
    for p in clients:
        p.join()

    server.stop()
    logger.info("=== federated training done ===")


if __name__ == "__main__":
    main()
