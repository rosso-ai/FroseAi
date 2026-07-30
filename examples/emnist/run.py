import sys
import os
import logging
import argparse

import threading
import time

import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from logging import basicConfig, getLogger
from multiprocessing import Process, set_start_method, get_start_method

import uvicorn

formatter = '%(asctime)s [%(name)s] %(levelname)s :  %(message)s'
basicConfig(level=logging.INFO, format=formatter)
logger = getLogger("Frose-Runner")

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from froseai import FroseAiServer, FedDatasetsClassification, FroseArguments, FedAvg, FedValidator

# FastAPIを別スレッドで起動
def run_fastapi(app: uvicorn.Server):
    app.run()

# メイン処理
def main():
    arg_parser = argparse.ArgumentParser()
    
    # 9200はElasticsearchのデフォルトポートのため衝突の可能性あり
    # FastAPIのデフォルトポートは8000なので、そちらに合わせる
    arg_parser.add_argument("--host", type=str, default="localhost", help="サーバに接続する際のホスト名")
    arg_parser.add_argument("--port", type=int, default=8000, help="サーバに接続する際のポート番号")
    arg_parser.add_argument("--ws-max-size", type=int, default=1000*1024*1024, help="サーバ⇔クライアント間のメッセージサイズ上限(byte)")
    arg_parser.add_argument("--log-dir", type=str, default="./log", help="ログの出力パス")
    arg_parser.add_argument("--data-dir", type=str, default="./data", help="データセットのキャッシュの出力パス")
    
    args = arg_parser.parse_args()

    # サーバの起動
    server = FroseAiServer(
        host = args.host,
        port = args.port,
        ws_max_size = args.ws_max_size,
        log_dir = args.log_dir,
        data_dir = args.data_dir,
        device = "cpu"
    )
    
    # サーバをメインスレッドで起動
    # コンテナはプロセスが終了すると停止してしまうため、
    # 将来的なコンテナ化も見据えてフォアグラウンド方式を採用
    # Uvicorn/FastAPIは標準でCtrl+C受信時に安全にクローズする仕組みがある
    logger.info("FroseAI Server を起動します...")
    server.run()

if __name__ == "__main__":
    main()
