import sys
import os
import logging
import argparse

#import threading
import time

#import torch.nn as nn
#from torchvision import datasets
#from torchvision.transforms import ToTensor
from logging import basicConfig, getLogger
#from multiprocessing import Process, set_start_method, get_start_method

import uvicorn

formatter = '%(asctime)s [%(name)s] %(levelname)s :  %(message)s'
basicConfig(level=logging.INFO, format=formatter)
logger = getLogger("Frose-Runner")

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from froseai import FroseAiServer #, FedDatasetsClassification, FroseArguments, FedAvg, FedValidator

# FastAPIを別スレッドで起動
def run_fastapi(app: uvicorn.Server):
    app.run()

# ヘルプ関連の設定
class ArgParseFormatter(
    argparse.ArgumentDefaultsHelpFormatter,    # デフォルト値をヘルプに表示
    argparse.RawDescriptionHelpFormatter,    # ヘルプ文章の改行を維持
):
    pass

# メイン処理
def main():
    arg_parser = argparse.ArgumentParser(
        # --helpを指定した際の説明文
        prog = "froserun",    # 実行コマンド名
        description = "FroseAi: 連合学習フレームワーク",
        epilog = (    # ヘルプの末尾でGitHubのREADMEに飛ばす
            "詳細な仕様は以下のリンクをご覧ください:\n"
            "https://github.com/rosso-ai/FroseAi/blob/main/README.md"
        ),
        add_help = False, # ヘルプメッセージを日本語で表示するため個別定義
        formatter_class = ArgParseFormatter
    )
    # ヘルプメッセージを日本語で表示するため個別定義
    arg_parser.add_argument(
        "-h",
        "--help",
        action = "help",
        help="ヘルプメッセージを表示して終了"
    )
    # ホスト、ポートはよく使うため短縮形あり
    # IPv6(::1)との解釈揺れを避けるため、127.0.0.1で指定
    arg_parser.add_argument(
        "-H",
        "--host",
        type = str,
        default = "127.0.0.1",
        help="サーバに接続する際のホスト名"
    )
    # 9200はElasticsearchのデフォルトポートのため衝突の可能性あり
    # FastAPIのデフォルトポートは8000なので、そちらに合わせる
    arg_parser.add_argument(
        "-p",
        "--port",
        type = int,
        default = 8000,
        help = "サーバに接続する際のポート番号"
    )
    # モデル重みをやり取りするため、メッセージサイズ上限は1GBでとる
    arg_parser.add_argument(
        "--ws-max-size",
        type = int,
        default = 1000 * 1024 * 1024,
        help = "サーバ⇔クライアント間のメッセージサイズ上限(bytes)"
    )
    # ログやキャッシュのデフォルトパスは実行ファイルからの相対パスで定義
    arg_parser.add_argument(
        "--log-dir",
        type=str,
        default="./log",
        help="ログの出力パス"
    )
    arg_parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="データセットのキャッシュの出力パス"
    )
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
