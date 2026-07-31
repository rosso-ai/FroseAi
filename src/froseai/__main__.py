import sys
import os

import time

import logging
from logging import basicConfig, getLogger

import uvicorn

formatter = '%(asctime)s [%(name)s] %(levelname)s :  %(message)s'
basicConfig(level=logging.INFO, format=formatter)
logger = getLogger("Frose-Runner")

from context import ServerConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from froseai import FroseAiServer

from pydantic_settings import CliApp

# メイン処理
def main():
    config = CliApp.run(ServerConfig)
    # サーバの起動
    server = FroseAiServer(
        host = config.host,
        port = config.port,
        ws_max_size = config.ws_max_size,
        log_dir = config.log_dir,
        data_dir = config.data_dir,
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
