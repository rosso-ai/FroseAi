import sys
import os
import logging
import argparse

import time

from logging import basicConfig, getLogger

import uvicorn

formatter = '%(asctime)s [%(name)s] %(levelname)s :  %(message)s'
basicConfig(level=logging.INFO, format=formatter)
logger = getLogger("Frose-Runner")

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from froseai import FroseAiServer

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

# 将来的な環境変数化の可能性も考慮してBaseSettingsを採用
class ServerConfig(BaseSettings):
    """FroseAi: 連合学習フレームワーク"""
    # 先頭のコメントのみdocstringとして読み込まれヘルプの説明文となる
    
    model_config = SettingsConfigDict(
        cli_prog_name = "froserun",    # 実行コマンド名
        cli_epilog = (    # ヘルプの末尾でGitHubのREADMEに飛ばす
            "詳細な仕様は以下のリンクをご覧ください:\n"
            "https://github.com/rosso-ai/FroseAi/blob/main/README.md"
        ),
        cli_kebab_case = True,    # スネークケース(log_dir)からケバブケース(--log-dir)に割り当て
        cli_exit_on_error = True,    # 無効な引数が渡された場合にシステム終了する
        case_sensitive = True,    # -hと-Hが衝突しないよう、大文字小文字を区別
    )
    
    # 引数の定義
    
    # ホスト、ポートはよく使うため短縮形あり
    # IPv6(::1)との解釈揺れを避けるため、127.0.0.1で指定
    host : str = Field(
        validation_alias = AliasChoices("H", "host"),
        default = "127.0.0.1",
        description="サーバに接続する際のホスト名"
    )
    # 9200はElasticsearchのデフォルトポートのため衝突の可能性あり
    # FastAPIのデフォルトポートは8000なので、そちらに合わせる
    port : int = Field(
        validation_alias = AliasChoices("p", "port"),
        default = 8000,
        description = "サーバに接続する際のポート番号"
    )
    # モデル重みをやり取りするため、メッセージサイズ上限は1GBでとる
    ws_max_size : int = Field(
        default = 1000 * 1024 * 1024,
        description = "サーバ⇔クライアント間のメッセージサイズ上限(bytes)"
    )
    # ログやキャッシュのデフォルトパスは実行ファイルからの相対パスで定義
    log_dir : str = Field(
        default="./log",
        description="ログの出力パス"
    )
    data_dir : str = Field(
        default="./data",
        description="データセットのキャッシュの出力パス"
    )

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
