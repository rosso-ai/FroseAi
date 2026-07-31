
from logging import INFO, basicConfig, Logger, getLogger

# フォーマットされたロガーを返す
def get_logger(name: str) -> Logger:
    formatter = '%(asctime)s [%(name)s] %(levelname)s :  %(message)s'
    basicConfig(level=INFO, format=formatter)
    return getLogger(name)


from enum import StrEnum
from pydantic import BaseModel

# サーバのステータス状態の列挙クラス
class PhaseServer(StrEnum):
    READY = "ready"    # クライアント起動前
    TRAINING = "training"    # クライアント学習中
    AGGREGATING = "aggregating"    # 学習結果集約中
    COMPLETED = "completed"    # 正常終了
    ERROR = "error"    # エラー終了

# クライアントのステータス状態の列挙クラス
class PhaseClient(StrEnum):
    READY = "ready"    # クライアント起動前
    TRAINING = "training"    # クライアント学習中
    COMPLETED = "completed"    # クライアント学習終了
    ERROR = "error"    # エラー終了

class FroseArguments(BaseModel):
    repo_name: str = "fedavg_cifar10_hetero1.0"
    model_name: str = "logistic_regression"
    random_seed: int = 42
    device: str = "cpu"
    log_output_path: str = "./log"
    round: int = 10
    batch_size: int = 100
    inner_loop: int = 100
    partition_method: str =  "hetero"
    partition_alpha: float =  10.0
    worker_num: int = 1

class ClientInfo(BaseModel):
    status: PhaseClient
    round: int
    last_seen: float    # 最終アクセス時刻



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

import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        outputs = self.linear(x)
        return outputs


# モデル生成用のファクトリ関数
# torchvision.modelsに含まれていないモデルを利用したい場合はここに登録
MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
}


# RestAPI用のクラスオブジェクト
# データ検証の自動化のため、BaseModelを継承
# GET STATUS
class ResponseGetStatus(BaseModel):
    # 列挙クラスで取りうる値を定義
    status: PhaseServer    # ステータス
    total_round: int    # 総ラウンド数
    current_round: int    # 現在のラウンド数
    total_clients: int    # 総クライアント数
    complete_clients: int    # 学習結果を返したクライアント数
    uptime_seconds: float    # 連続稼働時間(秒)
    # 文字列またはNone、指定なしの場合はNone
    latest_metrics: str | None = None    # 最新のメトリクス値

# GET HEALTHZ
class ResponseGetHealthz(BaseModel):
    status: str

# GET CLIENT LIST
class ResponseGetClientList(BaseModel):
    total_clients: int    # 総クライアント数
    current_round: int    # 現在のラウンド数
    clients: dict    # クライアントのリスト

# GET CLIENT STATUS
class ResponseGetClientStatus(BaseModel):
    client_id: str    # クライアントの識別ID
    status: str    # クライアントのステータス
    last_seen: float    # 最終アクセス時刻
    current_round: int    # 現在のラウンド数

# POST SESSION START
class RequestPostSessionStart(BaseModel):
    repo_name: str = "fedavg_cifar10_hetero1.0"
    model_name: str = "logistic_regression"
    model_args: dict = {"input_dim": 784, "output_dim": 10}
    dataset_name: str = "EMNIST"
    dataset_args: dict = {"split": "digits"}
    criterion_name: str = "CrossEntropyLoss"
    criterion_args: dict = {}
    random_seed: int = 0
    device: str = "cpu"
    round: int = 1
    batch_size: int = 100
    inner_loop: int = 1
    partition_method: str = "hetero"
    partition_alpha: float = 1.0
    worker_num: int = 1





