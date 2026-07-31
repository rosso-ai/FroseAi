""" FroseAi Federated Learning Server Module
このモジュールは、連合学習における中央サーバの
REST APIエンドポイント(フロントエンド⇒サーバ)及び
WebSocket通信インタフェース(サーバ⇔クライアント)を提供するモジュールです。
"""

import contextlib
import io
import json
import pickle
import queue
import threading
import time
import torch
import uvicorn
import inspect
from enum import StrEnum
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status, Response
from fastapi.middleware.cors import CORSMiddleware
from logging import INFO, basicConfig, getLogger
from multiprocessing import Process, set_start_method, get_start_method
from pydantic import BaseModel
from typing import Optional
from .aggregator import FedAvgAggregator
from .context import FroseArguments
from .pb.froseai_pb2 import FroseAiPiece, FroseAiParams, FroseAiStatus
from torchvision import models
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor

formatter = '%(asctime)s [%(name)s] %(levelname)s :  %(message)s'
basicConfig(level=INFO, format=formatter)

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

# サーバのステータス状態の列挙クラス
class PhaseStatus(StrEnum):
    UNINITIALIZED = "uninitialized"    # サーバ起動前
    READY = "ready"    # クライアント起動前
    TRAINING = "training"    # クライアント学習中
    AGGREGATING = "aggregating"    # 学習結果集約中
    COMPLETED = "completed"    # 正常終了
    ERROR = "error"    # エラー終了

# RestAPI用のクラスオブジェクト
# データ検証の自動化のため、BaseModelを継承
# GET STATUS
class ResponseGetStatus(BaseModel):
    # 列挙クラスで取りうる値を定義
    status: PhaseStatus    # ステータス
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

# サーバ側(Aggregator)⇔クライアント側(WebSocket)のゲートウェイクラス
class FroseAiGateway:
    # 初期化
    def __init__(self, agg: FedAvgAggregator):
        # 引数を取得
        self._agg = agg
        # ロガーを取得
        self._logger = getLogger("FroseAi-Gateway")
        # クライアント情報管理用の辞書を初期化
        self._clients: dict[str, dict] = {}
        # 起動時刻を記録(UNIX時刻形式)
        self._start_time: float = time.time()
        # 接続済みのクライアント数
        self._connected_clients : int = 0
        # 学習中のクライアント数
        self._uncomplete_clients : int = 0
        # クライアント情報
        self._clients_info = {}
        # 現在のステータス
        self._status : PhaseStatus = PhaseStatus.READY
        self._logger.info("FroseAi-Gatewayを初期化しました")

    # リセット
    def reset(self):
        self._agg.reset()
        self._clients_info = {}
        self._status = PhaseStatus.READY

    # 読み取り用プロパティ
    # 引数
    @property
    def agg(self) -> FedAvgAggregator:
        return self._agg
    
    # AIモデル
    @property
    def model(self):
        return self._agg.model
    
    # ステータス
    @property
    def status(self):
        return self._status
    
    # 学習が完了したクライアント数
    @property
    def complete_clients(self) -> int:
        return self._connected_clients - self._uncomplete_clients
    
    # 稼働時間(秒)
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    # 接続/切断処理
    # サーバ⇔クライアント間の接続を新規作成
    async def connect(self, websocket: WebSocket, client_id: str):
        # 接続要求の受け入れ
        await websocket.accept()
        # クライアント情報を初期化して辞書に追加
        self._clients[client_id] = {
            "ws": websocket,
            "round": 0,
            "waiting": False
        }
        self._logger.info(f"クライアントを接続しました クライアントID: {client_id}")

    # サーバ⇔クライアント間の接続が切断された際の処理
    def disconnect(self, client_id: str):
        # 辞書から該当のクライアント情報を削除
        self._clients.pop(client_id, None)
        self._logger.info(f"クライアントを切断しました クライアントID: {client_id}")

    # クライアントからのリクエスト種別ごとの処理
    # クライアント処理開始時の最初のハンドシェイク
    def hello(self, req: FroseAiParams) -> FroseAiParams:
        # ステータスを学習中に変更
        self._status = PhaseStatus.TRAINING
        # ラウンド数を初期化
        self._agg.round = 1
        # 接続済みのモデル数を1増やす
        self._connected_clients = self._connected_clients + 1
        # 計算中のモデル数を1増やす
        self._uncomplete_clients = self._uncomplete_clients + 1
        # クライアント情報の追加
        self._clients_info[str(req.src)] = {
            "status": "training",
            "last_seen": time.time()
        }
        # サーバの保持するAIモデル重みをCPUに配置
        ret_model_state = self.model.cpu().state_dict()
        # AIモデルを配置できていれば重みをバイナリ化して返却メッセージに格納
        # AIモデルを配置できていなければリクエストのメッセージをそのまま返却
        if ret_model_state is None:
            messages = req.messages
        else:
            messages = pickle.dumps({"model": ret_model_state})
        # レスポンスを生成
        res = FroseAiParams()
        res.src = req.src
        res.messages = messages
        res.round = self._agg.round
        return res

    # クライアントから送付された重みを受け取る
    def push(self, req: FroseAiParams) -> FroseAiPiece:
        # srcをint型に統一する
        client_id = int(req.src) if isinstance(req.src, (int, str)) else req.src
        # クライアントから送付された重みをバイナリから戻し、集約ロジックに登録
        self._agg.push(client_id, pickle.loads(req.messages), req.round)
        # 計算中のモデル数を1減らす
        self._uncomplete_clients = self._uncomplete_clients - 1
        # クライアント情報の更新
        self._clients_info[str(req.src)] = {
            "status": "complete",
            "last_seen": time.time()
        }
        # 重みをすべて受け取り済みならステータスを変更
        if self.agg.is_all_received:
            self._status = PhaseStatus.AGGREGATING
        # レスポンスを生成
        res = FroseAiPiece()
        res.src = req.src
        res.status = 202
        return res

    # 集約後の重みをクライアントが要求する
    def pull(self, req: FroseAiParams) -> FroseAiParams:
        # レスポンス初期化
        status = 204
        messages = b""
        # クライアント情報の更新
        self._clients_info[str(req.src)] = {
            "status": "complete",
            "last_seen": time.time()
        }
        # client_idの型について、整数と文字列の両方で判定できるようにチェック
        client_id = req.src
        # int 型 / str 型の両方のキーで snd_q を探索
        target_key = None
        if client_id in self._agg.snd_q:
            target_key = client_id
        elif str(client_id) in self._agg.snd_q:
            target_key = str(client_id)
        elif client_id != "" and int(client_id) in self._agg.snd_q:
            target_key = int(client_id)
        # 送信キューに該当クライアント用のデータがあれば返却メッセージに格納
        if target_key is not None and not self._agg.snd_q[target_key].empty():
            status = 200
            messages = self._agg.snd_q[target_key].get()
            # 学習中のクライアント数を1増やす
            self._uncomplete_clients = self._uncomplete_clients + 1
            # クライアント情報の更新
            self._clients_info[str(req.src)] = {
                "status": "training",
                "last_seen": time.time()
            }
            # ステータスを学習中に変更
            self._status = PhaseStatus.TRAINING
            
            # 【重要】特定のクライアントが取得しただけで全体の集約状態を消去してしまわないよう、
            # 全てのキューが空になった場合のみクリアを実行する
            all_empty = all(q.empty() for q in self._agg.snd_q.values())
            if all_empty:
                self._agg.clear_aggregator()
        # レスポンスを生成
        res = FroseAiParams()
        res.src = req.src
        res.status = status
        res.messages = messages if messages else b""
        res.round = self._agg.round
        return res

    # ステータス確認をクライアントが要求する
    def status(self, req: FroseAiParams) -> FroseAiStatus:
        # レスポンスを生成
        res = FroseAiStatus()
        res.src = req.src
        res.status = 200
        return res

# クライアント管理用のグローバルオブジェクト
client_processes = []
client_lock = threading.Lock()

# クライアントプロセスの実行関数
def _proc_run(
    conf: FroseArguments,
    client_id: int,
    model,
    host = "localhost",
    port = "8000",
    dataset = None,
    criterion = None,
    device = "cpu"
):
    from .optimizer import FedAvg
    import torch.nn as nn
    import logging
    from logging import basicConfig, getLogger
    formatter = '%(asctime)s [%(name)s] %(levelname)s :  %(message)s'
    basicConfig(level=logging.INFO, format=formatter)
    logger = getLogger("Frose-Client")
    optimizer = FedAvg(
        parameters = model.parameters(),
        client_id = client_id,
        job_name = conf.repo_name,
        host = host,
        port = port,
        lr = 0.1,
        weight_decay = 0.01,
        train_data_num = dataset["num"]
    )

    optimizer.hello(model)

    while optimizer.round <= conf.round:
        logger.info("[Client:%4d]  Round-%d Start!!" % (client_id, optimizer.round))
        model.train().to(device)
        batch_loss = []
        for batch_idx, (x, labels) in enumerate(dataset["data"]):
            x, labels = x.to(device), labels.to(device)

            optimizer.zero_grad()
            labels = labels.long()
            log_probs = model(x)
            loss = criterion(log_probs, labels)  # pylint: disable=E1102

            loss.backward()
            batch_loss.append(loss.item())
            optimizer.step()

        if len(batch_loss) > 0:
            logger.info("[Client:%4d]    Loss: %.8f" % (client_id, sum(batch_loss) / len(batch_loss)))

        optimizer.update(model)

    logger.info("[Client:%4d]  Training Finished!!" % (client_id,))

# インタフェースアプリケーションを作成
app = FastAPI(title="FroseAI Server API & WebSocket", version="1.0.0")

# CORSミドルウェアを追加
# これを設定しないとセキュリティ上の理由からWebSocketがアクセスを拒否して403エラーとなる
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ゲートウェイはサーバ全体で使用するため、グローバル変数として定義
# 初期化は後で実施するため、エラー防止用にNoneの可能性もあるように定義
gateway: Optional[FroseAiGateway] = None

# RESTエンドポイントの定義
# GET /api/v1/status でサーバ状態を返却
# レスポンスのフォーマットはResponseGetStatusクラスで定義
@app.get(
    "/api/v1/status",
    response_model = ResponseGetStatus,
    responses = {
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "ゲートウェイまたはアグリゲータが未初期化"
        }
    }
)
def get_status() -> ResponseGetStatus:
    # ゲートウェイまたはアグリゲータが未初期化の場合は 503 を返す
    if gateway is None or gateway._agg is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "ゲートウェイまたはアグリゲータが未初期化"
        )
    
    return {
        "status": gateway._status,
        "total_round": gateway._agg.round_num,
        "current_round": gateway._agg.round,
        "total_clients": gateway._agg.client_num,
        "complete_clients": gateway.complete_clients,
        "uptime_seconds": gateway.uptime_seconds,
        "latest_metrics": gateway._agg.last_metrics
    }

# GET /api/v1/clients でクライアントの一覧を返却
# いったんエンドポイントだけ作成
@app.get(
    "/api/v1/clients",
    response_model = ResponseGetClientList,
    responses = {
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "ゲートウェイまたはアグリゲータが未初期化"
        }
    }
)
def get_client_list():
    # ゲートウェイまたはアグリゲータが未初期化の場合は 503 を返す
    if gateway is None or gateway._agg is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "ゲートウェイまたはアグリゲータが未初期化"
        )
    return {
        "total_clients": gateway._agg.client_num,
        "current_round": gateway._agg.round,
        "clients": gateway._clients_info
    }

# GET /api/v1/clients/{client_id} で個別のクライアント状態を返却
# いったんエンドポイントだけ作成
@app.get(
    "/api/v1/clients/{client_id}",
    response_model = ResponseGetClientStatus,
    responses = {
        status.HTTP_404_NOT_FOUND: {
            "description": "指定のクライアントが不在"
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "ゲートウェイまたはアグリゲータが未初期化"
        }
    }
)
def get_client_status(client_id: str):
    # ゲートウェイまたはアグリゲータが未初期化の場合は 503 を返す
    if gateway is None or gateway._agg is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "ゲートウェイまたはアグリゲータが未初期化"
        )
    # 指定したクライアントが存在しない場合は 404 を返す
    if client_id not in gateway._clients_info:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail = "指定のクライアントが不在"
        )
    return {
        "client_id": client_id,
        "status": gateway._clients_info[client_id]["status"],
        "last_seen": gateway._clients_info[client_id]["last_seen"],
        "current_round": gateway._agg.round
    }

# GET /api/v1/config で現在適用されている設定値を返却
@app.get(
    "/api/v1/config",
    responses = {
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "ゲートウェイまたはアグリゲータが未初期化"
        }
    }
)
def get_config():
    # ゲートウェイまたはアグリゲータが未初期化の場合は 503 を返す
    if gateway is None or gateway._agg is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "ゲートウェイまたはアグリゲータが未初期化"
        )
    return {
        "config": gateway._agg._conf
    }

# GET /api/v1/model/latest で最新のAIモデル重みを返却
@app.get(
    "/api/v1/model/latest",
    summary = "メモリ上に保持されている最新のAIモデル重み(バイナリ形式)の取得",
    responses = {
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "ゲートウェイまたはアグリゲータが未初期化"
        }
    }
)
def get_model_latest():
    # ゲートウェイまたはアグリゲータが未初期化の場合は 503 を返す
    if gateway is None or gateway._agg is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "ゲートウェイまたはアグリゲータが未初期化"
        )
    
    # 最新のモデル重みをメモリバッファに保存
    buffer = io.BytesIO()
    torch.save(gateway._agg._model.state_dict(), buffer)
    buffer.seek(0)
    
    # 値をバイナリ形式で返却
    return Response(
        content = buffer.getvalue(),
        media_type = "application/octet-stream",
        headers = {
            "Content-Disposition": "attachment; filename=latest_model.pt"
        }
    )

# GET /api/v1/metrics でメトリクスを返却
@app.get(
    "/api/v1/metrics",
    responses = {
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "ゲートウェイまたはアグリゲータが未初期化"
        }
    }
)
def get_metrics():
    # ゲートウェイまたはアグリゲータが未初期化の場合は 503 を返す
    if gateway is None or gateway._agg is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "ゲートウェイまたはアグリゲータが未初期化"
        )
    return {
        "metrics": gateway._agg.last_metrics
    }

# GET /metrics でメトリクスを返却(Prometheus用)
# いったんエンドポイントだけ作成
@app.get(
    "/metrics",
    responses = {
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "ゲートウェイまたはアグリゲータが未初期化"
        }
    }
)
def get_metrics_prom():
    # ゲートウェイまたはアグリゲータが未初期化の場合は 503 を返す
    if gateway is None or gateway._agg is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "ゲートウェイまたはアグリゲータが未初期化"
        )
    
    # メトリクス(JSON)の取得
    last_metrics_json = gateway._agg.last_metrics
    # Prometheus用にメトリクスを変換
    # メトリクスが存在しない(集約前)の場合、存在しない旨を返す
    if not last_metrics_json:
        return Response(
            content = "# No metrics available yet\n",
            media_type = "text/plain"
        )
    # JSON文字列を一旦辞書オブジェクトに変換
    try:
        metrics_dict = json.loads(last_metrics_json)
    except json.JSONDecodeError:
        return Response(
            content = "# Error parsing metrics\n",
            media_type = "text/plain"
        )
    # Prometheus用のフォーマットに組み立て
    prom_lines = []
    for key, val in metrics_dict.items():
        metric_name = f"froseai_{key}"
        prom_lines.append(f"# HELP {metric_name} Latest round {key}")
        prom_lines.append(f"# TYPE {metric_name} gauge")
        prom_lines.append(f"{metric_name} {val}")
        prom_lines.append("")    # 空行
    content = "\n".join(prom_lines)
    
    return Response(
        content = content,
        media_type = "text/plain"
    )

# GET /healthz/live でヘルスチェック(生存確認,Kubernetes用)
# いったんエンドポイントだけ作成
@app.get(
    "/healthz/live",
    response_model = ResponseGetHealthz,
    status_code = status.HTTP_200_OK,
    summary = "Liveness Probe (生存確認)",
    tags = ["Health Check"]
)
def get_healthz_live():
    return {
        "status": "alive"
    }

# GET /healthz/ready でヘルスチェック(準備完了確認,Kubernetes用)
# いったんエンドポイントだけ作成
@app.get(
    "/healthz/ready",
    response_model = ResponseGetHealthz,
    status_code = status.HTTP_200_OK,
    responses = {
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "ゲートウェイまたはアグリゲータが未初期化"
        }
    },
    summary = "Readiness Probe (準備完了確認)",
    tags = ["Health Check"]
)
def get_healthz_ready():
    # ゲートウェイまたはアグリゲータが未初期化の場合は 503 を返す
    if gateway is None or gateway._agg is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "ゲートウェイまたはアグリゲータが未初期化"
        )
    return {
        "status": "ready"
    }

# GET /api/v1/models で利用可能なモデルのリストを返却
@app.get(
    "/api/v1/models"
)
def get_models():
    tv_model_names = set(models.list_models())
    custom_model_names = set(MODEL_REGISTRY.keys())
    
    all_models = []
    for name in tv_model_names:
        if name in custom_model_names:
            continue
        model_fn = models.get_model_builder(name)
        params = list(inspect.signature(model_fn).parameters.keys())
        all_models.append({"name": name, "type": "torchvision", "parameters": params})
    for name, model_cls in MODEL_REGISTRY.items():
        params = list(inspect.signature(model_cls).parameters.keys())
        all_models.append({"name": name, "type": "custom", "parameters": params})
    
    return {
        "models": sorted(all_models, key=lambda x: x["name"]),
        "document": "https://docs.pytorch.org/vision/stable/models.html"
    }

# GET /api/v1/datasets で利用可能なデータセットのリストを返却
@app.get(
    "/api/v1/datasets"
)
def get_datasets():
    tv_dataset_names = getattr(datasets, "__all__", dir(datasets))
    
    all_datasets = []
    for name in tv_dataset_names:
        dataset_fn = getattr(datasets, name)
        params = list(inspect.signature(dataset_fn).parameters.keys())
        all_datasets.append({"name": name, "type": "torchvision", "parameters": params})
    return {
        "datasets": sorted(all_datasets, key=lambda x: x["name"]),
        "document": "https://docs.pytorch.org/vision/stable/datasets.html"
    }

# GET /api/v1/criterions で利用可能な損失関数のリストを返却
@app.get(
    "/api/v1/criterions"
)
def get_criterions():
    tn_criterion_names = [
        name for name, obj in inspect.getmembers(nn, inspect.isclass)
        if name.endswith("Loss")
    ]
    
    all_criterions = []
    for name in tn_criterion_names:
        criterion_fn = getattr(nn, name)
        params = list(inspect.signature(criterion_fn).parameters.keys())
        all_criterions.append({"name": name, "type": "torch", "parameters": params})
    return {
        "criterions": sorted(all_criterions, key=lambda x: x["name"]),
        "document": "https://docs.pytorch.org/docs/stable/nn.html#loss-functions"
    }

# POST /api/v1/session/start で連合学習を開始
@app.post(
    "/api/v1/session/start",
    summary = "連合学習セッションの開始",
    responses = {
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "ゲートウェイまたはアグリゲータが未初期化"
        },
        status.HTTP_400_BAD_REQUEST: {
            "description": "連合学習セッションが実行中"
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "指定のモデルが不在"
        }
    }
)
def post_start(req: RequestPostSessionStart):
    global client_processes, gateway
    
    # ゲートウェイまたはアグリゲータが未初期化の場合は 503 を返す
    if gateway is None or gateway._agg is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "ゲートウェイまたはアグリゲータが未初期化"
        )
    
    with client_lock:
        # 実行中のセッションがある場合は 400 を返す
        if any(p.is_alive() for p in client_processes):
            raise HTTPException(
                status_code = status.HTTP_400_BAD_REQUEST,
                detail = "連合学習セッションが実行中"
            )
        # プロセス開始方式の設定
        if get_start_method() == 'fork':
            set_start_method('spawn', force=True)
        # リクエストのJSONからパラメータを構築
        conf = FroseArguments(
            repo_name = req.repo_name,
            random_seed = req.random_seed,
            device = req.device,
            round = req.round,
            batch_size = req.batch_size,
            inner_loop = req.inner_loop,
            partition_method = req.partition_method,
            partition_alpha = req.partition_alpha,
            worker_num = req.worker_num
        )
        
        # 指定したモデル名からモデルを検索
        # 標準モデルの挙動をカスタマイズできるよう、
        # モデルレジストリ側に名前がある場合はmodelsより優先する
        model = None
        if req.model_name in MODEL_REGISTRY:
            model = MODEL_REGISTRY[req.model_name](**req.model_args)
            if hasattr(models, req.model_name):
                print(f"Warning: '{name}' は torchvision.models にも存在する名前です。自作モデルで上書きされます。")
        elif hasattr(models, req.model_name):
            model_fn = getattr(models, req.model_name)
            model = model_fn(**req.model_args)
        else:
            raise HTTPException(
                status_code = status.HTTP_404_NOT_FOUND,
                detail = "指定のモデルが不在"
            )
        
        # 指定したデータセット名からデータセットを検索
        train_data = None
        valid_data = None
        if hasattr(datasets, req.dataset_name):
            dataset_fn = getattr(datasets, req.dataset_name)
            train_data = dataset_fn(
                root = gateway._agg._data_dir,
                train = True,
                download = True,
                transform = ToTensor(),
                **req.dataset_args
            )
            valid_data = dataset_fn(
                root = gateway._agg._data_dir,
                train = False,
                download = True,
                transform = ToTensor(),
                **req.dataset_args
            )
        else:
            raise HTTPException(
                status_code = status.HTTP_404_NOT_FOUND,
                detail = "指定のデータセットが不在"
            )
        
        # 指定した評価関数名から評価関数を検索
        criterion = None
        if hasattr(nn, req.criterion_name):
            criterion_fn = getattr(nn, req.criterion_name)
            criterion = criterion_fn(**req.criterion_args)
        else:
            raise HTTPException(
                status_code = status.HTTP_404_NOT_FOUND,
                detail = "指定の評価関数が不在"
            )
        
        gateway._agg._model = model
        gateway._agg._criterion = criterion
        gateway._agg._train_data = train_data
        gateway._agg._valid_data = valid_data
        
        gateway._agg._conf = conf
        gateway._agg.start()
        
        client_processes = []
        model = gateway._agg._model
        fed_datasets = gateway._agg._fed_data
        # クライアントプロセスの起動
        for client_id in range(conf.worker_num):
            # クライアント起動
            client = Process(
                target = _proc_run,
                kwargs = {
                    "conf": conf,
                    "client_id": client_id,
                    "model": gateway._agg._model,
                    "host": gateway._agg._host,
                    "port": gateway._agg._port,
                    "dataset": fed_datasets.fed_dataset(client_id),
                    "criterion": gateway._agg._criterion,
                    "device": conf.device
                }
            )
            client.start()
            client_processes.append(client)
    
    return {
        "status": "started",
        "message": f"{conf.worker_num} 台のクライアントプロセスを起動しました"
    }

# POST /api/v1/session/stop で連合学習を停止
# 学習を早期終了したい場合ややり直したい場合、
# システム異常などで緊急停止したい場合に使用
@app.post(
    "/api/v1/session/stop",
    summary = "連合学習セッションの停止",
    responses = {
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "ゲートウェイまたはアグリゲータが未初期化"
        },
        status.HTTP_400_BAD_REQUEST: {
            "description": "連合学習セッションが不在"
        }
    }
)
def post_stop():
    global client_processes, gateway
    
    # ゲートウェイまたはアグリゲータが未初期化の場合は 503 を返す
    if gateway is None or gateway._agg is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "ゲートウェイまたはアグリゲータが未初期化"
        )
    
    # 現在学習中でない場合はエラー
    current_status = gateway._status
    if current_status not in [PhaseStatus.TRAINING, PhaseStatus.AGGREGATING]:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = "連合学習セッションが不在"
        )
    
    # 停止中にエラーが起きる場合を考慮してtry-exceptを利用
    try:
        # 起動しているクライアントを1つずつ停止
        stopped_clients_count = 0
        if client_processes:
            for p in client_processes:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=2.0)
                    if p.is_alive():
                        p.kill()
                    stopped_clients_count += 1
        # サーバステータスを終了に更新
        gateway._status = PhaseStatus.COMPLETED
        return {
            "message": "連合学習セッションの停止に成功しました",
            "stoped_clients": stopped_clients_count,
            "final_status": gateway.status
        }
    except Exception as e:
        # エラー発生時はステータスをエラーに設定
        gateway._status = PhaseStatus.ERROR
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail = f"連合学習セッションの停止に失敗: {str(e)}"
        )

# POST /api/v1/session/reset でラウンド数やキューなどを初期化
@app.post(
    "/api/v1/session/reset",
    summary = "連合学習セッション情報の初期化",
    responses = {
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "ゲートウェイまたはアグリゲータが未初期化"
        },
        status.HTTP_400_BAD_REQUEST: {
            "description": "連合学習セッションが実行中"
        }
    }
)
def post_reset():
    global client_processes, gateway
    # ゲートウェイまたはアグリゲータが未初期化の場合は 503 を返す
    if gateway is None or gateway._agg is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "ゲートウェイまたはアグリゲータが未初期化"
        )
    
    # 現在学習中の場合はエラー
    current_status = gateway._status
    if current_status in [PhaseStatus.TRAINING, PhaseStatus.AGGREGATING]:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = "連合学習セッションが実行中"
        )
    
    # 残存プロセスの清掃
    if client_processes:
        for p in client_processes:
            if p.is_alive():
                p.terminate()
                p.join()
        client_processes = []
    
    # リセット実行
    gateway.reset()
    
    return {
        "message": "連合学習セッション情報の初期化に成功しました"
    }

# WebSocketエンドポイントの定義
# サーバ⇔クライアント間のエンドポイント
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str, op: str = "hello"):
    # ゲートウェイが初期化されていなければエラーコード1011を返却
    if gateway is None:
        await websocket.close(code=1011)
        return

    # クライアントとの接続を新規作成
    await gateway.connect(websocket, client_id)

    try:
        while True:
            # バイナリデータの受信
            data = await websocket.receive_bytes()
            # バイナリデータを復元
            req = FroseAiParams()
            req.ParseFromString(data)
            # opで処理を振り分け (Hello / Push / Pull / Status)
            if op == "hello":
                res = gateway.hello(req)
                await websocket.send_bytes(res.SerializeToString())
            elif op == "push":
                res = gateway.push(req)
                await websocket.send_bytes(res.SerializeToString())
            elif op == "pull":
                res = gateway.pull(req)
                await websocket.send_bytes(res.SerializeToString())
            elif op == "status":
                res = gateway.status(req)
                await websocket.send_bytes(res.SerializeToString())
            else:
                getLogger("FroseAi-Gateway").warning(f"不正な操作種別です 操作種別: {op}")
    except WebSocketDisconnect:
        # クライアントとの接続が切れた場合、辞書からクライアント情報を削除
        gateway.disconnect(client_id)

# サーバ管理クラス
class FroseAiServer(uvicorn.Server):
    # 初期化
    def __init__(
        self,
        host = "localhost",
        port = 8000,
        ws_max_size = 1000 * 1024 * 1024,
        log_dir = "./log",
        data_dir = "./data",
        device="cpu",
        **kwargs
    ):
        # ゲートウェイ用のグローバル変数を定義
        global gateway
        # ロガーを取得
        self._logger = getLogger("FroseAi-Server")
        # 引数を取得
        self._agg = FedAvgAggregator(
            host = host,
            port = port,
            ws_max_size = ws_max_size,
            log_dir = log_dir,
            data_dir = data_dir,
            device = device
        )
        # ゲートウェイの初期化
        gateway = FroseAiGateway(self._agg)

        # 各種設定をサーバ用のコンフィグオブジェクトに格納
        config = uvicorn.Config(
            app, 
            host=host, 
            port=port, 
            ws_max_size=ws_max_size, 
            access_log=False, 
            **kwargs
        )
        # サーバを初期化
        super().__init__(config)

    # シグナルハンドラの無効化
    # サーバが別のバックグラウンドスレッドで動く際、
    # Ctrl+CなどのOSシグナルを受け取れずエラーになるのを防ぐため、
    # デフォルトの設定を無効化している
    def install_signal_handlers(self):
        pass

    # 別スレッドでサーバを非同期起動
    @contextlib.contextmanager
    def run_in_thread(self):
        # 別スレッドでサーバを非同期起動
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            # サーバが稼働するまで待機
            while not self.started and thread.is_alive():
                time.sleep(1e-3)
            # サーバ稼働
            yield
        # 終了時の処理
        finally:
            # サーバに終了フラグを付ける
            self.should_exit = True
            # 完全に終了するまで待機
            thread.join()
