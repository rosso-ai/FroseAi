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
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status, Response, APIRouter, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from multiprocessing import Process, set_start_method, get_start_method
from typing import Optional
from .context import *
from .aggregator import FedAvgAggregator
from .context import FroseArguments
from .client import _proc_run
from .pb.froseai_pb2 import FroseAiPiece, FroseAiParams, FroseAiStatus
from torchvision import models
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from contextlib import asynccontextmanager
from starlette.requests import HTTPConnection

################################### アプリ生成関連 ###################################
# サーバ起動・終了時の処理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # サーバ起動時の処理
    # 既存コードで `FroseAiServer.__init__` で行っていた初期化ロジックをここに集約
    logger = get_logger("FroseAi-Server")
    logger.info("サーバを起動しています...")
    
    # app.stateからconfigを取得
    server_conf: ServerConfig = app.state.config
    # 集約インスタンスを作成
    aggregator = FedAvgAggregator(server_conf)
    # 通信インスタンスを作成し、集約インスタンスを注入
    app.state.gateway = FroseAiGateway(server_conf, aggregator)
    logger.info("サーバの起動が完了しました。")
    
    yield  # ここでAPIリクエストの受付状態（サーバー稼働中）になる
    
    # サーバ終了時の処理
    logger.info("サーバを終了しています...")
    # 必要に応じてクリーンアップを記述
    logger.info("サーバの終了が完了しました。")

# API生成をファクトリ関数内部で行うため、APIルータを使用
router = APIRouter()

# アプリ生成用のファクトリ関数
def create_app(config: ServerConfig) -> FastAPI:
    # インタフェースアプリケーションを作成
    app = FastAPI(title="FroseAI Server API & WebSocket", version="1.0.0", lifespan=lifespan)
    
    # CORSミドルウェアを追加
    # これを設定しないとセキュリティ上の理由からWebSocketがアクセスを拒否して403エラーとなる
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # configを取得できるように設定を注入
    app.state.config = config
    
    # ルータをアプリに登録
    app.include_router(router)
    
    return app

# 通信インスタンス参照用の関数
def get_gateway(connection: HTTPConnection) -> FroseAiGateway | None:
    gateway = getattr(connection.app.state, "gateway", None)
    return gateway

################################### 通信クラス ###################################
# サーバ側(Aggregator)⇔クライアント側(WebSocket)のゲートウェイクラス
class FroseAiGateway:
    # 初期化
    def __init__(self, server_conf: ServerConfig, agg: FedAvgAggregator):
        # ロガーを取得
        self._logger = get_logger("FroseAi-Gateway")
        self._logger.info("ゲートウェイを初期化しています...")
        # サーバ用のパラメータを取得
        self._server_conf = server_conf
        # アプリケーション用のパラメータを初期化
        self._app_conf = None
        # 集約インスタンスを取得
        self._agg = agg
        # クライアント情報管理用の辞書を初期化
        self._clients_info: dict[int, ClientInfo] = {}
        # クライアントプロセス格納用
        self._client_processes = []
        # サーバの起動時刻を記録(UNIX時刻形式)
        self._start_time: float = time.time()
        # サーバのステータスを初期化
        self._status : PhaseServer = PhaseServer.READY
        self._logger.info("ゲートウェイの初期化が完了しました。")

    # リセット
    def reset(self):
        self._logger.info("ゲートウェイを初期化しています...")
        # アプリケーション用のパラメータを初期化
        self._app_conf = None
        # 集約インスタンスを初期化
        self._agg.reset()
        # クライアント情報管理用の辞書を初期化
        self._clients_info = {}
        # サーバのステータスを初期化
        self._status = PhaseServer.READY
        self._logger.info("ゲートウェイの初期化が完了しました。")

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
    def status(self) -> str:
        return self._status
    
    # アプリケーション用コンフィグ
    @property
    def app_conf(self):
        return self._app_conf
    
    # サーバ用コンフィグ
    @property
    def server_conf(self):
        return self._server_conf
    
    # クライアントプロセス
    @property
    def client_processes(self):
        return self._client_processes
    
    # 総クライアント数
    @property
    def total_clients(self) -> int:
        return len(self._clients_info)
    
    # 学習が完了したクライアント数
    @property
    def complete_clients(self) -> int:
        complete_clients = sum(
            1 for client in self._clients_info.values()
            if client.status == PhaseClient.COMPLETED
        )
        return complete_clients
    
    # 稼働時間(秒)
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    # パラメータセット
    def set_app_conf(self, app_conf, model, criterion, train_data, valid_data):
        self._app_conf = app_conf
        self._model = model
        self._agg.set_app_conf(app_conf, model, criterion, train_data, valid_data)
    
    # クライアントプロセス追加
    def set_client_process(self, client_process):
        self._client_processes.append(client_process)
    
    # 接続/切断処理
    # サーバ⇔クライアント間の接続を新規作成
    async def connect(self, websocket: WebSocket, client_id: int):
        self._logger.info(f"クライアントに接続しています... クライアントID: {client_id}")
        # 接続要求の受け入れ
        await websocket.accept()
        # クライアント情報を初期化して辞書に追加
        self._clients_info[client_id] = ClientInfo(
            status = PhaseClient.READY,
            round = 0,
            last_seen = time.time()
        )
        self._logger.info(f"クライアントを接続しました。 クライアントID: {client_id}")

    # サーバ⇔クライアント間の接続が切断された際の処理
    def disconnect(self, client_id: int):
        self._logger.info(f"クライアントに接続しています... クライアントID: {client_id}")
        # 辞書から該当のクライアント情報を削除
        if client_id in self._clients_info:
            self._clients_info.pop(client_id, None)
        self._logger.info(f"クライアントを切断しました。 クライアントID: {client_id}")

    # クライアントからのリクエスト種別ごとの処理
    # クライアント処理開始時の最初のハンドシェイク
    def hello(self, req: FroseAiParams) -> FroseAiParams:
        self._logger.info(f"クライアント処理を開始しています... クライアントID: {req.src}")
        # サーバのステータスを学習中に変更
        self._status = PhaseServer.TRAINING
        # クライアント情報の更新
        if req.src in self._clients_info:
            self._clients_info[req.src].status = PhaseClient.TRAINING
            self._clients_info[req.src].round = 1
            self._clients_info[req.src].last_seen = time.time()
        # サーバの保持するAIモデル重みをCPUに配置
        ret_model_state = self._model.cpu().state_dict()
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
        res.round = self.agg.round
        self._logger.info(f"クライアント処理を開始しました。 クライアントID: {req.src}")
        return res

    # クライアントから送付された重みを受け取る
    def push(self, req: FroseAiParams) -> FroseAiPiece:
        # srcをint型に統一する
        client_id = int(req.src) if isinstance(req.src, (int, str)) else req.src
        # クライアントから送付された重みをバイナリから戻し、集約ロジックに登録
        self._agg.push(client_id, pickle.loads(req.messages), req.round)
        # クライアント情報の更新
        self._clients_info[str(req.src)] = {
            "status": "complete",
            "last_seen": time.time()
        }
        # 重みをすべて受け取り済みならステータスを変更
        if self.agg.is_all_received:
            self._status = PhaseServer.AGGREGATING
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
            # クライアント情報の更新
            self._clients_info[str(req.src)] = {
                "status": "training",
                "last_seen": time.time()
            }
            # ステータスを学習中に変更
            self._status = PhaseServer.TRAINING
            
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

################################### RESTエンドポイント(GET) ###################################
# GET /api/v1/status でサーバ状態を返却
# レスポンスのフォーマットはResponseGetStatusクラスで定義
@router.get(
    "/api/v1/status",
    response_model = ResponseGetStatus,
    responses = {
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "ゲートウェイが未初期化"
        }
    }
)
def get_status(gateway: FroseAiGateway = Depends(get_gateway)) -> ResponseGetStatus:
    # ゲートウェイが未初期化の場合は 503 を返す
    if not gateway:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "ゲートウェイが未初期化"
        )
    
    return {
        "status": gateway.status,
        "total_round": gateway.agg.round_num,
        "current_round": gateway.agg.round,
        "total_clients": gateway.total_clients,
        "complete_clients": gateway.complete_clients,
        "uptime_seconds": gateway.uptime_seconds,
        "latest_metrics": gateway.agg.last_metrics
    }

# GET /api/v1/clients でクライアントの一覧を返却
# いったんエンドポイントだけ作成
@router.get(
    "/api/v1/clients",
    response_model = ResponseGetClientList,
    responses = {
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "ゲートウェイが未初期化"
        }
    }
)
def get_client_list(gateway: FroseAiGateway = Depends(get_gateway)):
    # ゲートウェイが未初期化の場合は 503 を返す
    if not gateway:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "ゲートウェイが未初期化"
        )
    return {
        "total_clients": gateway._agg.client_num,
        "current_round": gateway._agg.round,
        "clients": gateway._clients_info
    }

# GET /api/v1/clients/{client_id} で個別のクライアント状態を返却
# いったんエンドポイントだけ作成
@router.get(
    "/api/v1/clients/{client_id}",
    response_model = ResponseGetClientStatus,
    responses = {
        status.HTTP_404_NOT_FOUND: {
            "description": "指定のクライアントが不在"
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "ゲートウェイが未初期化"
        }
    }
)
def get_client_status(client_id: str, gateway: FroseAiGateway = Depends(get_gateway)):
    # ゲートウェイが未初期化の場合は 503 を返す
    if not gateway:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "ゲートウェイが未初期化"
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
@router.get(
    "/api/v1/config",
    responses = {
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "ゲートウェイが未初期化"
        }
    }
)
def get_config(gateway: FroseAiGateway = Depends(get_gateway)):
    # ゲートウェイが未初期化の場合は 503 を返す
    if not gateway:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "ゲートウェイが未初期化"
        )
    return {
        "config": gateway._agg._conf
    }

# GET /api/v1/model/latest で最新のAIモデル重みを返却
@router.get(
    "/api/v1/model/latest",
    summary = "メモリ上に保持されている最新のAIモデル重み(バイナリ形式)の取得",
    responses = {
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "ゲートウェイが未初期化"
        }
    }
)
def get_model_latest(gateway: FroseAiGateway = Depends(get_gateway)):
    # ゲートウェイが未初期化の場合は 503 を返す
    if not gateway:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "ゲートウェイが未初期化"
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
@router.get(
    "/api/v1/metrics",
    responses = {
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "ゲートウェイが未初期化"
        }
    }
)
def get_metrics(gateway: FroseAiGateway = Depends(get_gateway)):
    # ゲートウェイが未初期化の場合は 503 を返す
    if not gateway:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "ゲートウェイが未初期化"
        )
    return {
        "metrics": gateway._agg.last_metrics
    }

# GET /metrics でメトリクスを返却(Prometheus用)
@router.get(
    "/metrics",
    responses = {
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "ゲートウェイが未初期化"
        }
    }
)
def get_metrics_prom(gateway: FroseAiGateway = Depends(get_gateway)):
    # ゲートウェイが未初期化の場合は 503 を返す
    if not gateway:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "ゲートウェイが未初期化"
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
@router.get(
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
@router.get(
    "/healthz/ready",
    response_model = ResponseGetHealthz,
    status_code = status.HTTP_200_OK,
    responses = {
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "ゲートウェイが未初期化"
        }
    },
    summary = "Readiness Probe (準備完了確認)",
    tags = ["Health Check"]
)
def get_healthz_ready(gateway: FroseAiGateway = Depends(get_gateway)):
    # ゲートウェイが未初期化の場合は 503 を返す
    if not gateway:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "ゲートウェイが未初期化"
        )
    return {
        "status": "ready"
    }

# GET /api/v1/models で利用可能なモデルのリストを返却
@router.get(
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
@router.get(
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
@router.get(
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

################################### RESTエンドポイント(POST) ###################################
# POST /api/v1/session/start で連合学習を開始
@router.post(
    "/api/v1/session/start",
    summary = "連合学習セッションの開始",
    responses = {
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "ゲートウェイが未初期化"
        },
        status.HTTP_400_BAD_REQUEST: {
            "description": "連合学習セッションが実行中"
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "指定のモデルが不在"
        }
    }
)
def post_start(req: RequestPostSessionStart, gateway: FroseAiGateway = Depends(get_gateway)):
    # ゲートウェイが未初期化の場合は 503 を返す
    if not gateway:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "ゲートウェイが未初期化"
        )
    
    # 実行中のセッションがある場合は 400 を返す
    if gateway.status in [PhaseServer.TRAINING, PhaseServer.AGGREGATING]:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = "連合学習セッションが実行中"
        )
    
    # プロセス開始方式の設定
    if get_start_method() == 'fork':
        set_start_method('spawn', force=True)
    
    # リクエストのJSONからパラメータを構築
    app_conf = FroseArguments(
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
            print(f"Warning: '{req.model_name}' は torchvision.models にも存在する名前です。自作モデルで上書きされます。")
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
    
    # 各種パラメータをセットする
    gateway.set_app_conf(app_conf, model, criterion, train_data, valid_data)
    
    # クライアントプロセスの起動
    for client_id in range(app_conf.worker_num):
        # クライアント起動
        client = Process(
            target = _proc_run,
            kwargs = {
                "client_id": client_id,
                "app_conf": app_conf,
                "server_conf": gateway._server_conf,
                "model": model,
                "criterion": criterion,
                "dataset": gateway.agg._fed_data.fed_dataset(client_id)
            }
        )
        client.start()
        gateway.set_client_process(client)
    
    return {
        "status": "started",
        "message": f"{app_conf.worker_num} 台のクライアントプロセスを起動しました"
    }

# POST /api/v1/session/stop で連合学習を停止
# 学習を早期終了したい場合ややり直したい場合、
# システム異常などで緊急停止したい場合に使用
@router.post(
    "/api/v1/session/stop",
    summary = "連合学習セッションの停止",
    responses = {
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "ゲートウェイが未初期化"
        },
        status.HTTP_400_BAD_REQUEST: {
            "description": "連合学習セッションが不在"
        }
    }
)
def post_stop(gateway: FroseAiGateway = Depends(get_gateway)):
    # ゲートウェイが未初期化の場合は 503 を返す
    if not gateway:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "ゲートウェイが未初期化"
        )
    
    # 現在学習中でない場合はエラー
    if gateway.status not in [PhaseServer.TRAINING, PhaseServer.AGGREGATING]:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = "連合学習セッションが不在"
        )
    
    # 停止中にエラーが起きる場合を考慮してtry-exceptを利用
    try:
        # 起動しているクライアントを1つずつ停止
        stopped_clients_count = 0
        if gateway.client_processes:
            for p in gateway.client_processes:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=2.0)
                    if p.is_alive():
                        p.kill()
                    stopped_clients_count += 1
        # サーバステータスを終了に更新
        gateway._status = PhaseServer.COMPLETED
        return {
            "message": "連合学習セッションの停止に成功しました",
            "stoped_clients": stopped_clients_count,
            "final_status": gateway.status
        }
    except Exception as e:
        # エラー発生時はステータスをエラーに設定
        gateway._status = PhaseServer.ERROR
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail = f"連合学習セッションの停止に失敗: {str(e)}"
        )

# POST /api/v1/session/reset でラウンド数やキューなどを初期化
@router.post(
    "/api/v1/session/reset",
    summary = "連合学習セッション情報の初期化",
    responses = {
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "ゲートウェイが未初期化"
        },
        status.HTTP_400_BAD_REQUEST: {
            "description": "連合学習セッションが実行中"
        }
    }
)
def post_reset(gateway: FroseAiGateway = Depends(get_gateway)):
    # ゲートウェイが未初期化の場合は 503 を返す
    if not gateway:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "ゲートウェイが未初期化"
        )
    
    # 現在学習中の場合はエラー
    if gateway.status in [PhaseServer.TRAINING, PhaseServer.AGGREGATING]:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = "連合学習セッションが実行中"
        )
    
    # 残存プロセスの清掃
    if gateway.client_processes:
        for p in gateway.client_processes:
            if p.is_alive():
                p.terminate()
                p.join()
    
    # リセット実行
    gateway.reset()
    
    return {
        "message": "連合学習セッション情報の初期化に成功しました"
    }

################################### WebSocketエンドポイント ###################################
# サーバ⇔クライアント間のエンドポイント
@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str, op: str = "hello", gateway: FroseAiGateway = Depends(get_gateway)):
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
                get_logger("FroseAi-Gateway").warning(f"不正な操作種別です 操作種別: {op}")
    except WebSocketDisconnect:
        # クライアントとの接続が切れた場合、辞書からクライアント情報を削除
        gateway.disconnect(client_id)

