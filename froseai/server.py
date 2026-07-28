import pickle

import queue
import time
import threading
import contextlib

import json

from logging import INFO, basicConfig, getLogger

from typing import Optional

from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from enum import StrEnum

from .aggregator import FedAvgAggregator
from .context import FroseArguments
from .pb.froseai_pb2 import FroseAiPiece, FroseAiParams, FroseAiStatus

formatter = '%(asctime)s [%(name)s] %(levelname)s :  %(message)s'
basicConfig(level=INFO, format=formatter)

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
        # 現在のステータス
        self._status : PhaseStatus = PhaseStatus.READY
        self._logger.info("FroseAi-Gatewayを初期化しました")

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
        self._agg.push(req.src, pickle.loads(req.messages), req.round)
        # 計算中のモデル数を1減らす
        self._uncomplete_clients = self._uncomplete_clients - 1
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
@app.get("/api/v1/clients")
def get_client_list():
    return {
        "test_message": "GET CLIENT LIST"
    }

# GET /api/v1/clients/{client_id} で個別のクライアント状態を返却
# いったんエンドポイントだけ作成
@app.get("/api/v1/clients/{client_id}")
def get_client_status(client_id: str):
    return {
        "test_message": "GET CLIENT STATUS",
        "client_id": client_id
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
@app.get("/api/v1/model/latest")
def get_model_latest():
    print(gateway.model)
    print(type(gateway.model))
    return {
        "model": gateway.model
    }

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
    print(last_metrics_json)
    print(type(last_metrics_json))
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
    
    return {
        Response(
            content = content,
            media_type = "text/plain"
        )
    }

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

# POST /api/v1/session/start で連合学習を開始
# いったんエンドポイントだけ作成
@app.post("/api/v1/session/start")
def post_start():
    return {
        "test_message": "POST START"
    }

# POST /api/v1/session/stop で連合学習を停止
# いったんエンドポイントだけ作成
@app.post("/api/v1/session/stop")
def post_stop():
    return {
        "test_message": "POST STOP"
    }

# POST /api/v1/session/reset でラウンド数やキューなどを初期化
# いったんエンドポイントだけ作成
@app.post("/api/v1/session/reset")
def post_reset():
    return {
        "test_message": "POST RESET"
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
                self._logger.warning(f"不正な操作種別です 操作種別: {op}")
    except WebSocketDisconnect:
        # クライアントとの接続が切れた場合、辞書からクライアント情報を削除
        gateway.disconnect(client_id)

# サーバ管理クラス
class FroseAiServer(uvicorn.Server):
    # 初期化
    def __init__(self, conf: FroseArguments, model, test_data=None, device="cpu", **kwargs):
        # ゲートウェイ用のグローバル変数を定義
        global gateway
        # コンフィグを取得
        self._conf = conf
        # ロガーを取得
        self._logger = getLogger("FroseAi-Server")
        # 引数を取得
        self._agg = FedAvgAggregator(conf, model, test_data=test_data, device=device)
        # ゲートウェイの初期化
        gateway = FroseAiGateway(self._agg)

        # コンフィグの内容からポート番号を抽出
        port_num = int(self._conf.server_url.split(":")[1])
        # AIモデル重みをやり取りできるよう、最大通信サイズを1GBに拡張
        ws_max_size = 1000 * 1024 * 1024
        # 各種設定をサーバ用のコンフィグオブジェクトに格納
        config = uvicorn.Config(
            app, 
            host="0.0.0.0", 
            port=port_num, 
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
