import time
import pickle

import torch

import asyncio
import websockets

from typing import Dict
from torch import nn
from torch.optim.optimizer import Optimizer
from logging import getLogger
from abc import ABCMeta, abstractmethod
from ..pb.froseai_pb2 import FroseAiPiece, FroseAiParams, FroseAiStatus



class FroseAiOptFrame(Optimizer, metaclass=ABCMeta):
    def __init__(self, params, defaults, client_id: int, job_name: str, host: str, port: str):
        super().__init__(params, defaults)
        self._client_id = client_id
        self._round = 0
        self._job_name = job_name
        self._server_url = f"{host}:{port}"
        
        self._logger = getLogger("FroseAiOptimizer")
        
        # WebSocket用のURL"ws://サーバURL/ws/クライアントID"を生成
        # サーバURLにもともとws://が付与されている場合はws://の付与処理はスキップする
        url_base = self._server_url if self._server_url.startswith("ws://") else f"ws://{self._server_url}"
        self._ws_url = f"{url_base}/ws/{self._client_id}"

    @property
    def job_name(self) -> str:
        return self._job_name

    @property
    def server_url(self) -> str:
        return self._server_url

    @property
    def client_id(self):
        return self._client_id

    @property
    def round(self) -> int:
        return self._round

    # Optimizer側のhelloやupdateは同期関数だが、
    # WebSocketは非同期関数でないと動作しないため、
    # ブリッジ用の関数を作成する
    def _send_and_recv(self, req: FroseAiParams, op: str) -> FroseAiParams:
        # サーバとのデータ送受信
        async def _communicate():
            # AIモデル重みをやり取りできるよう、最大通信サイズを1GBに拡張
            ws_max_size = 1000 * 1024 * 1024
            # URLにクエリパラメータとして操作種別を付与
            target_url = f"{self._ws_url}?op={op}"
            # 非同期で接続を確立
            # この接続はWith句が終わると閉じる
            # タイムアウトはいったんなしにする
            async with websockets.connect(
                target_url,
                max_size = ws_max_size,
                open_timeout = None
            ) as ws:
                # データをバイナリ化してサーバへ送信
                await ws.send(req.SerializeToString())
                # サーバからバイナリデータを受信
                res_data = await ws.recv()
                # 受信したバイナリデータを復元して返す
                res = FroseAiParams()
                res.ParseFromString(res_data)
                return res
        # サーバとのデータ送受信を一時的に非同期で立ち上げる
        return asyncio.run(_communicate())

    @torch.no_grad()
    def hello(self, model: nn.modules):
        messages = pickle.dumps({"model": model.cpu().state_dict()})
        
        req = FroseAiParams(src=self._client_id, messages=messages)
        op = "hello" # サーバ側の分岐用の操作種別文字列
        
        # サーバ側へリクエストを送り、レスポンスを受け取る
        rsp = self._send_and_recv(req, op)

        messages = pickle.loads(rsp.messages)
        model.load_state_dict(messages["model"])
        self._round = rsp.round

    @torch.no_grad()
    def update(self, model: nn.modules):
        # サーバへ学習したAIモデル重みをPush
        messages = self.snd_params()
        messages["model"] = model.cpu().state_dict()
        
        push_req = FroseAiParams(
            src=self._client_id,
            messages=pickle.dumps(messages),
            round=self._round
        )
        op = "push" # サーバ側の分岐用の操作種別文字列
        # サーバ側へリクエストを送り、レスポンスは破棄する
        self._send_and_recv(push_req, op)

        # サーバから集約後のAIモデル重みをPull
        ret_code = 204
        while ret_code != 200:
            pull_req = FroseAiParams(src=self._client_id)
            op = "pull" # サーバ側の分岐用の操作種別文字列
            
            # サーバ側へリクエストを送り、レスポンスを受け取る
            rsp = self._send_and_recv(pull_req, op)
            ret_code = rsp.status
            self._round = rsp.round

            if ret_code == 200:
                # サーバ側の処理が完了している場合はAIモデル重みを登録して終了
                messages = pickle.loads(rsp.messages)
                model.load_state_dict(messages["model"])
                self.rcv_params(messages)

            else:
                # サーバ側がまだ処理中の場合は待機
                # 待機時間は0.5秒にする
                time.sleep(0.5)

    @abstractmethod
    def snd_params(self) -> Dict:
        return {}

    @abstractmethod
    def rcv_params(self, others: Dict):
        pass

