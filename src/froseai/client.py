from .optimizer import FedAvg
import torch.nn as nn
import logging
from logging import basicConfig, getLogger
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
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status, Response
from fastapi.middleware.cors import CORSMiddleware
from logging import INFO, basicConfig, getLogger
from multiprocessing import Process, set_start_method, get_start_method
from typing import Optional
from .context import *
from .aggregator import FedAvgAggregator
from .pb.froseai_pb2 import FroseAiPiece, FroseAiParams, FroseAiStatus
from torchvision import models
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from contextlib import asynccontextmanager

################################### 通信クラス ###################################
# サーバ側(Aggregator)⇔クライアント側(WebSocket)のゲートウェイクラス
class FroseAiClient:
    # 初期化
    def __init__(self,
        client_id,
        app_conf,
        server_conf,
        model,
        criterion,
        dataset
    ):
        # ロガーを取得
        self._logger = get_logger("FroseAi-Client")
        self._model = model
        self._client_id = client_id
        self._device = app_conf.device
        self._round = app_conf.round
        self._dataset = dataset
        self._criterion = criterion
        self._optimizer = FedAvg(
            parameters = model.parameters(),
            client_id = client_id,
            job_name = app_conf.repo_name,
            host = server_conf.host,
            port = server_conf.port,
            lr = 0.1,
            weight_decay = 0.01,
            train_data_num = dataset["num"]
        )

    def run(self):

        self._optimizer.hello(self._model)

        while self._optimizer.round <= self._round:
            self._logger.info("[Client:%4d]  Round-%d Start!!" % (self._client_id, self._optimizer.round))
            self._model.train().to(self._device)
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(self._dataset["data"]):
                x, labels = x.to(self._device), labels.to(self._device)

                self._optimizer.zero_grad()
                labels = labels.long()
                log_probs = self._model(x)
                loss = self._criterion(log_probs, labels)  # pylint: disable=E1102

                loss.backward()
                batch_loss.append(loss.item())
                self._optimizer.step()

            if len(batch_loss) > 0:
                self._logger.info("[Client:%4d]    Loss: %.8f" % (self._client_id, sum(batch_loss) / len(batch_loss)))

            self._optimizer.update(self._model)

        self._logger.info("[Client:%4d]  Training Finished!!" % (self._client_id,))

################################### クライアントプロセスの起動 ###################################
# クライアントプロセスの実行関数
def _proc_run(
    client_id: int,
    app_conf: FroseArguments,
    server_conf: ServerConfig,
    model,
    criterion,
    dataset
):
    client = FroseAiClient(
        client_id = client_id,
        app_conf = app_conf,
        server_conf = server_conf,
        model = model,
        criterion = criterion,
        dataset = dataset
    )
    client.run()

