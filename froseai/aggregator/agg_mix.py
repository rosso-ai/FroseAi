import pickle
import torch
import copy
from queue import Queue
from logging import getLogger
from typing import Dict, Optional
from abc import ABCMeta, abstractmethod
from threading import Thread
from ..context import FroseArguments
from ..datasets import FedDatasetsClassification
from ..validator import FedValidator


class FroseAiAggFrame(metaclass=ABCMeta):
    def __init__(
        self,
        host = "localhost",
        port = 8000,
        ws_max_size = 1000 * 1024 * 1024,
        log_dir = "./log",
        data_dir = "./data",
        train_data=None,
        valid_data=None,
        device="cpu",
        **kwargs
    ):
        self._conf : FroseArguments | None = None
        self._device = device
        self._round = 0
        self._model = None
        self._rsp_messages = {"model": None}
        self._train_data = train_data
        self._valid_data = valid_data
        self._fed_data = None
        self._flag_client_uploaded_round = []
        self._aggregator = None
        self._received = []
        self._snd_q = {}
        self._log_dir = log_dir
        self._host = host
        self._port = port

        self._logger = getLogger("FroseAi-ServerAgg")
        self._logger.info("Initialize!!")

    def start(self):
        # コンフィグを後から読み込ませる形にしたため、
        # 学習関連の初期化を一部移動
        
        # データセットの分割
        fed_datasets = FedDatasetsClassification(
            self._conf.worker_num,
            self._conf.batch_size,
            self._conf.inner_loop,
            self._conf.partition_method,
            self._conf.partition_alpha,
            self._train_data,
            self._valid_data,
            10
        )
        self._fed_data = fed_datasets
        # いずれは外に出すが一旦Aggregator内で定義する
        self._validator = FedValidator(self._conf, fed_datasets.valid_data_loader, self._log_dir)
        # 途中のクライアント増減など将来的な拡張性を意識して、
        # キュー管理をリストから辞書型に変更
        self._received = []
        self._snd_q = {}
        for idx in range(self.client_num):
            self._flag_client_uploaded_round.append(self._round)
            self._received.append({})
            self._snd_q[idx] = Queue()

    def reset(self):
        # アグリゲータの学習状態を初期化
        self._round = 0
        self._rsp_messages = {"model": None}
        self._received = []
        self._snd_q = {}
        self._validator.reset()

    @property
    def model(self):
        return self._model

    @property
    def messages(self):
        return self._rsp_messages

    @messages.setter
    def messages(self, val):
        self._rsp_messages = val

    @property
    def snd_q(self):
        return self._snd_q

    @property
    def is_all_received(self):
        for idx in range(self.client_num):
            if self._flag_client_uploaded_round[idx] < self._round:
                return False
        return True

    @property
    def round(self):
        return self._round

    @round.setter
    def round(self, val):
        self._round = val

    @property
    def device(self):
        return self._device

    @property
    def client_num(self):
        return self._conf.worker_num

    @property
    def round_num(self):
        return self._conf.round

    @property
    def validator(self) -> FedValidator:
        return self._validator

    @property
    def metrics(self) -> list[dict]:
        return self._validator.metrics

    @property
    def last_metrics(self) -> Optional[str]:
        return self._validator.last_metrics

    @abstractmethod
    def aggregate(self):
        pass

    def push(self, client_id: int, message: Dict, round_cnt: int):
        def _aggregate():
            self.aggregate()
            self._validator.test(self.model, self.round, self.device)

            for idx in range(self.client_num):
                self._snd_q[idx].put(pickle.dumps(self.messages))

            self._round += 1

        self._flag_client_uploaded_round[client_id] = round_cnt
        self._received[client_id] = message

        if self.is_all_received:
            if self._aggregator is None:
                self._aggregator = Thread(target=_aggregate)
                self._aggregator.start()

    def clear_aggregator(self):
        if self._aggregator is not None:
            self._aggregator.join()
        self._aggregator = None


class FedAvgAggregator(FroseAiAggFrame):
    def aggregate(self):
        sample_num = 0
        for i in range(self.client_num):
            sample_num += self._received[i]["sample_num"]

        with torch.no_grad():
            average_params = self.model.cpu().state_dict()
            for i in range(self.client_num):

                sample_rate = 1
                if sample_num != 0:
                    sample_rate = self._received[i]["sample_num"] / sample_num

                for k in average_params.keys():
                    if i == 0:
                        average_params[k] = self._received[i]["model"][k] * sample_rate
                    else:
                        average_params[k] += self._received[i]["model"][k] * sample_rate

            self.model.load_state_dict(average_params)
            self.messages["model"] = copy.deepcopy(self.model).cpu().state_dict()
