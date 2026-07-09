import pickle
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from queue import Queue
from logging import getLogger
from typing import Dict
from abc import ABCMeta, abstractmethod
from threading import Thread
from ..context import FroseArguments
from ..validator import FedValidator

class FroseAiAggFrame(metaclass=ABCMeta):
    def __init__(self, conf: FroseArguments, model, test_data=None, device="cpu"):
        self._conf = conf
        self._device = device
        self._round = 0
        self._model = model
        self._rsp_messages = {"model": None}

        self._validator = None
        if test_data is not None:
            self._validator = FedValidator(conf, test_data)

        self._flag_client_uploaded_round = []
        self._aggregator = None
        self._received = []
        self._snd_q = []
        for idx in range(self.client_num):
            self._flag_client_uploaded_round.append(self._round)
            self._received.append({})
            self._snd_q.append(Queue())

        self._logger = getLogger("FroseAi-ServerAgg")
        self._logger.info("Initialize!!")

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
    def metrics(self):
        # test_data=None(サーバ側評価なし)のとき validator が無いので空を返す
        if self._validator is None:
            return "{}"
        return self._validator.metrics

    @abstractmethod
    def aggregate(self):
        pass

    def push(self, client_id: int, message: Dict, round_cnt: int):
        def _aggregate():
            self.aggregate()

            if self._validator is not None:
                self._validator.test(self.model, self.round, self.device)
                self._validator.write_log(self.round)

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
            # 届いたアダプター(LoRA)のキーだけを回して平均する
            keys = self._received[0]["model"].keys()
            average_params = {}
            for i in range(self.client_num):
                sample_rate = 1 / self.client_num
                if sample_num != 0:
                    sample_rate = self._received[i]["sample_num"] / sample_num

                for k in keys:
                    contrib = self._received[i]["model"][k] * sample_rate
                    if i == 0:
                        average_params[k] = contrib
                    else:
                        average_params[k] += contrib

            # 平均アダプターをサーバの器(グローバルモデル)にはめてから配る
            set_peft_model_state_dict(self.model, average_params)
            self.messages["model"] = {
                k: v.cpu() for k, v in get_peft_model_state_dict(self.model).items()
            }
