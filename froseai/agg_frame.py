import os
import json
import pickle
import csv
from queue import Queue
from omegaconf import OmegaConf
from logging import getLogger
from typing import Dict, Optional
from abc import ABCMeta, abstractmethod
from datetime import datetime
from threading import Thread


class FroseAiAggFrame(metaclass=ABCMeta):
    def __init__(self, config_pass: str, model, test_data=None, device="cpu"):
        self._conf = OmegaConf.load(config_pass)
        self._device = device
        self._round = 0
        self._model = model
        self._test_data = test_data
        self._rsp_messages = {"model": None}
        self._metrics = {}

        self._flag_client_uploaded_round = []
        self._aggregator = None
        self._received = []
        self._snd_q = []
        for idx in range(self.client_num):
            self._flag_client_uploaded_round.append(self._round)
            self._received.append({})
            self._snd_q.append(Queue())

        dt_now = datetime.now()
        job_name = self._conf.common.job_name + "_" + dt_now.strftime('%Y%m%d%H%M%S')
        log_output_path = os.path.join(self._conf.common.log_output_path, job_name)
        os.makedirs(log_output_path, exist_ok=True)
        OmegaConf.save(self._conf, os.path.join(str(log_output_path), "config.yml"))

        file_name = os.path.join(str(log_output_path), "metrics.csv")
        self._metrics_f = open(file_name, "w", encoding="utf-8")
        self._metrics_writer = csv.writer(self._metrics_f)
        self._log_no_header = True

        self._logger = getLogger("FroseAi-ServerAgg")
        self._logger.info("Initialize!!")

    def __del__(self):
        self._metrics_f.close()

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
    def test_data(self):
        return self._test_data

    @property
    def device(self):
        return self._device

    @property
    def client_num(self):
        return self._conf.common.client_num

    @property
    def round_num(self):
        return self._conf.train.round

    @property
    def metrics(self):
        return json.dumps(self._metrics)

    @abstractmethod
    def aggregate(self):
        pass

    @abstractmethod
    def test(self):
        pass

    def push(self, client_id: int, message: Dict, round_cnt: int):
        def _aggregate():
            self.aggregate()

            if self._test_data is not None:
                self._metrics = self.test()
                self._write_log()

            for idx in range(self.client_num):
                self._snd_q[idx].put(pickle.dumps(self.messages))

            self._round += 1

        self._flag_client_uploaded_round[client_id] = round_cnt
        self._received[client_id] = message

        if self.is_all_received:
            if self._aggregator is None:
                self._aggregator = Thread(target=_aggregate)
                self._aggregator.start()

    def _write_log(self):
        metrics_key = ["round"]
        metrics_val = [self._round]
        for k, v in self._metrics.items():
            metrics_key.append(k)
            metrics_val.append(v)

        if self._log_no_header:
            self._metrics_writer.writerow(metrics_key)
            self._log_no_header = False

        self._metrics_writer.writerow(metrics_val)
        self._metrics_f.flush()

    def clear_aggregator(self):
        if self._aggregator is not None:
            self._aggregator.join()
        self._aggregator = None
