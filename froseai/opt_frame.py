import time
import pickle
import grpc
import torch
from typing import Dict
from torch import nn
from torch.optim.optimizer import Optimizer
from omegaconf import OmegaConf
from logging import getLogger
from abc import ABCMeta, abstractmethod
from .pb.froseai_pb2 import FroseAiPiece, FroseAiParams, FroseAiStatus
from .pb.froseai_pb2_grpc import FroseAiStub


class FroseAiOptFrame(Optimizer, metaclass=ABCMeta):
    def __init__(self, params, defaults, client_id: int, config_pass: str):
        super().__init__(params, defaults)
        self._client_id = client_id
        self._round = 0
        self._conf = OmegaConf.load(config_pass)
        self._grpc_opts = [
            ("grpc.max_send_message_length", 1000 * 1024 * 1024),
            ("grpc.max_receive_message_length", 1000 * 1024 * 1024),
        ]
        self._logger = getLogger("FroseAiOptimizer")

    @property
    def job_name(self) -> str:
        return self._conf.common.job_name

    @property
    def server_url(self) -> str:
        return self._conf.common.server_url

    @property
    def client_id(self):
        return self._client_id

    @property
    def round(self) -> int:
        return self._round

    @torch.no_grad()
    def hello(self, model: nn.modules):
        with grpc.insecure_channel(self.server_url, options=self._grpc_opts) as channel:
            stub = FroseAiStub(channel)
            messages = pickle.dumps({"model": model.cpu().state_dict()})
            rsp = stub.Hello(FroseAiParams(src=self._client_id, messages=messages))

            messages = pickle.loads(rsp.messages)
            model.load_state_dict(messages["model"])
            self._round = rsp.round

    @torch.no_grad()
    def update(self, model: nn.modules):
        with grpc.insecure_channel(self.server_url, options=self._grpc_opts) as channel:
            stub = FroseAiStub(channel)
            messages = self.snd_params()
            messages["model"] = model.cpu().state_dict()
            stub.Push(FroseAiParams(src=self.client_id, messages=pickle.dumps(messages), round=self._round))

            ret_code = 204
            while ret_code != 200:
                rsp = stub.Pull(FroseAiPiece(src=self.client_id))
                ret_code = rsp.status
                self._round = rsp.round

                if ret_code == 200:
                    messages = pickle.loads(rsp.messages)
                    model.load_state_dict(messages["model"])
                    self.rcv_params(messages)

                else:
                    time.sleep(0.01)

    @abstractmethod
    def snd_params(self) -> Dict:
        return {}

    @abstractmethod
    def rcv_params(self, others: Dict):
        pass

