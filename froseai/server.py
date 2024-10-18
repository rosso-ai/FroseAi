import grpc
import pickle
from logging import INFO, basicConfig, getLogger
from concurrent import futures
from omegaconf import OmegaConf
from .flow import FroseAiAggregator
from .pb.froseai_pb2 import FroseAiPiece, FroseAiParams, FroseAiStatus
from .pb.froseai_pb2_grpc import FroseAiServicer, add_FroseAiServicer_to_server

formatter = '%(asctime)s [%(name)s] %(levelname)s :  %(message)s'
basicConfig(level=INFO, format=formatter)


class FroseAiGrpcGateway(FroseAiServicer):
    def __init__(self, config_pass: str, model, test_data=None, device="cpu"):
        self._agg = FroseAiAggregator(config_pass, model, test_data=test_data, device=device)
        self._logger = getLogger("FroseAi-Gateway")
        self._logger.info("Initialize!!")

    @property
    def model(self):
        return self._agg.model

    @property
    def metrics(self):
        return self._agg.metrics

    def Hello(self, request, context):
        self._agg.round = 1
        ret_model_state = self.model.cpu().state_dict()
        if ret_model_state is None:
            messages = request.messages
        else:
            messages = pickle.dumps({"model": ret_model_state})
        return FroseAiParams(src=request.src, messages=messages, metrics=self.metrics, round=self._agg.round)

    def Push(self, request, context):
        self._agg.push(request.src, pickle.loads(request.messages), request.round)
        return FroseAiPiece(src=request.src, status=202)

    def Pull(self, request, context):
        status = 204
        messages = None
        if not self._agg.snd_q[request.src].empty():
            status = 200
            messages = self._agg.snd_q[request.src].get()
            self._agg.clear_aggregator()

        return FroseAiParams(src=request.src, status=status, messages=messages, round=self._agg.round, metrics=self.metrics)

    def Status(self, request, context):
        return FroseAiStatus(src=request.src, status=200, metrics=self.metrics)


class FroseAiServer:
    def __init__(self, config_pass: str, model, test_data=None, device="cpu", max_workers=4):
        self._conf = OmegaConf.load(config_pass)
        self._logger = getLogger("FroseAi-Srv")

        grpc_opts = [
            ("grpc.max_send_message_length", 1000 * 1024 * 1024),
            ("grpc.max_receive_message_length", 1000 * 1024 * 1024),
        ]
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=grpc_opts, )
        self._servicer = FroseAiGrpcGateway(config_pass, model, test_data=test_data, device=device)

    def start(self):
        add_FroseAiServicer_to_server(self._servicer, self._server)
        port_num = int(self._conf.common.server_url.split(":")[1])
        port_str = '[::]:' + str(port_num)
        self._server.add_insecure_port(port_str)

        self._server.start()
        #self._server.wait_for_termination()
        self._logger.info("gPRC Server START : %s" % (port_str,))

    def stop(self):
        self._server.stop(grace=1)

