from dataclasses import dataclass
from omegaconf import OmegaConf


@dataclass
class FroseArguments:
    repo_name: str
    server_url: str = "localhost:9200"
    ws_max_size: int = 1048576000
    random_seed: int = 42
    device: str = "cpu"
    log_output_path: str = "./log"

    # train args
    round: int = 10
    batch_size: int = 100
    inner_loop: int = 100

    # data args
    data_cache_dir: str = "./data"
    partition_method: str = "hetero"
    partition_alpha: float =  10.0

    # poc-mode
    worker_num: int = 1

    @classmethod
    def from_yml(cls, yml_path: str):
        loaded = OmegaConf.load(yml_path)
        read_conf = FroseArguments(**loaded)

        base_conf = OmegaConf.structured(FroseArguments)
        merged = OmegaConf.merge(base_conf, read_conf)
        return cls(**merged)
