import logging
import argparse
import torch.nn as nn
from omegaconf import OmegaConf
from torchvision import models
from torchvision import datasets
from torchvision.transforms import ToTensor
from logging import basicConfig, getLogger
from multiprocessing import Process, set_start_method, get_start_method
from froseai import FroseAiServer, FroseAiOptimizer, FedDatasetsClassification

formatter = '%(asctime)s [%(name)s] %(levelname)s :  %(message)s'
basicConfig(level=logging.INFO, format=formatter)
logger = getLogger("Frose-Runner")


def _proc_run(config_path: str, client_id: int, model, dataset, device="cpu"):
    conf = OmegaConf.load(config_path)
    optimizer = FroseAiOptimizer(model.parameters(), client_id, config_path,
                                 lr=conf.train.learning_rate, weight_decay=conf.train.weight_decay,
                                 train_data_num=dataset["num"])
    optimizer.hello(model)

    criterion = nn.CrossEntropyLoss()
    while optimizer.round <= conf.train.round:
        logger.info("[Client:%4d]  Round-%d Start!!" % (client_id, optimizer.round))
        model.train().to(device)
        batch_loss = []
        for batch_idx, (x, labels) in enumerate(dataset["data"]):
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()
            labels = labels.long()

            log_probs = model(x)
            loss = criterion(log_probs, labels)  # pylint: disable=E1102
            loss.backward()
            batch_loss.append(loss.item())

            optimizer.step()

        if len(batch_loss) > 0:
            logger.info("[Client:%4d]    Loss: %.8f" % (client_id, sum(batch_loss) / len(batch_loss)))

        # round update
        optimizer.update(model)

    # close
    logger.info("[Client:%4d]  Training Finished!!" % (client_id,))


def run():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config_path", type=str, help="path of config file")
    args = arg_parser.parse_args()
    conf = OmegaConf.load(args.config_path)

    if conf.data.dataset == "CIFAR10":
        train_data = datasets.CIFAR10(root=conf.data.data_cache_dir, train=True, download=True, transform=ToTensor())
        valid_data = datasets.CIFAR10(root=conf.data.data_cache_dir, train=False, download=True, transform=ToTensor())
    else:
        raise Exception("Frose-Runner does not currently support such dataset")

    fed_datasets = FedDatasetsClassification(conf.common.client_num, conf.train.batch_size, conf.train.inner_loop,
                                             conf.data.partition_method, conf.data.partition_alpha,
                                             train_data, valid_data, 10)

    if conf.model.model == "resnet18":
        model = models.resnet18()
    else:
        raise Exception("Frose-Runner does not currently support such model")

    # Server Start
    server = FroseAiServer(args.config_path, model, test_data=fed_datasets.valid_data_loader, device=conf.common.device)
    server.start()

    if get_start_method() == 'fork':
        set_start_method('spawn', force=True)

    clients = []
    for client_id in range(conf.common.client_num):
        client = Process(target=_proc_run,
                         args=(args.config_path, client_id, model, fed_datasets.fed_dataset(client_id), conf.common.device,))

        # client start
        client.start()
        clients.append(client)

    # wait for stop clients
    for client in clients:
        client.join()

    server.stop()

