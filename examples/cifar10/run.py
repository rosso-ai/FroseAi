import sys
import os
import logging
import argparse
import torch.nn as nn
from torchvision import models
from torchvision import datasets
from torchvision.transforms import ToTensor
from logging import basicConfig, getLogger
from multiprocessing import Process, set_start_method, get_start_method

formatter = '%(asctime)s [%(name)s] %(levelname)s :  %(message)s'
basicConfig(level=logging.INFO, format=formatter)
logger = getLogger("Frose-Runner")

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from froseai import FroseAiServer, FedDatasetsClassification, FroseArguments, FroseAiOptimizer


def _proc_run(conf: FroseArguments, client_id: int, model, dataset, device="cpu"):
    optimizer = FroseAiOptimizer(model.parameters(), client_id, conf.repo_name, conf.server_url,
                                 lr=0.1, weight_decay=0.01,
                                 train_data_num=dataset["num"])
    optimizer.hello(model)

    criterion = nn.CrossEntropyLoss()
    while optimizer.round <= conf.round:
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


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config_path", type=str, help="path of config file")
    args = arg_parser.parse_args()
    conf = FroseArguments.from_yml(args.config_path)

    train_data = datasets.CIFAR10(root=conf.data_cache_dir, train=True, download=True, transform=ToTensor())
    valid_data = datasets.CIFAR10(root=conf.data_cache_dir, train=False, download=True, transform=ToTensor())

    fed_datasets = FedDatasetsClassification(conf.worker_num, conf.batch_size, conf.inner_loop,
                                             conf.partition_method, conf.partition_alpha,
                                             train_data, valid_data, 10)
    model = models.resnet18()

    # Server Start
    server = FroseAiServer(conf, model, test_data=fed_datasets.valid_data_loader, device=conf.device)
    server.start()

    if get_start_method() == 'fork':
        set_start_method('spawn', force=True)

    clients = []
    for client_id in range(conf.worker_num):
        client = Process(target=_proc_run,
                         args=(conf, client_id, model, fed_datasets.fed_dataset(client_id), conf.device,))

        # client start
        client.start()
        clients.append(client)

    # wait for stop clients
    for client in clients:
        client.join()

    server.stop()


if __name__ == "__main__":
    main()

