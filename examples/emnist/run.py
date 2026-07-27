import sys
import os
import logging
import argparse

import threading
import time

import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from logging import basicConfig, getLogger
from multiprocessing import Process, set_start_method, get_start_method

import uvicorn

#import matplotlib.pyplot as plt

formatter = '%(asctime)s [%(name)s] %(levelname)s :  %(message)s'
basicConfig(level=logging.INFO, format=formatter)
logger = getLogger("Frose-Runner")

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from froseai import FroseAiServer, FedDatasetsClassification, FroseArguments, FedAvg, FedValidator, create_front_if


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        outputs = self.linear(x)
        return outputs


def _proc_run(conf: FroseArguments, client_id: int, model, dataset, device="cpu"):
    optimizer = FedAvg(model.parameters(), client_id, conf.repo_name, conf.server_url,
                       lr=0.1, weight_decay=0.01, train_data_num=dataset["num"])

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

        optimizer.update(model)

    logger.info("[Client:%4d]  Training Finished!!" % (client_id,))


#def plot_metrics(validator: FedValidator):
#    rounds = []
#    losses = []
#    for i, m in enumerate(validator.metrics):
#        rounds.append(i + 1)
#        losses.append(m["loss"])
#
#    # グラフの描画
#    if losses:
#        plt.figure(figsize=(8, 5))
#        plt.plot(rounds, losses, marker='o', color='blue', linestyle='-', linewidth=2, label='Server Loss')
#
#        plt.title('Server Aggregated Loss per Round', fontsize=14)
#        plt.xlabel('Round', fontsize=12)
#        plt.ylabel('Loss', fontsize=12)
#        plt.grid(True, linestyle='--', alpha=0.7)
#        plt.legend(fontsize=11)
#
#        # 横軸（ラウンド数）の目盛りを整数にする
#        plt.xticks(rounds)
#
#        # 画像として保存
#        save_path = os.path.join(validator.log_path, 'server_loss_graph.png')
#        plt.savefig(save_path, dpi=300, bbox_inches='tight')
#        print(f'グラフを {save_path} として保存しました。')
#
#        # 画面に表示
#        # plt.show()
#    else:
#        print('警告: LossのデータがCSVから読み込めませんでした。')

# FastAPIを別スレッドで起動
def run_fastapi(app: uvicorn.Server):
    app.run()

# メイン処理
def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config_path", type=str, help="path of config file")
    
    arg_parser.add_argument("--rest-port", type=int, default=8000, help="Port for REST API")
    
    args = arg_parser.parse_args()
    conf = FroseArguments.from_yml(args.config_path)

    # データセットの取得
    emnist_split='digits'
    train_data = datasets.EMNIST(root=conf.data_cache_dir, split=emnist_split, train=True, download=True, transform=ToTensor())
    valid_data = datasets.EMNIST(root=conf.data_cache_dir, split=emnist_split, train=False, download=True, transform=ToTensor())

    # データセットの分割
    fed_datasets = FedDatasetsClassification(conf.worker_num, conf.batch_size, conf.inner_loop,
                                             conf.partition_method, conf.partition_alpha,
                                             train_data, valid_data, 10)

    input_dim = 1 * 28 * 28
    output_dim = 10
    model = LogisticRegression(input_dim=input_dim, output_dim=output_dim)

    # サーバの起動
    server = FroseAiServer(conf, model, test_data=fed_datasets.valid_data_loader, device=conf.device)
    # フロント⇒サーバのインタフェースを生成
    app = create_front_if(server)
    # フロント⇒サーバのインタフェースを受け付けるWebサーバを起動
    # サーバも自動的に起動される
    config = uvicorn.Config(app, host="0.0.0.0", port=args.rest_port, log_level="info")
    uvicorn_server = uvicorn.Server(config)
    server_thread = threading.Thread(target=run_fastapi, args=(uvicorn_server,), daemon=True)
    server_thread.start()
    # 起動を少し待機
    time.sleep(2)

    if get_start_method() == 'fork':
        set_start_method('spawn', force=True)

    clients = []
    for client_id in range(conf.worker_num):
        # クライアント起動
        client = Process(target=_proc_run,
                         args=(conf, client_id, model, fed_datasets.fed_dataset(client_id), conf.device,))
        client.start()
        clients.append(client)

    # クライアントの停止待ち
    for client in clients:
        client.join()

    # Webサーバに終了フラグを付ける
    # サーバも自動的に終了する
    uvicorn_server.should_exit = True
    # 完全に終了するまで待機
    server_thread.join()
#    plot_metrics(server.aggregator.validator)


if __name__ == "__main__":
    main()
