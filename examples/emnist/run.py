# インポート処理
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

import matplotlib.pyplot as plt

formatter = '%(asctime)s [%(name)s] %(levelname)s :  %(message)s'
basicConfig(level=logging.INFO, format=formatter)
logger = getLogger("Frose-Runner")

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from froseai import FroseAiServer, FedDatasetsClassification, FroseArguments, FedAvg



# ==========================================
# FedValidator の不具合を修正するパッチ処理
# ==========================================
from froseai.validator import FedValidator

# 元の test メソッドを退避
_original_test = FedValidator.test

# 正しく self._metrics を更新する新しい test メソッドを定義
def _patched_test(self, model, round_num, device="cpu"):
    res_metrics = _original_test(self, model, round_num, device)
    self._metrics = res_metrics  # 計算結果を self._metrics に保持させる
    return res_metrics

# メソッドを差し替え
FedValidator.test = _patched_test



# 多項ロジスティック回帰モデル、CPU負荷を抑えるため選定
# 本来は多クラス分類には向いていないが、
# EMNISTデータセットは比較的単純であるのと、
# 損失関数側でSoftmax変換を行っているため、
# これでも90%程度の精度は出る
# 基本クラスtorch.nn.Moduleを継承
# 以下のソースより引用
# https://github.com/FedML-AI/FedML/blob/master/python/fedml/model/linear/lr.py
class LogisticRegression(nn.Module):
    # 初期化処理
    # 入力サイズと出力サイズはインスタンス作成時に指定
    def __init__(self, input_dim, output_dim):
        # 親クラスtorch.nn.Moduleの初期化処理を呼び出し
        super(LogisticRegression, self).__init__()
        # Input→Outputの計算を行う全結合層を定義
        self.linear = nn.Linear(input_dim, output_dim)

    # 順伝播処理
    def forward(self, x):
        # 入力データを1次元ベクトルに変換
        # 1つ目の値(バッチサイズ)はそのまま、
        # 2つ目以降の値をすべて掛け合わせて1列にする
        x = x.view(x.size(0), -1)
        # 線形層に通す
        # 今回は多クラス分類のため、シグモイド関数は利用しない
        #outputs = torch.sigmoid(self.linear(x))
        outputs = self.linear(x)
        # 予測結果を返す
        return outputs

# クライアント処理
def _proc_run(conf: FroseArguments, client_id: int, model, dataset, device="cpu"):
    # オプティマイザを初期化
    optimizer = FedAvg(model.parameters(), client_id, conf.repo_name, conf.server_url,
                       lr=0.1, weight_decay=0.01, train_data_num=dataset["num"])

    # クライアントが参加したことをサーバに通知し、初期モデルを取得
    optimizer.hello(model)

    # 交差エントロピー誤差損失関数を初期化
    criterion = nn.CrossEntropyLoss()
    
    # コンフィグに指定されたラウンド数ループ
    while optimizer.round <= conf.round:
        logger.info("[Client:%4d]  Round-%d Start!!" % (client_id, optimizer.round))
        print(f"クライアント:{client_id} ラウンド:{optimizer.round}")
        # モデルを訓練モードにする
        model.train().to(device)
        # 全バッチの損失値の初期化
        batch_loss = []
        # 入力画像xと正解labelsからなるミニバッチを取り出す
        for batch_idx, (x, labels) in enumerate(dataset["data"]):
            x, labels = x.to(device), labels.to(device)
            # モデルのパラメータ勾配を0にリセット
            optimizer.zero_grad()
            # 損失計算でエラーとならないよう、正解ラベルの型を整数値に変換
            labels = labels.long()
            # 画像データをモデルに入力し、各クラスの予測確率を出力
            log_probs = model(x)
            # 予測結果と正解ラベルの差から損失を計算
            loss = criterion(log_probs, labels)  # pylint: disable=E1102
            # 損失値から逆伝播を行い、各パラメータの勾配を計算
            loss.backward()
            # 計算されたバッチの損失値をリストに追加
            batch_loss.append(loss.item())
            # 計算された勾配からモデルの重みを更新
            optimizer.step()

        # 今回のラウンドで損失が発生していた場合、
        # 平均の損失値を表示
        if len(batch_loss) > 0:
            logger.info("[Client:%4d]    Loss: %.8f" % (client_id, sum(batch_loss) / len(batch_loss)))

        # ローカル学習後の重みパラメータをサーバに送信
        # 次のラウンドのグローバル重みを受け取ってモデルを更新
        optimizer.update(model)

    # クライアント処理終了
    logger.info("[Client:%4d]  Training Finished!!" % (client_id,))

# メイン処理
def main():
    print('処理開始')
    # 引数の取得
    # コンフィグのパスのみ指定できる
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config_path", type=str, help="path of config file")
    args = arg_parser.parse_args()
    conf = FroseArguments.from_yml(args.config_path)

    # データセットの取得
    # テンソルに変換し、コンフィグに指定した場所にデータを一時保存
    # 今回は処理が低負荷で済むEmnistのdigits(数字)を使用
    #train_data = datasets.CIFAR10(root=conf.data_cache_dir, train=True, download=True, transform=ToTensor())
    #valid_data = datasets.CIFAR10(root=conf.data_cache_dir, train=False, download=True, transform=ToTensor())
    print('データセット取得中...')
    emnist_split='digits'
    train_data = datasets.EMNIST(root=conf.data_cache_dir, split=emnist_split, train=True, download=True, transform=ToTensor())
    valid_data = datasets.EMNIST(root=conf.data_cache_dir, split=emnist_split, train=False, download=True, transform=ToTensor())
    print('データセット取得完了')
    print('データセット分割中...')
    # データセットの分割
    # コンフィグに指定した条件でデータを分割
    fed_datasets = FedDatasetsClassification(conf.worker_num, conf.batch_size, conf.inner_loop,
                                             conf.partition_method, conf.partition_alpha,
                                             train_data, valid_data, 10)

    print('データセット分割完了')
    print('モデル読み込み中...')
    # モデルの読み込み
    # 今回使用するEMNISTのdigitsの場合、
    # 入力サイズは1チャンネル*28ピクセル*28ピクセル
    # 出力サイズは10(0～9の数字に分類)
    input_dim = 1 * 28 * 28
    output_dim = 10
    model = LogisticRegression(input_dim=input_dim, output_dim=output_dim)
    print('モデル読み込み完了')

    # サーバの起動
    # コンフィグ、モデル、データセット、デバイスを与えて実行
    print('サーバ起動中...')
    server = FroseAiServer(conf, model, test_data=fed_datasets.valid_data_loader, device=conf.device)
    server.start()
    print('サーバ起動完了')

    # マルチプロセス生成方式がfork(コピー、Unixデフォルト)の場合、
    # 子プロセスを生成する際に不正なロックなどが混入することがあるため、
    # spawn(新規作成、Windows/Macデフォルト)にする
    if get_start_method() == 'fork':
        set_start_method('spawn', force=True)

    # クライアントの起動
    # コンフィグに指定した数だけ並列でプロセスを起動
    # プロセスの実行内容は_proc_runで定義
    print('クライアント起動中...')
    clients = []
    for client_id in range(conf.worker_num):
        client = Process(target=_proc_run,
                         args=(conf, client_id, model, fed_datasets.fed_dataset(client_id), conf.device,))
        client.start()
        clients.append(client)
        print(client_id)

    print('クライアント起動完了')
    print('処理中...')
    # クライアントの停止待ち
    for client in clients:
        client.join()



    # ---------------------------------------------------------
    # グラフ作成処理の追加箇所
    # ---------------------------------------------------------
    print('グラフ作成中...')

    # FedValidator が出力した CSV ファイルのパスを取得
    # （FedValidator 内で指定されているログ出力フォルダーを探します）
    validator = server._servicer._agg._validator
    csv_file_path = None

    if validator and hasattr(validator, '_metrics_f'):
        csv_file_path = validator._metrics_f.name

    rounds = []
    losses = []

    # CSV ファイルが存在し、書き込まれている場合は読み込む
    if csv_file_path and os.path.exists(csv_file_path):
        # ファイル更新を確実にするため一旦 flush しておく
        validator._metrics_f.flush()
        
        import csv
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'round' in row and 'loss' in row:
                    rounds.append(int(row['round']))
                    losses.append(float(row['loss']))

    # グラフの描画
    if losses:
        plt.figure(figsize=(8, 5))
        plt.plot(rounds, losses, marker='o', color='blue', linestyle='-', linewidth=2, label='Server Loss')
        
        plt.title('Server Aggregated Loss per Round', fontsize=14)
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=11)
        
        # 横軸（ラウンド数）の目盛りを整数にする
        plt.xticks(rounds)

        # 画像として保存
        save_path = 'server_loss_graph.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'グラフを {save_path} として保存しました。')
        
        # 画面に表示
        #plt.show()
    else:
        print('警告: LossのデータがCSVから読み込めませんでした。')




    # サーバの停止
    server.stop()

    print('処理完了')

if __name__ == "__main__":
    main()
