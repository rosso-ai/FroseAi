import numpy as np
from typing import Dict
from torch.utils.data import Dataset, DataLoader
from ._datasets import FedDatasetsBase, FedInnerLoopSampler


class FedDatasetsClassification(FedDatasetsBase):
    def __init__(self, clients_num: int, batch_size: int, inner_loop: int, partition_method: str, partition_alpha: float,
                 train_data: Dataset, valid_data: Dataset, class_num: int, min_len=10):
        super().__init__(batch_size, clients_num, train_data, valid_data, class_num)
        self._partition_method = partition_method
        self._partition_alpha = partition_alpha

        indices = self._partition_data(self._train_data, self._train_data_num, min_len)
        self._fed_train_data_loader, self._fed_train_data_num = self._build_datasets(batch_size, inner_loop, self._train_data, indices)

        indices = self._partition_data(self._valid_data, self._valid_data_num, min_len)
        self._fed_valid_data_loader, _ = self._build_datasets(batch_size, inner_loop, self._valid_data, indices)

    def _partition_data(self, dataset: Dataset, n_data: int, min_len: int):
        net_data_idx_map = {}
        np.random.seed(self._random_seed)
        target = np.array(dataset.targets)

        if self._partition_method == "hetero":
            min_size = 0
            while min_size < min_len:
                idx_batch = [[] for _ in range(self._clients_num)]
                # for each class in the dataset
                for k in range(self._class_num):
                    idx_k = np.where(target == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(self._partition_alpha, self._clients_num))
                    # Balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < n_data / self._clients_num)
                            for p, idx_j in zip(proportions, idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                    ]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(self._clients_num):
                np.random.shuffle(idx_batch[j])
                net_data_idx_map[j] = idx_batch[j]
                self._logger.info("partition data hetero alpha= %.1f  CL=%d: datasize= %d / %d" %
                                  (self._partition_alpha, j, len(net_data_idx_map[j]), n_data))

        else:
            # partition_method = homo
            total_num = n_data
            indices = np.random.permutation(total_num)
            batch_indices = np.array_split(indices, self._clients_num)
            net_data_idx_map = {i: batch_indices[i].tolist() for i in range(self._clients_num)}
            for i in range(self._clients_num):
                self._logger.info("partition data homo CL=%d: datasize= %d / %d" %
                                  (i, len(net_data_idx_map[i]), n_data))
        return net_data_idx_map

    def _build_datasets(self, batch_size: int, inner_loop: int, dataset: Dataset, indices: Dict) -> (Dict, Dict):
        dataloader = {}
        datasize = {}

        for client_idx in range(self._clients_num):
            _sampler = FedInnerLoopSampler(batch_size, inner_loop, indices[client_idx])
            dataloader[client_idx] = DataLoader(dataset=dataset, batch_size=self._batch_size, sampler=_sampler)
            datasize[client_idx] = len(indices[client_idx])

        return dataloader, datasize
