from typing import List
from logging import getLogger
from abc import ABCMeta
from torch.utils.data import Dataset, DataLoader, sampler


class FedDatasetsBase(metaclass=ABCMeta):
    def __init__(self, batch_size, clients_num, train_data: Dataset, valid_data: Dataset, class_num=0, random_seed=0):
        self._train_data = train_data
        self._valid_data = valid_data
        self._class_num = class_num
        self._batch_size = batch_size
        self._clients_num = clients_num
        self._random_seed = random_seed

        self._train_data_loader = DataLoader(dataset=self._train_data, batch_size=self._batch_size,
                                             shuffle=True, drop_last=True)
        self._valid_data_loader = DataLoader(dataset=self._valid_data, batch_size=self._batch_size,
                                             shuffle=False, drop_last=True)
        self._train_data_num = len(self._train_data)
        self._valid_data_num = len(self._valid_data)

        self._fed_train_data_num = {}
        self._fed_train_data_loader = {}
        self._fed_valid_data_loader = {}

        self._logger = getLogger("FedDatasets")

    def fed_dataset(self, client_id: int):
        return {"data": self._fed_train_data_loader[client_id], "num": self._fed_train_data_num[client_id]}

    @property
    def train_data_loader(self) -> DataLoader:
        return self._train_data_loader

    @property
    def valid_data_loader(self) -> DataLoader:
        return self._valid_data_loader

    @property
    def class_num(self):
        return self._class_num


class FedInnerLoopSampler(sampler.Sampler[int]):
    def __init__(self, batch_size: int, inner_loop: int, indices: List):
        super().__init__()
        self._n_data = len(indices)
        self._batch_size = batch_size
        self._inner_loop = inner_loop

        self._indices = []
        self._data_indices = indices
        self._n_data_batch = self._batch_size * self._inner_loop if self._inner_loop is not None else self._n_data
        self._n_offset = 0

    def __len__(self):
        return self._n_data_batch

    def __iter__(self):
        # Removal of used data
        self._indices = self._indices[self._n_offset:]
        self._n_offset = 0  # reset start index

        # Prepare at least 1 epoch's worth of data in the index list that stores the data call order.
        while len(self._indices) <= self._n_data_batch:
            self._indices += self._data_indices

        sidx, eidx = self._n_offset, self._n_offset + self._n_data_batch
        indices = self._indices[sidx:eidx].copy()
        self._n_offset = eidx
        yield from indices
