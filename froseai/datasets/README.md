# Federated Datasets

This module provides functionality for data partitioning methods used in Federated learning trials.  
The implementation method was referred to in [FedML](https://github.com/FedML-AI/FedML).  

## Function
* FedDatasetsClassification: Partitioning the dataset for classification tasks

## Usage
Class Dataloader of PyTorch is returned.

```python
from omegaconf import OmegaConf
from torchvision import datasets
from torchvision.transforms import ToTensor
from froseai import FedDatasetsClassification

conf = OmegaConf.load("./config.yml")

train_data = datasets.CIFAR10(root=conf.data.data_cache_dir, train=True, download=True, transform=ToTensor())
valid_data = datasets.CIFAR10(root=conf.data.data_cache_dir, train=False, download=True, transform=ToTensor())
fed_datasets = FedDatasetsClassification(conf.common.client_num, conf.train.batch_size, conf.train.inner_loop,
                                         conf.data.partition_method, conf.data.partition_alpha,
                                         train_data, valid_data, 10)

# gotten Class Dataloader for federated
dataloader_for_client1 = fed_datasets.fed_dataset(1)
```
