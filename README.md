# FroseAi

## RUN
To run the CIFAR10 example, run the following command:
```shell
python run_cifar10.py conf/froseai_conf_cifar10.yml
``` 

## ProtoBuf compile for gRPC
```shell
python -m grpc_tools.protoc -I./froseai/pb --python_out=./froseai/pb --grpc_python_out=./froseai/pb froseai.proto
```
