# ConLAi

## RUN
それぞれの動作方法は apps 下をご確認ください   

## ProtoBuf compile for gRPC
```shell
python -m grpc_tools.protoc -I./froseai/pb --python_out=./froseai/pb --grpc_python_out=./froseai/pb froseai.proto
```
