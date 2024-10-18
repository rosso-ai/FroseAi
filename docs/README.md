# Documents Guide


## Build Guide
### ProtoBuf compile for gRPC
```shell
python -m grpc_tools.protoc -I./froseai/pb --python_out=./froseai/pb --grpc_python_out=./froseai/pb froseai.proto
```
