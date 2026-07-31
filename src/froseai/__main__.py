# Webサーバ起動用
import uvicorn
# 引数取得用
from pydantic_settings import CliApp
# 引数の定義
from .context import ServerConfig
# サーバ起動関数
from .server import create_app

# アプリケーションとWebサーバを分離するため、uvicorn.runで起動に変更
if __name__ == "__main__":
    # 引数の取得
    config = CliApp.run(ServerConfig)
    # サーバインスタンスの生成
    app = create_app(config)
    # 生成したサーバインスタンスを渡して起動
    uvicorn.run(app, host=config.host, port=config.port, ws_max_size=config.ws_max_size)

