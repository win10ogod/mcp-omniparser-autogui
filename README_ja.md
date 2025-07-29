# omniparser-autogui-mcp

これは[OmniParser](https://github.com/microsoft/OmniParser)で画面を解析し、GUIを自動で操作させるための[MCP server](https://modelcontextprotocol.io/introduction)です。  
Windowsで動作確認しております.

## ライセンスについて

これはMIT licenseですが、サブモジュールとパッケージはそれらのライセンスに従います。  
OmniParserのリポジトリ（サブモジュール）はCC-BY-4.0です。  
OmniParserのモデルはそれぞれ異なるライセンスに従います([参照](https://github.com/microsoft/OmniParser?tab=readme-ov-file#model-weights-license)).

## インストール方法

1. 以下を実行してください。

```
git clone --recursive https://github.com/NON906/omniparser-autogui-mcp.git
cd omniparser-autogui-mcp
uv sync
uv run download_models.py
```

(``langchain_example.py``を動作させたい場合は代わりに``uv sync --extra langchain``を実行してください)

2. ``claude_desktop_config.json``に以下を追加してください。

```claude_desktop_config.json
{
  "mcpServers": {
    "omniparser_autogui_mcp": {
      "command": "uv",
      "args": [
        "--directory",
        "D:\\CLONED_PATH\\omniparser-autogui-mcp",
        "run",
        "omniparser-autogui-mcp"
      ],
      "env": {
        "PYTHONIOENCODING": "utf-8"
      }
    }
  }
}
```

(``D:\\CLONED_PATH\\omniparser-autogui-mcp``はクローンしたディレクトリに置き換えてください)

``env``には追加で以下の設定が出来ます。

- ``OMNI_PARSER_BACKEND_LOAD``  
他のクライアント（[LibreChat](https://github.com/danny-avila/LibreChat)など）で動作しない場合、``1``と指定してください

- ``TARGET_WINDOW_NAME``  
操作させるウィンドウを指定したい場合、ウィンドウ名を指定してください  
指定しない場合、画面全体に対して動作します

- ``OMNI_PARSER_SERVER``  
他のデバイスでOmniParserの処理を行う場合、``127.0.0.1:8000``のようにサーバーのアドレスとポートを指定してください  
サーバーは``uv run omniparserserver``で開始できます

- ``SSE_HOST``, ``SSE_PORT``  
指定するとstdioではなくSSEで通信を行うようになります

- ``SOM_MODEL_PATH``, ``CAPTION_MODEL_NAME``, ``CAPTION_MODEL_PATH``, ``OMNI_PARSER_DEVICE``, ``BOX_TRESHOLD``  
OmniParserの設定用です  
通常は不要です

## プロンプト例

- 画面を確認し、ブラウザから「MCPサーバー」と入力して検索してください

など