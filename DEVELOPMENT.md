# 開發文檔

本文檔為 `mcp-omniparser-autogui` 專案的開發提供指南，旨在幫助開發人員快速了解專案結構、核心技術和開發流程。

## 1. 專案概覽

`mcp-omniparser-autogui` 是一個基於 [MCP (Model-driven Co-pilot Protocol)](https://modelcontextprotocol.io/introduction) 的伺服器，它結合了 [OmniParser](https://github.com/microsoft/OmniParser) 的螢幕分析能力和 `PyAutoGUI` 的自動化操作能力，實現了強大的 GUI 自動化功能。

**核心功能:**
- **AI 螢幕分析**: 使用 OmniParser 將 GUI 螢幕解析為結構化的元素。
- **進階鍵鼠操作**: 提供人性化的滑鼠移動、精確點擊、拖拽和滾輪操作。
- **巨集系統**: 支援錄製、播放和管理複雜的操作序列。
- **遊戲輔助**: 包含像素檢測、快速點擊和連招系統等遊戲專用功能。
- **視窗管理**: 能夠列舉、切換和管理應用程式視窗。
- **高影格率截圖**: 針對遊戲和高效能場景優化的截圖模式。

## 2. 核心技術棧

- **主要框架**: MCP (Model-driven Co-pilot Protocol)
- **Python 版本**: >=3.12
- **套件管理**: `uv`
- **GUI 自動化**: `pyautogui`, `pygetwindow`
- **螢幕解析 (AI)**: `OmniParser` (基於 `torch`, `ultralytics`, `transformers`)
- **OCR**: `easyocr`, `paddleocr`
- **異步處理**: `asyncio`
- **Web 服務 (OmniParser Server)**: `uvicorn`

## 3. 專案結構

```
/
├── src/
│   ├── mcp_autogui/
│   │   ├── __init__.py
│   │   ├── mcp_autogui_main.py  # MCP 伺服器核心邏輯
│   │   └── high_fps_capture.py  # 高影格率截圖模組
│   └── omniparserserver/
│       └── __init__.py          # OmniParser 伺服器 (未使用)
├── OmniParser/                  # Git 子模組，包含 OmniParser 的實現
│   ├── util/
│   │   └── omniparser.py        # OmniParser 核心解析類
│   └── requirements.txt         # OmniParser 的依賴
├── .venv/                       # Python 虛擬環境
├── pyproject.toml               # 專案依賴和配置
├── README.md                    # 專案介紹
├── ADVANCED_USAGE.md            # 進階功能使用指南
└── DEVELOPMENT.md               # 開發文檔 (本文件)
```

## 4. 關鍵模組詳解

### `src/mcp_autogui/mcp_autogui_main.py`

這是專案的核心，實現了 MCP 伺服器的所有工具 (Tool)。

- **初始化 (`mcp_autogui_main`)**:
  - 設置日誌、載入環境變數 (`TARGET_WINDOW_NAME`, `OMNI_PARSER_SERVER` 等)。
  - 初始化 `OmniParser` 模型 (可選擇後台載入)。
  - 初始化巨集系統 (`MacroSystem`) 和 `OmniParser` 快取 (`OmniParserCache`)。
  - 設置 `PyAutoGUI` 的安全機制。

- **核心工具**:
  - `omniparser_details_on_screen()`: 截取螢幕，使用 `OmniParser` 進行分析，並返回帶有標籤的圖片和元素詳情。此工具整合了快取機制以提高效能。
  - `omniparser_click()`, `omniparser_drags()`, `omniparser_write()`: 基於 `OmniParser` 分析結果的元素級別操作。
  - `mouse_click_coordinate()`, `mouse_move_coordinate()`: 新增的基於精確座標的操作，支援人性化模擬。
  - `keyboard_type_text()`, `keyboard_press_keys()`: 增強的鍵盤操作，支援多語言和複雜組合鍵。
  - `macro_*` 系列工具: 實現巨集的錄製、播放和管理。
  - `get_pixel_color()`, `wait_for_pixel_color()`: 遊戲專用的像素級操作。
  - `start_high_fps_capture()`: 啟動高效能截圖模式。

- **錯誤處理與穩定性**:
  - 使用 `@retry_on_error` 裝飾器為關鍵操作增加了重試機制。
  - 使用 `ThreadSafeSingleton` 模式確保 `MacroSystem` 和 `OmniParserCache` 的執行緒安全。
  - 增加了詳細的日誌記錄和資源管理 (`managed_resource`)。

### `OmniParser` (子模組)

- 這是一個獨立的專案，用於將 GUI 螢幕截圖解析為結構化的資料。
- `util/omniparser.py` 中的 `Omniparser` 類是其核心，封裝了模型載入和圖像解析的邏輯。
- 它依賴於 `PyTorch` 和 `YOLO` (`ultralytics`) 模型來檢測介面元素。

## 5. 開發環境設置

1.  **克隆儲存庫**:
    ```bash
    git clone --recursive https://github.com/NON906/omniparser-autogui-mcp.git
    cd omniparser-autogui-mcp
    ```

2.  **安裝依賴**:
    本專案使用 `uv` 進行環境和依賴管理。
    ```bash
    uv sync
    ```
    如果需要 `langchain` 相關功能，請執行：
    ```bash
    uv sync --extra langchain
    ```

3.  **下載 AI 模型**:
    ```bash
    set OCR_LANG=en  # Windows
    # export OCR_LANG=en # Linux/macOS
    uv run download_models.py
    ```

4.  **配置 MCP 客戶端**:
    將此專案作為 MCP 伺服器添加到您的 MCP 客戶端設定中 (例如 `claude_desktop_config.json`)。請參考 `README.md` 中的範例。

## 6. 運行與測試

### 運行主應用

MCP 客戶端會根據設定自動啟動 `mcp-omniparser-autogui` 伺服器。

### 獨立運行 OmniParser 伺服器

如果您希望將 `OmniParser` 的處理卸載到另一台設備上，可以單獨運行其伺服器：
```bash
uv run omniparserserver
```
然後在主應用中設定 `OMNI_PARSER_SERVER` 環境變數。

### 測試

專案中包含一些測試腳本，可以用於驗證特定功能：
- `test_advanced_features.py`
- `test_high_fps.py`
- `test_improvements.py`

執行測試:
```bash
uv run python test_advanced_features.py
```

## 7. 開發指南與注意事項

- **程式碼風格**:
  - 遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/)。
  - 使用 `ruff` 進行程式碼格式化和檢查。

- **新增 MCP 工具**:
  - 在 `src/mcp_autogui/mcp_autogui_main.py` 中，使用 `@mcp.tool()` 裝飾器定義新的非同步函式。
  - 為函式提供清晰的 docstring，因為這會直接影響 AI Agent 的使用效果。
  - 考慮為可能失敗的操作添加 `@retry_on_error` 裝飾器。

- **執行緒安全**:
  - 所有共享狀態 (如 `state` 字典、`MacroSystem` 實例) 都必須以執行緒安全的方式進行存取。
  - 避免使用全域變數，盡量將狀態封裝在類或字典中。

- **子模組管理**:
  - `OmniParser` 是一個 Git 子模組。如果需要更新，請使用 `git submodule update --remote`。

- **依賴管理**:
  - 所有依賴都應在 `pyproject.toml` 中定義。
  - 修改後，執行 `uv sync` 來更新虛擬環境。
