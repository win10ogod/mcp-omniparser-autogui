# 高級鍵鼠操作與巨集系統使用指南

## 概述

本 MCP 伺服器提供了完整的鍵鼠操作和巨集系統，支援人性化輸入模擬、遊戲自動化和複雜的操作序列。

## 🎯 核心功能

### 1. 精確座標操作

#### 滑鼠點擊
```python
# 基本點擊
await mouse_click_coordinate(x=100, y=200, button='left', clicks=1, human_like=True)

# 雙擊
await mouse_click_coordinate(x=100, y=200, clicks=2)

# 右鍵點擊
await mouse_click_coordinate(x=100, y=200, button='right')
```

#### 滑鼠移動
```python
# 人性化移動
await mouse_move_coordinate(x=500, y=300, human_like=True, duration=0.5)

# 快速移動
await mouse_move_coordinate(x=500, y=300, human_like=False, duration=0.1)
```

#### 拖拽操作
```python
# 拖拽文件
await mouse_drag_coordinate(
    from_x=100, from_y=100, 
    to_x=300, to_y=300, 
    button='left', 
    human_like=True, 
    duration=1.0
)
```

### 2. 增強鍵盤操作

#### 文字輸入
```python
# 人性化輸入
await keyboard_type_text("Hello World!", human_like=True)

# 使用剪貼簿輸入中文
await keyboard_type_text("你好世界！", use_clipboard=True)
```

#### 組合鍵操作
```python
# 複製
await keyboard_hotkey(['ctrl', 'c'])

# 複雜組合鍵
await keyboard_press_keys(['ctrl', 'shift', 's'], hold_duration=0.2)
```

#### 高級滾輪操作
```python
# 在指定位置滾動
await scroll_advanced(direction='up', clicks=5, x=400, y=300)

# 水平滾動
await scroll_advanced(direction='left', clicks=3)
```

### 3. 巨集系統

#### 錄製巨集
```python
# 開始錄製
await macro_start_recording("my_macro")

# 執行一些操作...
await mouse_click_coordinate(100, 100)
await keyboard_type_text("test")

# 停止錄製
await macro_stop_recording()
```

#### 播放巨集
```python
# 播放一次
await macro_play("my_macro")

# 重複播放
await macro_play("my_macro", repeat_count=5, delay_between_repeats=2.0)
```

#### 巨集管理
```python
# 列出所有巨集
macros = await macro_list()

# 獲取巨集信息
info = await macro_get_info("my_macro")

# 刪除巨集
await macro_delete("my_macro")
```

### 4. 遊戲專用功能

#### 像素檢測
```python
# 獲取像素顏色
color = await get_pixel_color(x=100, y=100)
print(f"RGB: {color['rgb']}, HEX: {color['hex']}")

# 等待特定顏色出現
found = await wait_for_pixel_color(
    x=100, y=100, 
    target_color="#FF0000", 
    timeout=10.0
)
```

#### 快速點擊
```python
# 遊戲中的快速點擊
await rapid_click(x=200, y=200, clicks=10, interval=0.05)
```

#### 連招系統
```python
# 定義連招序列
combo_actions = [
    {"type": "key", "keys": ["q"]},
    {"type": "wait", "duration": 0.1},
    {"type": "key", "keys": ["w"]},
    {"type": "wait", "duration": 0.1},
    {"type": "click", "x": 300, "y": 300, "button": "left"},
    {"type": "key", "keys": ["e"]}
]

# 執行連招
await combo_sequence(combo_actions, human_like=True)
```

### 5. 視窗管理

#### 視窗操作
```python
# 列出所有視窗
windows = await list_windows()

# 切換到指定視窗
await switch_to_window("記事本")

# 獲取螢幕尺寸
size = await get_screen_size()
```

#### 區域截圖
```python
# 截取指定區域
screenshot = await take_screenshot_region(x=0, y=0, width=800, height=600)
```

## 🤖 人性化模擬配置

### 時間配置
- **滑鼠移動**: 0.1-0.8秒，根據距離自動調整
- **點擊延遲**: 0.05-0.15秒隨機延遲
- **打字間隔**: 0.02-0.12秒，包含暫停和快速輸入

### 軌跡模擬
- 使用貝塞爾曲線生成自然移動軌跡
- 添加隨機抖動和緩動效果
- 避免直線移動被檢測

## 🎮 遊戲自動化最佳實踐

### 1. 防檢測策略
```python
# 使用人性化操作
await mouse_click_coordinate(x, y, human_like=True)

# 添加隨機延遲
await asyncio.sleep(random.uniform(0.5, 2.0))

# 使用像素檢測確認狀態
color = await get_pixel_color(x, y)
if color['hex'] == "#00FF00":  # 綠色表示可點擊
    await mouse_click_coordinate(x, y)
```

### 2. 連招優化
```python
# 精確時間控制的連招
combo = [
    {"type": "key", "keys": ["q"]},
    {"type": "wait", "duration": 0.05},  # 技能冷卻
    {"type": "key", "keys": ["w"]},
    {"type": "wait", "duration": 0.1},
    {"type": "key", "keys": ["e"]},
]
await combo_sequence(combo, human_like=False)  # 遊戲中使用精確時間
```

### 3. 狀態監控
```python
# 等待技能冷卻完成（顏色變化）
await wait_for_pixel_color(
    x=skill_icon_x, 
    y=skill_icon_y, 
    target_color="#FFFFFF",  # 技能可用時的顏色
    timeout=30.0
)
```

## 📝 預定義巨集模板

```python
# 創建常用巨集
await create_predefined_macro("copy_paste", "quick_copy")
await create_predefined_macro("save_file", "quick_save")
await create_predefined_macro("refresh_page", "refresh")
await create_predefined_macro("close_window", "close")
```

## ⚠️ 注意事項

1. **安全設置**: PyAutoGUI 的 FAILSAFE 已啟用，移動滑鼠到螢幕左上角可緊急停止
2. **視窗焦點**: 確保目標應用程式視窗處於前台
3. **座標系統**: 支援絕對座標和相對於目標視窗的座標
4. **錯誤處理**: 所有操作都包含異常處理和錯誤日誌

## 🔧 系統信息

```python
# 獲取完整系統狀態
info = await get_system_info()
print(f"螢幕尺寸: {info['screen_size']}")
print(f"當前滑鼠位置: {info['mouse_position']}")
print(f"巨集數量: {info['macro_count']}")
```

## 📚 進階用法示例

### 自動化瀏覽器操作
```python
# 切換到瀏覽器
await switch_to_window("Chrome")

# 點擊地址欄
await mouse_click_coordinate(400, 50)

# 輸入網址
await keyboard_type_text("https://example.com", human_like=True)

# 按下回車
await keyboard_hotkey(['enter'])
```

### 遊戲自動戰鬥
```python
# 開始錄製戰鬥巨集
await macro_start_recording("auto_battle")

# 使用技能1
await mouse_click_coordinate(100, 500)
await asyncio.sleep(0.5)

# 使用技能2
await keyboard_hotkey(['q'])
await asyncio.sleep(1.0)

# 停止錄製
await macro_stop_recording()

# 重複戰鬥
await macro_play("auto_battle", repeat_count=10)
```

這個系統提供了完整的電腦自動化解決方案，適用於辦公自動化、遊戲輔助和測試自動化等多種場景。
