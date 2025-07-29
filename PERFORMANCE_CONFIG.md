# 🚀 性能優化配置指南

## 📋 概述

本文檔提供了 MCP OmniParser AutoGUI 工具的性能優化配置和最佳實踐。

## ⚡ 核心性能優化

### 1. OmniParser 快取系統

#### 快取配置
```python
# 快取設置（已實現）
CACHE_TTL = 30.0  # 快取存活時間（秒）
MAX_CACHE_SIZE = 10  # 最大快取條目數
CACHE_CLEANUP_INTERVAL = 60.0  # 快取清理間隔（秒）
```

#### 快取策略
- **圖像指紋**: 使用 MD5 雜湊識別相同圖像
- **LRU 淘汰**: 最近最少使用的條目優先淘汰
- **自動清理**: 定期清理過期快取條目
- **記憶體限制**: 防止快取無限增長

### 2. 線程優化

#### 線程池配置
```python
# 建議的線程配置
MAX_WORKER_THREADS = 4  # 最大工作線程數
THREAD_TIMEOUT = 30.0   # 線程超時時間
DAEMON_THREADS = True   # 使用守護線程
```

#### 線程安全改進
- **可重入鎖**: 防止死鎖問題
- **線程本地存儲**: 避免全局變量競爭
- **原子操作**: 確保數據一致性

### 3. 記憶體優化

#### 圖像處理優化
```python
# 圖像壓縮設置
IMAGE_QUALITY = 85      # JPEG 品質 (1-100)
MAX_IMAGE_SIZE = 1920   # 最大圖像尺寸
COMPRESSION_FORMAT = 'PNG'  # 壓縮格式
```

#### 記憶體管理
- **及時釋放**: 處理完成後立即釋放圖像記憶體
- **弱引用**: 使用 weakref 避免循環引用
- **垃圾回收**: 定期觸發垃圾回收

## 🔧 環境變數配置

### 基本性能設置
```bash
# OmniParser 設置
export OMNI_PARSER_DEVICE="cuda"  # 使用 GPU 加速
export OMNI_PARSER_BACKEND_LOAD="1"  # 後台載入模型
export BOX_TRESHOLD="0.05"  # 檢測閾值

# 快取設置
export CACHE_ENABLED="1"
export CACHE_TTL="30"
export MAX_CACHE_SIZE="10"

# 線程設置
export MAX_WORKER_THREADS="4"
export THREAD_TIMEOUT="30"

# 日誌設置
export LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR
export LOG_FILE="mcp_autogui.log"
```

### 高性能配置
```bash
# 針對高性能需求的配置
export OMNI_PARSER_DEVICE="cuda"
export OMNI_PARSER_BACKEND_LOAD="1"
export CACHE_TTL="60"  # 更長的快取時間
export MAX_CACHE_SIZE="20"  # 更大的快取
export MAX_WORKER_THREADS="8"  # 更多線程
export PYAUTOGUI_PAUSE="0.001"  # 更快的操作間隔
```

### 低資源配置
```bash
# 針對資源受限環境的配置
export OMNI_PARSER_DEVICE="cpu"
export CACHE_TTL="15"  # 較短的快取時間
export MAX_CACHE_SIZE="5"  # 較小的快取
export MAX_WORKER_THREADS="2"  # 較少線程
export IMAGE_QUALITY="70"  # 較低的圖像品質
```

## 📊 性能監控

### 關鍵指標
```python
# 監控指標（建議實現）
METRICS = {
    'cache_hit_rate': 0.0,      # 快取命中率
    'avg_response_time': 0.0,   # 平均響應時間
    'memory_usage': 0.0,        # 記憶體使用量
    'thread_count': 0,          # 活動線程數
    'error_rate': 0.0,          # 錯誤率
}
```

### 性能基準
- **快取命中率**: > 70%
- **平均響應時間**: < 2 秒
- **記憶體使用**: < 1GB
- **錯誤率**: < 5%

## 🎯 最佳實踐

### 1. 使用模式優化

#### 批量操作
```python
# 好的做法：批量處理
actions = [
    {"type": "click", "x": 100, "y": 100},
    {"type": "wait", "duration": 0.1},
    {"type": "type", "text": "Hello"},
]
await combo_sequence(actions)

# 避免：頻繁的單個操作
await mouse_click_coordinate(100, 100)
await asyncio.sleep(0.1)
await keyboard_type_text("Hello")
```

#### 智能快取利用
```python
# 好的做法：重用螢幕分析結果
screen_data = await omniparser_details_on_screen()
# 在30秒內進行多次操作，會使用快取

# 避免：頻繁重新分析相同畫面
for i in range(5):
    await omniparser_details_on_screen()  # 不必要的重複調用
```

### 2. 資源管理

#### 正確的資源清理
```python
# 使用上下文管理器
with managed_resource(temp_file, cleanup_func) as resource:
    # 使用資源
    pass
# 自動清理

# 避免：手動管理容易遺漏
temp_file = create_temp_file()
try:
    # 使用文件
    pass
finally:
    # 可能忘記清理
    pass
```

### 3. 錯誤處理優化

#### 智能重試
```python
# 好的做法：針對性重試
@retry_on_error(
    config=RetryConfig(max_attempts=3, base_delay=1.0),
    error_types=(NetworkError, TimeoutError)
)
async def network_operation():
    pass

# 避免：盲目重試所有錯誤
try:
    operation()
except:
    # 重試所有錯誤可能導致無限循環
    pass
```

## 🔍 故障排除

### 常見性能問題

#### 1. 響應時間慢
**症狀**: 操作響應時間超過 5 秒
**原因**: 
- OmniParser 模型載入慢
- 網路連接問題
- 快取未命中

**解決方案**:
```bash
# 啟用後台載入
export OMNI_PARSER_BACKEND_LOAD="1"

# 使用 GPU 加速
export OMNI_PARSER_DEVICE="cuda"

# 增加快取時間
export CACHE_TTL="60"
```

#### 2. 記憶體使用過高
**症狀**: 記憶體使用超過 2GB
**原因**:
- 快取過大
- 圖像未及時釋放
- 線程洩漏

**解決方案**:
```bash
# 減少快取大小
export MAX_CACHE_SIZE="5"

# 降低圖像品質
export IMAGE_QUALITY="70"

# 限制線程數
export MAX_WORKER_THREADS="2"
```

#### 3. 快取命中率低
**症狀**: 快取命中率 < 50%
**原因**:
- 快取時間太短
- 畫面變化頻繁
- 快取大小不足

**解決方案**:
```bash
# 增加快取時間和大小
export CACHE_TTL="45"
export MAX_CACHE_SIZE="15"

# 調整檢測閾值
export BOX_TRESHOLD="0.1"
```

## 📈 性能測試

### 基準測試腳本
```python
import time
import asyncio
from statistics import mean

async def benchmark_screen_analysis(iterations=10):
    """測試螢幕分析性能"""
    times = []
    for i in range(iterations):
        start = time.time()
        await omniparser_details_on_screen()
        end = time.time()
        times.append(end - start)
    
    return {
        'avg_time': mean(times),
        'min_time': min(times),
        'max_time': max(times),
        'total_time': sum(times)
    }

async def benchmark_click_operations(iterations=100):
    """測試點擊操作性能"""
    times = []
    for i in range(iterations):
        start = time.time()
        await mouse_click_coordinate(100, 100)
        end = time.time()
        times.append(end - start)
    
    return {
        'avg_time': mean(times),
        'operations_per_second': iterations / sum(times)
    }
```

### 性能報告範例
```
=== 性能測試報告 ===
螢幕分析:
- 平均時間: 1.2 秒
- 最小時間: 0.8 秒
- 最大時間: 2.1 秒
- 快取命中率: 75%

點擊操作:
- 平均時間: 0.05 秒
- 每秒操作數: 20 次

記憶體使用:
- 峰值記憶體: 512 MB
- 平均記憶體: 256 MB

錯誤統計:
- 總操作數: 1000
- 失敗次數: 12
- 錯誤率: 1.2%
```

## 🎯 下一步優化

### 短期改進 (1-2週)
1. **實現性能監控儀表板**
2. **添加自動性能調優**
3. **優化圖像處理管道**

### 中期改進 (1個月)
1. **實現分散式快取**
2. **添加負載均衡**
3. **優化模型載入策略**

### 長期改進 (3個月)
1. **實現預測性快取**
2. **添加機器學習優化**
3. **開發專用硬體支援**
