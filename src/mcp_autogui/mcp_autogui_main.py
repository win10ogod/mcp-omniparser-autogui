#coding: utf-8

import os
import sys
import threading
import io
import asyncio
import tempfile
from contextlib import redirect_stdout, contextmanager
import base64
import json
import pyautogui
import pyperclip
from mcp.server.fastmcp import Image
import PIL
import pygetwindow as gw
import requests
import random
import time
import math
import logging
import traceback
import weakref
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from functools import wraps
from dataclasses import dataclass
from enum import Enum

# 導入高FPS截圖模組
try:
    from .high_fps_capture import (
        HighFPSCapture, GameOptimizedCapture, FrameRateSync,
        LowLatencyProcessor, CaptureConfig, CaptureMethod
    )
    HIGH_FPS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"高FPS截圖模組不可用: {e}")
    HIGH_FPS_AVAILABLE = False

omniparser_path = os.path.join(os.path.dirname(__file__), '..', '..', 'OmniParser')
sys.path = [omniparser_path, ] + sys.path
from util.omniparser import Omniparser
sys.path = sys.path[1:]

INPUT_IMAGE_SIZE = 960

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler('mcp_autogui.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 錯誤類型枚舉
class ErrorType(Enum):
    NETWORK_ERROR = "network_error"
    PERMISSION_ERROR = "permission_error"
    RESOURCE_ERROR = "resource_error"
    VALIDATION_ERROR = "validation_error"
    SYSTEM_ERROR = "system_error"
    TIMEOUT_ERROR = "timeout_error"

# 錯誤處理配置
@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 10.0
    backoff_factor: float = 2.0

# 全局重試配置
DEFAULT_RETRY_CONFIG = RetryConfig()

# 重試裝飾器
def retry_on_error(config: RetryConfig = None, error_types: Tuple = None):
    """重試裝飾器，支援指定錯誤類型和重試配置"""
    if config is None:
        config = DEFAULT_RETRY_CONFIG
    if error_types is None:
        error_types = (Exception,)

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except error_types as e:
                    last_exception = e
                    if attempt == config.max_attempts - 1:
                        logger.error(f"函數 {func.__name__} 在 {config.max_attempts} 次嘗試後失敗: {e}")
                        break

                    delay = min(config.base_delay * (config.backoff_factor ** attempt), config.max_delay)
                    logger.warning(f"函數 {func.__name__} 第 {attempt + 1} 次嘗試失敗: {e}，{delay:.2f}秒後重試")
                    await asyncio.sleep(delay)

            raise last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except error_types as e:
                    last_exception = e
                    if attempt == config.max_attempts - 1:
                        logger.error(f"函數 {func.__name__} 在 {config.max_attempts} 次嘗試後失敗: {e}")
                        break

                    delay = min(config.base_delay * (config.backoff_factor ** attempt), config.max_delay)
                    logger.warning(f"函數 {func.__name__} 第 {attempt + 1} 次嘗試失敗: {e}，{delay:.2f}秒後重試")
                    time.sleep(delay)

            raise last_exception

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# 資源管理器
@contextmanager
def managed_resource(resource, cleanup_func=None):
    """資源管理上下文管理器"""
    try:
        yield resource
    finally:
        if cleanup_func:
            try:
                cleanup_func(resource)
            except Exception as e:
                logger.warning(f"清理資源時發生錯誤: {e}")

# 線程安全的單例模式
class ThreadSafeSingleton:
    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]

# 人性化輸入模擬配置
class HumanLikeConfig:
    """人性化操作配置"""
    # 滑鼠移動配置
    MOUSE_MOVE_MIN_DURATION = 0.1  # 最小移動時間
    MOUSE_MOVE_MAX_DURATION = 0.8  # 最大移動時間
    MOUSE_MOVE_CURVE_STRENGTH = 0.3  # 曲線強度

    # 點擊配置
    CLICK_MIN_DELAY = 0.05  # 最小點擊延遲
    CLICK_MAX_DELAY = 0.15  # 最大點擊延遲
    DOUBLE_CLICK_INTERVAL = 0.1  # 雙擊間隔

    # 鍵盤輸入配置
    TYPING_MIN_INTERVAL = 0.02  # 最小打字間隔
    TYPING_MAX_INTERVAL = 0.12  # 最大打字間隔
    TYPING_BURST_CHANCE = 0.3  # 快速輸入機率
    TYPING_PAUSE_CHANCE = 0.1  # 暫停機率
    TYPING_PAUSE_DURATION = 0.5  # 暫停時間

    # 滾輪配置
    SCROLL_MIN_DELAY = 0.05
    SCROLL_MAX_DELAY = 0.2

# 改進的巨集系統
class MacroSystem(ThreadSafeSingleton):
    """線程安全的巨集錄製和回放系統"""
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.macros: Dict[str, List[Dict[str, Any]]] = {}
        self.recording_macro: Optional[str] = None
        self.current_recording: List[Dict[str, Any]] = []
        self._lock = threading.RLock()  # 使用可重入鎖

    def start_recording(self, name: str):
        """開始錄製巨集"""
        with self._lock:
            if self.recording_macro:
                logger.warning(f"已在錄製巨集 '{self.recording_macro}'，停止當前錄製")
                self.stop_recording()
            self.recording_macro = name
            self.current_recording = []
            logger.info(f"開始錄製巨集: {name}")

    def stop_recording(self):
        """停止錄製巨集"""
        with self._lock:
            if self.recording_macro:
                self.macros[self.recording_macro] = self.current_recording.copy()
                logger.info(f"完成錄製巨集: {self.recording_macro}，共 {len(self.current_recording)} 個動作")
                self.recording_macro = None
                self.current_recording = []

    def add_action(self, action_type: str, **kwargs):
        """添加動作到當前錄製"""
        with self._lock:
            if self.recording_macro:
                action = {
                    'type': action_type,
                    'timestamp': time.time(),
                    **kwargs
                }
                self.current_recording.append(action)

    def get_macro(self, name: str) -> Optional[List[Dict[str, Any]]]:
        """獲取巨集"""
        with self._lock:
            return self.macros.get(name)

    def list_macros(self) -> List[str]:
        """列出所有巨集"""
        with self._lock:
            return list(self.macros.keys())

    def delete_macro(self, name: str) -> bool:
        """刪除巨集"""
        with self._lock:
            if name in self.macros:
                del self.macros[name]
                logger.info(f"刪除巨集: {name}")
                return True
            return False

# OmniParser 快取管理
class OmniParserCache(ThreadSafeSingleton):
    """OmniParser 結果快取"""
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.RLock()
        self.cache_ttl = 30.0  # 快取30秒
        self.max_cache_size = 10

    def _generate_key(self, image_data: bytes) -> str:
        """生成快取鍵"""
        import hashlib
        return hashlib.md5(image_data).hexdigest()

    def get(self, image_data: bytes) -> Optional[Tuple[Any, Any]]:
        """獲取快取結果"""
        with self._lock:
            key = self._generate_key(image_data)
            if key in self._cache:
                result, timestamp = self._cache[key]
                if time.time() - timestamp < self.cache_ttl:
                    logger.debug(f"快取命中: {key}")
                    return result
                else:
                    del self._cache[key]
                    logger.debug(f"快取過期: {key}")
            return None

    def set(self, image_data: bytes, result: Tuple[Any, Any]):
        """設置快取結果"""
        with self._lock:
            key = self._generate_key(image_data)

            # 清理過期快取
            current_time = time.time()
            expired_keys = [k for k, (_, ts) in self._cache.items()
                          if current_time - ts >= self.cache_ttl]
            for k in expired_keys:
                del self._cache[k]

            # 限制快取大小
            if len(self._cache) >= self.max_cache_size:
                oldest_key = min(self._cache.keys(),
                               key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

            self._cache[key] = (result, current_time)
            logger.debug(f"快取設置: {key}")

# 全局快取實例
omniparser_cache = OmniParserCache()

# 高FPS截圖實例
high_fps_capture = None
frame_rate_sync = None
low_latency_processor = None

def generate_human_like_curve(start_x: int, start_y: int, end_x: int, end_y: int,
                             duration: float, curve_strength: float = 0.3) -> List[Tuple[int, int, float]]:
    """生成人性化的滑鼠移動軌跡"""
    distance = math.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

    # 根據距離調整點數
    num_points = max(10, min(50, int(distance / 10)))

    points = []
    for i in range(num_points + 1):
        t = i / num_points

        # 使用貝塞爾曲線創建自然軌跡
        # 添加隨機控制點
        mid_x = (start_x + end_x) / 2 + random.uniform(-distance * curve_strength, distance * curve_strength)
        mid_y = (start_y + end_y) / 2 + random.uniform(-distance * curve_strength, distance * curve_strength)

        # 二次貝塞爾曲線
        x = (1 - t) ** 2 * start_x + 2 * (1 - t) * t * mid_x + t ** 2 * end_x
        y = (1 - t) ** 2 * start_y + 2 * (1 - t) * t * mid_y + t ** 2 * end_y

        # 添加微小的隨機抖動
        x += random.uniform(-2, 2)
        y += random.uniform(-2, 2)

        # 計算時間點（使用緩動函數）
        time_point = duration * (t ** 0.8)  # 緩動效果

        points.append((int(x), int(y), time_point))

    return points

async def human_like_mouse_move(x: int, y: int, duration: Optional[float] = None):
    """人性化滑鼠移動"""
    current_x, current_y = pyautogui.position()

    if duration is None:
        distance = math.sqrt((x - current_x) ** 2 + (y - current_y) ** 2)
        duration = random.uniform(
            HumanLikeConfig.MOUSE_MOVE_MIN_DURATION,
            HumanLikeConfig.MOUSE_MOVE_MAX_DURATION
        )
        # 根據距離調整時間
        duration *= min(2.0, distance / 500)

    # 生成軌跡點
    points = generate_human_like_curve(
        current_x, current_y, x, y, duration, HumanLikeConfig.MOUSE_MOVE_CURVE_STRENGTH
    )

    start_time = time.time()
    for point_x, point_y, time_offset in points:
        target_time = start_time + time_offset
        current_time = time.time()

        if current_time < target_time:
            await asyncio.sleep(target_time - current_time)

        pyautogui.moveTo(point_x, point_y)

async def human_like_click(x: int, y: int, button: str = 'left', clicks: int = 1):
    """人性化點擊"""
    await human_like_mouse_move(x, y)

    # 點擊前的小延遲
    await asyncio.sleep(random.uniform(HumanLikeConfig.CLICK_MIN_DELAY, HumanLikeConfig.CLICK_MAX_DELAY))

    if clicks == 1:
        pyautogui.click(button=button)
    elif clicks == 2:
        pyautogui.click(button=button)
        await asyncio.sleep(HumanLikeConfig.DOUBLE_CLICK_INTERVAL)
        pyautogui.click(button=button)
    else:
        for i in range(clicks):
            pyautogui.click(button=button)
            if i < clicks - 1:
                await asyncio.sleep(random.uniform(0.05, 0.15))

@retry_on_error(error_types=(pyperclip.PyperclipException, Exception))
async def human_like_type(text: str, use_clipboard: bool = False):
    """人性化文字輸入，帶錯誤處理和重試"""
    try:
        if use_clipboard or not text.isascii():
            # 使用剪貼簿輸入非ASCII字符
            prev_clip = None
            try:
                prev_clip = pyperclip.paste()
            except Exception as e:
                logger.warning(f"獲取剪貼簿內容失敗: {e}")

            pyperclip.copy(text)
            await asyncio.sleep(0.1)  # 等待剪貼簿更新
            pyautogui.hotkey('ctrl', 'v')

            # 恢復原剪貼簿內容
            if prev_clip:
                try:
                    await asyncio.sleep(0.1)
                    pyperclip.copy(prev_clip)
                except Exception as e:
                    logger.warning(f"恢復剪貼簿內容失敗: {e}")
            return

        # 逐字符輸入
        for i, char in enumerate(text):
            try:
                pyautogui.write(char)
            except Exception as e:
                logger.warning(f"輸入字符 '{char}' 失敗: {e}")
                continue

            # 隨機延遲
            if random.random() < HumanLikeConfig.TYPING_PAUSE_CHANCE:
                # 偶爾暫停（模擬思考）
                await asyncio.sleep(HumanLikeConfig.TYPING_PAUSE_DURATION)
            elif random.random() < HumanLikeConfig.TYPING_BURST_CHANCE:
                # 快速輸入
                await asyncio.sleep(HumanLikeConfig.TYPING_MIN_INTERVAL)
            else:
                # 正常輸入
                await asyncio.sleep(random.uniform(
                    HumanLikeConfig.TYPING_MIN_INTERVAL,
                    HumanLikeConfig.TYPING_MAX_INTERVAL
                ))
    except Exception as e:
        logger.error(f"人性化輸入失敗: {e}")
        raise


async def _activate_window_with_retry(window, max_retries=3, delay=0.2) -> bool:
    """帶重試機制的視窗啟用函數"""
    if not window:
        return True  # 沒有目標視窗，無需啟用
    for attempt in range(max_retries):
        try:
            window.activate()
            await asyncio.sleep(delay)  # 等待視窗穩定
            if window.isActive:
                return True
            logger.warning(f"第 {attempt + 1} 次啟用視窗 '{window.title}' 失敗，正在重試...")
        except Exception as e:
            logger.warning(f"啟用視窗時發生錯誤 (第 {attempt + 1} 次): {e}")
        await asyncio.sleep(delay * (attempt + 1))
    logger.error(f"'{window.title}' 在 {max_retries} 次嘗試後仍無法啟用。")
    return False

def _get_absolute_coords(state, x, y):
    """計算並驗證絕對座標"""
    if state['is_set_target_window'] and state['current_window']:
        try:
            abs_x = x + state['current_window'].left
            abs_y = y + state['current_window'].top
        except Exception as e:
            logger.warning(f"計算絕對座標時視窗失效: {e}")
            return None, None
    else:
        abs_x, abs_y = x, y

    # 關鍵檢查：防止 PyAutoGUI 安全機制觸發
    if abs_x == 0 and abs_y == 0:
        logger.warning("偵測到目標座標為 (0, 0)，為防止意外觸發安全機制，已中止操作。")
        return None, None

    return abs_x, abs_y


def mcp_autogui_main(mcp):
    """主要MCP工具初始化函數，帶改進的錯誤處理和資源管理"""
    global omniparser
    omniparser = None

    # 使用局部變量而非全局變量
    state = {
        'input_image_path': '',
        'output_dir_path': '',
        'omniparser_thread': None,
        'result_image': None,
        'input_image_resized_path': None,
        'detail': None,
        'is_finished': False,
        'current_mouse_x': 0,
        'current_mouse_y': 0,
        'is_set_target_window': False,
        'current_window': None
    }

    try:
        # 初始化滑鼠位置
        state['current_mouse_x'], state['current_mouse_y'] = pyautogui.position()

        # 設置目標視窗
        match_windows = None
        if 'TARGET_WINDOW_NAME' in os.environ:
            try:
                match_windows = gw.getWindowsWithTitle(os.environ['TARGET_WINDOW_NAME'])
            except Exception as e:
                logger.warning(f"獲取目標視窗失敗: {e}")

        if match_windows:
            state['current_window'] = match_windows[0]
            state['is_set_target_window'] = True
            logger.info(f"設置目標視窗: {state['current_window'].title}")
        else:
            try:
                state['current_window'] = gw.getActiveWindow()
            except Exception as e:
                logger.warning(f"獲取活動視窗失敗: {e}")

        # 初始化巨集系統
        macro_system = MacroSystem()

        # 設置PyAutoGUI安全設置
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.01  # 基本延遲

        logger.info("MCP AutoGUI 初始化完成")

    except Exception as e:
        logger.error(f"MCP AutoGUI 初始化失敗: {e}")
        raise
    # 改進的配置管理
    try:
        with redirect_stdout(sys.stderr):
            config = {
                'som_model_path': os.environ.get('SOM_MODEL_PATH',
                    os.path.join(omniparser_path, 'weights/icon_detect/model.pt')),
                'caption_model_name': os.environ.get('CAPTION_MODEL_NAME', 'florence2'),
                'caption_model_path': os.environ.get('CAPTION_MODEL_PATH',
                    os.path.join(omniparser_path, 'weights/icon_caption_florence')),
                'device': os.environ.get('OMNI_PARSER_DEVICE', 'cuda'),
                'BOX_TRESHOLD': float(os.environ.get('BOX_TRESHOLD', '0.05')),
            }

            # 驗證配置
            for key, value in config.items():
                if key in ['som_model_path', 'caption_model_path'] and not os.path.exists(value):
                    logger.warning(f"配置路徑不存在: {key}={value}")

            # 初始化 OmniParser
            if 'OMNI_PARSER_SERVER' not in os.environ:
                def omniparser_start_thread_func():
                    global omniparser
                    try:
                        sys.path = [os.path.join(os.path.dirname(__file__), '..', '..'), ] + sys.path
                        from download_models import download_omniparser_models
                        download_omniparser_models()
                        sys.path = sys.path[1:]

                        omniparser = Omniparser(config)
                        logger.info("OmniParser 載入完成")
                    except Exception as e:
                        logger.error(f"OmniParser 載入失敗: {e}")
                        raise

                if os.environ.get('OMNI_PARSER_BACKEND_LOAD'):
                    omniparser_start_thread = threading.Thread(
                        target=omniparser_start_thread_func,
                        name="OmniParserLoader",
                        daemon=True  # 設為守護線程
                    )
                    omniparser_start_thread.start()
                    logger.info("OmniParser 後台載入已啟動")
                else:
                    omniparser_start_thread_func()
    except Exception as e:
        logger.error(f"OmniParser 配置失敗: {e}")
        raise

    # 改進的臨時目錄管理
    temp_dir = tempfile.TemporaryDirectory()
    dname = temp_dir.name

    # 註冊清理函數
    def cleanup_resources():
        """清理資源"""
        try:
            temp_dir.cleanup()
            logger.info("臨時目錄已清理")
        except Exception as e:
            logger.warning(f"清理臨時目錄失敗: {e}")

    # 使用 weakref 確保清理
    weakref.finalize(mcp, cleanup_resources)

    @mcp.tool()
    @retry_on_error(error_types=(requests.RequestException, ConnectionError, TimeoutError))
    async def omniparser_details_on_screen() -> list:
        """Get the screen and analyze its details with improved error handling and caching.

        If a timeout occurs, you can continue by running it again.
        Uses caching to improve performance for similar screenshots.

        Return value:
            - Details such as the content of text.
            - Screen capture with ID number added.
        """
        nonlocal state

        try:
            # 等待 OmniParser 載入
            if 'OMNI_PARSER_SERVER' not in os.environ:
                timeout_counter = 0
                while omniparser is None and timeout_counter < 300:  # 30秒超時
                    await asyncio.sleep(0.1)
                    timeout_counter += 1

                if omniparser is None:
                    raise TimeoutError("OmniParser 載入超時")

            detail_text = ''

            with redirect_stdout(sys.stderr):
                def omniparser_thread_func():
                    nonlocal state, detail_text

                    try:
                        with redirect_stdout(sys.stderr):
                            # 激活目標視窗
                            if state['is_set_target_window'] and state['current_window']:
                                try:
                                    state['current_window'].activate()
                                    time.sleep(0.1)  # 等待視窗激活
                                except Exception as e:
                                    logger.warning(f"激活視窗失敗: {e}")

                            # 截圖
                            screenshot_image = pyautogui.screenshot()

                            # 裁剪到目標視窗
                            if state['is_set_target_window'] and state['current_window']:
                                try:
                                    screenshot_image = screenshot_image.crop((
                                        state['current_window'].left,
                                        state['current_window'].top,
                                        state['current_window'].right,
                                        state['current_window'].bottom
                                    ))
                                except Exception as e:
                                    logger.warning(f"裁剪視窗失敗: {e}")

                            # 轉換為字節用於快取
                            buffered = io.BytesIO()
                            screenshot_image.save(buffered, format='PNG')
                            image_bytes = buffered.getvalue()

                            # 檢查快取
                            cached_result = omniparser_cache.get(image_bytes)
                            if cached_result:
                                dino_labled_img, state['detail'] = cached_result
                                logger.debug("使用快取結果")
                            else:
                                # 處理圖像
                                if 'OMNI_PARSER_SERVER' in os.environ:
                                    # 使用遠程服務器
                                    send_img = base64.b64encode(image_bytes).decode('ascii')
                                    json_data = json.dumps({'base64_image': send_img})

                                    response = requests.post(
                                        f"http://{os.environ['OMNI_PARSER_SERVER']}/parse/",
                                        data=json_data,
                                        headers={"Content-Type": "application/json"},
                                        timeout=30  # 30秒超時
                                    )
                                    response.raise_for_status()
                                    response_json = response.json()
                                    dino_labled_img = response_json['som_image_base64']
                                    state['detail'] = response_json['parsed_content_list']
                                else:
                                    # 使用本地 OmniParser
                                    dino_labled_img, state['detail'] = omniparser.parse_raw(screenshot_image)

                                # 快取結果
                                omniparser_cache.set(image_bytes, (dino_labled_img, state['detail']))

                    except Exception as e:
                        logger.error(f"OmniParser 處理失敗: {e}")
                        state['detail'] = []
                        dino_labled_img = None
                        raise

                    # 處理結果圖像
                    if dino_labled_img:
                        try:
                            image_bytes = base64.b64decode(dino_labled_img)
                            result_image_local = PIL.Image.open(io.BytesIO(image_bytes))

                            # 調整圖像大小
                            width, height = result_image_local.size
                            if width > height:
                                new_size = (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE * height // width)
                            else:
                                new_size = (INPUT_IMAGE_SIZE * width // height, INPUT_IMAGE_SIZE)

                            result_image_local = result_image_local.resize(new_size, PIL.Image.Resampling.LANCZOS)

                            state['result_image'] = io.BytesIO()
                            result_image_local.save(state['result_image'], format='PNG')
                        except Exception as e:
                            logger.error(f"處理結果圖像失敗: {e}")
                            state['result_image'] = io.BytesIO()

                    # 生成詳細文本
                    detail_text = ''
                    if state['detail']:
                        for loop, content in enumerate(state['detail']):
                            try:
                                detail_text += f'ID: {loop}, {content["type"]}: {content["content"]}\n'
                            except (KeyError, TypeError) as e:
                                logger.warning(f"處理詳細信息失敗 (ID: {loop}): {e}")
                                detail_text += f'ID: {loop}, 錯誤: 無法解析內容\n'

                    state['is_finished'] = True

                # 線程安全的執行
                if state['omniparser_thread'] is None:
                    state['result_image'] = None
                    state['detail'] = None
                    state['is_finished'] = False

                    state['omniparser_thread'] = threading.Thread(
                        target=omniparser_thread_func,
                        name="OmniParserWorker",
                        daemon=True
                    )
                    state['omniparser_thread'].start()

                # 等待完成，帶超時
                timeout_counter = 0
                while not state['is_finished'] and timeout_counter < 600:  # 60秒超時
                    await asyncio.sleep(0.1)
                    timeout_counter += 1

                if not state['is_finished']:
                    logger.error("OmniParser 處理超時")
                    raise TimeoutError("OmniParser 處理超時")

                state['omniparser_thread'] = None

                # 返回結果
                if state['result_image']:
                    return [detail_text, Image(data=state['result_image'].getvalue(), format="png")]
                else:
                    return [detail_text, Image(data=b'', format="png")]

        except Exception as e:
            logger.error(f"螢幕分析失敗: {e}")
            return [f"錯誤: {str(e)}", Image(data=b'', format="png")]

    @mcp.tool()
    @retry_on_error(error_types=(pyautogui.FailSafeException, Exception))
    async def omniparser_click(id: int, button: str = 'left', clicks: int = 1) -> bool:
        """Click on anything on the screen with improved error handling.

        Args:
            id: The element on the screen to click. Check with "omniparser_details_on_screen".
            button: Button to click. 'left', 'middle', or 'right'.
            clicks: Number of clicks. 2 for double click.

        Return value:
            True if successful, False if element not found or click failed.
        """
        try:
            # 驗證輸入參數
            if not isinstance(id, int) or id < 0:
                logger.error(f"無效的元素ID: {id}")
                return False

            if button not in ['left', 'right', 'middle']:
                logger.error(f"無效的按鈕類型: {button}")
                return False

            if not isinstance(clicks, int) or clicks < 1:
                logger.error(f"無效的點擊次數: {clicks}")
                return False

            # 檢查是否有詳細信息
            if not state['detail'] or len(state['detail']) <= id:
                logger.error(f"元素ID {id} 不存在，當前有 {len(state['detail']) if state['detail'] else 0} 個元素")
                return False

            screen_width, screen_height = pyautogui.size()

            # 激活目標視窗
            if not await _activate_window_with_retry(state.get('current_window')):
                return False

            # 獲取視窗偏移
            left = state['current_window'].left if state.get('current_window') else 0
            top = state['current_window'].top if state.get('current_window') else 0

            # 計算點擊座標
            try:
                compos = state['detail'][id]['bbox']
                click_x = int((compos[0] + compos[2]) * screen_width) // 2 + left
                click_y = int((compos[1] + compos[3]) * screen_height) // 2 + top

                # 驗證座標是否在螢幕範圍內
                if not (0 <= click_x <= screen_width and 0 <= click_y <= screen_height):
                    logger.error(f"計算的座標超出螢幕範圍: ({click_x}, {click_y})")
                    return False

            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"解析元素座標失敗: {e}")
                return False

            # 執行點擊
            pyautogui.click(x=click_x, y=click_y, button=button, clicks=clicks)

            # 更新滑鼠位置
            state['current_mouse_x'] = click_x
            state['current_mouse_y'] = click_y

            logger.info(f"成功點擊元素 ID:{id} 座標:({click_x},{click_y}) 按鈕:{button} 次數:{clicks}")
            return True

        except pyautogui.FailSafeException:
            logger.error("PyAutoGUI 安全機制觸發，滑鼠移動到螢幕角落")
            return False
        except Exception as e:
            logger.error(f"點擊操作失敗: {e}")
            return False
            if not state['is_set_target_window']:
                try:
                    state['current_window'] = gw.getActiveWindow()
                except Exception as e:
                    logger.warning(f"獲取活動視窗失敗: {e}")
            return True
        return False

    @mcp.tool()
    async def omniparser_drags(from_id: int, to_id: int, button: str = 'left', key: str = '') -> bool:
        """Drag and drop on the screen.

Args:
    from_id: The element on the screen that it start to drag. You can check it with "omniparser_details_on_screen".
    to_id: The element on the screen that it end to drag. You can check it with "omniparser_details_on_screen".
    button: Button to click. 'left', 'middle', or 'right'.
    key: The name of the keyboard key if you hold down it while dragging. You can check key's name with "omniparser_get_keys_list".
Return value:
    True is success. False is means "this is not found".
"""
        try:
            screen_width, screen_height = pyautogui.size()

            if not await _activate_window_with_retry(state.get('current_window')):
                return False

            left = state['current_window'].left if state.get('current_window') else 0
            top = state['current_window'].top if state.get('current_window') else 0

            # 檢查元素ID有效性
            if not state['detail'] or len(state['detail']) <= from_id or len(state['detail']) <= to_id:
                logger.error(f"無效的元素ID: from_id={from_id}, to_id={to_id}")
                return False

            # 計算起始位置
            from_compos = state['detail'][from_id]['bbox']
            from_x = int((from_compos[0] + from_compos[2]) * screen_width) // 2 + left
            from_y = int((from_compos[1] + from_compos[3]) * screen_height) // 2 + top

            # 計算結束位置
            to_compos = state['detail'][to_id]['bbox']
            to_x = int((to_compos[0] + to_compos[2]) * screen_width) // 2 + left
            to_y = int((to_compos[1] + to_compos[3]) * screen_height) // 2 + top

            # 按下修飾鍵
            if key is not None and key != '':
                pyautogui.keyDown(key)

            # 執行拖拽
            pyautogui.moveTo(from_x, from_y)
            pyautogui.dragTo(to_x, to_y, button=button)

            # 釋放修飾鍵
            if key is not None and key != '':
                pyautogui.keyUp(key)

            # 更新滑鼠位置
            state['current_mouse_x'] = to_x
            state['current_mouse_y'] = to_y

            # 更新當前視窗
            if not state['is_set_target_window']:
                try:
                    state['current_window'] = gw.getActiveWindow()
                except Exception as e:
                    logger.warning(f"獲取活動視窗失敗: {e}")

            logger.info(f"拖拽操作完成: ({from_x},{from_y}) -> ({to_x},{to_y})")
            return True

        except Exception as e:
            logger.error(f"拖拽操作失敗: {e}")
            return False

    @mcp.tool()
    async def omniparser_mouse_move(id: int) -> bool:
        """Moves the mouse cursor over the specified element.

Args:
    id: The element on the screen that it move. You can check it with "omniparser_details_on_screen".
Return value:
    True is success. False is means "this is not found".
"""
        try:
            screen_width, screen_height = pyautogui.size()

            # 檢查元素ID有效性
            if not state['detail'] or len(state['detail']) <= id:
                logger.error(f"無效的元素ID: {id}")
                return False

            compos = state['detail'][id]['bbox']

            if state['is_set_target_window'] and state['current_window']:
                try:
                    state['current_window'].activate()
                    left = state['current_window'].left
                    top = state['current_window'].top
                except Exception as e:
                    logger.warning(f"激活視窗失敗: {e}")
                    left = top = 0
            else:
                left = top = 0

            # 計算目標位置
            target_x = int((compos[0] + compos[2]) * screen_width) // 2 + left
            target_y = int((compos[1] + compos[3]) * screen_height) // 2 + top

            # 移動滑鼠
            pyautogui.moveTo(target_x, target_y)

            # 更新滑鼠位置
            state['current_mouse_x'] = target_x
            state['current_mouse_y'] = target_y

            # 更新當前視窗
            if not state['is_set_target_window']:
                try:
                    state['current_window'] = gw.getActiveWindow()
                except Exception as e:
                    logger.warning(f"獲取活動視窗失敗: {e}")

            logger.info(f"滑鼠移動到元素 ID:{id} 位置:({target_x},{target_y})")
            return True

        except Exception as e:
            logger.error(f"滑鼠移動失敗: {e}")
            return False

    @mcp.tool()
    async def omniparser_scroll(clicks: int) -> None:
        """The mouse scrolling wheel behavior.

Args:
    clicks: Amount of scrolling. 1000 is scroll up 1000 "clicks" and -1000 is scroll down 1000 "clicks".
"""
        try:
            if state['current_window']:
                state['current_window'].activate()
            pyautogui.moveTo(state['current_mouse_x'], state['current_mouse_y'])
            pyautogui.scroll(clicks)
            logger.info(f"滾動操作: {clicks} 點擊")
        except Exception as e:
            logger.error(f"滾動操作失敗: {e}")

    @mcp.tool()
    async def omniparser_write(content: str, id: int = -1) -> None:
        """Type the characters in the string that is passed.

Args:
    content: What to enter.
    id: Click on the target before typing. You can check it with "omniparser_details_on_screen".
"""
        try:
            if id >= 0:
                await omniparser_click(id)
            else:
                if state['current_window']:
                    state['current_window'].activate()
                pyautogui.moveTo(state['current_mouse_x'], state['current_mouse_y'])

            if content.isascii():
                pyautogui.write(content)
            else:
                prev_clip = pyperclip.paste()
                pyperclip.copy(content)
                pyautogui.hotkey('ctrl', 'v')
                if prev_clip:
                    pyperclip.copy(prev_clip)
        except Exception as e:
            logger.error(f"文字輸入失敗: {e}")

    @mcp.tool()
    async def omniparser_get_keys_list() -> list[str]:
        """List of keyboard keys. Used in "omniparser_input_key" etc.

Return value:
    List of keyboard keys.
"""
        return pyautogui.KEYBOARD_KEYS

    @mcp.tool()
    async def omniparser_input_key(key1: str, key2: str = '', key3: str = '') -> None:
        """Press of keyboard keys. 

Args:
    key1-3: Press of keyboard keys. You can check key's name with "omniparser_get_keys_list". If you specify multiple, keys will be pressed down in order, and then released in reverse order.
"""
        try:
            if state['current_window']:
                state['current_window'].activate()
            pyautogui.moveTo(state['current_mouse_x'], state['current_mouse_y'])

            if key2 is not None and key2 != '' and key3 is not None and key3 != '':
                pyautogui.hotkey(key1, key2, key3)
                logger.info(f"按鍵組合: {key1}+{key2}+{key3}")
            elif key2 is not None and key2 != '':
                pyautogui.hotkey(key1, key2)
                logger.info(f"按鍵組合: {key1}+{key2}")
            else:
                pyautogui.hotkey(key1)
                logger.info(f"按鍵: {key1}")
        except Exception as e:
            logger.error(f"按鍵操作失敗: {e}")

    @mcp.tool()
    async def omniparser_wait(time: float = 1.0) -> None:
        """Waits for the specified number of seconds.

Args:
    time: Waiting time (seconds).
"""
        await asyncio.sleep(time)

    # ==================== 新增：精確座標操作 ====================

    @mcp.tool()
    @retry_on_error(error_types=(pyautogui.FailSafeException, Exception))
    async def mouse_click_coordinate(x: int, y: int, button: str = 'left', clicks: int = 1, human_like: bool = True) -> bool:
        """在指定座標執行精確的滑鼠點擊操作。

        Args:
            x: X座標，相對於當前視窗或螢幕
            y: Y座標，相對於當前視窗或螢幕
            button: 滑鼠按鈕 ('left', 'right', 'middle')，預設 'left'
            clicks: 點擊次數，預設 1，設為 2 表示雙擊
            human_like: 是否使用人性化移動，預設 True

        Return value:
            True表示成功，False表示失敗
        """
        try:
            # 參數驗證
            if not isinstance(x, int) or not isinstance(y, int):
                logger.error(f"座標必須為整數: x={x}, y={y}")
                return False

            if button not in ['left', 'right', 'middle']:
                logger.error(f"無效的按鈕類型: {button}")
                return False

            if not isinstance(clicks, int) or clicks < 1:
                logger.error(f"點擊次數必須為正整數: {clicks}")
                return False

            # 啟用視窗
            if not await _activate_window_with_retry(state.get('current_window')):
                return False

            # 獲取並驗證座標
            abs_x, abs_y = _get_absolute_coords(state, x, y)
            if abs_x is None or abs_y is None:
                return False

            # 獲取螢幕尺寸並進行邊界檢查
            screen_width, screen_height = pyautogui.size()
            abs_x = max(0, min(abs_x, screen_width - 1))
            abs_y = max(0, min(abs_y, screen_height - 1))

            # 執行點擊
            if human_like:
                await human_like_click(abs_x, abs_y, button, clicks)
            else:
                pyautogui.click(x=abs_x, y=abs_y, button=button, clicks=clicks)

            # 更新當前位置
            state['current_mouse_x'] = abs_x
            state['current_mouse_y'] = abs_y

            # 記錄巨集動作
            if macro_system.recording_macro:
                macro_system.add_action('click_coordinate', x=x, y=y, button=button, clicks=clicks, human_like=human_like)

            logger.info(f"成功點擊座標 ({abs_x}, {abs_y})，按鈕: {button}，次數: {clicks}")
            return True

        except pyautogui.FailSafeException:
            logger.error("PyAutoGUI 安全機制觸發")
            return False
        except Exception as e:
            logger.error(f"座標點擊失敗: {e}")
            return False

    @mcp.tool()
    async def mouse_move_coordinate(x: int, y: int, human_like: bool = True, duration: float = 0.5) -> bool:
        """移動滑鼠到指定座標

        Args:
            x: X座標
            y: Y座標
            human_like: 是否使用人性化移動
            duration: 移動持續時間（秒）

        Return value:
            True表示成功
        """
        try:
            # 啟用視窗
            if not await _activate_window_with_retry(state.get('current_window')):
                return False

            # 獲取並驗證座標
            abs_x, abs_y = _get_absolute_coords(state, x, y)
            if abs_x is None or abs_y is None:
                return False

            if human_like:
                await human_like_mouse_move(abs_x, abs_y, duration)
            else:
                pyautogui.moveTo(abs_x, abs_y, duration=duration)

            # 更新滑鼠位置
            state['current_mouse_x'] = abs_x
            state['current_mouse_y'] = abs_y

            # 記錄巨集動作
            if macro_system.recording_macro:
                macro_system.add_action('move_coordinate', x=x, y=y, duration=duration, human_like=human_like)

            logger.info(f"滑鼠移動到座標 ({abs_x}, {abs_y})")
            return True

        except Exception as e:
            logger.error(f"滑鼠移動失敗: {e}")
            return False

    @mcp.tool()
    async def mouse_drag_coordinate(from_x: int, from_y: int, to_x: int, to_y: int,
                                  button: str = 'left', human_like: bool = True, duration: float = 1.0) -> bool:
        """在兩個座標之間拖拽

        Args:
            from_x: 起始X座標
            from_y: 起始Y座標
            to_x: 結束X座標
            to_y: 結束Y座標
            button: 滑鼠按鈕
            human_like: 是否使用人性化移動
            duration: 拖拽持續時間

        Return value:
            True表示成功
        """
        try:
            # 啟用視窗
            if not await _activate_window_with_retry(state.get('current_window')):
                return False

            # 獲取並驗證起始和結束座標
            abs_from_x, abs_from_y = _get_absolute_coords(state, from_x, from_y)
            abs_to_x, abs_to_y = _get_absolute_coords(state, to_x, to_y)

            if abs_from_x is None or abs_to_x is None:
                return False

            # 移動到起始位置
            if human_like:
                await human_like_mouse_move(abs_from_x, abs_from_y)
            else:
                pyautogui.moveTo(abs_from_x, abs_from_y)

            # 執行拖拽
            if human_like:
                pyautogui.mouseDown(button=button)
                await human_like_mouse_move(abs_to_x, abs_to_y, duration)
                pyautogui.mouseUp(button=button)
            else:
                pyautogui.dragTo(abs_to_x, abs_to_y, duration=duration, button=button)

            # 更新滑鼠位置
            state['current_mouse_x'] = abs_to_x
            state['current_mouse_y'] = abs_to_y

            # 記錄巨集動作
            if macro_system.recording_macro:
                macro_system.add_action('drag_coordinate',
                                      from_x=from_x, from_y=from_y,
                                      to_x=to_x, to_y=to_y,
                                      button=button, duration=duration, human_like=human_like)

            logger.info(f"拖拽操作完成: ({abs_from_x},{abs_from_y}) -> ({abs_to_x},{abs_to_y})")
            return True

        except Exception as e:
            logger.error(f"拖拽操作失敗: {e}")
            return False

    @mcp.tool()
    async def get_mouse_position() -> dict:
        """獲取當前滑鼠位置

        Return value:
            包含x, y座標的字典
        """
        x, y = pyautogui.position()

        # 如果有目標視窗，轉換為相對座標
        if state['is_set_target_window']:
            rel_x = x - state['current_window'].left
            rel_y = y - state['current_window'].top
            return {
                'absolute': {'x': x, 'y': y},
                'relative': {'x': rel_x, 'y': rel_y},
                'window': state['current_window'].title if state['current_window'] else None
            }
        else:
            return {
                'absolute': {'x': x, 'y': y},
                'relative': {'x': x, 'y': y},
                'window': None
            }

    # ==================== 新增：增強鍵盤操作 ====================

    @mcp.tool()
    @retry_on_error(error_types=(pyperclip.PyperclipException, Exception))
    async def keyboard_type_text(text: str, human_like: bool = True, use_clipboard: bool = False) -> bool:
        """智能文字輸入，支援多語言和人性化模擬。

        功能說明：
        - 自動檢測文字類型並選擇最佳輸入方法
        - 支援英文直接輸入和中文剪貼簿輸入
        - 人性化模擬包含變化的打字速度和隨機暫停
        - 自動備份和恢復剪貼簿內容

        Args:
            text: 要輸入的文字內容，支援任何 Unicode 字符
            human_like: 是否使用人性化輸入模式，預設 True
                - True: 模擬真實打字節奏，包含隨機延遲和暫停
                - False: 快速輸入，適用於批量操作
            use_clipboard: 是否強制使用剪貼簿輸入，預設 False
                - True: 所有文字都通過剪貼簿輸入
                - False: 自動選擇最佳輸入方法

        輸入模式：
        1. 直接輸入：適用於英文、數字和基本符號
        2. 剪貼簿輸入：適用於中文、表情符號和特殊字符
        3. 混合模式：自動選擇最佳方式

        人性化特性：
        - 變化的打字速度（0.02-0.12秒間隔）
        - 隨機暫停模擬思考（10%機率，0.5秒）
        - 快速輸入模擬熟練操作（30%機率）

        Return value:
            True表示輸入成功，False表示輸入失敗
        """
        try:
            if state['is_set_target_window']:
                state['current_window'].activate()

            if human_like:
                await human_like_type(text, use_clipboard)
            else:
                if use_clipboard or not text.isascii():
                    prev_clip = pyperclip.paste()
                    pyperclip.copy(text)
                    pyautogui.hotkey('ctrl', 'v')
                    if prev_clip:
                        pyperclip.copy(prev_clip)
                else:
                    pyautogui.write(text)

            # 記錄巨集動作
            if macro_system.recording_macro:
                macro_system.add_action('type_text', text=text, human_like=human_like)

            return True
        except Exception as e:
            print(f"輸入文字失敗: {e}", file=sys.stderr)
            return False

    @mcp.tool()
    async def keyboard_press_keys(keys: List[str], hold_duration: float = 0.1) -> bool:
        """按下多個按鍵（支援組合鍵）

        Args:
            keys: 按鍵列表，例如 ['ctrl', 'c'] 或 ['alt', 'tab']
            hold_duration: 按鍵保持時間

        Return value:
            True表示成功
        """
        try:
            if state['is_set_target_window']:
                state['current_window'].activate()

            # 按下所有按鍵
            for key in keys:
                pyautogui.keyDown(key)
                await asyncio.sleep(0.01)  # 小延遲確保按鍵順序

            # 保持按鍵
            await asyncio.sleep(hold_duration)

            # 釋放所有按鍵（反向順序）
            for key in reversed(keys):
                pyautogui.keyUp(key)
                await asyncio.sleep(0.01)

            # 記錄巨集動作
            if macro_system.recording_macro:
                macro_system.add_action('press_keys', keys=keys, hold_duration=hold_duration)

            return True
        except Exception as e:
            print(f"按鍵操作失敗: {e}", file=sys.stderr)
            return False

    @mcp.tool()
    async def keyboard_hotkey(keys: List[str]) -> bool:
        """執行快捷鍵組合

        Args:
            keys: 快捷鍵列表，例如 ['ctrl', 'shift', 's']

        Return value:
            True表示成功
        """
        try:
            if state['is_set_target_window']:
                state['current_window'].activate()

            pyautogui.hotkey(*keys)

            # 記錄巨集動作
            if macro_system.recording_macro:
                macro_system.add_action('hotkey', keys=keys)

            return True
        except Exception as e:
            print(f"快捷鍵操作失敗: {e}", file=sys.stderr)
            return False

    @mcp.tool()
    async def scroll_advanced(direction: str, clicks: int = 3, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """高級滾輪操作

        Args:
            direction: 滾動方向 ('up', 'down', 'left', 'right')
            clicks: 滾動量
            x: 滾動位置X座標（可選）
            y: 滾動位置Y座標（可選）

        Return value:
            True表示成功
        """
        try:
            if state['is_set_target_window']:
                state['current_window'].activate()

            # 移動到指定位置（如果提供）
            if x is not None and y is not None:
                if state['is_set_target_window']:
                    abs_x = x + state['current_window'].left
                    abs_y = y + state['current_window'].top
                else:
                    abs_x = x
                    abs_y = y
                pyautogui.moveTo(abs_x, abs_y)

            # 執行滾動
            if direction == 'up':
                pyautogui.scroll(clicks)
            elif direction == 'down':
                pyautogui.scroll(-clicks)
            elif direction == 'left':
                pyautogui.hscroll(-clicks)
            elif direction == 'right':
                pyautogui.hscroll(clicks)
            else:
                return False

            # 記錄巨集動作
            if macro_system.recording_macro:
                macro_system.add_action('scroll_advanced', direction=direction, clicks=clicks, x=x, y=y)

            return True
        except Exception as e:
            print(f"滾動操作失敗: {e}", file=sys.stderr)
            return False

    # ==================== 新增：巨集系統 ====================

    @mcp.tool()
    async def macro_start_recording(name: str) -> bool:
        """開始錄製巨集

        Args:
            name: 巨集名稱

        Return value:
            True表示成功開始錄製
        """
        try:
            macro_system.start_recording(name)
            return True
        except Exception as e:
            print(f"開始錄製巨集失敗: {e}", file=sys.stderr)
            return False

    @mcp.tool()
    async def macro_stop_recording() -> bool:
        """停止錄製巨集

        Return value:
            True表示成功停止錄製
        """
        try:
            macro_system.stop_recording()
            return True
        except Exception as e:
            print(f"停止錄製巨集失敗: {e}", file=sys.stderr)
            return False

    @mcp.tool()
    async def macro_play(name: str, repeat_count: int = 1, delay_between_repeats: float = 1.0) -> bool:
        """播放巨集

        Args:
            name: 巨集名稱
            repeat_count: 重複次數
            delay_between_repeats: 重複之間的延遲時間

        Return value:
            True表示成功播放
        """
        try:
            macro_actions = macro_system.get_macro(name)
            if not macro_actions:
                print(f"找不到巨集: {name}", file=sys.stderr)
                return False

            for repeat in range(repeat_count):
                if repeat > 0:
                    await asyncio.sleep(delay_between_repeats)

                for action in macro_actions:
                    action_type = action['type']

                    if action_type == 'click_coordinate':
                        await mouse_click_coordinate(
                            action['x'], action['y'],
                            action.get('button', 'left'),
                            action.get('clicks', 1),
                            human_like=True
                        )
                    elif action_type == 'move_coordinate':
                        await mouse_move_coordinate(
                            action['x'], action['y'],
                            human_like=True,
                            duration=action.get('duration', 0.5)
                        )
                    elif action_type == 'drag_coordinate':
                        await mouse_drag_coordinate(
                            action['from_x'], action['from_y'],
                            action['to_x'], action['to_y'],
                            action.get('button', 'left'),
                            human_like=True,
                            duration=action.get('duration', 1.0)
                        )
                    elif action_type == 'type_text':
                        await keyboard_type_text(
                            action['text'],
                            human_like=action.get('human_like', True)
                        )
                    elif action_type == 'press_keys':
                        await keyboard_press_keys(
                            action['keys'],
                            action.get('hold_duration', 0.1)
                        )
                    elif action_type == 'hotkey':
                        await keyboard_hotkey(action['keys'])
                    elif action_type == 'scroll_advanced':
                        await scroll_advanced(
                            action['direction'],
                            action.get('clicks', 3),
                            action.get('x'),
                            action.get('y')
                        )
                    elif action_type == 'wait':
                        await asyncio.sleep(action.get('duration', 1.0))

                    # 動作之間的小延遲
                    await asyncio.sleep(0.1)

            return True
        except Exception as e:
            print(f"播放巨集失敗: {e}", file=sys.stderr)
            return False

    @mcp.tool()
    async def macro_list() -> List[str]:
        """列出所有可用的巨集

        Return value:
            巨集名稱列表
        """
        return macro_system.list_macros()

    @mcp.tool()
    async def macro_delete(name: str) -> bool:
        """刪除指定的巨集

        Args:
            name: 要刪除的巨集名稱

        Return value:
            True表示成功刪除
        """
        return macro_system.delete_macro(name)

    @mcp.tool()
    async def macro_get_info(name: str) -> Optional[Dict[str, Any]]:
        """獲取巨集詳細信息

        Args:
            name: 巨集名稱

        Return value:
            巨集信息字典，包含動作列表和統計信息
        """
        macro_actions = macro_system.get_macro(name)
        if not macro_actions:
            return None

        action_types = {}
        for action in macro_actions:
            action_type = action['type']
            action_types[action_type] = action_types.get(action_type, 0) + 1

        return {
            'name': name,
            'action_count': len(macro_actions),
            'action_types': action_types,
            'actions': macro_actions
        }

    # ==================== 新增：遊戲專用功能 ====================

    @mcp.tool()
    async def get_pixel_color(x: int, y: int) -> Dict[str, Any]:
        """獲取指定座標的像素顏色

        Args:
            x: X座標
            y: Y座標

        Return value:
            包含RGB值和十六進制顏色的字典
        """
        try:
            if state['is_set_target_window']:
                abs_x = x + state['current_window'].left
                abs_y = y + state['current_window'].top
            else:
                abs_x = x
                abs_y = y

            # 獲取像素顏色
            screenshot = pyautogui.screenshot()
            pixel = screenshot.getpixel((abs_x, abs_y))

            return {
                'x': x,
                'y': y,
                'rgb': {'r': pixel[0], 'g': pixel[1], 'b': pixel[2]},
                'hex': f"#{pixel[0]:02x}{pixel[1]:02x}{pixel[2]:02x}"
            }
        except Exception as e:
            print(f"獲取像素顏色失敗: {e}", file=sys.stderr)
            return {'error': str(e)}

    @mcp.tool()
    async def wait_for_pixel_color(x: int, y: int, target_color: str, timeout: float = 10.0,
                                 check_interval: float = 0.1) -> bool:
        """等待指定座標的像素變為目標顏色

        Args:
            x: X座標
            y: Y座標
            target_color: 目標顏色（十六進制格式，如 "#FF0000"）
            timeout: 超時時間（秒）
            check_interval: 檢查間隔（秒）

        Return value:
            True表示找到目標顏色，False表示超時
        """
        try:
            start_time = time.time()
            target_color = target_color.lower().replace('#', '')

            while time.time() - start_time < timeout:
                pixel_info = await get_pixel_color(x, y)
                if 'hex' in pixel_info:
                    current_color = pixel_info['hex'].lower().replace('#', '')
                    if current_color == target_color:
                        return True

                await asyncio.sleep(check_interval)

            return False
        except Exception as e:
            print(f"等待像素顏色失敗: {e}", file=sys.stderr)
            return False

    @mcp.tool()
    async def rapid_click(x: int, y: int, clicks: int = 10, interval: float = 0.05,
                         button: str = 'left') -> bool:
        """快速連續點擊（適用於遊戲）

        Args:
            x: X座標
            y: Y座標
            clicks: 點擊次數
            interval: 點擊間隔（秒）
            button: 滑鼠按鈕

        Return value:
            True表示成功
        """
        try:
            if state['is_set_target_window']:
                state['current_window'].activate()
                abs_x = x + state['current_window'].left
                abs_y = y + state['current_window'].top
            else:
                abs_x = x
                abs_y = y

            # 移動到目標位置
            pyautogui.moveTo(abs_x, abs_y)

            # 快速點擊
            for i in range(clicks):
                pyautogui.click(button=button)
                if i < clicks - 1:  # 最後一次點擊後不需要等待
                    await asyncio.sleep(interval)

            return True
        except Exception as e:
            print(f"快速點擊失敗: {e}", file=sys.stderr)
            return False

    @mcp.tool()
    async def combo_sequence(actions: List[Dict[str, Any]], human_like: bool = False) -> bool:
        """執行連招序列（適用於遊戲）

        Args:
            actions: 動作序列，每個動作包含type和相關參數
            human_like: 是否使用人性化時間

        動作類型:
        - click: {type: "click", x: int, y: int, button: str}
        - key: {type: "key", keys: List[str]}
        - wait: {type: "wait", duration: float}
        - type: {type: "type", text: str}

        Return value:
            True表示成功執行
        """
        try:
            for action in actions:
                action_type = action.get('type')

                if action_type == 'click':
                    x = action.get('x', 0)
                    y = action.get('y', 0)
                    button = action.get('button', 'left')
                    await mouse_click_coordinate(x, y, button, human_like=human_like)

                elif action_type == 'key':
                    keys = action.get('keys', [])
                    await keyboard_hotkey(keys)

                elif action_type == 'wait':
                    duration = action.get('duration', 0.1)
                    await asyncio.sleep(duration)

                elif action_type == 'type':
                    text = action.get('text', '')
                    await keyboard_type_text(text, human_like=human_like)

                # 動作間的基本延遲
                base_delay = 0.05 if not human_like else random.uniform(0.05, 0.15)
                await asyncio.sleep(base_delay)

            return True
        except Exception as e:
            print(f"執行連招序列失敗: {e}", file=sys.stderr)
            return False

    # ==================== 新增：視窗管理功能 ====================

    @mcp.tool()
    async def list_windows() -> List[Dict[str, Any]]:
        """列出所有可見視窗

        Return value:
            視窗信息列表
        """
        try:
            windows = gw.getAllWindows()
            window_list = []

            for window in windows:
                if window.title and window.visible:
                    window_list.append({
                        'title': window.title,
                        'left': window.left,
                        'top': window.top,
                        'width': window.width,
                        'height': window.height,
                        'is_active': window == gw.getActiveWindow(),
                        'is_maximized': window.isMaximized if hasattr(window, 'isMaximized') else False
                    })

            return window_list
        except Exception as e:
            print(f"列出視窗失敗: {e}", file=sys.stderr)
            return []

    @mcp.tool()
    async def switch_to_window(title_pattern: str) -> bool:
        """切換到指定視窗

        Args:
            title_pattern: 視窗標題模式（支援部分匹配）

        Return value:
            True表示成功切換
        """
        try:
            windows = gw.getWindowsWithTitle(title_pattern)
            if not windows:
                # 嘗試模糊匹配
                all_windows = gw.getAllWindows()
                for window in all_windows:
                    if title_pattern.lower() in window.title.lower():
                        windows = [window]
                        break

            if windows:
                target_window = windows[0]
                target_window.activate()

                # 更新狀態中的視窗變量
                state['current_window'] = target_window
                state['is_set_target_window'] = True

                logger.info(f"成功切換到視窗: {target_window.title}")
                return True

            return False
        except Exception as e:
            print(f"切換視窗失敗: {e}", file=sys.stderr)
            return False

    @mcp.tool()
    async def get_screen_size() -> Dict[str, int]:
        """獲取螢幕尺寸

        Return value:
            包含寬度和高度的字典
        """
        width, height = pyautogui.size()
        return {'width': width, 'height': height}

    @mcp.tool()
    async def take_screenshot_region(x: int, y: int, width: int, height: int) -> Image:
        """截取指定區域的螢幕截圖

        Args:
            x: 起始X座標
            y: 起始Y座標
            width: 寬度
            height: 高度

        Return value:
            截圖圖像
        """
        try:
            if state['is_set_target_window']:
                abs_x = x + state['current_window'].left
                abs_y = y + state['current_window'].top
            else:
                abs_x = x
                abs_y = y

            screenshot = pyautogui.screenshot(region=(abs_x, abs_y, width, height))

            # 轉換為MCP Image格式
            buffered = io.BytesIO()
            screenshot.save(buffered, format='PNG')

            return Image(data=buffered.getvalue(), format="png")
        except Exception as e:
            print(f"區域截圖失敗: {e}", file=sys.stderr)
            # 返回空圖像
            return Image(data=b'', format="png")

    # ==================== 新增：實用工具 ====================

    @mcp.tool()
    async def create_predefined_macro(macro_type: str, name: str, **kwargs) -> bool:
        """創建預定義的巨集模板

        Args:
            macro_type: 巨集類型 ('copy_paste', 'alt_tab', 'save_file', 'refresh_page', 'close_window')
            name: 巨集名稱
            **kwargs: 額外參數

        Return value:
            True表示成功創建
        """
        try:
            actions = []

            if macro_type == 'copy_paste':
                actions = [
                    {'type': 'hotkey', 'keys': ['ctrl', 'c']},
                    {'type': 'wait', 'duration': 0.1},
                    {'type': 'hotkey', 'keys': ['ctrl', 'v']}
                ]
            elif macro_type == 'alt_tab':
                actions = [
                    {'type': 'hotkey', 'keys': ['alt', 'tab']}
                ]
            elif macro_type == 'save_file':
                actions = [
                    {'type': 'hotkey', 'keys': ['ctrl', 's']}
                ]
            elif macro_type == 'refresh_page':
                actions = [
                    {'type': 'hotkey', 'keys': ['f5']}
                ]
            elif macro_type == 'close_window':
                actions = [
                    {'type': 'hotkey', 'keys': ['alt', 'f4']}
                ]
            else:
                return False

            # 將動作添加到巨集系統
            macro_system.macros[name] = actions
            return True

        except Exception as e:
            print(f"創建預定義巨集失敗: {e}", file=sys.stderr)
            return False

    @mcp.tool()
    async def get_system_info() -> Dict[str, Any]:
        """獲取系統信息

        Return value:
            系統信息字典
        """
        try:
            screen_width, screen_height = pyautogui.size()
            mouse_x, mouse_y = pyautogui.position()

            return {
                'screen_size': {'width': screen_width, 'height': screen_height},
                'mouse_position': {'x': mouse_x, 'y': mouse_y},
                'current_window': {
                    'title': state['current_window'].title if state['current_window'] else None,
                    'size': {
                        'width': state['current_window'].width if state['current_window'] else None,
                        'height': state['current_window'].height if state['current_window'] else None
                    },
                    'position': {
                        'left': state['current_window'].left if state['current_window'] else None,
                        'top': state['current_window'].top if state['current_window'] else None
                    }
                } if state['current_window'] else None,
                'target_window_set': state['is_set_target_window'],
                'available_keys': len(pyautogui.KEYBOARD_KEYS),
                'macro_count': len(macro_system.macros),
                'recording_macro': macro_system.recording_macro
            }
        except Exception as e:
            print(f"獲取系統信息失敗: {e}", file=sys.stderr)
            return {'error': str(e)}

    # ==================== 新增：高FPS截圖功能 ====================

    if HIGH_FPS_AVAILABLE:
        @mcp.tool()
        async def start_high_fps_capture(target_fps: int = 180, method: str = "mss",
                                       region: Optional[List[int]] = None,
                                       enable_compression: bool = True) -> bool:
            """啟動高FPS截圖模式，專為競技遊戲優化。

            功能說明：
            - 支援高達180FPS的截圖頻率
            - 使用硬體加速和多線程優化
            - 專為競技遊戲場景設計
            - 支援動作檢測和智能快取

            Args:
                target_fps: 目標幀率，預設180FPS
                method: 截圖方法 ['mss', 'win32', 'opencv', 'pyautogui']
                region: 截圖區域 [x, y, width, height]，None表示全螢幕
                enable_compression: 是否啟用圖像壓縮

            適用場景：
            - 180Hz競技遊戲監控
            - 高頻率螢幕分析
            - 實時遊戲輔助
            - 性能測試和調優

            Return value:
                True表示啟動成功，False表示啟動失敗
            """
            global high_fps_capture, frame_rate_sync, low_latency_processor

            try:
                if high_fps_capture and high_fps_capture.is_running:
                    logger.warning("高FPS截圖已在運行中")
                    return False

                # 驗證參數
                if target_fps <= 0 or target_fps > 300:
                    logger.error(f"無效的目標FPS: {target_fps}")
                    return False

                # 創建配置
                try:
                    capture_method = CaptureMethod(method)
                except ValueError:
                    logger.error(f"無效的截圖方法: {method}")
                    return False

                config = CaptureConfig(
                    method=capture_method,
                    target_fps=target_fps,
                    compression_quality=85 if enable_compression else 100,
                    use_hardware_acceleration=True,
                    enable_region_optimization=True
                )

                # 轉換區域格式
                capture_region = None
                if region and len(region) == 4:
                    capture_region = tuple(region)

                # 創建高FPS截圖實例
                high_fps_capture = GameOptimizedCapture(config)
                frame_rate_sync = FrameRateSync(target_fps)
                low_latency_processor = LowLatencyProcessor()

                # 啟動截圖
                high_fps_capture.start_capture(capture_region)

                logger.info(f"高FPS截圖已啟動: {target_fps}FPS, 方法: {method}")
                return True

            except Exception as e:
                logger.error(f"啟動高FPS截圖失敗: {e}")
                return False

        @mcp.tool()
        async def stop_high_fps_capture() -> bool:
            """停止高FPS截圖模式。

            Return value:
                True表示停止成功，False表示停止失敗
            """
            global high_fps_capture

            try:
                if high_fps_capture:
                    high_fps_capture.stop_capture()
                    high_fps_capture = None
                    logger.info("高FPS截圖已停止")
                    return True
                else:
                    logger.warning("高FPS截圖未在運行")
                    return False

            except Exception as e:
                logger.error(f"停止高FPS截圖失敗: {e}")
                return False

        @mcp.tool()
        async def get_high_fps_frame() -> Optional[Image]:
            """獲取最新的高FPS截圖幀。

            功能說明：
            - 獲取最新捕獲的高質量幀
            - 支援低延遲訪問（<5ms）
            - 自動處理幀同步
            - 包含性能統計信息

            使用場景：
            - 實時遊戲分析
            - 快速反應系統
            - 性能監控
            - 競技遊戲輔助

            Return value:
                最新的截圖幀，如果沒有可用幀則返回None
            """
            global high_fps_capture

            try:
                if not high_fps_capture or not high_fps_capture.is_running:
                    logger.warning("高FPS截圖未啟動")
                    return None

                # 獲取最新幀
                frame_data = high_fps_capture.get_latest_frame(timeout=0.01)

                if frame_data is None:
                    return None

                # 轉換為MCP Image格式
                if len(frame_data.image.shape) == 3:
                    pil_image = Image.fromarray(frame_data.image, 'RGB')
                else:
                    pil_image = Image.fromarray(frame_data.image)

                buffer = io.BytesIO()
                pil_image.save(buffer, format='PNG')

                return Image(data=buffer.getvalue(), format="png")

            except Exception as e:
                logger.error(f"獲取高FPS幀失敗: {e}")
                return None

        @mcp.tool()
        async def get_fps_stats() -> Dict[str, Any]:
            """獲取高FPS截圖的性能統計。

            Return value:
                包含FPS、延遲、幀數等統計信息的字典
            """
            global high_fps_capture, frame_rate_sync

            try:
                stats = {}

                if high_fps_capture:
                    capture_stats = high_fps_capture.get_stats()
                    stats.update(capture_stats)

                if frame_rate_sync:
                    fps_stats = frame_rate_sync.get_frame_time_stats()
                    stats['actual_fps'] = frame_rate_sync.get_actual_fps()
                    stats['frame_time_ms'] = fps_stats

                # 添加系統性能信息
                import psutil
                stats['cpu_percent'] = psutil.cpu_percent()
                stats['memory_percent'] = psutil.virtual_memory().percent

                return stats

            except Exception as e:
                logger.error(f"獲取FPS統計失敗: {e}")
                return {'error': str(e)}

        @mcp.tool()
        async def add_game_region(name: str, x: int, y: int, width: int, height: int, priority: int = 1) -> bool:
            """添加遊戲特定監控區域。

            Args:
                name: 區域名稱
                x: 起始X座標
                y: 起始Y座標
                width: 寬度
                height: 高度
                priority: 優先級（1-10，數字越大優先級越高）

            Return value:
                True表示添加成功，False表示添加失敗
            """
            global high_fps_capture

            try:
                if not high_fps_capture:
                    logger.error("高FPS截圖未啟動")
                    return False

                if not isinstance(high_fps_capture, GameOptimizedCapture):
                    logger.error("當前截圖模式不支援遊戲區域")
                    return False

                region = (x, y, width, height)
                high_fps_capture.add_game_region(name, region, priority)

                logger.info(f"添加遊戲區域: {name} {region}")
                return True

            except Exception as e:
                logger.error(f"添加遊戲區域失敗: {e}")
                return False

        @mcp.tool()
        async def enable_low_latency_mode(max_latency_ms: float = 5.0) -> bool:
            """啟用低延遲模式，專為競技遊戲優化。

            功能說明：
            - 最小化處理延遲
            - 優化記憶體使用
            - 減少系統開銷
            - 提高響應速度

            Args:
                max_latency_ms: 最大允許延遲（毫秒），預設5ms

            適用場景：
            - 競技FPS遊戲
            - 實時策略遊戲
            - 格鬥遊戲
            - 電競比賽

            Return value:
                True表示啟用成功，False表示啟用失敗
            """
            global high_fps_capture, low_latency_processor

            try:
                if not high_fps_capture:
                    logger.error("高FPS截圖未啟動")
                    return False

                # 創建低延遲處理器
                max_processing_time = max_latency_ms / 1000.0  # 轉換為秒
                low_latency_processor = LowLatencyProcessor(max_processing_time)

                # 優化截圖配置
                if hasattr(high_fps_capture, 'config'):
                    high_fps_capture.config.max_queue_size = 2  # 減少隊列大小
                    high_fps_capture.config.compression_quality = 100  # 關閉壓縮
                    high_fps_capture.config.cache_duration = 0.01  # 減少快取時間

                logger.info(f"低延遲模式已啟用，最大延遲: {max_latency_ms}ms")
                return True

            except Exception as e:
                logger.error(f"啟用低延遲模式失敗: {e}")
                return False

        @mcp.tool()
        async def capture_game_region_fast(region_name: str) -> Optional[Image]:
            """快速捕獲遊戲特定區域。

            功能說明：
            - 極低延遲區域截圖
            - 專為遊戲UI元素設計
            - 支援預定義遊戲區域
            - 自動優化處理

            Args:
                region_name: 區域名稱（需要先用add_game_region添加）

            常用區域：
            - "minimap": 小地圖
            - "health_bar": 血條
            - "ammo_counter": 彈藥計數
            - "crosshair": 準星區域
            - "chat": 聊天區域

            Return value:
                區域截圖，如果區域不存在則返回None
            """
            global high_fps_capture

            try:
                if not high_fps_capture:
                    logger.error("高FPS截圖未啟動")
                    return None

                if not isinstance(high_fps_capture, GameOptimizedCapture):
                    logger.error("當前截圖模式不支援遊戲區域")
                    return None

                # 檢查區域是否存在
                if region_name not in high_fps_capture.game_regions:
                    logger.error(f"遊戲區域不存在: {region_name}")
                    return None

                region_info = high_fps_capture.game_regions[region_name]
                region = region_info['region']

                # 快速截圖指定區域
                image_array = high_fps_capture.capture_func(region)

                # 轉換為MCP Image格式
                if len(image_array.shape) == 3:
                    pil_image = Image.fromarray(image_array, 'RGB')
                else:
                    pil_image = Image.fromarray(image_array)

                buffer = io.BytesIO()
                pil_image.save(buffer, format='PNG')

                logger.debug(f"快速捕獲區域: {region_name} {region}")
                return Image(data=buffer.getvalue(), format="png")

            except Exception as e:
                logger.error(f"快速捕獲遊戲區域失敗: {e}")
                return None

        @mcp.tool()
        async def detect_game_fps() -> Dict[str, Any]:
            """檢測當前遊戲的實際幀率。

            功能說明：
            - 自動檢測遊戲FPS
            - 分析幀變化模式
            - 提供穩定性評估
            - 建議最佳截圖設置

            Return value:
                包含檢測到的FPS和相關統計信息的字典
            """
            global high_fps_capture

            try:
                if not high_fps_capture:
                    logger.error("高FPS截圖未啟動")
                    return {'error': '高FPS截圖未啟動'}

                # 收集幀數據進行分析
                frame_samples = []
                sample_duration = 2.0  # 採樣2秒
                start_time = time.time()

                while time.time() - start_time < sample_duration:
                    frame_data = high_fps_capture.get_latest_frame(timeout=0.1)
                    if frame_data:
                        frame_samples.append(frame_data.timestamp)
                    await asyncio.sleep(0.01)

                if len(frame_samples) < 10:
                    return {'error': '採樣數據不足'}

                # 分析FPS
                time_diffs = [frame_samples[i] - frame_samples[i-1]
                             for i in range(1, len(frame_samples))]

                if not time_diffs:
                    return {'error': '無法計算幀間隔'}

                import statistics
                avg_interval = statistics.mean(time_diffs)
                std_interval = statistics.stdev(time_diffs) if len(time_diffs) > 1 else 0

                detected_fps = 1.0 / avg_interval if avg_interval > 0 else 0
                stability = max(0, 1 - (std_interval / avg_interval)) if avg_interval > 0 else 0

                # 生成建議
                recommendations = []
                if detected_fps > 165:
                    recommendations.append("建議使用180FPS截圖模式")
                elif detected_fps > 120:
                    recommendations.append("建議使用144FPS截圖模式")
                elif detected_fps > 90:
                    recommendations.append("建議使用120FPS截圖模式")
                else:
                    recommendations.append("建議使用60FPS截圖模式")

                if stability < 0.8:
                    recommendations.append("遊戲幀率不穩定，建議降低畫質設置")

                return {
                    'detected_fps': detected_fps,
                    'stability': stability,
                    'avg_frame_time_ms': avg_interval * 1000,
                    'frame_time_std_ms': std_interval * 1000,
                    'sample_count': len(frame_samples),
                    'sample_duration': sample_duration,
                    'recommendations': recommendations
                }

            except Exception as e:
                logger.error(f"檢測遊戲FPS失敗: {e}")
                return {'error': str(e)}

        @mcp.tool()
        async def optimize_for_game(game_name: str) -> bool:
            """為特定遊戲優化截圖設置。

            支援的遊戲：
            - "cs2": Counter-Strike 2
            - "valorant": Valorant
            - "lol": League of Legends
            - "overwatch": Overwatch 2
            - "apex": Apex Legends
            - "fortnite": Fortnite

            Args:
                game_name: 遊戲名稱

            Return value:
                True表示優化成功，False表示優化失敗
            """
            global high_fps_capture

            try:
                if not high_fps_capture:
                    logger.error("高FPS截圖未啟動")
                    return False

                game_configs = {
                    "cs2": {
                        "target_fps": 180,
                        "compression": False,
                        "regions": {
                            "crosshair": (960, 540, 100, 100),
                            "minimap": (1670, 850, 250, 250),
                            "health": (50, 950, 200, 50),
                            "ammo": (1700, 950, 200, 50)
                        }
                    },
                    "valorant": {
                        "target_fps": 180,
                        "compression": False,
                        "regions": {
                            "crosshair": (960, 540, 80, 80),
                            "minimap": (1650, 100, 270, 270),
                            "health": (100, 950, 150, 40),
                            "abilities": (1400, 950, 400, 80)
                        }
                    },
                    "lol": {
                        "target_fps": 144,
                        "compression": True,
                        "regions": {
                            "minimap": (1650, 750, 270, 270),
                            "health": (50, 950, 300, 100),
                            "items": (1400, 950, 400, 100)
                        }
                    },
                    "overwatch": {
                        "target_fps": 165,
                        "compression": False,
                        "regions": {
                            "crosshair": (960, 540, 120, 120),
                            "health": (960, 900, 200, 80),
                            "abilities": (800, 950, 320, 70)
                        }
                    }
                }

                if game_name not in game_configs:
                    logger.error(f"不支援的遊戲: {game_name}")
                    return False

                config = game_configs[game_name]

                # 應用配置
                if hasattr(high_fps_capture, 'config'):
                    high_fps_capture.config.target_fps = config["target_fps"]
                    if not config["compression"]:
                        high_fps_capture.config.compression_quality = 100

                # 添加遊戲區域
                if isinstance(high_fps_capture, GameOptimizedCapture):
                    for region_name, region_coords in config["regions"].items():
                        high_fps_capture.add_game_region(region_name, region_coords, priority=8)

                logger.info(f"已為遊戲 {game_name} 優化截圖設置")
                return True

            except Exception as e:
                logger.error(f"遊戲優化失敗: {e}")
                return False
