# -*- coding: utf-8 -*-
"""
高FPS截圖引擎
專為180Hz競技遊戲環境優化的高性能截圖系統
"""

import os
import sys
import time
import threading
import asyncio
import queue
import logging
from typing import Optional, Tuple, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
from PIL import Image
import io
import base64
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

# 嘗試導入高性能截圖庫
try:
    import mss  # 高性能截圖庫
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    
try:
    import cv2  # OpenCV 用於圖像處理
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import win32gui
    import win32ui
    import win32con
    import win32api
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False

logger = logging.getLogger(__name__)

class CaptureMethod(Enum):
    """截圖方法枚舉"""
    MSS = "mss"              # 最快的跨平台方法
    WIN32 = "win32"          # Windows 原生 API
    OPENCV = "opencv"        # OpenCV 方法
    PYAUTOGUI = "pyautogui"  # 備用方法

@dataclass
class CaptureConfig:
    """截圖配置"""
    method: CaptureMethod = CaptureMethod.MSS
    target_fps: int = 180
    max_queue_size: int = 10
    compression_quality: int = 85
    use_hardware_acceleration: bool = True
    enable_region_optimization: bool = True
    cache_duration: float = 0.1  # 100ms 快取
    worker_threads: int = 4

@dataclass
class FrameData:
    """幀數據結構"""
    timestamp: float
    image: np.ndarray
    region: Optional[Tuple[int, int, int, int]] = None
    frame_id: int = 0
    processing_time: float = 0.0

class HighFPSCapture:
    """高FPS截圖引擎"""
    
    def __init__(self, config: CaptureConfig = None):
        self.config = config or CaptureConfig()
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=self.config.max_queue_size)
        self.processed_queue = queue.Queue(maxsize=self.config.max_queue_size)
        self.capture_thread = None
        self.processing_threads = []
        self.frame_counter = 0
        self.last_capture_time = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # 性能統計
        self.stats = {
            'frames_captured': 0,
            'frames_processed': 0,
            'frames_dropped': 0,
            'avg_capture_time': 0.0,
            'avg_processing_time': 0.0,
            'current_fps': 0.0
        }
        
        # 初始化截圖方法
        self._init_capture_method()
        
        # 線程池
        self.executor = ThreadPoolExecutor(max_workers=self.config.worker_threads)
        
        logger.info(f"高FPS截圖引擎初始化完成，方法: {self.config.method.value}")

    def _init_capture_method(self):
        """初始化截圖方法"""
        if self.config.method == CaptureMethod.MSS and MSS_AVAILABLE:
            self.sct = mss.mss()
            self.capture_func = self._capture_mss
            logger.info("使用 MSS 截圖方法")
        elif self.config.method == CaptureMethod.WIN32 and WIN32_AVAILABLE:
            self.capture_func = self._capture_win32
            logger.info("使用 Win32 截圖方法")
        elif self.config.method == CaptureMethod.OPENCV and CV2_AVAILABLE:
            self.capture_func = self._capture_opencv
            logger.info("使用 OpenCV 截圖方法")
        else:
            # 備用方法
            import pyautogui
            self.capture_func = self._capture_pyautogui
            logger.warning("使用 PyAutoGUI 備用截圖方法")

    def _capture_mss(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """MSS 截圖方法（最快）"""
        if region:
            monitor = {
                "top": region[1],
                "left": region[0], 
                "width": region[2],
                "height": region[3]
            }
        else:
            monitor = self.sct.monitors[1]  # 主顯示器
        
        screenshot = self.sct.grab(monitor)
        return np.array(screenshot)

    def _capture_win32(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Win32 API 截圖方法"""
        if region:
            x, y, w, h = region
        else:
            x, y = 0, 0
            w = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
            h = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
        
        hdesktop = win32gui.GetDesktopWindow()
        desktop_dc = win32gui.GetWindowDC(hdesktop)
        img_dc = win32ui.CreateDCFromHandle(desktop_dc)
        mem_dc = img_dc.CreateCompatibleDC()
        
        screenshot = win32ui.CreateBitmap()
        screenshot.CreateCompatibleBitmap(img_dc, w, h)
        mem_dc.SelectObject(screenshot)
        mem_dc.BitBlt((0, 0), (w, h), img_dc, (x, y), win32con.SRCCOPY)
        
        bmpinfo = screenshot.GetInfo()
        bmpstr = screenshot.GetBitmapBits(True)
        
        img = np.frombuffer(bmpstr, dtype='uint8')
        img.shape = (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)
        
        # 清理資源
        mem_dc.DeleteDC()
        win32gui.ReleaseDC(hdesktop, desktop_dc)
        
        return img[:, :, :3]  # 移除 alpha 通道

    def _capture_opencv(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """OpenCV 截圖方法"""
        # OpenCV 不直接支援螢幕截圖，這裡使用其他方法然後轉換
        import pyautogui
        if region:
            screenshot = pyautogui.screenshot(region=region)
        else:
            screenshot = pyautogui.screenshot()
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    def _capture_pyautogui(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """PyAutoGUI 截圖方法（備用）"""
        import pyautogui
        if region:
            screenshot = pyautogui.screenshot(region=region)
        else:
            screenshot = pyautogui.screenshot()
        return np.array(screenshot)

    def start_capture(self, region: Optional[Tuple[int, int, int, int]] = None):
        """開始高FPS截圖"""
        if self.is_running:
            logger.warning("截圖已在運行中")
            return
        
        self.is_running = True
        self.region = region
        
        # 啟動截圖線程
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            name="HighFPSCapture",
            daemon=True
        )
        self.capture_thread.start()
        
        # 啟動處理線程
        for i in range(self.config.worker_threads):
            thread = threading.Thread(
                target=self._processing_loop,
                name=f"FrameProcessor-{i}",
                daemon=True
            )
            thread.start()
            self.processing_threads.append(thread)
        
        logger.info(f"高FPS截圖已啟動，目標FPS: {self.config.target_fps}")

    def stop_capture(self):
        """停止截圖"""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        
        # 清空隊列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.processed_queue.empty():
            try:
                self.processed_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("高FPS截圖已停止")

    def _capture_loop(self):
        """截圖循環"""
        target_interval = 1.0 / self.config.target_fps
        
        while self.is_running:
            start_time = time.time()
            
            try:
                # 截圖
                image_array = self.capture_func(self.region)
                
                # 創建幀數據
                frame_data = FrameData(
                    timestamp=start_time,
                    image=image_array,
                    region=self.region,
                    frame_id=self.frame_counter
                )
                
                # 添加到隊列
                try:
                    self.frame_queue.put_nowait(frame_data)
                    self.stats['frames_captured'] += 1
                except queue.Full:
                    # 隊列滿，丟棄最舊的幀
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame_data)
                        self.stats['frames_dropped'] += 1
                    except queue.Empty:
                        pass
                
                self.frame_counter += 1
                
                # 更新FPS統計
                self._update_fps_stats()
                
                # 計算處理時間
                processing_time = time.time() - start_time
                self.stats['avg_capture_time'] = (
                    self.stats['avg_capture_time'] * 0.9 + processing_time * 0.1
                )
                
                # 控制幀率
                elapsed = time.time() - start_time
                sleep_time = max(0, target_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"截圖錯誤: {e}")
                time.sleep(0.01)  # 短暫延遲避免錯誤循環

    def _processing_loop(self):
        """幀處理循環"""
        while self.is_running:
            try:
                # 獲取幀數據
                frame_data = self.frame_queue.get(timeout=0.1)
                
                # 處理幀
                processed_frame = self._process_frame(frame_data)
                
                # 添加到處理完成隊列
                try:
                    self.processed_queue.put_nowait(processed_frame)
                    self.stats['frames_processed'] += 1
                except queue.Full:
                    # 處理隊列滿，丟棄
                    self.stats['frames_dropped'] += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"幀處理錯誤: {e}")

    def _process_frame(self, frame_data: FrameData) -> FrameData:
        """處理單個幀"""
        start_time = time.time()
        
        try:
            # 這裡可以添加圖像處理邏輯
            # 例如：壓縮、格式轉換、特徵提取等
            
            # 如果需要壓縮
            if self.config.compression_quality < 100:
                # 轉換為 PIL Image 進行壓縮
                if frame_data.image.shape[2] == 4:  # RGBA
                    pil_image = Image.fromarray(frame_data.image, 'RGBA')
                else:  # RGB
                    pil_image = Image.fromarray(frame_data.image, 'RGB')
                
                # 壓縮
                buffer = io.BytesIO()
                pil_image.save(buffer, format='JPEG', quality=self.config.compression_quality)
                
                # 轉回 numpy array
                buffer.seek(0)
                compressed_image = Image.open(buffer)
                frame_data.image = np.array(compressed_image)
            
            frame_data.processing_time = time.time() - start_time
            
        except Exception as e:
            logger.error(f"幀處理失敗: {e}")
        
        return frame_data

    def _update_fps_stats(self):
        """更新FPS統計"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.stats['current_fps'] = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time

    def get_latest_frame(self, timeout: float = 0.1) -> Optional[FrameData]:
        """獲取最新幀"""
        try:
            return self.processed_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """獲取性能統計"""
        return self.stats.copy()

    def __del__(self):
        """析構函數"""
        if self.is_running:
            self.stop_capture()

        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


class GameOptimizedCapture(HighFPSCapture):
    """遊戲優化的截圖引擎"""

    def __init__(self, config: CaptureConfig = None):
        super().__init__(config)
        self.game_regions = {}  # 遊戲特定區域
        self.motion_detection_enabled = True
        self.last_frame_hash = None
        self.frame_diff_threshold = 0.05

    def add_game_region(self, name: str, region: Tuple[int, int, int, int], priority: int = 1):
        """添加遊戲特定區域"""
        self.game_regions[name] = {
            'region': region,
            'priority': priority,
            'last_update': 0
        }
        logger.info(f"添加遊戲區域: {name} {region}")

    def enable_motion_detection(self, threshold: float = 0.05):
        """啟用動作檢測"""
        self.motion_detection_enabled = True
        self.frame_diff_threshold = threshold
        logger.info(f"啟用動作檢測，閾值: {threshold}")

    def _process_frame(self, frame_data: FrameData) -> FrameData:
        """遊戲優化的幀處理"""
        start_time = time.time()

        try:
            # 動作檢測
            if self.motion_detection_enabled:
                frame_hash = self._calculate_frame_hash(frame_data.image)
                if self.last_frame_hash:
                    similarity = self._calculate_similarity(frame_hash, self.last_frame_hash)
                    if similarity > (1 - self.frame_diff_threshold):
                        # 幀變化很小，可以跳過處理
                        frame_data.processing_time = time.time() - start_time
                        return frame_data
                self.last_frame_hash = frame_hash

            # 調用父類處理
            frame_data = super()._process_frame(frame_data)

        except Exception as e:
            logger.error(f"遊戲幀處理失敗: {e}")

        return frame_data

    def _calculate_frame_hash(self, image: np.ndarray) -> str:
        """計算幀雜湊值"""
        # 縮小圖像以加快計算
        small_image = cv2.resize(image, (64, 64)) if CV2_AVAILABLE else image[::8, ::8]
        return hashlib.md5(small_image.tobytes()).hexdigest()

    def _calculate_similarity(self, hash1: str, hash2: str) -> float:
        """計算兩個雜湊值的相似度"""
        if hash1 == hash2:
            return 1.0

        # 簡單的漢明距離計算
        diff = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        return 1.0 - (diff / len(hash1))


class FrameRateSync:
    """幀率同步器"""

    def __init__(self, target_fps: int = 180):
        self.target_fps = target_fps
        self.target_interval = 1.0 / target_fps
        self.last_frame_time = 0
        self.frame_times = []
        self.max_history = 60  # 保留60幀的歷史

    def wait_for_next_frame(self) -> float:
        """等待下一幀時間"""
        current_time = time.time()

        if self.last_frame_time > 0:
            elapsed = current_time - self.last_frame_time
            sleep_time = max(0, self.target_interval - elapsed)

            if sleep_time > 0:
                time.sleep(sleep_time)
                current_time = time.time()

        # 記錄幀時間
        if self.last_frame_time > 0:
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)

            if len(self.frame_times) > self.max_history:
                self.frame_times.pop(0)

        self.last_frame_time = current_time
        return current_time

    def get_actual_fps(self) -> float:
        """獲取實際FPS"""
        if len(self.frame_times) < 2:
            return 0.0

        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    def get_frame_time_stats(self) -> Dict[str, float]:
        """獲取幀時間統計"""
        if not self.frame_times:
            return {'min': 0, 'max': 0, 'avg': 0, 'std': 0}

        import statistics
        return {
            'min': min(self.frame_times) * 1000,  # 轉換為毫秒
            'max': max(self.frame_times) * 1000,
            'avg': statistics.mean(self.frame_times) * 1000,
            'std': statistics.stdev(self.frame_times) * 1000 if len(self.frame_times) > 1 else 0
        }


class LowLatencyProcessor:
    """低延遲處理器"""

    def __init__(self, max_processing_time: float = 0.005):  # 5ms 最大處理時間
        self.max_processing_time = max_processing_time
        self.processing_queue = queue.Queue(maxsize=1)  # 只保留最新的一幀
        self.result_cache = {}
        self.cache_ttl = 0.1  # 100ms 快取

    def process_frame_async(self, frame_data: FrameData, callback: Callable = None):
        """異步處理幀"""
        # 如果隊列滿，丟棄舊幀
        if self.processing_queue.full():
            try:
                self.processing_queue.get_nowait()
            except queue.Empty:
                pass

        try:
            self.processing_queue.put_nowait((frame_data, callback))
        except queue.Full:
            pass

    def _fast_process(self, frame_data: FrameData) -> Any:
        """快速處理邏輯"""
        start_time = time.time()

        # 檢查快取
        frame_hash = hashlib.md5(frame_data.image.tobytes()).hexdigest()[:16]  # 短雜湊

        if frame_hash in self.result_cache:
            cache_time, result = self.result_cache[frame_hash]
            if time.time() - cache_time < self.cache_ttl:
                return result

        # 快速處理邏輯（例如：關鍵點檢測、顏色檢測等）
        result = self._extract_key_features(frame_data.image)

        # 更新快取
        self.result_cache[frame_hash] = (time.time(), result)

        # 清理過期快取
        current_time = time.time()
        expired_keys = [k for k, (t, _) in self.result_cache.items()
                       if current_time - t > self.cache_ttl]
        for k in expired_keys:
            del self.result_cache[k]

        processing_time = time.time() - start_time
        if processing_time > self.max_processing_time:
            logger.warning(f"處理時間超過限制: {processing_time:.3f}s > {self.max_processing_time:.3f}s")

        return result

    def _extract_key_features(self, image: np.ndarray) -> Dict[str, Any]:
        """提取關鍵特徵（快速版本）"""
        # 這裡實現快速的特徵提取
        # 例如：顏色直方圖、邊緣檢測、關鍵點等

        features = {}

        # 快速顏色統計
        if len(image.shape) == 3:
            features['avg_color'] = np.mean(image, axis=(0, 1)).tolist()
            features['color_std'] = np.std(image, axis=(0, 1)).tolist()

        # 快速亮度統計
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        features['brightness'] = float(np.mean(gray))
        features['contrast'] = float(np.std(gray))

        return features


class AdaptiveFrameRateSync:
    """自適應幀率同步器"""

    def __init__(self, target_fps: int = 180, adaptation_rate: float = 0.1):
        self.target_fps = target_fps
        self.target_interval = 1.0 / target_fps
        self.adaptation_rate = adaptation_rate
        self.last_frame_time = 0
        self.frame_times = []
        self.max_history = 120  # 保留2秒的歷史
        self.adaptive_interval = self.target_interval
        self.performance_score = 1.0

    def sync_with_game_frame(self, game_fps_hint: Optional[float] = None) -> float:
        """與遊戲幀率同步"""
        current_time = time.time()

        if self.last_frame_time > 0:
            actual_interval = current_time - self.last_frame_time
            self.frame_times.append(actual_interval)

            if len(self.frame_times) > self.max_history:
                self.frame_times.pop(0)

            # 自適應調整
            if len(self.frame_times) >= 10:
                avg_interval = sum(self.frame_times[-10:]) / 10

                # 如果實際間隔與目標間隔差異較大，調整目標
                if abs(avg_interval - self.target_interval) > 0.001:  # 1ms 容差
                    adjustment = (avg_interval - self.adaptive_interval) * self.adaptation_rate
                    self.adaptive_interval += adjustment
                    self.adaptive_interval = max(0.001, min(0.1, self.adaptive_interval))  # 限制範圍

            # 如果有遊戲FPS提示，優先使用
            if game_fps_hint and game_fps_hint > 0:
                game_interval = 1.0 / game_fps_hint
                self.adaptive_interval = (self.adaptive_interval * 0.8 + game_interval * 0.2)

        # 計算等待時間
        if self.last_frame_time > 0:
            elapsed = current_time - self.last_frame_time
            sleep_time = max(0, self.adaptive_interval - elapsed)

            if sleep_time > 0:
                time.sleep(sleep_time)
                current_time = time.time()

        self.last_frame_time = current_time
        return current_time

    def get_performance_metrics(self) -> Dict[str, float]:
        """獲取性能指標"""
        if len(self.frame_times) < 2:
            return {'fps': 0, 'stability': 0, 'efficiency': 0}

        import statistics

        # 計算實際FPS
        avg_interval = statistics.mean(self.frame_times)
        actual_fps = 1.0 / avg_interval if avg_interval > 0 else 0

        # 計算穩定性（基於標準差）
        std_dev = statistics.stdev(self.frame_times) if len(self.frame_times) > 1 else 0
        stability = max(0, 1 - (std_dev / avg_interval)) if avg_interval > 0 else 0

        # 計算效率（實際FPS vs 目標FPS）
        efficiency = min(1.0, actual_fps / self.target_fps) if self.target_fps > 0 else 0

        return {
            'fps': actual_fps,
            'stability': stability,
            'efficiency': efficiency,
            'target_fps': self.target_fps,
            'adaptive_interval_ms': self.adaptive_interval * 1000
        }


class GameFrameDetector:
    """遊戲幀檢測器"""

    def __init__(self):
        self.last_frame_hash = None
        self.frame_change_times = []
        self.fps_estimates = []
        self.detection_window = 60  # 檢測窗口（幀數）

    def detect_game_fps(self, frame_data: FrameData) -> Optional[float]:
        """檢測遊戲實際FPS"""
        current_time = frame_data.timestamp

        # 計算幀雜湊
        frame_hash = self._fast_hash(frame_data.image)

        if self.last_frame_hash and frame_hash != self.last_frame_hash:
            # 幀發生變化
            self.frame_change_times.append(current_time)

            # 保持檢測窗口大小
            if len(self.frame_change_times) > self.detection_window:
                self.frame_change_times.pop(0)

            # 計算FPS估計
            if len(self.frame_change_times) >= 10:
                time_span = self.frame_change_times[-1] - self.frame_change_times[0]
                if time_span > 0:
                    fps_estimate = (len(self.frame_change_times) - 1) / time_span
                    self.fps_estimates.append(fps_estimate)

                    # 保持估計歷史
                    if len(self.fps_estimates) > 10:
                        self.fps_estimates.pop(0)

                    # 返回平均估計
                    return sum(self.fps_estimates) / len(self.fps_estimates)

        self.last_frame_hash = frame_hash
        return None

    def _fast_hash(self, image: np.ndarray) -> int:
        """快速雜湊計算"""
        # 縮小圖像並計算簡單雜湊
        if len(image.shape) == 3:
            small = image[::16, ::16, 0]  # 只取一個通道，大幅縮小
        else:
            small = image[::16, ::16]

        return hash(small.tobytes())


class PredictiveCapture:
    """預測性截圖"""

    def __init__(self, base_capture: HighFPSCapture):
        self.base_capture = base_capture
        self.frame_predictor = GameFrameDetector()
        self.adaptive_sync = AdaptiveFrameRateSync()
        self.prediction_buffer = queue.Queue(maxsize=5)
        self.is_predicting = False

    def start_predictive_capture(self):
        """啟動預測性截圖"""
        self.is_predicting = True

        # 啟動預測線程
        prediction_thread = threading.Thread(
            target=self._prediction_loop,
            name="PredictiveCapture",
            daemon=True
        )
        prediction_thread.start()

    def _prediction_loop(self):
        """預測循環"""
        while self.is_predicting and self.base_capture.is_running:
            try:
                # 獲取最新幀
                frame_data = self.base_capture.get_latest_frame(timeout=0.01)

                if frame_data:
                    # 檢測遊戲FPS
                    detected_fps = self.frame_predictor.detect_game_fps(frame_data)

                    # 同步幀率
                    self.adaptive_sync.sync_with_game_frame(detected_fps)

                    # 預測下一幀時間
                    next_frame_time = self._predict_next_frame_time()

                    # 添加到預測緩衝區
                    try:
                        self.prediction_buffer.put_nowait({
                            'predicted_time': next_frame_time,
                            'confidence': self._calculate_prediction_confidence(),
                            'detected_fps': detected_fps
                        })
                    except queue.Full:
                        # 緩衝區滿，移除最舊的預測
                        try:
                            self.prediction_buffer.get_nowait()
                            self.prediction_buffer.put_nowait({
                                'predicted_time': next_frame_time,
                                'confidence': self._calculate_prediction_confidence(),
                                'detected_fps': detected_fps
                            })
                        except queue.Empty:
                            pass

                time.sleep(0.001)  # 1ms 檢查間隔

            except Exception as e:
                logger.error(f"預測循環錯誤: {e}")
                time.sleep(0.01)

    def _predict_next_frame_time(self) -> float:
        """預測下一幀時間"""
        metrics = self.adaptive_sync.get_performance_metrics()
        predicted_interval = self.adaptive_sync.adaptive_interval
        return time.time() + predicted_interval

    def _calculate_prediction_confidence(self) -> float:
        """計算預測信心度"""
        metrics = self.adaptive_sync.get_performance_metrics()
        return metrics['stability'] * metrics['efficiency']

    def get_prediction(self) -> Optional[Dict[str, Any]]:
        """獲取幀預測"""
        try:
            return self.prediction_buffer.get_nowait()
        except queue.Empty:
            return None

    def stop_predictive_capture(self):
        """停止預測性截圖"""
        self.is_predicting = False
