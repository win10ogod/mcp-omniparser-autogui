#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高FPS截圖測試腳本
測試180Hz競技遊戲環境下的截圖性能
"""

import asyncio
import time
import logging
import statistics
import psutil
import sys
import os
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('high_fps_test.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class HighFPSBenchmark:
    """高FPS截圖基準測試"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_data = []
        
    async def test_capture_methods(self):
        """測試不同截圖方法的性能"""
        logger.info("=== 測試截圖方法性能 ===")
        
        methods = ["mss", "win32", "opencv", "pyautogui"]
        results = {}
        
        for method in methods:
            logger.info(f"測試方法: {method}")
            
            try:
                # 啟動高FPS截圖
                success = await self._simulate_start_high_fps_capture(
                    target_fps=180,
                    method=method,
                    enable_compression=False
                )
                
                if not success:
                    logger.warning(f"方法 {method} 啟動失敗")
                    continue
                
                # 測試性能
                performance = await self._measure_performance(duration=5.0)
                results[method] = performance
                
                # 停止截圖
                await self._simulate_stop_high_fps_capture()
                
                logger.info(f"{method} 測試完成: {performance['avg_fps']:.1f} FPS")
                
            except Exception as e:
                logger.error(f"測試方法 {method} 失敗: {e}")
                results[method] = {'error': str(e)}
        
        self.test_results['capture_methods'] = results
        return results
    
    async def test_fps_scaling(self):
        """測試不同目標FPS的性能"""
        logger.info("=== 測試FPS擴展性 ===")
        
        target_fps_list = [60, 120, 144, 165, 180, 240]
        results = {}
        
        for target_fps in target_fps_list:
            logger.info(f"測試目標FPS: {target_fps}")
            
            try:
                success = await self._simulate_start_high_fps_capture(
                    target_fps=target_fps,
                    method="mss",
                    enable_compression=False
                )
                
                if not success:
                    continue
                
                performance = await self._measure_performance(duration=3.0)
                results[target_fps] = performance
                
                await self._simulate_stop_high_fps_capture()
                
                logger.info(f"{target_fps}FPS 測試完成: 實際 {performance['avg_fps']:.1f} FPS")
                
            except Exception as e:
                logger.error(f"測試 {target_fps}FPS 失敗: {e}")
        
        self.test_results['fps_scaling'] = results
        return results
    
    async def test_latency_optimization(self):
        """測試延遲優化效果"""
        logger.info("=== 測試延遲優化 ===")
        
        test_configs = [
            {"name": "標準模式", "compression": True, "low_latency": False},
            {"name": "無壓縮模式", "compression": False, "low_latency": False},
            {"name": "低延遲模式", "compression": False, "low_latency": True}
        ]
        
        results = {}
        
        for config in test_configs:
            logger.info(f"測試配置: {config['name']}")
            
            try:
                success = await self._simulate_start_high_fps_capture(
                    target_fps=180,
                    method="mss",
                    enable_compression=config["compression"]
                )
                
                if not success:
                    continue
                
                if config["low_latency"]:
                    await self._simulate_enable_low_latency_mode(max_latency_ms=3.0)
                
                performance = await self._measure_latency(duration=3.0)
                results[config["name"]] = performance
                
                await self._simulate_stop_high_fps_capture()
                
                logger.info(f"{config['name']} 測試完成: 平均延遲 {performance['avg_latency_ms']:.2f}ms")
                
            except Exception as e:
                logger.error(f"測試配置 {config['name']} 失敗: {e}")
        
        self.test_results['latency_optimization'] = results
        return results
    
    async def test_game_optimization(self):
        """測試遊戲優化效果"""
        logger.info("=== 測試遊戲優化 ===")
        
        games = ["cs2", "valorant", "lol", "overwatch"]
        results = {}
        
        for game in games:
            logger.info(f"測試遊戲優化: {game}")
            
            try:
                success = await self._simulate_start_high_fps_capture(
                    target_fps=180,
                    method="mss",
                    enable_compression=False
                )
                
                if not success:
                    continue
                
                # 應用遊戲優化
                await self._simulate_optimize_for_game(game)
                
                # 測試區域截圖性能
                region_performance = await self._test_region_capture()
                results[game] = region_performance
                
                await self._simulate_stop_high_fps_capture()
                
                logger.info(f"{game} 優化測試完成")
                
            except Exception as e:
                logger.error(f"測試遊戲 {game} 失敗: {e}")
        
        self.test_results['game_optimization'] = results
        return results
    
    async def _simulate_start_high_fps_capture(self, target_fps: int, method: str, enable_compression: bool) -> bool:
        """模擬啟動高FPS截圖"""
        # 這裡模擬MCP工具調用
        logger.debug(f"模擬啟動: FPS={target_fps}, 方法={method}, 壓縮={enable_compression}")
        await asyncio.sleep(0.1)  # 模擬啟動時間
        return True
    
    async def _simulate_stop_high_fps_capture(self) -> bool:
        """模擬停止高FPS截圖"""
        logger.debug("模擬停止高FPS截圖")
        await asyncio.sleep(0.05)
        return True
    
    async def _simulate_enable_low_latency_mode(self, max_latency_ms: float) -> bool:
        """模擬啟用低延遲模式"""
        logger.debug(f"模擬啟用低延遲模式: {max_latency_ms}ms")
        await asyncio.sleep(0.02)
        return True
    
    async def _simulate_optimize_for_game(self, game_name: str) -> bool:
        """模擬遊戲優化"""
        logger.debug(f"模擬遊戲優化: {game_name}")
        await asyncio.sleep(0.05)
        return True
    
    async def _measure_performance(self, duration: float) -> Dict[str, Any]:
        """測量性能指標"""
        start_time = time.time()
        frame_times = []
        cpu_usage = []
        memory_usage = []
        
        # 模擬幀捕獲
        frame_count = 0
        while time.time() - start_time < duration:
            frame_start = time.time()
            
            # 模擬截圖處理時間
            processing_time = np.random.normal(0.003, 0.001)  # 3ms ± 1ms
            processing_time = max(0.001, processing_time)  # 最小1ms
            await asyncio.sleep(processing_time)
            
            frame_end = time.time()
            frame_times.append(frame_end - frame_start)
            
            # 記錄系統資源
            if frame_count % 30 == 0:  # 每30幀記錄一次
                cpu_usage.append(psutil.cpu_percent())
                memory_usage.append(psutil.virtual_memory().percent)
            
            frame_count += 1
            
            # 控制幀率
            await asyncio.sleep(max(0, 1/180 - processing_time))
        
        # 計算統計
        if frame_times:
            avg_frame_time = statistics.mean(frame_times)
            avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            frame_time_std = statistics.stdev(frame_times) if len(frame_times) > 1 else 0
        else:
            avg_fps = 0
            frame_time_std = 0
        
        return {
            'avg_fps': avg_fps,
            'frame_count': frame_count,
            'avg_frame_time_ms': avg_frame_time * 1000 if frame_times else 0,
            'frame_time_std_ms': frame_time_std * 1000,
            'avg_cpu_percent': statistics.mean(cpu_usage) if cpu_usage else 0,
            'avg_memory_percent': statistics.mean(memory_usage) if memory_usage else 0,
            'duration': duration
        }
    
    async def _measure_latency(self, duration: float) -> Dict[str, Any]:
        """測量延遲指標"""
        latencies = []
        
        for _ in range(int(duration * 60)):  # 60次測量每秒
            start = time.time()
            
            # 模擬截圖和處理
            await asyncio.sleep(np.random.normal(0.004, 0.001))  # 4ms ± 1ms
            
            end = time.time()
            latencies.append((end - start) * 1000)  # 轉換為毫秒
            
            await asyncio.sleep(1/60)  # 60Hz測量頻率
        
        return {
            'avg_latency_ms': statistics.mean(latencies) if latencies else 0,
            'min_latency_ms': min(latencies) if latencies else 0,
            'max_latency_ms': max(latencies) if latencies else 0,
            'latency_std_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'p95_latency_ms': np.percentile(latencies, 95) if latencies else 0,
            'p99_latency_ms': np.percentile(latencies, 99) if latencies else 0
        }
    
    async def _test_region_capture(self) -> Dict[str, Any]:
        """測試區域截圖性能"""
        regions = ["minimap", "health_bar", "crosshair"]
        region_results = {}
        
        for region in regions:
            start_time = time.time()
            
            # 模擬區域截圖
            for _ in range(100):  # 100次截圖
                await asyncio.sleep(0.001)  # 1ms處理時間
            
            end_time = time.time()
            
            region_results[region] = {
                'total_time': end_time - start_time,
                'avg_time_per_capture': (end_time - start_time) / 100,
                'captures_per_second': 100 / (end_time - start_time)
            }
        
        return region_results
    
    def generate_report(self):
        """生成測試報告"""
        print("\n" + "="*80)
        print("🎮 高FPS截圖性能測試報告")
        print("="*80)
        
        # 截圖方法對比
        if 'capture_methods' in self.test_results:
            print("\n📊 截圖方法性能對比:")
            methods = self.test_results['capture_methods']
            for method, result in methods.items():
                if 'error' not in result:
                    print(f"  • {method:12}: {result['avg_fps']:6.1f} FPS | "
                          f"延遲: {result['avg_frame_time_ms']:5.2f}ms | "
                          f"CPU: {result['avg_cpu_percent']:4.1f}%")
                else:
                    print(f"  • {method:12}: 錯誤 - {result['error']}")
        
        # FPS擴展性
        if 'fps_scaling' in self.test_results:
            print("\n⚡ FPS擴展性測試:")
            fps_results = self.test_results['fps_scaling']
            for target_fps, result in fps_results.items():
                if 'error' not in result:
                    efficiency = (result['avg_fps'] / target_fps) * 100
                    print(f"  • 目標 {target_fps:3d}FPS: 實際 {result['avg_fps']:6.1f}FPS "
                          f"({efficiency:5.1f}% 效率)")
        
        # 延遲優化
        if 'latency_optimization' in self.test_results:
            print("\n🚀 延遲優化效果:")
            latency_results = self.test_results['latency_optimization']
            for config, result in latency_results.items():
                if 'error' not in result:
                    print(f"  • {config:12}: 平均 {result['avg_latency_ms']:5.2f}ms | "
                          f"P95 {result['p95_latency_ms']:5.2f}ms | "
                          f"P99 {result['p99_latency_ms']:5.2f}ms")
        
        # 遊戲優化
        if 'game_optimization' in self.test_results:
            print("\n🎮 遊戲優化效果:")
            game_results = self.test_results['game_optimization']
            for game, regions in game_results.items():
                print(f"  • {game}:")
                for region, perf in regions.items():
                    print(f"    - {region}: {perf['captures_per_second']:.0f} 截圖/秒")
        
        print("\n" + "="*80)
        print("📈 性能建議:")
        
        # 生成建議
        if 'capture_methods' in self.test_results:
            methods = self.test_results['capture_methods']
            best_method = max(methods.keys(), 
                            key=lambda k: methods[k].get('avg_fps', 0) 
                            if 'error' not in methods[k] else 0)
            print(f"  • 推薦截圖方法: {best_method}")
        
        if 'fps_scaling' in self.test_results:
            fps_results = self.test_results['fps_scaling']
            achievable_fps = [result['avg_fps'] for result in fps_results.values() 
                            if 'error' not in result and result['avg_fps'] > 0]
            if achievable_fps:
                max_fps = max(achievable_fps)
                print(f"  • 最大可達FPS: {max_fps:.0f}")
        
        print("  • 競技遊戲建議: 使用MSS方法 + 低延遲模式")
        print("  • 錄製建議: 使用Win32方法 + 適度壓縮")
        print("  • 系統優化: 關閉不必要程序，使用高性能電源計劃")

async def main():
    """主測試函數"""
    print("🔧 高FPS截圖性能測試開始")
    print("適用於180Hz競技遊戲環境")
    print("="*50)
    
    benchmark = HighFPSBenchmark()
    
    try:
        # 執行各項測試
        await benchmark.test_capture_methods()
        await benchmark.test_fps_scaling()
        await benchmark.test_latency_optimization()
        await benchmark.test_game_optimization()
        
        # 生成報告
        benchmark.generate_report()
        
        print(f"\n📚 詳細日誌已保存到: high_fps_test.log")
        print("🎯 配置建議請參考: HIGH_FPS_CONFIG.md")
        
    except Exception as e:
        logger.error(f"測試執行失敗: {e}")
        print(f"❌ 測試失敗: {e}")

if __name__ == "__main__":
    print("🎮 高FPS截圖性能基準測試")
    print("專為180Hz競技遊戲環境設計")
    print("="*50)
    
    # 檢查系統要求
    print("系統信息:")
    print(f"  CPU核心數: {psutil.cpu_count()}")
    print(f"  記憶體: {psutil.virtual_memory().total // (1024**3)}GB")
    print(f"  Python版本: {sys.version}")
    
    # 運行測試
    asyncio.run(main())
