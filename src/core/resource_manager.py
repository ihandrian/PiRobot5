import psutil
import gc
import time
import logging
import threading
from typing import Dict, Optional
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger('PiRobot-Resource')

@dataclass
class ResourceMetrics:
    """Resource usage metrics"""
    cpu_percent: float
    memory_percent: float
    temperature: float
    frame_skip_rate: float
    processing_time: float

class ResourceManager:
    """Optimized resource manager with frame skipping and memory management."""
    
    def __init__(self,
                 max_cpu_percent: float = 80.0,
                 max_memory_percent: float = 80.0,
                 max_temperature: float = 80.0,
                 check_interval: float = 1.0,
                 buffer_size: int = 10):
        """Initialize resource manager with optimized settings."""
        self.logger = logging.getLogger('PiRobot.Resource')
        
        # Configuration
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.max_temperature = max_temperature
        self.check_interval = check_interval
        self.buffer_size = buffer_size
        
        # State tracking
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.frame_skip_count = 0
        self.total_frames = 0
        self.processing_times = deque(maxlen=buffer_size)
        self.last_check_time = 0
        
        # Performance optimization
        self._setup_performance_optimization()
        
        # Start monitoring
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True
        )
        self.monitor_thread.start()
        
    def _setup_performance_optimization(self):
        """Setup performance optimization features."""
        try:
            # Disable garbage collector for real-time operations
            gc.disable()
            
            # Set process priority
            p = psutil.Process()
            p.nice(psutil.HIGH_PRIORITY_CLASS)
            
            # Set CPU affinity
            p.cpu_affinity([0])  # Use first CPU core
            
            logger.info("Performance optimization enabled")
            
        except Exception as e:
            logger.warning(f"Failed to setup performance optimization: {e}")
            
    def _monitor_resources(self):
        """Monitor system resources."""
        while self.running:
            try:
                current_time = time.time()
                
                # Check if it's time for an update
                if current_time - self.last_check_time < self.check_interval:
                    time.sleep(0.001)  # Small sleep to prevent CPU hogging
                    continue
                    
                # Get resource metrics
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                temperature = self._get_cpu_temperature()
                
                # Calculate frame skip rate
                frame_skip_rate = self.frame_skip_count / max(1, self.total_frames)
                
                # Calculate average processing time
                processing_time = (
                    sum(self.processing_times) / len(self.processing_times)
                    if self.processing_times else 0
                )
                
                # Store metrics
                metrics = ResourceMetrics(
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    temperature=temperature,
                    frame_skip_rate=frame_skip_rate,
                    processing_time=processing_time
                )
                self.metrics_buffer.append(metrics)
                
                # Optimize resources if needed
                self._optimize_resources(metrics)
                
                # Update timing
                self.last_check_time = current_time
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(0.1)
                
    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature."""
        try:
            # Implement actual temperature reading
            # For now using simulated value
            return 50.0 + psutil.cpu_percent() * 0.1
        except Exception as e:
            logger.error(f"Temperature reading error: {e}")
            return 0.0
            
    def _optimize_resources(self, metrics: ResourceMetrics):
        """Optimize system resources based on current metrics."""
        try:
            # Check CPU usage
            if metrics.cpu_percent > self.max_cpu_percent:
                self._reduce_cpu_usage()
                
            # Check memory usage
            if metrics.memory_percent > self.max_memory_percent:
                self._reduce_memory_usage()
                
            # Check temperature
            if metrics.temperature > self.max_temperature:
                self._reduce_temperature()
                
        except Exception as e:
            logger.error(f"Resource optimization error: {e}")
            
    def _reduce_cpu_usage(self):
        """Reduce CPU usage."""
        try:
            # Increase frame skip rate
            self.frame_skip_count += 1
            
            # Reduce process priority
            p = psutil.Process()
            p.nice(psutil.NORMAL_PRIORITY_CLASS)
            
            logger.warning("Reducing CPU usage")
            
        except Exception as e:
            logger.error(f"CPU reduction error: {e}")
            
    def _reduce_memory_usage(self):
        """Reduce memory usage."""
        try:
            # Force garbage collection
            gc.collect()
            
            # Clear buffers
            self.metrics_buffer.clear()
            self.processing_times.clear()
            
            logger.warning("Reducing memory usage")
            
        except Exception as e:
            logger.error(f"Memory reduction error: {e}")
            
    def _reduce_temperature(self):
        """Reduce system temperature."""
        try:
            # Reduce CPU frequency
            # Implement actual frequency reduction
            # For now just log warning
            logger.warning("System temperature too high")
            
        except Exception as e:
            logger.error(f"Temperature reduction error: {e}")
            
    def should_process_frame(self) -> bool:
        """Determine if current frame should be processed."""
        try:
            self.total_frames += 1
            
            # Get current metrics
            if not self.metrics_buffer:
                return True
                
            metrics = self.metrics_buffer[-1]
            
            # Check resource usage
            if (metrics.cpu_percent > self.max_cpu_percent or
                metrics.memory_percent > self.max_memory_percent or
                metrics.temperature > self.max_temperature):
                self.frame_skip_count += 1
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Frame processing check error: {e}")
            return True
            
    def record_processing_time(self, processing_time: float):
        """Record frame processing time."""
        try:
            self.processing_times.append(processing_time)
        except Exception as e:
            logger.error(f"Processing time recording error: {e}")
            
    def get_resource_status(self) -> Dict:
        """Get current resource status."""
        try:
            if not self.metrics_buffer:
                return {}
                
            metrics = self.metrics_buffer[-1]
            return {
                'cpu_percent': metrics.cpu_percent,
                'memory_percent': metrics.memory_percent,
                'temperature': metrics.temperature,
                'frame_skip_rate': metrics.frame_skip_rate,
                'processing_time': metrics.processing_time
            }
        except Exception as e:
            logger.error(f"Status retrieval error: {e}")
            return {}
            
    def cleanup(self):
        """Cleanup resources."""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        self.metrics_buffer.clear()
        self.processing_times.clear()
        
    def __del__(self):
        """Cleanup resources."""
        self.cleanup() 