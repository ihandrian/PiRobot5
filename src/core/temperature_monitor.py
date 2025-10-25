import time
import logging
from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np
from collections import deque
import threading

logger = logging.getLogger('PiRobot-Temperature')

@dataclass
class TemperatureConfig:
    """Temperature configuration with optimized defaults."""
    adc_channel: int = 0
    sample_rate: int = 10  # Hz
    buffer_size: int = 10  # Number of samples to average
    power_save_mode: bool = True
    idle_timeout: int = 30  # Seconds
    warning_threshold: float = 70.0  # °C
    critical_threshold: float = 80.0  # °C
    conversion_factor: float = 0.1  # °C per ADC unit

class TemperatureMonitor:
    """Optimized temperature monitor with power saving features."""
    
    def __init__(self, adc_channels: Dict[str, int],
                 temp_coefficient: float = 0.01,
                 max_temp: float = 80.0,
                 warning_temp: float = 70.0,
                 sample_rate: float = 1.0,  # Hz
                 buffer_size: int = 5):
        """Initialize temperature monitor with optimized settings.
        
        Args:
            adc_channels: Dictionary of motor IDs to ADC channels
            temp_coefficient: Temperature coefficient for ADC conversion
            max_temp: Maximum allowed temperature in °C
            warning_temp: Temperature for warning in °C
            sample_rate: Sampling rate in Hz
            buffer_size: Number of samples to average
        """
        self.logger = logging.getLogger('PiRobot.Temperature')
        
        # Configuration
        self.adc_channels = adc_channels
        self.temp_coefficient = temp_coefficient
        self.max_temp = max_temp
        self.warning_temp = warning_temp
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        
        # Performance optimization
        self._temperature_buffers = {
            motor_id: deque(maxlen=buffer_size) 
            for motor_id in adc_channels
        }
        self._last_temperatures = {
            motor_id: 0.0 for motor_id in adc_channels
        }
        self._last_check_time = time.time()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_temperature,
            daemon=True
        )
        self._monitor_thread.start()
        
    def _read_adc(self, channel: int) -> float:
        """Read ADC value with error handling."""
        try:
            # Implement actual ADC reading here
            # For now using simulated value
            return np.random.normal(500, 10)
        except Exception as e:
            self.logger.error(f"ADC read error: {e}")
            return 0.0
            
    def _convert_to_temperature(self, adc_value: float) -> float:
        """Convert ADC value to temperature."""
        return adc_value * self.temp_coefficient
        
    def _monitor_temperature(self):
        """Monitor temperature in background thread."""
        while True:
            try:
                current_time = time.time()
                
                # Read temperatures for all motors
                for motor_id, channel in self.adc_channels.items():
                    adc_value = self._read_adc(channel)
                    temperature = self._convert_to_temperature(adc_value)
                    
                    with self._lock:
                        self._temperature_buffers[motor_id].append(temperature)
                        self._last_temperatures[motor_id] = temperature
                        
                        # Check thresholds
                        if temperature >= self.max_temp:
                            self.logger.critical(
                                f"Critical temperature for {motor_id}: {temperature}°C"
                            )
                            self._handle_critical_temperature(motor_id)
                        elif temperature >= self.warning_temp:
                            self.logger.warning(
                                f"High temperature for {motor_id}: {temperature}°C"
                            )
                            
                # Sleep for sample interval
                time.sleep(1.0 / self.sample_rate)
                
            except Exception as e:
                self.logger.error(f"Temperature monitoring error: {e}")
                time.sleep(1.0)  # Sleep on error
                
    def _handle_critical_temperature(self, motor_id: str):
        """Handle critical temperature condition."""
        # Implement safety measures
        self.logger.critical(f"Implementing safety measures for {motor_id}")
        
    def get_temperature(self, motor_id: str) -> float:
        """Get current temperature with averaging."""
        with self._lock:
            if motor_id not in self._temperature_buffers:
                return 0.0
            buffer = self._temperature_buffers[motor_id]
            if not buffer:
                return 0.0
            return np.mean(buffer)
            
    def get_temperature_stats(self, motor_id: str) -> Dict:
        """Get temperature statistics for a motor."""
        with self._lock:
            if motor_id not in self._temperature_buffers:
                return {}
            buffer = self._temperature_buffers[motor_id]
            if not buffer:
                return {}
            return {
                'current': self._last_temperatures[motor_id],
                'average': np.mean(buffer),
                'min': np.min(buffer),
                'max': np.max(buffer),
                'is_warning': self._last_temperatures[motor_id] >= self.warning_temp,
                'is_critical': self._last_temperatures[motor_id] >= self.max_temp
            }
            
    def reset(self):
        """Reset temperature monitor."""
        with self._lock:
            for buffer in self._temperature_buffers.values():
                buffer.clear()
            self._last_temperatures = {
                motor_id: 0.0 for motor_id in self.adc_channels
            }
            self._last_check_time = time.time()
            
    def __del__(self):
        """Cleanup resources."""
        self.reset() 