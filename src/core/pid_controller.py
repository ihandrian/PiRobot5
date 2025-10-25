import time
import logging
from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np
from collections import deque

logger = logging.getLogger('PiRobot-PID')

@dataclass
class PIDConfig:
    """PID configuration with optimized defaults."""
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    min_output: float = -100.0
    max_output: float = 100.0
    sample_time: float = 0.01  # 10ms
    buffer_size: int = 10  # Number of samples to average
    power_save_mode: bool = True
    idle_timeout: int = 30  # Seconds
    anti_windup: bool = True
    derivative_filter: bool = True

class PIDController:
    """Optimized PID controller with power saving features."""
    
    def __init__(self, config: Dict):
        """Initialize PID controller with power optimization."""
        self.config = PIDConfig(**config)
        
        # Performance optimization
        self._last_time = time.time()
        self._last_error = 0.0
        self._integral = 0.0
        self._last_output = 0.0
        
        # Buffers for averaging
        self._error_buffer = deque(maxlen=self.config.buffer_size)
        self._output_buffer = deque(maxlen=self.config.buffer_size)
        
        # Anti-windup
        self._integral_limit = 100.0
        
    def compute(self, setpoint: float, measured_value: float) -> float:
        """Compute PID output with optimization."""
        current_time = time.time()
        dt = current_time - self._last_time
        
        # Skip computation if sample time not elapsed
        if dt < self.config.sample_time:
            return self._last_output
            
        # Calculate error
        error = setpoint - measured_value
        self._error_buffer.append(error)
        
        # Calculate P term
        p_term = self.config.kp * error
        
        # Calculate I term with anti-windup
        if self.config.anti_windup:
            if abs(self._integral) < self._integral_limit:
                self._integral += error * dt
            else:
                self._integral = np.sign(self._integral) * self._integral_limit
        else:
            self._integral += error * dt
            
        i_term = self.config.ki * self._integral
        
        # Calculate D term with filtering
        if self.config.derivative_filter:
            # Use moving average for derivative
            if len(self._error_buffer) > 1:
                derivative = (error - self._last_error) / dt
                derivative = np.mean(list(self._error_buffer)[-3:])  # 3-point moving average
            else:
                derivative = 0.0
        else:
            derivative = (error - self._last_error) / dt
            
        d_term = self.config.kd * derivative
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Clamp output
        output = np.clip(output, self.config.min_output, self.config.max_output)
        
        # Update buffers
        self._output_buffer.append(output)
        
        # Update state
        self._last_time = current_time
        self._last_error = error
        self._last_output = output
        
        return output
        
    def reset(self):
        """Reset controller state."""
        self._integral = 0.0
        self._last_error = 0.0
        self._last_output = 0.0
        self._error_buffer.clear()
        self._output_buffer.clear()
        self._last_time = time.time()
        
    def get_average_error(self) -> float:
        """Get average error from buffer."""
        if not self._error_buffer:
            return 0.0
        return np.mean(self._error_buffer)
        
    def get_average_output(self) -> float:
        """Get average output from buffer."""
        if not self._output_buffer:
            return 0.0
        return np.mean(self._output_buffer)
        
    def set_integral_limit(self, limit: float):
        """Set integral anti-windup limit."""
        self._integral_limit = abs(limit)
        
    def __del__(self):
        """Cleanup resources."""
        self.reset() 