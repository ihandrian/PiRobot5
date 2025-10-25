import RPi.GPIO as GPIO
import time
import logging
from typing import Dict, Optional
from dataclasses import dataclass
import threading
from collections import deque

logger = logging.getLogger('PiRobot-Encoder')

@dataclass
class EncoderConfig:
    """Encoder configuration with optimized defaults."""
    pin_a: int = 0
    pin_b: int = 0
    ppr: int = 20  # Pulses per revolution
    sample_rate: int = 100  # Hz
    buffer_size: int = 10  # Number of samples to average
    debounce_time: int = 1  # ms
    power_save_mode: bool = True
    idle_timeout: int = 30  # Seconds

class EncoderHandler:
    """Optimized encoder handler with power saving features."""
    
    def __init__(self, config: Dict):
        """Initialize encoder handler with power optimization."""
        self.config = EncoderConfig(**config)
        self._setup_gpio()
        self._setup_power_management()
        
        # Performance optimization
        self._count = 0
        self._last_time = time.time()
        self._speed_buffer = deque(maxlen=self.config.buffer_size)
        self._last_position = 0
        self._position_buffer = deque(maxlen=self.config.buffer_size)
        
        # Thread safety
        self._lock = threading.Lock()
        
    def _setup_gpio(self):
        """Setup GPIO with optimized configuration."""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Setup pins with pull-down for power saving
        GPIO.setup(self.config.pin_a, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(self.config.pin_b, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        
        # Add hardware debouncing
        GPIO.add_event_detect(self.config.pin_a, GPIO.RISING, 
                            callback=self._encoder_callback, 
                            bouncetime=self.config.debounce_time)
        GPIO.add_event_detect(self.config.pin_b, GPIO.RISING, 
                            callback=self._encoder_callback, 
                            bouncetime=self.config.debounce_time)
                            
    def _setup_power_management(self):
        """Setup power management system."""
        self.power_save_thread = threading.Thread(target=self._power_save_monitor, 
                                                daemon=True)
        self.power_save_thread.start()
        
    def _power_save_monitor(self):
        """Monitor and manage power saving mode."""
        while True:
            if (time.time() - self._last_time > self.config.idle_timeout and 
                self.config.power_save_mode):
                self._enter_power_save()
            time.sleep(1)
            
    def _enter_power_save(self):
        """Enter power saving mode."""
        with self._lock:
            # Clear buffers to save memory
            self._speed_buffer.clear()
            self._position_buffer.clear()
            logger.info("Encoder entered power save mode")
            
    def _encoder_callback(self, channel):
        """Optimized encoder callback with debouncing."""
        current_time = time.time()
        
        with self._lock:
            # Determine direction based on which pin triggered
            direction = 1 if channel == self.config.pin_a else -1
            
            # Update count with direction
            self._count += direction
            
            # Calculate time delta
            dt = current_time - self._last_time
            if dt > 0:
                # Calculate speed in RPM
                speed = (direction * 60) / (self.config.ppr * dt)
                self._speed_buffer.append(speed)
                
                # Update position
                position = self._count / self.config.ppr
                self._position_buffer.append(position)
                
            self._last_time = current_time
            
    def get_speed(self) -> float:
        """Get current speed with averaging."""
        with self._lock:
            if not self._speed_buffer:
                return 0.0
            return sum(self._speed_buffer) / len(self._speed_buffer)
            
    def get_position(self) -> float:
        """Get current position with averaging."""
        with self._lock:
            if not self._position_buffer:
                return 0.0
            return sum(self._position_buffer) / len(self._position_buffer)
            
    def reset(self):
        """Reset encoder counts."""
        with self._lock:
            self._count = 0
            self._speed_buffer.clear()
            self._position_buffer.clear()
            self._last_time = time.time()
            
    def __del__(self):
        """Cleanup resources."""
        try:
            GPIO.remove_event_detect(self.config.pin_a)
            GPIO.remove_event_detect(self.config.pin_b)
            GPIO.cleanup([self.config.pin_a, self.config.pin_b])
        except Exception as e:
            logger.error(f"Cleanup error: {e}") 