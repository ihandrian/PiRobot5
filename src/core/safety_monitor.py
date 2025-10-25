import time
import logging
import threading
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum, auto
from collections import deque, PriorityQueue
import numpy as np
import cv2
import RPi.GPIO as GPIO
import spidev  # For ADC communication

logger = logging.getLogger('PiRobot-Safety')

class SafetyLevel(Enum):
    """Safety levels for different conditions"""
    NORMAL = auto()
    WARNING = auto()
    CAUTION = auto()
    CRITICAL = auto()
    EMERGENCY = auto()

@dataclass
class SafetyEvent:
    """Safety event data structure"""
    timestamp: float
    level: SafetyLevel
    source: str
    message: str
    data: Dict
    priority: int

class SafetyMonitor:
    """Optimized safety monitor with priority-based checks and hardware acceleration."""
    
    def __init__(self, 
                 emergency_stop_callback: Callable[[], None],
                 warning_callback: Optional[Callable[[str], None]] = None,
                 check_interval: float = 0.1,
                 buffer_size: int = 10,
                 power_save_mode: bool = True):
        """Initialize safety monitor with optimized settings."""
        self.logger = logging.getLogger('PiRobot.Safety')
        
        # Callbacks
        self.emergency_stop_callback = emergency_stop_callback
        self.warning_callback = warning_callback
        
        # Configuration
        self.check_interval = check_interval
        self.buffer_size = buffer_size
        self.power_save_mode = power_save_mode
        
        # Hardware setup
        self._setup_gpio()
        self._setup_adc()
        self._setup_watchdog()
        
        # State
        self.safety_level = SafetyLevel.NORMAL
        self.event_history = deque(maxlen=1000)
        self.active_warnings: Dict[str, SafetyLevel] = {}
        self.is_emergency_stop = False
        
        # Performance optimization
        self._battery_buffer = deque(maxlen=buffer_size)
        self._obstacle_buffer = deque(maxlen=buffer_size)
        self._last_check_time = time.time()
        
        # Priority queue for safety checks
        self._safety_queue = PriorityQueue()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Start monitoring
        self._monitor_thread = threading.Thread(
            target=self._monitor_safety,
            daemon=True
        )
        self._monitor_thread.start()
        
    def _setup_gpio(self):
        """Setup GPIO pins for safety monitoring."""
        GPIO.setmode(GPIO.BCM)
        
        # Emergency stop button
        GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(27, GPIO.FALLING, 
                            callback=self._emergency_stop_callback,
                            bouncetime=100)
        
        # Status LEDs
        GPIO.setup(5, GPIO.OUT)  # Status LED
        GPIO.setup(6, GPIO.OUT)  # Warning LED
        
    def _setup_adc(self):
        """Setup ADC for battery monitoring."""
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)  # Open SPI bus 0, device 0
        self.spi.max_speed_hz = 1000000  # Set SPI speed
        
    def _setup_watchdog(self):
        """Setup hardware watchdog timer."""
        try:
            with open('/dev/watchdog', 'w') as f:
                f.write('1')
            self.watchdog_enabled = True
        except Exception as e:
            self.logger.warning(f"Watchdog setup failed: {e}")
            self.watchdog_enabled = False
            
    def _read_adc(self, channel: int) -> int:
        """Read ADC value from specified channel."""
        if not (0 <= channel <= 7):
            raise ValueError("Channel must be between 0 and 7")
            
        # MCP3008 protocol
        r = self.spi.xfer2([1, (8 + channel) << 4, 0])
        return ((r[1] & 3) << 8) + r[2]
        
    def _read_battery_voltage(self) -> float:
        """Read battery voltage using ADC."""
        try:
            # Read ADC value
            adc_value = self._read_adc(0)  # Channel 0 for battery
            
            # Convert to voltage (assuming 3.3V reference and voltage divider)
            voltage = (adc_value / 1023.0) * 3.3 * 4.0  # 4:1 voltage divider
            
            # Update status LED
            GPIO.output(5, GPIO.HIGH if voltage > 11.0 else GPIO.LOW)
            
            return voltage
        except Exception as e:
            self.logger.error(f"Battery read error: {e}")
            return 0.0
            
    def _read_obstacle_distance(self) -> float:
        """Read obstacle distance from ultrasonic sensor."""
        try:
            # Trigger pulse
            GPIO.output(23, GPIO.HIGH)
            time.sleep(0.00001)
            GPIO.output(23, GPIO.LOW)
            
            # Wait for echo
            pulse_start = time.time()
            while GPIO.input(24) == 0:
                if time.time() - pulse_start > 0.1:
                    return float('inf')
                    
            pulse_start = time.time()
            while GPIO.input(24) == 1:
                if time.time() - pulse_start > 0.1:
                    return float('inf')
                    
            pulse_duration = time.time() - pulse_start
            
            # Calculate distance (speed of sound = 343 m/s)
            distance = (pulse_duration * 343) / 2
            
            return distance
        except Exception as e:
            self.logger.error(f"Obstacle read error: {e}")
            return float('inf')
            
    def _check_emergency_stop(self) -> bool:
        """Check emergency stop button state."""
        return GPIO.input(27) == GPIO.LOW
        
    def _monitor_safety(self):
        """Monitor safety conditions with priority-based checks."""
        while True:
            try:
                current_time = time.time()
                
                # Process high priority safety checks first
                while not self._safety_queue.empty():
                    priority, check_func = self._safety_queue.get()
                    check_func()
                
                # Check battery with priority
                voltage = self._read_battery_voltage()
                with self._lock:
                    self._battery_buffer.append(voltage)
                    
                    if voltage <= 10.5:  # Critical battery threshold
                        self.add_warning(
                            "battery",
                            SafetyLevel.CRITICAL,
                            f"Critical battery voltage: {voltage}V",
                            priority=3
                        )
                    elif voltage <= 11.0:  # Warning battery threshold
                        self.add_warning(
                            "battery",
                            SafetyLevel.WARNING,
                            f"Low battery voltage: {voltage}V",
                            priority=2
                        )
                        
                # Check obstacles with priority
                distance = self._read_obstacle_distance()
                with self._lock:
                    self._obstacle_buffer.append(distance)
                    
                    if distance <= 0.5:  # Obstacle threshold
                        self.add_warning(
                            "obstacle",
                            SafetyLevel.CAUTION,
                            f"Obstacle detected at {distance}m",
                            priority=1
                        )
                        
                # Check emergency stop with highest priority
                if self._check_emergency_stop():
                    with self._lock:
                        self.add_warning(
                            "emergency_stop",
                            SafetyLevel.EMERGENCY,
                            "Emergency stop triggered",
                            priority=4
                        )
                        
                # Update watchdog
                if self.watchdog_enabled:
                    try:
                        with open('/dev/watchdog', 'w') as f:
                            f.write('1')
                    except Exception as e:
                        self.logger.error(f"Watchdog update failed: {e}")
                        
                # Update last check time
                self._last_check_time = current_time
                
                # Sleep for check interval
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Safety monitoring error: {e}")
                time.sleep(1.0)  # Sleep on error
                
    def add_warning(self, source: str, level: SafetyLevel, message: str, 
                   data: Dict = None, priority: int = 0):
        """Add a safety warning with priority."""
        # Create event
        event = SafetyEvent(
            timestamp=time.time(),
            level=level,
            source=source,
            message=message,
            data=data or {},
            priority=priority
        )
        
        # Store event
        with self._lock:
            self.event_history.append(event)
            self.active_warnings[source] = level
            
            # Update safety level
            if level.value > self.safety_level.value:
                self.safety_level = level
                self._handle_safety_level_change(level)
                
        # Log event
        if level == SafetyLevel.EMERGENCY:
            self.logger.critical(f"EMERGENCY: {message} (from {source})")
            self._trigger_emergency_stop()
        elif level == SafetyLevel.CRITICAL:
            self.logger.error(f"CRITICAL: {message} (from {source})")
        elif level == SafetyLevel.CAUTION:
            self.logger.warning(f"CAUTION: {message} (from {source})")
        else:
            self.logger.info(f"WARNING: {message} (from {source})")
            
    def add_safety_check(self, check_func: Callable, priority: int = 0):
        """Add a safety check to the priority queue."""
        self._safety_queue.put((priority, check_func))
        
    def clear_warning(self, source: str):
        """Clear a safety warning."""
        with self._lock:
            if source in self.active_warnings:
                del self.active_warnings[source]
                self._update_safety_level()
                
    def _handle_safety_level_change(self, new_level: SafetyLevel):
        """Handle safety level changes."""
        if new_level == SafetyLevel.EMERGENCY:
            self._trigger_emergency_stop()
        elif new_level == SafetyLevel.CRITICAL:
            if self.warning_callback:
                self.warning_callback("Critical safety condition")
        elif new_level == SafetyLevel.CAUTION:
            if self.warning_callback:
                self.warning_callback("Caution safety condition")
        elif new_level == SafetyLevel.WARNING:
            if self.warning_callback:
                self.warning_callback("Warning safety condition")
                
    def _trigger_emergency_stop(self):
        """Trigger emergency stop."""
        if not self.is_emergency_stop:
            self.is_emergency_stop = True
            self.logger.critical("EMERGENCY STOP TRIGGERED")
            self.emergency_stop_callback()
            
    def _update_safety_level(self):
        """Update overall safety level based on active warnings."""
        if not self.active_warnings:
            self.safety_level = SafetyLevel.NORMAL
            return
            
        # Get highest safety level from active warnings
        max_level = max(self.active_warnings.values())
        
        # Update safety level
        if max_level != self.safety_level:
            self.safety_level = max_level
            self._handle_safety_level_change(max_level)
            
    def get_safety_status(self) -> Dict:
        """Get current safety status."""
        with self._lock:
            return {
                'level': self.safety_level.name,
                'is_emergency_stop': self.is_emergency_stop,
                'active_warnings': {
                    source: level.name
                    for source, level in self.active_warnings.items()
                },
                'battery_voltage': np.mean(self._battery_buffer) if self._battery_buffer else 0.0,
                'obstacle_distance': np.mean(self._obstacle_buffer) if self._obstacle_buffer else float('inf')
            }
            
    def get_event_history(self) -> List[SafetyEvent]:
        """Get safety event history."""
        with self._lock:
            return list(self.event_history)
            
    def reset_emergency_stop(self):
        """Reset emergency stop state."""
        with self._lock:
            if self.is_emergency_stop:
                self.is_emergency_stop = False
                self.logger.info("Emergency stop reset")
                
    def cleanup(self):
        """Cleanup resources."""
        try:
            self.spi.close()
            GPIO.cleanup([5, 6, 23, 24, 27])
            if self.watchdog_enabled:
                with open('/dev/watchdog', 'w') as f:
                    f.write('V')  # Disable watchdog
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
            
    def __del__(self):
        """Cleanup resources."""
        self.cleanup() 