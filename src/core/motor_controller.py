import RPi.GPIO as GPIO
import time
import logging
from typing import Dict, Optional
import threading
from dataclasses import dataclass
import numpy as np
import spidev  # For ADC communication
from enum import Enum, auto
import os

logger = logging.getLogger('PiRobot-Motor')

class MotorError(Exception):
    """Custom exception for motor controller errors."""
    pass

class MotorState(Enum):
    """Motor states"""
    IDLE = auto()
    RUNNING = auto()
    STALLED = auto()
    ERROR = auto()
    OVERHEATED = auto()

@dataclass
class MotorMetrics:
    """Motor performance metrics"""
    current: float
    temperature: float
    speed: float
    encoder_count: int
    state: MotorState

@dataclass
class MotorConfig:
    """Motor configuration with optimized defaults."""
    pwm_frequency: int = 1000  # Optimized PWM frequency
    pwm_duty_cycle: int = 0
    direction_pin: int = 0
    enable_pin: int = 0
    encoder_pin_a: int = 0
    encoder_pin_b: int = 0
    max_speed: int = 100
    acceleration: float = 0.5  # Gradual acceleration
    deceleration: float = 0.5  # Gradual deceleration
    power_save_mode: bool = True  # Enable power saving
    idle_timeout: int = 30  # Seconds before entering power save mode

class MotorController:
    """Enhanced motor controller with comprehensive safety features."""
    
    def __init__(self,
                 motor_pins: Dict[str, int],
                 encoder_pins: Dict[str, int],
                 current_sense_pin: int,
                 temp_sense_pin: int,
                 max_current: float = 2.0,
                 max_temp: float = 60.0,
                 update_rate: float = 0.1,
                 pwm_frequency: int = 1000,
                 acceleration_rate: float = 0.1):
        """Initialize motor controller with safety features."""
        self.logger = logging.getLogger('PiRobot.Motor')
        
        # Validate pin numbers
        self._validate_pins(motor_pins, encoder_pins, current_sense_pin, temp_sense_pin)
        
        # Pin configuration
        self.motor_pins = motor_pins
        self.encoder_pins = encoder_pins
        self.current_sense_pin = current_sense_pin
        self.temp_sense_pin = temp_sense_pin
        
        # Safety thresholds
        self.max_current = max_current
        self.max_temp = max_temp
        
        # Performance monitoring
        self.update_rate = update_rate
        self.pwm_frequency = pwm_frequency
        self.acceleration_rate = acceleration_rate
        self.target_speed = 0.0
        self.current_speed = 0.0
        
        self.metrics = MotorMetrics(
            current=0.0,
            temperature=0.0,
            speed=0.0,
            encoder_count=0,
            state=MotorState.IDLE
        )
        
        # Setup hardware
        try:
            self._setup_gpio()
            self._setup_adc()
            self._setup_encoders()
            self._setup_pwm()
        except Exception as e:
            self.cleanup()
            raise MotorError(f"Failed to initialize motor controller: {e}")
        
        # Thread safety
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
        # Start monitoring
        self._monitor_thread = threading.Thread(
            target=self._monitor_motors,
            daemon=True
        )
        self._monitor_thread.start()
        
    def _validate_pins(self, motor_pins: Dict[str, int], encoder_pins: Dict[str, int],
                      current_sense_pin: int, temp_sense_pin: int):
        """Validate pin numbers and configurations."""
        # Check if running on Raspberry Pi
        if not os.path.exists('/proc/device-tree/model'):
            raise MotorError("This code must run on a Raspberry Pi")
            
        # Validate pin numbers
        for name, pin in motor_pins.items():
            if not isinstance(pin, int) or pin < 0 or pin > 27:
                raise ValueError(f"Invalid motor pin number for {name}: {pin}")
                
        for name, pin in encoder_pins.items():
            if not isinstance(pin, int) or pin < 0 or pin > 27:
                raise ValueError(f"Invalid encoder pin number for {name}: {pin}")
                
        if not isinstance(current_sense_pin, int) or current_sense_pin < 0 or current_sense_pin > 27:
            raise ValueError(f"Invalid current sense pin number: {current_sense_pin}")
            
        if not isinstance(temp_sense_pin, int) or temp_sense_pin < 0 or temp_sense_pin > 27:
            raise ValueError(f"Invalid temperature sense pin number: {temp_sense_pin}")
            
        # Check for pin conflicts
        all_pins = list(motor_pins.values()) + list(encoder_pins.values()) + [current_sense_pin, temp_sense_pin]
        if len(all_pins) != len(set(all_pins)):
            raise ValueError("Pin numbers must be unique")
            
    def _setup_gpio(self):
        """Setup GPIO pins for motor control."""
        try:
            GPIO.setmode(GPIO.BCM)
            
            # Motor control pins
            for pin in self.motor_pins.values():
                GPIO.setup(pin, GPIO.OUT)
                
            # Current and temperature sense pins
            GPIO.setup(self.current_sense_pin, GPIO.IN)
            GPIO.setup(self.temp_sense_pin, GPIO.IN)
        except Exception as e:
            raise MotorError(f"Failed to setup GPIO: {e}")
            
    def _setup_pwm(self):
        """Setup PWM for motor speed control."""
        try:
            self.pwm = {}
            for name, pin in self.motor_pins.items():
                if 'pwm' in name.lower():
                    self.pwm[name] = GPIO.PWM(pin, self.pwm_frequency)
                    self.pwm[name].start(0)
        except Exception as e:
            raise MotorError(f"Failed to setup PWM: {e}")
            
    def _setup_adc(self):
        """Setup ADC for current and temperature monitoring."""
        try:
            self.spi = spidev.SpiDev()
            self.spi.open(0, 0)  # Open SPI bus 0, device 0
            self.spi.max_speed_hz = 1000000  # Set SPI speed
        except Exception as e:
            raise MotorError(f"Failed to setup ADC: {e}")
            
    def _setup_encoders(self):
        """Setup encoder pins and interrupts."""
        try:
            for pin in self.encoder_pins.values():
                GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
                GPIO.add_event_detect(pin, GPIO.BOTH,
                                    callback=self._encoder_callback,
                                    bouncetime=1)
        except Exception as e:
            raise MotorError(f"Failed to setup encoders: {e}")
            
    def _read_adc(self, channel: int) -> int:
        """Read ADC value from specified channel."""
        if not (0 <= channel <= 7):
            raise ValueError("Channel must be between 0 and 7")
            
        try:
            # MCP3008 protocol
            r = self.spi.xfer2([1, (8 + channel) << 4, 0])
            return ((r[1] & 3) << 8) + r[2]
        except Exception as e:
            self.logger.error(f"ADC read error: {e}")
            return 0
            
    def _read_current(self) -> float:
        """Read motor current using ADC."""
        try:
            # Read ADC value
            adc_value = self._read_adc(0)  # Channel 0 for current
            
            # Convert to current (assuming 3.3V reference and 0.1V/A sensitivity)
            current = (adc_value / 1023.0) * 3.3 / 0.1
            
            return current
        except Exception as e:
            self.logger.error(f"Current read error: {e}")
            return 0.0
            
    def _read_temperature(self) -> float:
        """Read motor temperature using ADC."""
        try:
            # Read ADC value
            adc_value = self._read_adc(1)  # Channel 1 for temperature
            
            # Convert to temperature (assuming 10mV/°C)
            temperature = (adc_value / 1023.0) * 3.3 / 0.01
            
            return temperature
        except Exception as e:
            self.logger.error(f"Temperature read error: {e}")
            return 0.0
            
    def _encoder_callback(self, channel):
        """Handle encoder interrupts."""
        with self._lock:
            self.metrics.encoder_count += 1
            
    def _detect_stall(self) -> bool:
        """Detect motor stall condition."""
        with self._lock:
            # Check if motor is running but encoder count hasn't changed
            if (self.metrics.state == MotorState.RUNNING and
                self.metrics.encoder_count == 0):
                return True
            return False
            
    def _update_speed(self):
        """Update motor speed with acceleration control."""
        with self._lock:
            try:
                if abs(self.current_speed - self.target_speed) > self.acceleration_rate:
                    if self.current_speed < self.target_speed:
                        self.current_speed += self.acceleration_rate
                    else:
                        self.current_speed -= self.acceleration_rate
                else:
                    self.current_speed = self.target_speed
                    
                # Update PWM duty cycle
                duty_cycle = abs(self.current_speed) * 100
                for pwm in self.pwm.values():
                    pwm.ChangeDutyCycle(duty_cycle)
                    
                # Set direction
                direction = GPIO.HIGH if self.current_speed > 0 else GPIO.LOW
                for name, pin in self.motor_pins.items():
                    if 'direction' in name.lower():
                        GPIO.output(pin, direction)
                        
                self.metrics.speed = self.current_speed
            except Exception as e:
                self.logger.error(f"Speed update error: {e}")
                self._emergency_stop()
                
    def _monitor_motors(self):
        """Monitor motor safety conditions."""
        while not self._stop_event.is_set():
            try:
                # Read current and temperature
                current = self._read_current()
                temperature = self._read_temperature()
                
                with self._lock:
                    # Update metrics
                    self.metrics.current = current
                    self.metrics.temperature = temperature
                    
                    # Check for overcurrent
                    if current > self.max_current:
                        self.metrics.state = MotorState.ERROR
                        self._emergency_stop()
                        self.logger.error(f"Overcurrent detected: {current}A")
                        
                    # Check for overheating
                    elif temperature > self.max_temp:
                        self.metrics.state = MotorState.OVERHEATED
                        self._emergency_stop()
                        self.logger.error(f"Overheating detected: {temperature}°C")
                        
                    # Check for stall
                    elif self._detect_stall():
                        self.metrics.state = MotorState.STALLED
                        self._emergency_stop()
                        self.logger.error("Motor stall detected")
                        
                    # Normal operation
                    else:
                        self.metrics.state = MotorState.RUNNING
                        self._update_speed()
                        
                time.sleep(self.update_rate)
                
            except Exception as e:
                self.logger.error(f"Motor monitoring error: {e}")
                time.sleep(1.0)
                
    def set_speed(self, speed: float):
        """Set motor speed with safety checks."""
        if not (-1.0 <= speed <= 1.0):
            raise ValueError("Speed must be between -1.0 and 1.0")
            
        with self._lock:
            if self.metrics.state != MotorState.ERROR:
                self.target_speed = speed
                
    def _emergency_stop(self):
        """Emergency stop all motors."""
        with self._lock:
            try:
                self.target_speed = 0.0
                self.current_speed = 0.0
                for pwm in self.pwm.values():
                    pwm.ChangeDutyCycle(0)
                for pin in self.motor_pins.values():
                    GPIO.output(pin, GPIO.LOW)
            except Exception as e:
                self.logger.error(f"Emergency stop error: {e}")
                
    def cleanup(self):
        """Cleanup resources."""
        self._stop_event.set()
        try:
            for pwm in self.pwm.values():
                pwm.stop()
            self.spi.close()
            GPIO.cleanup(list(self.motor_pins.values()) +
                        list(self.encoder_pins.values()) +
                        [self.current_sense_pin, self.temp_sense_pin])
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
            
    def __del__(self):
        """Cleanup resources."""
        self.cleanup() 