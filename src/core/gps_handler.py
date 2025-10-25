"""
GPS Handler for PiRobot V.4
Handles GPS data collection and processing
"""

import logging
import threading
import time
from typing import Optional, Dict, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import serial
import pynmea2
import gpsd


@dataclass
class GPSData:
    """GPS data container"""
    timestamp: datetime
    latitude: Optional[float]
    longitude: Optional[float]
    altitude: Optional[float]
    speed: Optional[float]
    heading: Optional[float]
    satellites: Optional[int]
    fix_quality: Optional[int]
    hdop: Optional[float]  # Horizontal Dilution of Precision


class GPSHandler:
    """GPS data handler for PiRobot"""
    
    def __init__(self, 
                 serial_port: str = "/dev/ttyUSB0",
                 baud_rate: int = 9600,
                 logger_name: str = "PiRobot.GPSHandler"):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.logger = logging.getLogger(logger_name)
        
        self.current_data: Optional[GPSData] = None
        self.data_callbacks: list[Callable[[GPSData], None]] = []
        
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Try to connect to gpsd if available
        self.gpsd_available = False
        try:
            gpsd.connect()
            self.gpsd_available = True
            self.logger.info("Connected to gpsd")
        except Exception as e:
            self.logger.warning(f"gpsd not available: {e}")
            
    def register_callback(self, callback: Callable[[GPSData], None]) -> None:
        """Register a callback for GPS data updates"""
        self.data_callbacks.append(callback)
        
    def start(self) -> None:
        """Start GPS data collection"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._collect_data, daemon=True)
        self.thread.start()
        self.logger.info("GPS handler started")
        
    def stop(self) -> None:
        """Stop GPS data collection"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.logger.info("GPS handler stopped")
        
    def _collect_data(self) -> None:
        """Main data collection loop"""
        while self.running:
            try:
                if self.gpsd_available:
                    self._collect_from_gpsd()
                else:
                    self._collect_from_serial()
                    
                time.sleep(1.0)  # Update every second
                
            except Exception as e:
                self.logger.error(f"Error in GPS data collection: {e}")
                time.sleep(5.0)  # Wait longer on error
                
    def _collect_from_gpsd(self) -> None:
        """Collect GPS data from gpsd"""
        try:
            packet = gpsd.get_current()
            if packet.mode >= 2:  # 2D or 3D fix
                data = GPSData(
                    timestamp=datetime.now(),
                    latitude=packet.lat,
                    longitude=packet.lon,
                    altitude=packet.alt,
                    speed=packet.speed,
                    heading=packet.track,
                    satellites=packet.sats,
                    fix_quality=packet.mode,
                    hdop=packet.eph
                )
                self._update_data(data)
        except Exception as e:
            self.logger.debug(f"gpsd error: {e}")
            
    def _collect_from_serial(self) -> None:
        """Collect GPS data from serial port"""
        try:
            with serial.Serial(self.serial_port, self.baud_rate, timeout=1) as ser:
                line = ser.readline().decode('ascii', errors='ignore').strip()
                if line.startswith('$'):
                    try:
                        msg = pynmea2.parse(line)
                        if isinstance(msg, pynmea2.RMC):  # Recommended Minimum
                            if msg.latitude and msg.longitude:
                                data = GPSData(
                                    timestamp=datetime.now(),
                                    latitude=msg.latitude,
                                    longitude=msg.longitude,
                                    altitude=None,
                                    speed=msg.spd_over_grnd,
                                    heading=msg.true_course,
                                    satellites=None,
                                    fix_quality=1 if msg.status == 'A' else 0,
                                    hdop=None
                                )
                                self._update_data(data)
                    except pynmea2.ParseError:
                        pass  # Ignore parse errors
        except Exception as e:
            self.logger.debug(f"Serial GPS error: {e}")
            
    def _update_data(self, data: GPSData) -> None:
        """Update current data and notify callbacks"""
        self.current_data = data
        
        # Notify callbacks
        for callback in self.data_callbacks:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Error in GPS callback: {e}")
                
    def get_current_data(self) -> Optional[GPSData]:
        """Get current GPS data"""
        return self.current_data
        
    def get_position(self) -> Optional[Tuple[float, float]]:
        """Get current position as (latitude, longitude)"""
        if self.current_data and self.current_data.latitude and self.current_data.longitude:
            return (self.current_data.latitude, self.current_data.longitude)
        return None
        
    def get_distance_to(self, target_lat: float, target_lon: float) -> Optional[float]:
        """Calculate distance to target position in meters"""
        current_pos = self.get_position()
        if not current_pos:
            return None
            
        # Haversine formula for distance calculation
        import math
        
        lat1, lon1 = math.radians(current_pos[0]), math.radians(current_pos[1])
        lat2, lon2 = math.radians(target_lat), math.radians(target_lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in meters
        r = 6371000
        return c * r
        
    def get_bearing_to(self, target_lat: float, target_lon: float) -> Optional[float]:
        """Calculate bearing to target position in degrees"""
        current_pos = self.get_position()
        if not current_pos:
            return None
            
        import math
        
        lat1, lon1 = math.radians(current_pos[0]), math.radians(current_pos[1])
        lat2, lon2 = math.radians(target_lat), math.radians(target_lon)
        
        dlon = lon2 - lon1
        
        y = math.sin(dlon) * math.cos(lat2)
        x = (math.cos(lat1) * math.sin(lat2) - 
             math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
        
        bearing = math.atan2(y, x)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing


# Global GPS handler instance
gps_handler = GPSHandler()


def get_gps_handler() -> GPSHandler:
    """Get the global GPS handler instance"""
    return gps_handler
