"""
Waypoint Navigation for PiRobot V.4
Handles waypoint-based navigation and path planning
"""

import logging
import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import json


class NavigationState(Enum):
    """Navigation state enumeration"""
    IDLE = auto()
    NAVIGATING = auto()
    ARRIVED = auto()
    LOST = auto()
    ERROR = auto()


@dataclass
class Waypoint:
    """Waypoint data structure"""
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    name: Optional[str] = None
    tolerance: float = 2.0  # meters
    speed_limit: Optional[float] = None


@dataclass
class NavigationInfo:
    """Navigation information"""
    current_waypoint: Optional[Waypoint]
    next_waypoint: Optional[Waypoint]
    distance_to_target: float
    bearing_to_target: float
    navigation_state: NavigationState
    progress_percentage: float


class WaypointNavigator:
    """Waypoint-based navigation system"""
    
    def __init__(self, 
                 waypoint_file: Optional[str] = None,
                 logger_name: str = "PiRobot.WaypointNavigator"):
        self.logger = logging.getLogger(logger_name)
        self.waypoint_file = waypoint_file or "waypoints.json"
        
        # Navigation state
        self.waypoints: List[Waypoint] = []
        self.current_waypoint_index = 0
        self.navigation_state = NavigationState.IDLE
        
        # Navigation parameters
        self.max_speed = 1.0  # m/s
        self.turn_speed = 0.5  # m/s
        self.arrival_threshold = 2.0  # meters
        
        # Load waypoints if file exists
        if Path(self.waypoint_file).exists():
            self.load_waypoints()
        else:
            self.logger.info("No waypoint file found, starting with empty waypoint list")
            
    def add_waypoint(self, 
                    latitude: float, 
                    longitude: float, 
                    altitude: Optional[float] = None,
                    name: Optional[str] = None,
                    tolerance: float = 2.0,
                    speed_limit: Optional[float] = None) -> None:
        """Add a waypoint to the navigation list"""
        waypoint = Waypoint(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            name=name,
            tolerance=tolerance,
            speed_limit=speed_limit
        )
        
        self.waypoints.append(waypoint)
        self.logger.info(f"Added waypoint: {name or f'({latitude}, {longitude})'}")
        
    def remove_waypoint(self, index: int) -> bool:
        """Remove a waypoint by index"""
        if 0 <= index < len(self.waypoints):
            removed = self.waypoints.pop(index)
            self.logger.info(f"Removed waypoint: {removed.name or f'({removed.latitude}, {removed.longitude})'}")
            return True
        return False
        
    def clear_waypoints(self) -> None:
        """Clear all waypoints"""
        self.waypoints.clear()
        self.current_waypoint_index = 0
        self.navigation_state = NavigationState.IDLE
        self.logger.info("Cleared all waypoints")
        
    def load_waypoints(self) -> bool:
        """Load waypoints from file"""
        try:
            with open(self.waypoint_file, 'r') as f:
                data = json.load(f)
                
            self.waypoints.clear()
            for wp_data in data.get('waypoints', []):
                waypoint = Waypoint(
                    latitude=wp_data['latitude'],
                    longitude=wp_data['longitude'],
                    altitude=wp_data.get('altitude'),
                    name=wp_data.get('name'),
                    tolerance=wp_data.get('tolerance', 2.0),
                    speed_limit=wp_data.get('speed_limit')
                )
                self.waypoints.append(waypoint)
                
            self.logger.info(f"Loaded {len(self.waypoints)} waypoints from {self.waypoint_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading waypoints: {e}")
            return False
            
    def save_waypoints(self) -> bool:
        """Save waypoints to file"""
        try:
            data = {
                'waypoints': [
                    {
                        'latitude': wp.latitude,
                        'longitude': wp.longitude,
                        'altitude': wp.altitude,
                        'name': wp.name,
                        'tolerance': wp.tolerance,
                        'speed_limit': wp.speed_limit
                    }
                    for wp in self.waypoints
                ]
            }
            
            with open(self.waypoint_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.logger.info(f"Saved {len(self.waypoints)} waypoints to {self.waypoint_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving waypoints: {e}")
            return False
            
    def start_navigation(self) -> bool:
        """Start waypoint navigation"""
        if not self.waypoints:
            self.logger.warning("No waypoints available for navigation")
            return False
            
        self.current_waypoint_index = 0
        self.navigation_state = NavigationState.NAVIGATING
        self.logger.info("Started waypoint navigation")
        return True
        
    def stop_navigation(self) -> None:
        """Stop waypoint navigation"""
        self.navigation_state = NavigationState.IDLE
        self.logger.info("Stopped waypoint navigation")
        
    def update_navigation(self, 
                         current_lat: float, 
                         current_lon: float,
                         current_heading: float) -> NavigationInfo:
        """Update navigation based on current position"""
        if self.navigation_state != NavigationState.NAVIGATING:
            return NavigationInfo(
                current_waypoint=None,
                next_waypoint=None,
                distance_to_target=0.0,
                bearing_to_target=0.0,
                navigation_state=self.navigation_state,
                progress_percentage=0.0
            )
            
        if self.current_waypoint_index >= len(self.waypoints):
            self.navigation_state = NavigationState.ARRIVED
            return NavigationInfo(
                current_waypoint=None,
                next_waypoint=None,
                distance_to_target=0.0,
                bearing_to_target=0.0,
                navigation_state=self.navigation_state,
                progress_percentage=100.0
            )
            
        current_waypoint = self.waypoints[self.current_waypoint_index]
        next_waypoint = (self.waypoints[self.current_waypoint_index + 1] 
                        if self.current_waypoint_index + 1 < len(self.waypoints) 
                        else None)
        
        # Calculate distance and bearing to current waypoint
        distance = self._calculate_distance(
            current_lat, current_lon,
            current_waypoint.latitude, current_waypoint.longitude
        )
        
        bearing = self._calculate_bearing(
            current_lat, current_lon,
            current_waypoint.latitude, current_waypoint.longitude
        )
        
        # Check if we've arrived at the current waypoint
        if distance <= current_waypoint.tolerance:
            self.current_waypoint_index += 1
            self.logger.info(f"Arrived at waypoint: {current_waypoint.name or f'({current_waypoint.latitude}, {current_waypoint.longitude})'}")
            
            if self.current_waypoint_index >= len(self.waypoints):
                self.navigation_state = NavigationState.ARRIVED
                self.logger.info("Navigation completed - all waypoints reached")
        
        # Calculate progress percentage
        progress = (self.current_waypoint_index / len(self.waypoints)) * 100.0
        
        return NavigationInfo(
            current_waypoint=current_waypoint,
            next_waypoint=next_waypoint,
            distance_to_target=distance,
            bearing_to_target=bearing,
            navigation_state=self.navigation_state,
            progress_percentage=progress
        )
        
    def get_navigation_commands(self, 
                               current_lat: float, 
                               current_lon: float,
                               current_heading: float) -> Tuple[float, float]:
        """Get navigation commands (steering, throttle) based on current position"""
        nav_info = self.update_navigation(current_lat, current_lon, current_heading)
        
        if nav_info.navigation_state != NavigationState.NAVIGATING:
            return 0.0, 0.0  # No movement
            
        # Calculate steering angle based on bearing difference
        bearing_diff = nav_info.bearing_to_target - current_heading
        
        # Normalize bearing difference to [-180, 180]
        while bearing_diff > 180:
            bearing_diff -= 360
        while bearing_diff < -180:
            bearing_diff += 360
            
        # Convert to steering command (-1 to 1)
        steering = np.clip(bearing_diff / 90.0, -1.0, 1.0)
        
        # Calculate throttle based on distance
        if nav_info.distance_to_target > 10.0:
            throttle = self.max_speed
        elif nav_info.distance_to_target > 5.0:
            throttle = self.max_speed * 0.7
        elif nav_info.distance_to_target > 2.0:
            throttle = self.turn_speed
        else:
            throttle = 0.1  # Slow approach
            
        return steering, throttle
        
    def _calculate_distance(self, 
                           lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in meters
        r = 6371000
        return c * r
        
    def _calculate_bearing(self, 
                          lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """Calculate bearing between two points"""
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlon = lon2 - lon1
        
        y = math.sin(dlon) * math.cos(lat2)
        x = (math.cos(lat1) * math.sin(lat2) - 
             math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
        
        bearing = math.atan2(y, x)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing
        
    def get_waypoint_list(self) -> List[Dict[str, Any]]:
        """Get list of waypoints with their information"""
        return [
            {
                'index': i,
                'latitude': wp.latitude,
                'longitude': wp.longitude,
                'altitude': wp.altitude,
                'name': wp.name,
                'tolerance': wp.tolerance,
                'speed_limit': wp.speed_limit,
                'is_current': i == self.current_waypoint_index
            }
            for i, wp in enumerate(self.waypoints)
        ]


# Global waypoint navigator instance
waypoint_navigator = WaypointNavigator()


def get_waypoint_navigator() -> WaypointNavigator:
    """Get the global waypoint navigator instance"""
    return waypoint_navigator
