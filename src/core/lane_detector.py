"""
Lane Detection for PiRobot V.4
Handles lane detection and following using computer vision
"""

import logging
import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum, auto


class LaneType(Enum):
    """Types of lanes detected"""
    SOLID = auto()
    DASHED = auto()
    DOUBLE = auto()
    UNKNOWN = auto()


@dataclass
class LaneInfo:
    """Lane detection information"""
    left_lane: Optional[np.ndarray]  # Left lane line points
    right_lane: Optional[np.ndarray]  # Right lane line points
    center_line: Optional[np.ndarray]  # Center line between lanes
    lane_type: LaneType
    confidence: float
    curvature: float
    offset: float  # Offset from center of lane


class LaneDetector:
    """Lane detection and following system"""
    
    def __init__(self, 
                 image_width: int = 640,
                 image_height: int = 480,
                 logger_name: str = "PiRobot.LaneDetector"):
        self.image_width = image_width
        self.image_height = image_height
        self.logger = logging.getLogger(logger_name)
        
        # Lane detection parameters
        self.roi_vertices = self._get_roi_vertices()
        self.canny_low = 50
        self.canny_high = 150
        self.hough_rho = 1
        self.hough_theta = np.pi/180
        self.hough_threshold = 50
        self.hough_min_line_len = 50
        self.hough_max_line_gap = 150
        
        # Lane tracking
        self.previous_left_lane = None
        self.previous_right_lane = None
        self.lane_history = []
        self.max_history = 5
        
    def _get_roi_vertices(self) -> np.ndarray:
        """Get region of interest vertices for lane detection"""
        # Define trapezoid for lane detection (focus on lower half of image)
        bottom_left = (0, self.image_height)
        top_left = (self.image_width * 0.4, self.image_height * 0.6)
        top_right = (self.image_width * 0.6, self.image_height * 0.6)
        bottom_right = (self.image_width, self.image_height)
        
        return np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        
    def detect_lanes(self, image: np.ndarray) -> LaneInfo:
        """Detect lanes in the given image"""
        try:
            # Preprocess image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
            
            # Apply region of interest mask
            mask = np.zeros_like(edges)
            cv2.fillPoly(mask, self.roi_vertices, 255)
            masked_edges = cv2.bitwise_and(edges, mask)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(
                masked_edges,
                self.hough_rho,
                self.hough_theta,
                self.hough_threshold,
                np.array([]),
                self.hough_min_line_len,
                self.hough_max_line_gap
            )
            
            if lines is not None:
                # Separate left and right lane lines
                left_lines, right_lines = self._separate_lines(lines)
                
                # Fit lines to points
                left_lane = self._fit_line(left_lines)
                right_lane = self._fit_line(right_lines)
                
                # Calculate lane information
                center_line = self._calculate_center_line(left_lane, right_lane)
                lane_type = self._classify_lane_type(left_lane, right_lane)
                confidence = self._calculate_confidence(left_lane, right_lane)
                curvature = self._calculate_curvature(left_lane, right_lane)
                offset = self._calculate_offset(center_line)
                
                # Update tracking
                self._update_tracking(left_lane, right_lane)
                
                return LaneInfo(
                    left_lane=left_lane,
                    right_lane=right_lane,
                    center_line=center_line,
                    lane_type=lane_type,
                    confidence=confidence,
                    curvature=curvature,
                    offset=offset
                )
            else:
                # No lanes detected, use previous data if available
                return self._get_previous_lane_info()
                
        except Exception as e:
            self.logger.error(f"Error in lane detection: {e}")
            return self._get_previous_lane_info()
            
    def _separate_lines(self, lines: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Separate lines into left and right lanes based on slope"""
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:  # Avoid division by zero
                slope = (y2 - y1) / (x2 - x1)
                if slope < -0.5:  # Left lane (negative slope)
                    left_lines.append(line[0])
                elif slope > 0.5:  # Right lane (positive slope)
                    right_lines.append(line[0])
                    
        return left_lines, right_lines
        
    def _fit_line(self, lines: List[np.ndarray]) -> Optional[np.ndarray]:
        """Fit a line to the given points"""
        if not lines:
            return None
            
        try:
            # Extract all points
            points = []
            for line in lines:
                points.extend([(line[0], line[1]), (line[2], line[3])])
                
            if len(points) < 2:
                return None
                
            # Convert to numpy array
            points = np.array(points)
            
            # Fit line using least squares
            if len(points) >= 2:
                # Use polyfit for line fitting
                x = points[:, 0]
                y = points[:, 1]
                
                # Fit polynomial of degree 1 (line)
                coeffs = np.polyfit(x, y, 1)
                
                # Create line points for visualization
                y1 = self.image_height
                y2 = int(self.image_height * 0.6)
                x1 = int((y1 - coeffs[1]) / coeffs[0]) if coeffs[0] != 0 else 0
                x2 = int((y2 - coeffs[1]) / coeffs[0]) if coeffs[0] != 0 else 0
                
                return np.array([x1, y1, x2, y2])
                
        except Exception as e:
            self.logger.debug(f"Error fitting line: {e}")
            
        return None
        
    def _calculate_center_line(self, 
                              left_lane: Optional[np.ndarray], 
                              right_lane: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Calculate center line between left and right lanes"""
        if left_lane is None or right_lane is None:
            return None
            
        try:
            # Calculate midpoint between lanes
            x1 = (left_lane[0] + right_lane[0]) // 2
            y1 = (left_lane[1] + right_lane[1]) // 2
            x2 = (left_lane[2] + right_lane[2]) // 2
            y2 = (left_lane[3] + right_lane[3]) // 2
            
            return np.array([x1, y1, x2, y2])
        except Exception:
            return None
            
    def _classify_lane_type(self, 
                           left_lane: Optional[np.ndarray], 
                           right_lane: Optional[np.ndarray]) -> LaneType:
        """Classify the type of lane detected"""
        if left_lane is None or right_lane is None:
            return LaneType.UNKNOWN
            
        # Simple classification based on lane characteristics
        # This could be enhanced with more sophisticated analysis
        return LaneType.SOLID  # Default to solid lane
        
    def _calculate_confidence(self, 
                             left_lane: Optional[np.ndarray], 
                             right_lane: Optional[np.ndarray]) -> float:
        """Calculate confidence in lane detection"""
        if left_lane is None or right_lane is None:
            return 0.0
            
        # Simple confidence calculation
        # Could be enhanced with more sophisticated metrics
        return 0.8  # Default confidence
        
    def _calculate_curvature(self, 
                           left_lane: Optional[np.ndarray], 
                           right_lane: Optional[np.ndarray]) -> float:
        """Calculate lane curvature"""
        if left_lane is None or right_lane is None:
            return 0.0
            
        # Simple curvature calculation
        # Could be enhanced with more sophisticated curve fitting
        return 0.0  # Default to straight lane
        
    def _calculate_offset(self, center_line: Optional[np.ndarray]) -> float:
        """Calculate offset from center of lane"""
        if center_line is None:
            return 0.0
            
        # Calculate offset from image center
        image_center = self.image_width // 2
        lane_center = (center_line[0] + center_line[2]) // 2
        
        return lane_center - image_center
        
    def _update_tracking(self, 
                        left_lane: Optional[np.ndarray], 
                        right_lane: Optional[np.ndarray]) -> None:
        """Update lane tracking history"""
        self.previous_left_lane = left_lane
        self.previous_right_lane = right_lane
        
        # Add to history
        self.lane_history.append((left_lane, right_lane))
        if len(self.lane_history) > self.max_history:
            self.lane_history.pop(0)
            
    def _get_previous_lane_info(self) -> LaneInfo:
        """Get previous lane information when no lanes are detected"""
        return LaneInfo(
            left_lane=self.previous_left_lane,
            right_lane=self.previous_right_lane,
            center_line=None,
            lane_type=LaneType.UNKNOWN,
            confidence=0.0,
            curvature=0.0,
            offset=0.0
        )
        
    def draw_lanes(self, image: np.ndarray, lane_info: LaneInfo) -> np.ndarray:
        """Draw detected lanes on the image"""
        result = image.copy()
        
        # Draw left lane
        if lane_info.left_lane is not None:
            x1, y1, x2, y2 = lane_info.left_lane
            cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 3)
            
        # Draw right lane
        if lane_info.right_lane is not None:
            x1, y1, x2, y2 = lane_info.right_lane
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
        # Draw center line
        if lane_info.center_line is not None:
            x1, y1, x2, y2 = lane_info.center_line
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
        return result


# Global lane detector instance
lane_detector = LaneDetector()


def get_lane_detector() -> LaneDetector:
    """Get the global lane detector instance"""
    return lane_detector
