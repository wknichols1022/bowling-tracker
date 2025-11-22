"""
Speed and trajectory calculation module for bowling analysis.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class CalibrationPoint:
    """Represents a calibration measurement."""
    pixel_distance: float
    actual_distance_feet: float
    confidence: float = 1.0


class SpeedCalculator:
    """Calculates ball speed and trajectory metrics from tracking data."""

    def __init__(self,
                 lane_length_feet: float = 60.0,
                 ball_diameter_inches: float = 8.5,
                 fps: float = 30.0):
        """
        Initialize speed calculator.

        Args:
            lane_length_feet: Distance from foul line to headpin (standard: 60 feet)
            ball_diameter_inches: Bowling ball diameter (standard: 8.5 inches)
            fps: Video frame rate (frames per second)

        Raises:
            ValueError: If parameters are invalid
        """
        if lane_length_feet <= 0 or lane_length_feet > 100:
            raise ValueError(f"Invalid lane_length_feet: {lane_length_feet}")

        if ball_diameter_inches <= 0 or ball_diameter_inches > 12:
            raise ValueError(f"Invalid ball_diameter_inches: {ball_diameter_inches}")

        if fps <= 0 or fps > 240:
            raise ValueError(f"Invalid fps: {fps}")

        self.lane_length_feet = lane_length_feet
        self.ball_diameter_inches = ball_diameter_inches
        self.fps = fps
        self.pixels_per_foot = None
        self.calibration_points: List[CalibrationPoint] = []
        self.calibration_confidence = 0.0

    def add_calibration_point(self,
                             pixel_distance: float,
                             actual_distance_feet: float,
                             confidence: float = 1.0) -> None:
        """
        Add a calibration measurement point.

        Args:
            pixel_distance: Distance in pixels
            actual_distance_feet: Actual distance in feet
            confidence: Confidence score (0.0 to 1.0)

        Raises:
            ValueError: If inputs are invalid
        """
        if pixel_distance <= 0:
            raise ValueError(f"Invalid pixel_distance: {pixel_distance}. Must be positive.")

        if actual_distance_feet <= 0:
            raise ValueError(f"Invalid actual_distance_feet: {actual_distance_feet}. Must be positive.")

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Invalid confidence: {confidence}. Must be between 0.0 and 1.0.")

        self.calibration_points.append(CalibrationPoint(
            pixel_distance=pixel_distance,
            actual_distance_feet=actual_distance_feet,
            confidence=confidence
        ))

    def calculate_calibration(self, min_points: int = 2) -> Dict:
        """
        Calculate calibration from multiple points using weighted average.

        Args:
            min_points: Minimum number of calibration points required

        Returns:
            Dictionary with calibration results

        Raises:
            ValueError: If insufficient calibration points
        """
        if len(self.calibration_points) < min_points:
            raise ValueError(f"Insufficient calibration points: {len(self.calibration_points)}. Need at least {min_points}.")

        # Calculate pixels-per-foot for each calibration point
        calibration_ratios = []
        weights = []

        for point in self.calibration_points:
            ratio = point.pixel_distance / point.actual_distance_feet
            calibration_ratios.append(ratio)
            weights.append(point.confidence)

        # Weighted average
        weights_array = np.array(weights)
        ratios_array = np.array(calibration_ratios)

        self.pixels_per_foot = np.average(ratios_array, weights=weights_array)

        # Calculate standard deviation for confidence estimate
        variance = np.average((ratios_array - self.pixels_per_foot)**2, weights=weights_array)
        std_dev = np.sqrt(variance)

        # Confidence based on consistency (lower std_dev = higher confidence)
        # Normalize std_dev to 0-1 range (assuming max reasonable std_dev is 20% of mean)
        max_acceptable_std = self.pixels_per_foot * 0.2
        consistency_score = max(0.0, 1.0 - (std_dev / max_acceptable_std))

        # Overall confidence is average of point confidences weighted by consistency
        avg_point_confidence = np.mean(weights)
        self.calibration_confidence = (consistency_score * 0.6 + avg_point_confidence * 0.4)

        return {
            "pixels_per_foot": self.pixels_per_foot,
            "num_points": len(self.calibration_points),
            "std_dev": std_dev,
            "confidence": self.calibration_confidence,
            "ratios": calibration_ratios
        }

    def calibrate_from_ball(self, ball_radius_pixels: int, confidence: float = 0.7) -> float:
        """
        Calibrate pixel-to-feet ratio using known ball diameter.

        Args:
            ball_radius_pixels: Ball radius in pixels
            confidence: Confidence in this measurement (0.0 to 1.0)

        Returns:
            Pixels per foot conversion factor

        Raises:
            ValueError: If ball_radius_pixels is invalid
        """
        if ball_radius_pixels <= 0:
            raise ValueError(f"Invalid ball_radius_pixels: {ball_radius_pixels}. Must be positive.")

        ball_diameter_feet = self.ball_diameter_inches / 12.0
        ball_diameter_pixels = ball_radius_pixels * 2

        # Add as calibration point
        self.add_calibration_point(
            pixel_distance=ball_diameter_pixels,
            actual_distance_feet=ball_diameter_feet,
            confidence=confidence
        )

        # Recalculate with all points
        try:
            cal_result = self.calculate_calibration(min_points=1)
            print(f"Calibrated: {self.pixels_per_foot:.2f} pixels/foot (confidence: {cal_result['confidence']:.2f})")
            return self.pixels_per_foot
        except ValueError:
            # Fallback for first point
            self.pixels_per_foot = ball_diameter_pixels / ball_diameter_feet
            self.calibration_confidence = confidence
            print(f"Calibrated: {self.pixels_per_foot:.2f} pixels/foot (single point)")
            return self.pixels_per_foot

    def calibrate_from_lane(self,
                           pixel_distance: float,
                           actual_distance_feet: float,
                           confidence: float = 1.0) -> float:
        """
        Calibrate using known lane distance.

        Args:
            pixel_distance: Distance in pixels
            actual_distance_feet: Actual distance in feet
            confidence: Confidence in this measurement (0.0 to 1.0)

        Returns:
            Pixels per foot conversion factor

        Raises:
            ValueError: If inputs are invalid
        """
        self.add_calibration_point(pixel_distance, actual_distance_feet, confidence)

        # Recalculate with all points
        try:
            cal_result = self.calculate_calibration(min_points=1)
            print(f"Calibrated: {self.pixels_per_foot:.2f} pixels/foot (confidence: {cal_result['confidence']:.2f})")
            return self.pixels_per_foot
        except ValueError:
            # Fallback for first point
            self.pixels_per_foot = pixel_distance / actual_distance_feet
            self.calibration_confidence = confidence
            print(f"Calibrated: {self.pixels_per_foot:.2f} pixels/foot (single point)")
            return self.pixels_per_foot

    def get_calibration_status(self) -> Dict:
        """
        Get current calibration status.

        Returns:
            Dictionary with calibration information
        """
        return {
            "is_calibrated": self.pixels_per_foot is not None,
            "pixels_per_foot": self.pixels_per_foot,
            "num_calibration_points": len(self.calibration_points),
            "confidence": self.calibration_confidence,
            "calibration_points": [
                {
                    "pixel_distance": cp.pixel_distance,
                    "actual_distance_feet": cp.actual_distance_feet,
                    "confidence": cp.confidence
                }
                for cp in self.calibration_points
            ]
        }
    
    def calculate_distance(self, point1: Tuple[int, int], 
                          point2: Tuple[int, int]) -> float:
        """
        Calculate distance between two points in pixels.
        
        Args:
            point1: (x, y) coordinates
            point2: (x, y) coordinates
            
        Returns:
            Distance in pixels
        """
        x1, y1 = point1
        x2, y2 = point2
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def calculate_speed_from_trajectory(self,
                                       trajectory: List[Tuple[int, int, int]],
                                       start_frame: int = None,
                                       end_frame: int = None) -> Dict[str, float]:
        """
        Calculate ball speed from trajectory data.

        For head-mounted cameras, we use the known lane length (60 feet) and time elapsed
        to calculate average speed, rather than pixel-to-pixel distance which is unreliable
        when the camera is moving.

        Args:
            trajectory: List of (x, y, frame_number) tuples
            start_frame: Optional starting frame for calculation
            end_frame: Optional ending frame for calculation

        Returns:
            Dictionary with speed in various units
        """
        if not trajectory or len(trajectory) < 2:
            return {"error": "Insufficient trajectory data"}

        # Filter trajectory by frame range if specified
        if start_frame is not None or end_frame is not None:
            start_frame = start_frame or 0
            end_frame = end_frame or float('inf')
            trajectory = [(x, y, f) for x, y, f in trajectory
                         if start_frame <= f <= end_frame]

        if len(trajectory) < 2:
            return {"error": "Insufficient trajectory data in specified range"}

        # For head-mounted cameras, use vertical (y-axis) pixel distance
        # as the primary measure of down-lane travel
        if self.pixels_per_foot is None:
            return {"error": "Must calibrate before calculating speed"}

        # Calculate vertical pixel distance (down-lane travel)
        y_start = trajectory[0][1]
        y_end = trajectory[-1][1]
        vertical_pixels = abs(y_end - y_start)

        # Convert to feet using calibration
        distance_feet = vertical_pixels / self.pixels_per_foot

        # Apply lane length scaling factor: typical bowling shot is 60 feet
        # This multiplier accounts for camera perspective and angle
        # Adjust this value based on actual measurements (start with 3.0x multiplier)
        perspective_multiplier = 3.0
        distance_feet = distance_feet * perspective_multiplier

        # Calculate time elapsed
        frame_start = trajectory[0][2]
        frame_end = trajectory[-1][2]
        frames_elapsed = frame_end - frame_start
        time_elapsed_seconds = frames_elapsed / self.fps

        if time_elapsed_seconds == 0:
            return {"error": "No time elapsed between trajectory points"}

        # Calculate speed in different units
        speed_fps = distance_feet / time_elapsed_seconds  # Feet per second
        speed_mph = speed_fps * 0.681818  # Miles per hour

        return {
            "speed_fps": round(speed_fps, 2),
            "speed_mph": round(speed_mph, 2),
            "distance_feet": round(distance_feet, 2),
            "time_seconds": round(time_elapsed_seconds, 2),
            "frames_analyzed": frames_elapsed,
        }
    
    def calculate_average_speed(self, 
                               trajectory: List[Tuple[int, int, int]],
                               window_frames: int = 5) -> List[float]:
        """
        Calculate rolling average speed throughout trajectory.
        
        Args:
            trajectory: List of (x, y, frame_number) tuples
            window_frames: Number of frames for rolling average
            
        Returns:
            List of average speeds (mph) at each point
        """
        if not trajectory or len(trajectory) < window_frames:
            return []
        
        speeds = []
        for i in range(len(trajectory) - window_frames):
            window = trajectory[i:i + window_frames]
            speed_data = self.calculate_speed_from_trajectory(window)
            if "speed_mph" in speed_data:
                speeds.append(speed_data["speed_mph"])
        
        return speeds
    
    def detect_hook_point(self,
                         trajectory: List[Tuple[int, int, int]],
                         smoothing_window: int = 3) -> Optional[Dict]:
        """
        Detect the breakpoint where the ball starts hooking.

        The hook point is where lateral velocity (x-direction) changes most dramatically,
        indicating the transition from skid phase to hook phase.

        Args:
            trajectory: List of (x, y, frame_number) tuples
            smoothing_window: Window size for smoothing lateral velocity

        Returns:
            Dictionary with hook point information or None if no hook detected
        """
        if len(trajectory) < 5:
            return None

        # Extract coordinates
        x_coords = np.array([p[0] for p in trajectory])
        y_coords = np.array([p[1] for p in trajectory])

        # Calculate lateral velocity (change in x per unit y distance)
        # Using y as the "time" axis since ball moves down the lane
        lateral_velocities = []
        for i in range(1, len(trajectory)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            if dy != 0:
                lateral_velocities.append(dx / dy)
            else:
                lateral_velocities.append(0)

        if len(lateral_velocities) < smoothing_window:
            return None

        # Smooth the lateral velocities
        smoothed_velocities = np.convolve(
            lateral_velocities,
            np.ones(smoothing_window)/smoothing_window,
            mode='valid'
        )

        # Calculate acceleration (change in lateral velocity)
        lateral_accelerations = np.diff(smoothed_velocities)

        if len(lateral_accelerations) == 0:
            return None

        # Find the point of maximum lateral acceleration (breakpoint)
        max_accel_idx = np.argmax(np.abs(lateral_accelerations))

        # Adjust index for smoothing offset
        hook_point_idx = max_accel_idx + smoothing_window

        # Ensure index is within bounds
        if hook_point_idx >= len(trajectory):
            hook_point_idx = len(trajectory) - 1

        hook_point = trajectory[hook_point_idx]

        # Calculate distance down lane (in pixels from start)
        distance_from_start = self.calculate_distance(
            trajectory[0][:2],
            hook_point[:2]
        )

        # Convert to feet if calibrated
        if self.pixels_per_foot:
            distance_feet = distance_from_start / self.pixels_per_foot
        else:
            distance_feet = None

        # Calculate total hook amount (lateral movement after breakpoint)
        hook_amount_pixels = abs(x_coords[-1] - hook_point[0])

        return {
            "hook_point_index": hook_point_idx,
            "hook_point_x": int(hook_point[0]),
            "hook_point_y": int(hook_point[1]),
            "hook_point_frame": int(hook_point[2]),
            "distance_from_start_pixels": round(distance_from_start, 2),
            "distance_from_start_feet": round(distance_feet, 2) if distance_feet else None,
            "hook_amount_pixels": round(hook_amount_pixels, 2),
            "total_trajectory_points": len(trajectory)
        }

    def analyze_trajectory_shape(self,
                                trajectory: List[Tuple[int, int, int]]) -> Dict:
        """
        Analyze trajectory shape (straight, hook, curve).

        Args:
            trajectory: List of (x, y, frame_number) tuples

        Returns:
            Dictionary with trajectory analysis
        """
        if len(trajectory) < 3:
            return {"error": "Insufficient data for trajectory analysis"}

        # Extract x and y coordinates
        x_coords = [p[0] for p in trajectory]
        y_coords = [p[1] for p in trajectory]

        # Calculate lateral movement (x-direction deviation)
        x_start = x_coords[0]
        x_end = x_coords[-1]
        x_range = max(x_coords) - min(x_coords)
        lateral_deviation = abs(x_end - x_start)

        # Fit a polynomial to detect curve
        if len(trajectory) > 5:
            poly_coeffs = np.polyfit(y_coords, x_coords, 2)
            curve_coefficient = poly_coeffs[0]  # Second-order coefficient
        else:
            curve_coefficient = 0

        # Determine trajectory type
        if abs(curve_coefficient) < 0.0001:
            trajectory_type = "straight"
        elif curve_coefficient < 0:
            trajectory_type = "hook_left"
        else:
            trajectory_type = "hook_right"

        # Detect hook point
        hook_point_data = self.detect_hook_point(trajectory)

        result = {
            "trajectory_type": trajectory_type,
            "lateral_deviation_pixels": round(lateral_deviation, 2),
            "x_range_pixels": round(x_range, 2),
            "curve_coefficient": round(curve_coefficient, 6),
            "total_points": len(trajectory),
        }

        # Add hook point data if detected
        if hook_point_data:
            result["hook_point"] = hook_point_data

        return result
    
    def calculate_release_speed(self, 
                               trajectory: List[Tuple[int, int, int]],
                               initial_frames: int = 10) -> Optional[Dict[str, float]]:
        """
        Calculate ball speed at release (first N frames).
        
        Args:
            trajectory: List of (x, y, frame_number) tuples
            initial_frames: Number of initial frames to analyze
            
        Returns:
            Dictionary with release speed data
        """
        if len(trajectory) < initial_frames:
            initial_frames = len(trajectory)
        
        release_trajectory = trajectory[:initial_frames]
        return self.calculate_speed_from_trajectory(release_trajectory)
    
    def get_statistics(self, 
                      speed_data_list: List[Dict[str, float]]) -> Dict:
        """
        Calculate aggregate statistics from multiple shots.
        
        Args:
            speed_data_list: List of speed dictionaries from multiple shots
            
        Returns:
            Dictionary with aggregate statistics
        """
        speeds_mph = [d["speed_mph"] for d in speed_data_list 
                     if "speed_mph" in d]
        
        if not speeds_mph:
            return {"error": "No valid speed data"}
        
        return {
            "average_speed_mph": round(np.mean(speeds_mph), 2),
            "median_speed_mph": round(np.median(speeds_mph), 2),
            "std_dev_mph": round(np.std(speeds_mph), 2),
            "min_speed_mph": round(min(speeds_mph), 2),
            "max_speed_mph": round(max(speeds_mph), 2),
            "total_shots": len(speeds_mph),
        }


def main():
    """Example usage of SpeedCalculator."""
    print("SpeedCalculator Example Usage:")
    print("\n# Initialize calculator")
    print("calc = SpeedCalculator(lane_length_feet=60.0, fps=30.0)")
    print("\n# Calibrate using ball size")
    print("calc.calibrate_from_ball(ball_radius_pixels=25)")
    print("\n# Calculate speed from trajectory")
    print("trajectory = [(100, 100, 0), (150, 200, 10), (200, 300, 20)]")
    print("speed = calc.calculate_speed_from_trajectory(trajectory)")
    print("print(f\"Speed: {speed['speed_mph']} mph\")")
    print("\n# Analyze trajectory shape")
    print("analysis = calc.analyze_trajectory_shape(trajectory)")
    print("print(f\"Type: {analysis['trajectory_type']}\")")


if __name__ == "__main__":
    main()
