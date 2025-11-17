"""
Ball tracking module using hybrid computer vision approach:
- Motion detection (background subtraction)
- Circular shape detection (Hough circles)
- Color validation (optional, not required)
- Size consistency validation
- Kalman filter for smoothing and prediction
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class DetectionCandidate:
    """Represents a potential ball detection."""
    x: int
    y: int
    radius: int
    confidence: float
    detection_method: str
    color_match: bool = False
    motion_score: float = 0.0
    shape_score: float = 0.0


class BallTracker:
    """Tracks bowling ball using hybrid detection approach."""

    def __init__(self,
                 color_lower: Optional[Tuple[int, int, int]] = None,
                 color_upper: Optional[Tuple[int, int, int]] = None,
                 min_radius: int = 15,
                 max_radius: int = 80,
                 enable_kalman: bool = True):
        """
        Initialize hybrid ball tracker.

        Args:
            color_lower: Optional lower HSV bound for ball color
            color_upper: Optional upper HSV bound for ball color
            min_radius: Minimum ball radius in pixels (default: 15)
            max_radius: Maximum ball radius in pixels (default: 80)
            enable_kalman: Use Kalman filter for smoothing (default: True)
        """
        # Color detection (optional)
        self.color_lower = np.array(color_lower) if color_lower else None
        self.color_upper = np.array(color_upper) if color_upper else None
        self.use_color = color_lower is not None and color_upper is not None

        # Size constraints
        self.min_radius = min_radius
        self.max_radius = max_radius

        # Trajectory storage
        self.trajectory = []

        # Motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100,
            varThreshold=50,
            detectShadows=False
        )

        # Kalman filter for smoothing
        self.enable_kalman = enable_kalman
        self.kalman = None
        self.last_detection = None

        if enable_kalman:
            self._initialize_kalman()

    def _initialize_kalman(self):
        """Initialize Kalman filter for ball position tracking."""
        # State: [x, y, dx, dy] (position and velocity)
        # Measurement: [x, y] (observed position)
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Process noise
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # Measurement noise
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10

    def _detect_by_motion(self, frame: np.ndarray,
                          prev_gray: Optional[np.ndarray]) -> List[DetectionCandidate]:
        """
        Detect ball using motion analysis.

        Args:
            frame: Current frame (BGR)
            prev_gray: Previous frame (grayscale) for optical flow

        Returns:
            List of detection candidates
        """
        candidates = []

        # Background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by size (typical ball area)
            min_area = np.pi * (self.min_radius ** 2)
            max_area = np.pi * (self.max_radius ** 2)

            if area < min_area or area > max_area:
                continue

            # Get enclosing circle
            ((x, y), radius) = cv2.minEnclosingCircle(contour)

            if not (self.min_radius <= radius <= self.max_radius):
                continue

            # Calculate motion score based on contour properties
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

            motion_score = circularity * 0.7 + (area / max_area) * 0.3

            candidates.append(DetectionCandidate(
                x=int(x),
                y=int(y),
                radius=int(radius),
                confidence=motion_score,
                detection_method="motion",
                motion_score=motion_score
            ))

        return candidates

    def _detect_by_shape(self, frame: np.ndarray) -> List[DetectionCandidate]:
        """
        Detect ball using Hough Circle Transform.

        Args:
            frame: Input frame (BGR)

        Returns:
            List of detection candidates
        """
        candidates = []

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        gray = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detect circles
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))

            for circle in circles[0]:
                x, y, r = circle

                # Calculate shape score based on edge strength
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, 2)

                edges = cv2.Canny(gray, 50, 150)
                edge_overlap = cv2.bitwise_and(edges, mask)
                shape_score = np.sum(edge_overlap > 0) / (2 * np.pi * r)

                # Normalize shape score
                shape_score = min(1.0, shape_score / 50)

                candidates.append(DetectionCandidate(
                    x=int(x),
                    y=int(y),
                    radius=int(r),
                    confidence=shape_score,
                    detection_method="shape",
                    shape_score=shape_score
                ))

        return candidates

    def _validate_color(self, frame: np.ndarray, x: int, y: int,
                       radius: int) -> Tuple[bool, float]:
        """
        Validate if detected circle matches ball color.

        Args:
            frame: Input frame (BGR)
            x, y: Center coordinates
            radius: Circle radius

        Returns:
            (color_match, color_score) tuple
        """
        if not self.use_color:
            return True, 1.0  # Skip color validation if not configured

        # Create mask for circle region
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, 255, -1)

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Check color match in circle region
        color_mask = cv2.inRange(hsv, self.color_lower, self.color_upper)
        color_match_pixels = cv2.bitwise_and(color_mask, mask)

        # Calculate percentage of matching pixels
        total_pixels = np.sum(mask > 0)
        matching_pixels = np.sum(color_match_pixels > 0)

        color_score = matching_pixels / total_pixels if total_pixels > 0 else 0.0

        # Consider it a match if >30% of pixels match
        color_match = color_score > 0.3

        return color_match, color_score

    def _merge_candidates(self, motion_candidates: List[DetectionCandidate],
                         shape_candidates: List[DetectionCandidate],
                         frame: np.ndarray) -> Optional[DetectionCandidate]:
        """
        Merge and select best detection from multiple methods.

        Args:
            motion_candidates: Candidates from motion detection
            shape_candidates: Candidates from shape detection
            frame: Current frame for color validation

        Returns:
            Best detection candidate or None
        """
        all_candidates = []

        # Validate color for all candidates
        for candidate in motion_candidates + shape_candidates:
            color_match, color_score = self._validate_color(
                frame, candidate.x, candidate.y, candidate.radius
            )
            candidate.color_match = color_match

            # Calculate combined confidence
            if candidate.detection_method == "motion":
                # Motion detection weighted score
                confidence = (
                    candidate.motion_score * 0.5 +
                    color_score * 0.3 +
                    0.2  # Base confidence for motion
                )
            else:  # shape detection
                confidence = (
                    candidate.shape_score * 0.5 +
                    color_score * 0.3 +
                    0.2  # Base confidence for shape
                )

            candidate.confidence = confidence
            all_candidates.append(candidate)

        if not all_candidates:
            return None

        # Filter by Kalman prediction if available
        if self.kalman is not None and self.last_detection is not None:
            prediction = self.kalman.predict()
            pred_x, pred_y = int(prediction[0]), int(prediction[1])

            # Filter candidates that are too far from prediction
            max_distance = 100  # pixels
            filtered = []

            for candidate in all_candidates:
                dist = np.sqrt((candidate.x - pred_x)**2 + (candidate.y - pred_y)**2)
                if dist < max_distance:
                    # Boost confidence if close to prediction
                    candidate.confidence *= (1 + 0.3 * (1 - dist / max_distance))
                    filtered.append(candidate)

            all_candidates = filtered if filtered else all_candidates

        # Merge candidates that are close to each other
        merged_candidates = []
        used = set()

        for i, c1 in enumerate(all_candidates):
            if i in used:
                continue

            close_candidates = [c1]
            for j, c2 in enumerate(all_candidates):
                if i != j and j not in used:
                    dist = np.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
                    if dist < 30:  # Within 30 pixels
                        close_candidates.append(c2)
                        used.add(j)

            # Average position weighted by confidence
            total_confidence = sum(c.confidence for c in close_candidates)
            if total_confidence > 0:
                avg_x = sum(c.x * c.confidence for c in close_candidates) / total_confidence
                avg_y = sum(c.y * c.confidence for c in close_candidates) / total_confidence
                avg_r = sum(c.radius * c.confidence for c in close_candidates) / total_confidence

                merged_candidates.append(DetectionCandidate(
                    x=int(avg_x),
                    y=int(avg_y),
                    radius=int(avg_r),
                    confidence=total_confidence / len(close_candidates),
                    detection_method="hybrid"
                ))

        # Return highest confidence candidate
        if merged_candidates:
            return max(merged_candidates, key=lambda c: c.confidence)

        return None

    def detect_ball(self, frame: np.ndarray,
                   prev_gray: Optional[np.ndarray] = None) -> Optional[Tuple[int, int, int, float]]:
        """
        Detect ball using hybrid approach (motion + shape + color).

        Args:
            frame: Input video frame (BGR format)
            prev_gray: Previous frame in grayscale (for motion analysis)

        Returns:
            (x, y, radius, confidence) tuple if ball detected, None otherwise
        """
        # Run both detection methods
        motion_candidates = self._detect_by_motion(frame, prev_gray)
        shape_candidates = self._detect_by_shape(frame)

        # Merge and select best candidate
        best = self._merge_candidates(motion_candidates, shape_candidates, frame)

        if best is None:
            return None

        # Update Kalman filter
        if self.enable_kalman and self.kalman is not None:
            measurement = np.array([[np.float32(best.x)], [np.float32(best.y)]])

            if self.last_detection is None:
                # First detection - initialize state
                self.kalman.statePre = np.array([
                    [np.float32(best.x)],
                    [np.float32(best.y)],
                    [0],
                    [0]
                ])

            self.kalman.correct(measurement)
            self.last_detection = (best.x, best.y)

        return (best.x, best.y, best.radius, best.confidence)

    def track_video(self, video_path: str,
                   save_output: bool = False,
                   output_path: Optional[str] = None,
                   min_confidence: float = 0.3) -> List[Tuple[int, int, int]]:
        """
        Track ball through entire video using hybrid detection.

        Args:
            video_path: Path to input video file
            save_output: Whether to save annotated video
            output_path: Path for output video (if save_output=True)
            min_confidence: Minimum confidence threshold (0.0 to 1.0)

        Returns:
            List of (x, y, frame_number) tuples representing ball trajectory
        """
        if not isinstance(video_path, str) or not video_path:
            raise ValueError(f"Invalid video path: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Video has no frames: {video_path}")

        self.trajectory = []
        frame_number = 0
        prev_gray = None

        # Setup video writer if saving output
        out = None
        if save_output:
            if output_path is None:
                output_path = video_path.replace('.mp4', '_tracked.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Tracking ball in video: {total_frames} frames at {fps:.1f} FPS")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect ball
                detection = self.detect_ball(frame, prev_gray)

                if detection and detection[3] >= min_confidence:
                    x, y, radius, confidence = detection
                    self.trajectory.append((x, y, frame_number))

                    # Draw on frame if saving
                    if save_output:
                        cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
                        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                        cv2.putText(frame, f"Conf: {confidence:.2f}",
                                  (x - 40, y - radius - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if save_output and out is not None:
                    out.write(frame)

                # Update previous frame
                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_number += 1

                # Progress indicator
                if frame_number % 30 == 0:
                    progress = (frame_number / total_frames) * 100
                    detected = len(self.trajectory)
                    print(f"  Progress: {progress:.1f}% - Detected in {detected}/{frame_number} frames")

        finally:
            cap.release()
            if out is not None:
                out.release()
                print(f"Annotated video saved to: {output_path}")

        detection_rate = (len(self.trajectory) / total_frames * 100) if total_frames > 0 else 0
        print(f"Tracking complete: Ball detected in {len(self.trajectory)}/{total_frames} frames ({detection_rate:.1f}%)")

        return self.trajectory

    def get_trajectory_data(self) -> Dict:
        """
        Get trajectory data as a dictionary.

        Returns:
            Dictionary with trajectory information
        """
        if not self.trajectory:
            return {"trajectory": [], "frames_detected": 0}

        return {
            "trajectory": self.trajectory,
            "frames_detected": len(self.trajectory),
            "start_position": self.trajectory[0][:2] if self.trajectory else None,
            "end_position": self.trajectory[-1][:2] if self.trajectory else None,
        }

    def clear_trajectory(self):
        """Clear stored trajectory data and reset Kalman filter."""
        self.trajectory = []
        self.last_detection = None
        if self.enable_kalman:
            self._initialize_kalman()

    def set_color_range(self, lower: Tuple[int, int, int],
                       upper: Tuple[int, int, int]):
        """
        Update color detection range.

        Args:
            lower: Lower HSV bound
            upper: Upper HSV bound
        """
        self.color_lower = np.array(lower)
        self.color_upper = np.array(upper)
        self.use_color = True


def main():
    """Example usage of BallTracker."""
    print("BallTracker - Hybrid Detection System")
    print("=" * 50)
    print("\nFeatures:")
    print("  • Motion detection (background subtraction)")
    print("  • Shape detection (Hough circles)")
    print("  • Color validation (optional)")
    print("  • Kalman filtering for smoothing")
    print("\nExample Usage:")
    print("\n# Initialize tracker (color optional)")
    print("tracker = BallTracker(")
    print("    color_lower=(0, 100, 100),  # Optional")
    print("    color_upper=(10, 255, 255),  # Optional")
    print("    min_radius=15,")
    print("    max_radius=80")
    print(")")
    print("\n# Track ball in video")
    print("trajectory = tracker.track_video('video.mp4', save_output=True)")
    print("\n# Get trajectory data")
    print("data = tracker.get_trajectory_data()")
    print(f"  Frames detected: {{data['frames_detected']}}")


if __name__ == "__main__":
    main()
