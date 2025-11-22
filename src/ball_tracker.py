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
                 enable_kalman: bool = True,
                 roi_top_ratio: float = 0.5,
                 roi_left_ratio: float = 0.3,
                 roi_right_ratio: float = 0.3):
        """
        Initialize hybrid ball tracker.

        Args:
            color_lower: Optional lower HSV bound for ball color
            color_upper: Optional upper HSV bound for ball color
            min_radius: Minimum ball radius in pixels (default: 15)
            max_radius: Maximum ball radius in pixels (default: 80)
            enable_kalman: Use Kalman filter for smoothing (default: True)
            roi_top_ratio: Ignore top portion of frame (0.5 = ignore top 50%)
            roi_left_ratio: Ignore left portion of frame (0.3 = ignore left 30%)
            roi_right_ratio: Ignore right portion of frame (0.3 = ignore right 30%)
        """
        # Color detection (optional)
        self.color_lower = np.array(color_lower) if color_lower else None
        self.color_upper = np.array(color_upper) if color_upper else None
        self.use_color = color_lower is not None and color_upper is not None

        # Size constraints
        self.min_radius = min_radius
        self.max_radius = max_radius

        # Region of interest (to ignore ceiling lights and gutters)
        self.roi_top_ratio = roi_top_ratio
        self.roi_left_ratio = roi_left_ratio
        self.roi_right_ratio = roi_right_ratio

        # Trajectory storage
        self.trajectory = []

        # Release detection (to ignore ball in hand)
        self.ball_released = False
        self.release_buffer = []  # Buffer of recent Y positions
        self.release_buffer_size = 5  # Frames needed for release confirmation
        self.min_downward_distance = 30  # Min Y increase to confirm release

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

        # Apply ROI to ignore top, left, and right portions
        height, width = frame.shape[:2]
        roi_start_y = int(height * self.roi_top_ratio)
        roi_start_x = int(width * self.roi_left_ratio)
        roi_end_x = int(width * (1 - self.roi_right_ratio))

        # Crop to ROI (focus on center lane area)
        frame_roi = frame[roi_start_y:, roi_start_x:roi_end_x]

        # Convert to grayscale
        gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)

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

                # Adjust coordinates back to original frame
                x_adjusted = x + roi_start_x
                y_adjusted = y + roi_start_y

                # Calculate shape score based on edge strength
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, 2)

                edges = cv2.Canny(gray, 50, 150)
                edge_overlap = cv2.bitwise_and(edges, mask)
                shape_score = np.sum(edge_overlap > 0) / (2 * np.pi * r)

                # Normalize shape score
                shape_score = min(1.0, shape_score / 50)

                candidates.append(DetectionCandidate(
                    x=int(x_adjusted),
                    y=int(y_adjusted),
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

        # Consider it a match if >50% of pixels match (stricter to avoid lights)
        color_match = color_score > 0.5

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

            # If color detection is enabled, require BOTH shape AND color match
            if self.use_color:
                # Only accept shape-detected candidates that match the color
                if candidate.detection_method == "shape" and not color_match:
                    continue  # Skip candidates that don't match color

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
            # Stricter threshold: bowling ball shouldn't jump > 40 pixels between frames
            max_distance = 40  # pixels (reduced from 100)
            filtered = []

            for candidate in all_candidates:
                dist = np.sqrt((candidate.x - pred_x)**2 + (candidate.y - pred_y)**2)
                if dist < max_distance:
                    # Boost confidence if close to prediction
                    candidate.confidence *= (1 + 0.5 * (1 - dist / max_distance))
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
            measurement = np.array([[np.float32(best.x)], [np.float32(best.y)]], dtype=np.float32)

            if self.last_detection is None:
                # First detection - initialize state
                self.kalman.statePre = np.array([
                    [np.float32(best.x)],
                    [np.float32(best.y)],
                    [np.float32(0)],
                    [np.float32(0)]
                ], dtype=np.float32)

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

        # Reset release detection for new video
        self.ball_released = False
        self.release_buffer = []

        # Initialize outlier rejection buffer (separate from trajectory)
        outlier_buffer = []
        trajectory_segment = []  # Track current segment for median-based outlier detection

        # Initialize optical flow tracker (simple custom tracker)
        tracker_bbox = None  # (x, y, width, height)
        tracker_initialized = False
        tracker_failures = 0
        max_tracker_failures = 10
        prev_frame_for_tracking = None

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

                x, y, radius = None, None, None

                # If tracker is active, use optical flow
                if tracker_initialized and tracker_bbox is not None and prev_frame_for_tracking is not None:
                    # Use optical flow to track the center point
                    prev_gray_track = cv2.cvtColor(prev_frame_for_tracking, cv2.COLOR_BGR2GRAY)
                    curr_gray_track = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Track the center point of the bbox
                    center_x = tracker_bbox[0] + tracker_bbox[2] // 2
                    center_y = tracker_bbox[1] + tracker_bbox[3] // 2
                    prev_pts = np.array([[[float(center_x), float(center_y)]]], dtype=np.float32)

                    # Calculate optical flow
                    next_pts, status, error = cv2.calcOpticalFlowPyrLK(
                        prev_gray_track, curr_gray_track, prev_pts, None,
                        winSize=(21, 21), maxLevel=3
                    )

                    if status[0][0] == 1:
                        # Successfully tracked
                        new_center_x = int(next_pts[0][0][0])
                        new_center_y = int(next_pts[0][0][1])

                        # Validate tracker position - check if out of bounds only
                        # (Don't restrict upper portion - ball naturally moves up in frame as it goes down lane)
                        frame_height = frame.shape[0]
                        frame_width = frame.shape[1]

                        # Check if tracker went out of bounds
                        if new_center_x < 0 or new_center_x >= frame_width or new_center_y < 0 or new_center_y >= frame_height:
                            # Tracker went out of bounds
                            tracker_initialized = False
                            tracker_bbox = None
                            print(f"  Tracker invalidated at frame {frame_number} - out of bounds")
                        else:
                            # Tracker position is valid
                            # Update bbox position
                            tracker_bbox = (
                                new_center_x - tracker_bbox[2] // 2,
                                new_center_y - tracker_bbox[3] // 2,
                                tracker_bbox[2],
                                tracker_bbox[3]
                            )

                            x = new_center_x
                            y = new_center_y
                            radius = tracker_bbox[2] // 2
                            tracker_failures = 0
                    else:
                        # Tracking failed
                        tracker_failures += 1
                        if tracker_failures >= max_tracker_failures:
                            tracker_initialized = False
                            tracker_bbox = None
                            print(f"  Tracker lost at frame {frame_number}, reinitializing...")

                # If no tracker or tracker failed, use detection
                if not tracker_initialized or tracker_failures > 0:
                    detection = self.detect_ball(frame, prev_gray)

                    if detection and detection[3] >= min_confidence:
                        x, y, radius, confidence = detection

                        # If ball released but tracker not yet initialized, try to initialize it
                        if self.ball_released and not tracker_initialized:
                            frame_height = frame.shape[0]
                            min_tracker_radius = 30  # Increased from 25
                            in_bottom_portion = y > (frame_height * 0.3)  # Bottom 70% only (stricter)

                            if radius >= min_tracker_radius and in_bottom_portion:
                                tracker_bbox = (x - radius, y - radius, radius * 2, radius * 2)
                                prev_frame_for_tracking = frame.copy()
                                tracker_initialized = True
                                # Clear buffers when tracker reinitializes (new tracking segment)
                                outlier_buffer = []
                                trajectory_segment = []  # Track current segment separately for outlier detection
                                print(f"  Tracker initialized at frame {frame_number} (delayed, r={radius}, y={y}/{frame_height})")

                        # Check if ball has been released
                        if not self.ball_released:
                            # Add Y position to buffer
                            self.release_buffer.append(y)

                            # Keep buffer at max size
                            if len(self.release_buffer) > self.release_buffer_size:
                                self.release_buffer.pop(0)

                            # Check for consistent downward motion
                            if len(self.release_buffer) >= self.release_buffer_size:
                                # Calculate Y distance from first to last position in buffer
                                y_movement = self.release_buffer[-1] - self.release_buffer[0]

                                # Check if all positions show increasing Y (moving down)
                                is_moving_down = all(
                                    self.release_buffer[i] < self.release_buffer[i + 1]
                                    for i in range(len(self.release_buffer) - 1)
                                )

                                if is_moving_down and y_movement > self.min_downward_distance:
                                    self.ball_released = True
                                    print(f"  Ball release detected at frame {frame_number}")

                                    # Initialize optical flow tracker on release
                                    # Only initialize if object is large enough and in bottom 70%
                                    frame_height = frame.shape[0]
                                    min_tracker_radius = 30  # Increased from 25
                                    in_bottom_portion = y > (frame_height * 0.3)  # Bottom 70% only

                                    if radius >= min_tracker_radius and in_bottom_portion:
                                        tracker_bbox = (x - radius, y - radius, radius * 2, radius * 2)
                                        prev_frame_for_tracking = frame.copy()
                                        tracker_initialized = True
                                        # Clear outlier buffer for new tracking segment
                                        outlier_buffer = []
                                        print(f"  Tracker initialized at frame {frame_number} (r={radius}, y={y}/{frame_height})")
                                    else:
                                        print(f"  Release detected but object too small/far (r={radius}, y={y}/{frame_height}), waiting for better detection...")

                # Record trajectory if we have a valid position after release
                if self.ball_released and x is not None and y is not None:
                    # Outlier rejection: check for sudden jumps within the current segment
                    # This allows gradual X changes due to perspective but rejects tracking errors
                    should_record = True

                    # Do outlier checking if we have at least 1 point in current segment
                    if len(trajectory_segment) >= 1:
                        # Compare to last point in the CURRENT segment only
                        last_x = trajectory_segment[-1][0]

                        # Use more lenient threshold for first few points, stricter after established
                        # This allows new segments to establish but still catches gross outliers
                        if len(trajectory_segment) < 5:
                            max_jump = 200  # Lenient for first few points of new segment
                        else:
                            max_jump = 80  # Strict once segment is established

                        x_jump = abs(x - last_x)

                        if x_jump > max_jump:
                            should_record = False
                            print(f"  Rejected outlier at frame {frame_number}: x={x}, segment_last_x={last_x}, jump={x_jump:.0f}px (max={max_jump}, seg_len={len(trajectory_segment)})")

                    if should_record:
                        self.trajectory.append((x, y, frame_number))
                        trajectory_segment.append((x, y, frame_number))
                        # Add to outlier buffer for current tracking segment
                        outlier_buffer.append((x, y, frame_number))
                        # Keep buffer at reasonable size
                        if len(outlier_buffer) > 50:
                            outlier_buffer.pop(0)

                    # Draw on frame if saving
                    if save_output:
                        color = (0, 255, 0) if tracker_initialized else (255, 0, 255)
                        cv2.circle(frame, (x, y), radius, color, 2)
                        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                        mode = "Tracker" if tracker_initialized else "Detect"
                        cv2.putText(frame, mode,
                                  (x - 40, y - radius - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                elif save_output and not self.ball_released and x is not None:
                    # Before release: draw in yellow
                    cv2.circle(frame, (x, y), radius, (0, 255, 255), 2)
                    cv2.putText(frame, "Pre-release",
                              (x - 40, y - radius - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                if save_output and out is not None:
                    out.write(frame)

                # Update previous frame for tracking
                if tracker_initialized:
                    prev_frame_for_tracking = frame.copy()

                # Update previous frame for motion detection
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
