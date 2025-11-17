"""
Video stabilization module for handling camera movement in head-mounted footage.
Uses feature tracking and motion compensation to stabilize shaky video.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class VideoStabilizer:
    """Stabilizes video by tracking features and compensating for camera motion."""

    def __init__(self,
                 smoothing_window: int = 30,
                 max_corners: int = 200):
        """
        Initialize video stabilizer.

        Args:
            smoothing_window: Number of frames for trajectory smoothing
            max_corners: Maximum number of feature points to track
        """
        self.smoothing_window = smoothing_window
        self.max_corners = max_corners

        # Parameters for feature detection (Shi-Tomasi corner detection)
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=7
        )

        # Parameters for optical flow (Lucas-Kanade)
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def detect_features(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Detect good features to track in a frame.

        Args:
            frame: Input frame (grayscale)
            mask: Optional mask to restrict feature detection

        Returns:
            Array of detected feature points
        """
        return cv2.goodFeaturesToTrack(
            frame,
            mask=mask,
            **self.feature_params
        )

    def calculate_camera_motion(self,
                                prev_frame: np.ndarray,
                                curr_frame: np.ndarray,
                                prev_pts: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate camera motion between frames using optical flow.

        Args:
            prev_frame: Previous frame (grayscale)
            curr_frame: Current frame (grayscale)
            prev_pts: Feature points from previous frame

        Returns:
            Tuple of (dx, dy, da) - translation x, y and rotation angle
        """
        # Calculate optical flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_frame,
            curr_frame,
            prev_pts,
            None,
            **self.lk_params
        )

        # Filter only valid points
        if curr_pts is None or prev_pts is None:
            return 0.0, 0.0, 0.0

        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        if len(prev_pts) < 4:
            return 0.0, 0.0, 0.0

        # Estimate affine transform
        try:
            transform = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]

            if transform is None:
                return 0.0, 0.0, 0.0

            # Extract translation and rotation
            dx = transform[0, 2]
            dy = transform[1, 2]
            da = np.arctan2(transform[1, 0], transform[0, 0])

            return dx, dy, da
        except:
            return 0.0, 0.0, 0.0

    def smooth_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Smooth camera trajectory using moving average.

        Args:
            trajectory: Array of (dx, dy, da) values

        Returns:
            Smoothed trajectory
        """
        smoothed = np.copy(trajectory)

        for i in range(3):  # For x, y, and angle
            kernel = np.ones(self.smoothing_window) / self.smoothing_window
            smoothed[:, i] = np.convolve(trajectory[:, i], kernel, mode='same')

        return smoothed

    def calculate_transforms(self, video_path: str) -> List[np.ndarray]:
        """
        Analyze video and calculate stabilization transforms without writing to disk.

        Args:
            video_path: Path to input video

        Returns:
            List of transformation matrices (one per frame)
        """
        if not isinstance(video_path, str) or not video_path:
            raise ValueError(f"Invalid video path: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if n_frames <= 0:
            cap.release()
            raise ValueError(f"Video has no frames: {video_path}")

        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError("Could not read first frame")

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Detect features in first frame
        prev_pts = self.detect_features(prev_gray)

        # Store camera motion trajectory
        transforms = []

        print(f"Analyzing camera motion in {n_frames} frames...")

        # Calculate motion between consecutive frames
        frame_idx = 0
        try:
            while True:
                ret, curr_frame = cap.read()
                if not ret:
                    break

                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

                # Calculate motion
                dx, dy, da = self.calculate_camera_motion(prev_gray, curr_gray, prev_pts)
                transforms.append([dx, dy, da])

                # Refresh feature points periodically
                if frame_idx % 30 == 0 or len(prev_pts) < 20:
                    prev_pts = self.detect_features(curr_gray)
                else:
                    # Track existing features
                    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                        prev_gray, curr_gray, prev_pts, None, **self.lk_params
                    )
                    if curr_pts is not None:
                        prev_pts = curr_pts[status == 1].reshape(-1, 1, 2)

                prev_gray = curr_gray
                frame_idx += 1

                if frame_idx % 10 == 0:
                    print(f"  Processed {frame_idx}/{n_frames} frames")

        finally:
            cap.release()

        if not transforms:
            raise ValueError("No motion transforms calculated from video")

        # Convert to cumulative trajectory
        trajectory = np.cumsum(transforms, axis=0)

        # Smooth trajectory
        smoothed_trajectory = self.smooth_trajectory(trajectory)

        # Calculate difference between smoothed and original
        difference = smoothed_trajectory - trajectory

        # Calculate stabilizing transforms
        transforms_smooth = transforms + difference

        # Generate stabilization matrices
        stabilization_matrices = []
        for dx, dy, da in transforms_smooth:
            # Create transformation matrix
            T = np.array([
                [np.cos(da), -np.sin(da), dx],
                [np.sin(da), np.cos(da), dy]
            ], dtype=np.float32)
            stabilization_matrices.append(T)

        return stabilization_matrices

    def stabilize_video(self,
                       video_path: str,
                       output_path: Optional[str] = None) -> Tuple[Optional[str], List[np.ndarray]]:
        """
        Calculate stabilization transforms and optionally save stabilized video.

        NOTE: This method is DEPRECATED for use with ball tracking.
        Use calculate_transforms() instead to avoid disk I/O.

        Args:
            video_path: Path to input video
            output_path: Optional path to save stabilized video

        Returns:
            Tuple of (output_path, list of transformation matrices)
        """
        # Calculate transforms (no disk I/O)
        stabilization_matrices = self.calculate_transforms(video_path)

        # Only write video if explicitly requested
        if output_path:
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            self._write_stabilized_video(
                video_path, output_path, stabilization_matrices, width, height, fps
            )
            print(f"Stabilized video saved to: {output_path}")
            return output_path, stabilization_matrices

        return None, stabilization_matrices

    def _write_stabilized_video(self,
                               input_path: str,
                               output_path: str,
                               transforms: List[np.ndarray],
                               width: int,
                               height: int,
                               fps: float):
        """Write stabilized video to file."""
        cap = cv2.VideoCapture(input_path)

        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply stabilization transform
            if frame_idx < len(transforms):
                T = transforms[frame_idx]
                frame_stabilized = cv2.warpAffine(frame, T, (width, height))
                out.write(frame_stabilized)
            else:
                out.write(frame)

            frame_idx += 1

        cap.release()
        out.release()

    def get_transform_for_frame(self,
                               frame_number: int,
                               transforms: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Get stabilization transform for a specific frame.

        Args:
            frame_number: Frame index
            transforms: List of transformation matrices

        Returns:
            Transformation matrix for the frame
        """
        if 0 <= frame_number < len(transforms):
            return transforms[frame_number]
        return None


def main():
    """Example usage of VideoStabilizer."""
    print("VideoStabilizer Example Usage:")
    print("\n# Initialize stabilizer")
    print("stabilizer = VideoStabilizer(smoothing_window=30)")
    print("\n# Stabilize video")
    print("output_path, transforms = stabilizer.stabilize_video(")
    print("    'input.mp4',")
    print("    output_path='stabilized.mp4'")
    print(")")
    print("\n# Use transforms for tracking")
    print("T = stabilizer.get_transform_for_frame(frame_num, transforms)")
    print("stabilized_point = cv2.transform(point, T)")


if __name__ == "__main__":
    main()
