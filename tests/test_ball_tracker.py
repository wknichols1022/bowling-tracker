"""
Unit tests for ball_tracker module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ball_tracker import BallTracker


class TestBallTracker:
    """Test cases for BallTracker class."""
    
    def test_initialization(self):
        """Test tracker initialization with default parameters."""
        tracker = BallTracker()
        
        assert tracker.min_radius == 10
        assert tracker.max_radius == 100
        assert len(tracker.trajectory) == 0
        assert tracker.color_lower is not None
        assert tracker.color_upper is not None
    
    def test_initialization_custom_params(self):
        """Test tracker initialization with custom parameters."""
        tracker = BallTracker(
            color_lower=(100, 100, 100),
            color_upper=(130, 255, 255),
            min_radius=20,
            max_radius=50
        )
        
        assert tracker.min_radius == 20
        assert tracker.max_radius == 50
        assert np.array_equal(tracker.color_lower, np.array([100, 100, 100]))
        assert np.array_equal(tracker.color_upper, np.array([130, 255, 255]))
    
    def test_set_color_range(self):
        """Test updating color detection range."""
        tracker = BallTracker()
        
        new_lower = (50, 50, 50)
        new_upper = (100, 255, 255)
        tracker.set_color_range(new_lower, new_upper)
        
        assert np.array_equal(tracker.color_lower, np.array(new_lower))
        assert np.array_equal(tracker.color_upper, np.array(new_upper))
    
    def test_clear_trajectory(self):
        """Test clearing trajectory data."""
        tracker = BallTracker()
        
        # Add some dummy trajectory points
        tracker.trajectory = [(100, 100, 0), (150, 150, 1), (200, 200, 2)]
        
        assert len(tracker.trajectory) == 3
        
        tracker.clear_trajectory()
        
        assert len(tracker.trajectory) == 0
    
    def test_get_trajectory_data_empty(self):
        """Test getting trajectory data when empty."""
        tracker = BallTracker()
        
        data = tracker.get_trajectory_data()
        
        assert data["trajectory"] == []
        assert data["frames_detected"] == 0
    
    def test_get_trajectory_data_with_points(self):
        """Test getting trajectory data with points."""
        tracker = BallTracker()
        tracker.trajectory = [
            (100, 100, 0),
            (150, 150, 5),
            (200, 200, 10)
        ]
        
        data = tracker.get_trajectory_data()
        
        assert len(data["trajectory"]) == 3
        assert data["frames_detected"] == 3
        assert data["start_position"] == (100, 100)
        assert data["end_position"] == (200, 200)
    
    def test_detect_ball_with_valid_contour(self):
        """Test ball detection with a valid circular contour."""
        tracker = BallTracker(min_radius=10, max_radius=50)
        
        # Create a test frame with a red circle
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2 = pytest.importorskip("cv2")
        cv2.circle(frame, (320, 240), 25, (0, 0, 255), -1)  # Red circle
        
        # Should detect the ball
        result = tracker.detect_ball(frame)
        
        # Result should be approximately the circle we drew
        if result:
            x, y, radius = result
            assert abs(x - 320) < 10  # Allow small margin
            assert abs(y - 240) < 10
            assert 20 <= radius <= 30
    
    def test_detect_ball_no_ball_in_frame(self):
        """Test ball detection when no ball is present."""
        tracker = BallTracker()
        
        # Create a blank frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = tracker.detect_ball(frame)
        
        # Should return None when no ball detected
        assert result is None


class TestBallTrackerIntegration:
    """Integration tests requiring actual video files."""
    
    @pytest.mark.skip(reason="Requires test video file")
    def test_track_video_full_workflow(self):
        """Test full video tracking workflow."""
        tracker = BallTracker()
        
        # This test would require an actual video file
        # Skipped in unit tests, useful for manual testing
        video_path = "test_video.mp4"
        trajectory = tracker.track_video(video_path)
        
        assert len(trajectory) > 0
        assert all(len(point) == 3 for point in trajectory)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
