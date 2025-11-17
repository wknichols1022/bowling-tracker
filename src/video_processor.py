"""
Main video processing module that orchestrates ball tracking, speed calculation,
and database storage.
"""

import os
import argparse
import json
from datetime import datetime
from typing import Dict, Optional

from ball_tracker import BallTracker
from speed_calculator import SpeedCalculator
from database import BowlingDatabase
from video_stabilizer import VideoStabilizer


class VideoProcessor:
    """Main processor for analyzing bowling videos."""

    def __init__(self,
                 db: Optional[BowlingDatabase] = None,
                 tracker: Optional[BallTracker] = None,
                 stabilizer: Optional[VideoStabilizer] = None,
                 db_path: str = "data/bowling_stats.db",
                 lane_length: float = 60.0,
                 ball_diameter: float = 8.5,
                 use_stabilization: bool = True):
        """
        Initialize video processor with dependency injection.

        Args:
            db: BowlingDatabase instance (creates default if None)
            tracker: BallTracker instance (creates default if None)
            stabilizer: VideoStabilizer instance (creates default if None and use_stabilization=True)
            db_path: Path to SQLite database (used only if db is None)
            lane_length: Bowling lane length in feet
            ball_diameter: Ball diameter in inches
            use_stabilization: Enable video stabilization for head-mounted cameras

        Raises:
            ValueError: If lane_length or ball_diameter are invalid
        """
        # Validate parameters
        if lane_length <= 0 or lane_length > 100:
            raise ValueError(f"Invalid lane_length: {lane_length}. Must be between 0 and 100 feet.")

        if ball_diameter <= 0 or ball_diameter > 12:
            raise ValueError(f"Invalid ball_diameter: {ball_diameter}. Must be between 0 and 12 inches.")

        # Dependency injection - use provided instances or create defaults
        self.db = db if db is not None else BowlingDatabase(db_path)
        self.tracker = tracker if tracker is not None else BallTracker()
        self.stabilizer = stabilizer if stabilizer is not None else (VideoStabilizer() if use_stabilization else None)
        self.speed_calc = None  # Will be initialized after calibration
        self.lane_length = lane_length
        self.ball_diameter = ball_diameter
        self.use_stabilization = use_stabilization
    
    def process_video(self,
                     video_path: str,
                     session_id: Optional[int] = None,
                     shot_number: Optional[int] = None,
                     location: str = None,
                     ball_color: str = "red",
                     auto_calibrate: bool = True,
                     save_annotated: bool = False) -> Dict:
        """
        Process a bowling video end-to-end.

        Args:
            video_path: Path to input video file
            session_id: Existing session ID (creates new if None)
            shot_number: Shot number within session
            location: Bowling alley name (for new sessions)
            ball_color: Color of bowling ball for detection
            auto_calibrate: Automatically calibrate speed calculation
            save_annotated: Save video with ball tracking visualization

        Returns:
            Dictionary with processing results

        Raises:
            ValueError: If input parameters are invalid
            FileNotFoundError: If video file doesn't exist
        """
        # Input validation
        if not isinstance(video_path, str) or not video_path.strip():
            raise ValueError(f"Invalid video path: {video_path}")

        if session_id is not None and (not isinstance(session_id, int) or session_id < 1):
            raise ValueError(f"Invalid session_id: {session_id}. Must be a positive integer.")

        if shot_number is not None and (not isinstance(shot_number, int) or shot_number < 1):
            raise ValueError(f"Invalid shot_number: {shot_number}. Must be a positive integer.")

        if not isinstance(ball_color, str) or ball_color.lower() not in ["red", "blue", "green", "black", "purple", "orange"]:
            raise ValueError(f"Invalid ball_color: {ball_color}. Must be one of: red, blue, green, black, purple, orange")

        print(f"\n{'='*60}")
        print(f"Processing video: {os.path.basename(video_path)}")
        print(f"{'='*60}\n")

        # Validate video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Validate it's a file, not a directory
        if not os.path.isfile(video_path):
            raise ValueError(f"Path is not a file: {video_path}")
        
        # Set ball color range
        self._set_ball_color(ball_color)

        # Calculate stabilization transforms if enabled (NO DISK I/O)
        stabilization_transforms = None

        if self.use_stabilization and self.stabilizer:
            print("Step 0: Calculating stabilization transforms...")
            try:
                stabilization_transforms = self.stabilizer.calculate_transforms(video_path)
                print(f"✓ Stabilization analysis complete ({len(stabilization_transforms)} frames)\n")
            except Exception as e:
                print(f"Warning: Stabilization failed ({str(e)}), proceeding without stabilization\n")
                stabilization_transforms = None

        # Create session if needed
        if session_id is None:
            date = datetime.now().strftime("%Y-%m-%d")
            session_id = self.db.create_session(date=date, location=location)
            print(f"Created new session: {session_id}")
        
        # Determine shot number
        if shot_number is None:
            existing_shots = self.db.get_session_shots(session_id)
            shot_number = len(existing_shots) + 1
        
        print(f"Session ID: {session_id}, Shot Number: {shot_number}\n")
        
        # Step 1: Track ball through video
        print("Step 1: Tracking ball...")
        output_path = None
        if save_annotated:
            output_dir = "data/processed"
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                raise ValueError(f"Cannot create output directory {output_dir}: {e}")

            filename = os.path.basename(video_path)
            output_path = os.path.join(output_dir,
                                      f"tracked_{filename}")

        try:
            trajectory = self.tracker.track_video(
                video_path,
                save_output=save_annotated,
                output_path=output_path
            )
        except Exception as e:
            print(f"ERROR: Ball tracking failed: {e}")
            return {
                "success": False,
                "error": f"Ball tracking failed: {str(e)}",
                "session_id": session_id,
                "shot_number": shot_number
            }

        if not trajectory or len(trajectory) < 3:
            print("ERROR: Insufficient ball detections in video!")
            return {
                "success": False,
                "error": f"Insufficient ball detections (found {len(trajectory) if trajectory else 0} frames, need at least 3)",
                "session_id": session_id,
                "shot_number": shot_number
            }
        
        print(f"✓ Ball tracked in {len(trajectory)} frames\n")
        
        # Step 2: Calculate speed
        print("Step 2: Calculating speed...")

        # Get video FPS
        import cv2
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            if fps <= 0 or fps > 240:
                raise ValueError(f"Invalid FPS: {fps}")
        except Exception as e:
            print(f"ERROR: Failed to read video properties: {e}")
            return {
                "success": False,
                "error": f"Failed to read video properties: {str(e)}",
                "session_id": session_id,
                "shot_number": shot_number
            }

        # Initialize speed calculator
        try:
            self.speed_calc = SpeedCalculator(
                lane_length_feet=self.lane_length,
                ball_diameter_inches=self.ball_diameter,
                fps=fps
            )
        except Exception as e:
            print(f"ERROR: Failed to initialize speed calculator: {e}")
            return {
                "success": False,
                "error": f"Speed calculator initialization failed: {str(e)}",
                "session_id": session_id,
                "shot_number": shot_number
            }

        # Auto-calibrate if enabled
        if auto_calibrate and trajectory:
            try:
                # Use average radius from trajectory for calibration
                avg_radius = 25  # Default fallback
                if len(trajectory) > 0:
                    # In real implementation, would extract radius from detection
                    # For now using reasonable default
                    avg_radius = 25

                self.speed_calc.calibrate_from_ball(avg_radius)
            except Exception as e:
                print(f"Warning: Calibration failed: {e}, using defaults")

        # Calculate speed
        try:
            speed_data = self.speed_calc.calculate_speed_from_trajectory(trajectory)
        except Exception as e:
            print(f"ERROR: Speed calculation failed: {e}")
            return {
                "success": False,
                "error": f"Speed calculation failed: {str(e)}",
                "session_id": session_id,
                "shot_number": shot_number
            }

        if "error" in speed_data:
            print(f"ERROR: {speed_data['error']}")
            return {
                "success": False,
                "error": speed_data["error"],
                "session_id": session_id,
                "shot_number": shot_number
            }
        
        print(f"✓ Speed: {speed_data['speed_mph']} mph "
              f"({speed_data['speed_fps']} ft/s)")
        print(f"  Distance: {speed_data['distance_feet']} feet")
        print(f"  Time: {speed_data['time_seconds']} seconds\n")
        
        # Step 3: Analyze trajectory
        print("Step 3: Analyzing trajectory...")
        try:
            trajectory_analysis = self.speed_calc.analyze_trajectory_shape(trajectory)

            if "error" in trajectory_analysis:
                print(f"Warning: Trajectory analysis incomplete: {trajectory_analysis['error']}")
                trajectory_analysis = {"trajectory_type": "unknown"}
            else:
                print(f"✓ Trajectory type: {trajectory_analysis['trajectory_type']}")
                print(f"  Lateral deviation: "
                      f"{trajectory_analysis.get('lateral_deviation_pixels', 'N/A')} pixels\n")
        except Exception as e:
            print(f"Warning: Trajectory analysis failed: {e}")
            trajectory_analysis = {"trajectory_type": "unknown", "error": str(e)}

        # Step 4: Save to database
        print("Step 4: Saving to database...")

        # Get release and impact points
        release_point = trajectory[0][:2] if trajectory else None
        impact_point = trajectory[-1][:2] if trajectory else None

        # Convert trajectory to JSON
        try:
            trajectory_json = json.dumps({
                "points": trajectory,
                "analysis": trajectory_analysis
            })
        except (TypeError, ValueError) as e:
            print(f"Warning: Failed to serialize trajectory data: {e}")
            trajectory_json = json.dumps({"points": [], "analysis": {"trajectory_type": "unknown"}})

        try:
            shot_id = self.db.add_shot(
                session_id=session_id,
                shot_number=shot_number,
                video_file=video_path,
                speed_mph=speed_data['speed_mph'],
                speed_fps=speed_data['speed_fps'],
                trajectory_data=trajectory_json,
                release_point=release_point,
                impact_point=impact_point
            )
            print(f"✓ Shot saved with ID: {shot_id}\n")
        except Exception as e:
            print(f"ERROR: Failed to save to database: {e}")
            return {
                "success": False,
                "error": f"Database save failed: {str(e)}",
                "session_id": session_id,
                "shot_number": shot_number
            }
        
        # Return comprehensive results
        result = {
            "success": True,
            "session_id": session_id,
            "shot_number": shot_number,
            "shot_id": shot_id,
            "speed_mph": speed_data['speed_mph'],
            "speed_fps": speed_data['speed_fps'],
            "distance_feet": speed_data['distance_feet'],
            "trajectory_type": trajectory_analysis['trajectory_type'],
            "frames_analyzed": len(trajectory),
            "output_video": output_path if save_annotated else None
        }
        
        print(f"{'='*60}")
        print("Processing complete!")
        print(f"{'='*60}\n")

        return result
    
    def _set_ball_color(self, color: str):
        """Set ball color detection range based on color name."""
        color_ranges = {
            "red": ((0, 100, 100), (10, 255, 255)),
            "blue": ((100, 100, 100), (130, 255, 255)),
            "green": ((40, 100, 100), (80, 255, 255)),
            "black": ((0, 0, 0), (180, 255, 30)),
            "purple": ((130, 100, 100), (160, 255, 255)),
            "orange": ((10, 100, 100), (25, 255, 255)),
        }
        
        if color.lower() in color_ranges:
            lower, upper = color_ranges[color.lower()]
            self.tracker.set_color_range(lower, upper)
            print(f"Set ball color detection to: {color}")
        else:
            print(f"Warning: Unknown color '{color}', using default (red)")
    
    def get_session_summary(self, session_id: int) -> Dict:
        """
        Get summary statistics for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dictionary with session statistics
        """
        shots = self.db.get_session_shots(session_id)
        
        if not shots:
            return {"error": "No shots found for session"}
        
        speeds = [s['speed_mph'] for s in shots if s['speed_mph']]
        
        import numpy as np
        
        return {
            "session_id": session_id,
            "total_shots": len(shots),
            "avg_speed_mph": round(np.mean(speeds), 2) if speeds else None,
            "min_speed_mph": round(min(speeds), 2) if speeds else None,
            "max_speed_mph": round(max(speeds), 2) if speeds else None,
            "shots": shots
        }
    
    def close(self):
        """Close database connection."""
        self.db.close()


def batch_process_videos(folder_path: str,
                        session_id: Optional[int] = None,
                        location: str = "Unknown",
                        ball_color: str = "red",
                        auto_calibrate: bool = True,
                        save_annotated: bool = False,
                        use_stabilization: bool = True) -> Dict:
    """
    Process all videos in a folder as a single session.

    Args:
        folder_path: Path to folder containing video files
        session_id: Existing session ID (creates new if None)
        location: Bowling alley name (for new sessions)
        ball_color: Color of bowling ball for detection
        auto_calibrate: Automatically calibrate speed calculation
        save_annotated: Save videos with ball tracking visualization

    Returns:
        Dictionary with batch processing results

    Raises:
        ValueError: If input parameters are invalid
        FileNotFoundError: If folder doesn't exist
    """
    # Input validation
    if not isinstance(folder_path, str) or not folder_path.strip():
        raise ValueError(f"Invalid folder path: {folder_path}")

    if session_id is not None and (not isinstance(session_id, int) or session_id < 1):
        raise ValueError(f"Invalid session_id: {session_id}. Must be a positive integer.")

    # Supported video extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.MP4', '.AVI', '.MOV'}

    # Find all video files in folder
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if not os.path.isdir(folder_path):
        raise ValueError(f"Path is not a directory: {folder_path}")

    video_files = []
    for file in os.listdir(folder_path):
        if any(file.endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(folder_path, file))

    # Sort files alphabetically for consistent processing order
    video_files.sort()

    if not video_files:
        print(f"No video files found in {folder_path}")
        return {
            "success": False,
            "error": "No video files found",
            "videos_processed": 0
        }

    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING: {len(video_files)} videos found")
    print(f"{'='*60}\n")

    for i, video_file in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] {os.path.basename(video_file)}")
    print()

    # Initialize processor
    processor = VideoProcessor(use_stabilization=use_stabilization)
    results = []
    successful = 0
    failed = 0

    try:
        for i, video_path in enumerate(video_files, 1):
            print(f"\n{'*'*60}")
            print(f"VIDEO {i}/{len(video_files)}")
            print(f"{'*'*60}")

            result = processor.process_video(
                video_path=video_path,
                session_id=session_id,
                location=location,
                ball_color=ball_color,
                auto_calibrate=auto_calibrate,
                save_annotated=save_annotated
            )

            # Use the session_id from first video for all subsequent videos
            if session_id is None and result.get("session_id"):
                session_id = result["session_id"]

            results.append({
                "video_file": os.path.basename(video_path),
                "result": result
            })

            if result.get("success"):
                successful += 1
            else:
                failed += 1

        # Print summary
        print(f"\n{'='*60}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total videos: {len(video_files)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Session ID: {session_id}")
        print(f"{'='*60}\n")

        return {
            "success": True,
            "session_id": session_id,
            "total_videos": len(video_files),
            "successful": successful,
            "failed": failed,
            "results": results
        }

    finally:
        processor.close()


def main():
    """Command-line interface for video processing."""
    parser = argparse.ArgumentParser(
        description="Process bowling videos to track ball and calculate speed"
    )
    parser.add_argument(
        "--video", "-v",
        help="Path to video file"
    )
    parser.add_argument(
        "--batch", "-b",
        help="Path to folder containing multiple videos to process as one session"
    )
    parser.add_argument(
        "--session-id", "-s",
        type=int,
        help="Existing session ID (creates new if not provided)"
    )
    parser.add_argument(
        "--location", "-l",
        default="Unknown",
        help="Bowling alley location"
    )
    parser.add_argument(
        "--color", "-c",
        default="red",
        choices=["red", "blue", "green", "black", "purple", "orange"],
        help="Ball color for detection"
    )
    parser.add_argument(
        "--save-video", "-o",
        action="store_true",
        help="Save annotated video with tracking visualization"
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Disable automatic calibration"
    )
    parser.add_argument(
        "--no-stabilization",
        action="store_true",
        help="Disable video stabilization (use for stationary camera)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.video and not args.batch:
        parser.error("Either --video or --batch must be specified")

    if args.video and args.batch:
        parser.error("Cannot use both --video and --batch at the same time")

    # Batch processing mode
    if args.batch:
        try:
            result = batch_process_videos(
                folder_path=args.batch,
                session_id=args.session_id,
                location=args.location,
                ball_color=args.color,
                auto_calibrate=not args.no_calibrate,
                save_annotated=args.save_video,
                use_stabilization=not args.no_stabilization
            )

            if not result["success"]:
                print(f"\nBatch processing failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"\nError during batch processing: {str(e)}")

    # Single video processing mode
    else:
        processor = VideoProcessor(use_stabilization=not args.no_stabilization)

        try:
            result = processor.process_video(
                video_path=args.video,
                session_id=args.session_id,
                location=args.location,
                ball_color=args.color,
                auto_calibrate=not args.no_calibrate,
                save_annotated=args.save_video
            )

            if result["success"]:
                print("\nResults:")
                print(f"  Speed: {result['speed_mph']} mph")
                print(f"  Trajectory: {result['trajectory_type']}")
                print(f"  Shot ID: {result['shot_id']}")
            else:
                print(f"\nProcessing failed: {result.get('error', 'Unknown error')}")

        finally:
            processor.close()


if __name__ == "__main__":
    main()
