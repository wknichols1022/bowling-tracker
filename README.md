# Bowling Shot Tracker

A computer vision-based bowling analysis system for tracking ball speed, trajectory, and performance metrics using footage from Meta Ray-Ban glasses.

## Features

- **Hybrid Ball Tracking**: Motion detection + shape recognition + optional color validation with Kalman filtering
- **Video Stabilization**: Automatic compensation for head-mounted camera movement
- **Speed Calculation**: Multi-point calibration system with confidence scoring
- **Trajectory Analysis**: Hook point detection, lateral deviation, and board-based visualization
- **Performance Dashboard**: Interactive Streamlit dashboard with session analytics
- **Batch Processing**: Process multiple videos as a single bowling session
- **Data Persistence**: SQLite database for long-term performance tracking

## Prerequisites

- **Python**: 3.8 or higher (tested on 3.9-3.11)
- **Operating System**: Windows, macOS, or Linux
- **Hardware**:
  - Minimum 8GB RAM (16GB recommended for video processing)
  - ~500MB disk space per hour of video footage
- **Video Source**: Meta Ray-Ban glasses or any bowling video shot from behind the bowler

## Quick Start

```bash
# 1. Clone and navigate to project
cd bowling-tracker

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize database
python src/database.py

# 5. Process your first video
python src/video_processor.py --video data/raw_videos/your_video.mp4 --location "Your Bowling Alley"

# 6. View results
streamlit run src/dashboard.py
```

## Project Structure

```
bowling-tracker/
├── data/
│   ├── raw_videos/          # Original video files from Ray-Ban glasses
│   ├── processed/           # Processed video outputs (optional)
│   └── bowling_stats.db     # SQLite database (auto-created)
├── src/
│   ├── video_processor.py   # Main video processing pipeline
│   ├── ball_tracker.py      # Hybrid ball detection (motion + shape + color)
│   ├── video_stabilizer.py  # Head-mounted camera stabilization
│   ├── speed_calculator.py  # Speed and trajectory calculations
│   ├── database.py          # Database operations
│   └── dashboard.py         # Streamlit visualization dashboard
├── notebooks/               # Jupyter notebooks for experimentation
├── tests/                   # Unit tests (future)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**
- OpenCV (computer vision)
- NumPy (numerical operations)
- Pandas (data manipulation)
- Streamlit (dashboard)
- Plotly (interactive visualizations)

### 3. Initialize Database

```bash
python src/database.py
```

This creates `data/bowling_stats.db` with the required schema.

## Usage

### Processing Videos

#### Single Video Processing

Process one video at a time:

```bash
python src/video_processor.py --video data/raw_videos/your_video.mp4
```

**With options:**
```bash
python src/video_processor.py \
  --video data/raw_videos/shot1.mp4 \
  --location "AMF Lanes" \
  --color blue \
  --save-video
```

#### Batch Processing (Recommended)

Process all videos in a folder as a single session:

```bash
# Basic batch processing
python src/video_processor.py --batch data/raw_videos/ --location "AMF Bowling"

# With optional parameters
python src/video_processor.py \
  --batch data/raw_videos/ \
  --location "AMF Bowling" \
  --color blue \
  --save-video
```

**The batch processor will:**
- Find all video files (`.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`) in the folder
- Process them in alphabetical order
- Automatically assign them to the same session with incrementing shot numbers
- Provide a summary of successful/failed processing
- Skip videos that fail to process and continue with the rest

#### Command-line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--video` | `-v` | Path to single video file | - |
| `--batch` | `-b` | Path to folder containing multiple videos | - |
| `--location` | `-l` | Bowling alley name | "Unknown" |
| `--color` | `-c` | Ball color (red, blue, green, black, purple, orange) | red |
| `--session-id` | `-s` | Use existing session ID instead of creating new one | None |
| `--save-video` | `-o` | Save annotated videos with ball tracking visualization | False |
| `--no-calibrate` | - | Disable automatic speed calibration | False |
| `--no-stabilization` | - | Disable video stabilization (use for stationary cameras only) | False |

**Note:** `--video` and `--batch` are mutually exclusive. Use one or the other.

### View Dashboard

Launch the interactive dashboard to view your bowling statistics:

```bash
streamlit run src/dashboard.py
```

**Dashboard Features:**
- **Overview**: Key metrics, speed distribution, and session summary
- **Speed Analysis**: Speed progression over time and statistics by session
- **Trajectory**: Visual overlay of shot trajectories on a bowling lane diagram
  - 39-board lane visualization
  - Hook point markers
  - Release and impact board positions
  - Date range filtering
- **Shot Details**: Detailed table of all shots with speeds, trajectories, and metadata

The dashboard will open in your browser at `http://localhost:8501`.

## Configuration

### Default Settings

The system uses these defaults:
- **Lane length**: 60 feet (foul line to headpin)
- **Ball diameter**: 8.5 inches (standard)
- **Video stabilization**: Enabled (for head-mounted cameras)
- **Ball color**: Red (optional, used for validation only)

### Customizing Settings

You can modify settings via command-line arguments or by editing the source files:

**Video Processor** (`src/video_processor.py`):
```python
processor = VideoProcessor(
    lane_length=60.0,      # Change lane length
    ball_diameter=8.5,     # Change ball diameter
    use_stabilization=True # Toggle stabilization
)
```

**Ball Color Ranges** (`src/video_processor.py`, line 237):
```python
color_ranges = {
    "red": ((0, 100, 100), (10, 255, 255)),      # HSV range
    "blue": ((100, 100, 100), (130, 255, 255)),
    # Add custom colors here
}
```

### Database Path

Default: `data/bowling_stats.db`

To change:
```python
processor = VideoProcessor(db_path="custom/path/bowling.db")
```

## How It Works

### 1. Video Stabilization (Optional)
- Analyzes camera motion using optical flow
- Calculates stabilization transforms (no disk I/O)
- Compensates for head movement during tracking

### 2. Ball Tracking
- **Motion Detection**: Background subtraction to find moving objects
- **Shape Detection**: Hough Circle Transform to identify circular objects
- **Color Validation**: Optional HSV color matching for additional confidence
- **Kalman Filtering**: Smooths trajectory and predicts next position
- **Confidence Scoring**: Each detection receives a confidence score (0.0-1.0)

### 3. Speed Calculation
- Calibrates pixel-to-feet ratio using ball diameter
- Supports multi-point calibration for improved accuracy
- Calculates speed from trajectory distance and video FPS
- Reports confidence based on calibration consistency

### 4. Trajectory Analysis
- Detects hook point (where ball starts curving)
- Calculates lateral deviation and board positions
- Classifies trajectory type (straight, hook_left, hook_right)

## Troubleshooting

### "No ball detected in video"
**Causes:**
- Ball not visible in footage
- Ball color doesn't match `--color` parameter
- Poor lighting conditions

**Solutions:**
- Try removing `--color` flag (makes color validation optional)
- Ensure ball is clearly visible throughout the shot
- Use better lighting or adjust camera angle

### "Video file not found"
**Solution:**
- Verify the file path is correct
- Use absolute paths if relative paths fail
- Check file extension is supported (`.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`)

### "Stabilization failed"
**Solution:**
- Use `--no-stabilization` flag if using a stationary camera
- Ensure video has sufficient features to track (not a blank wall)
- Check video isn't corrupted

### Slow Processing
**Solutions:**
- Disable `--save-video` if you don't need annotated output
- Use `--no-stabilization` for stationary camera footage
- Close other applications to free up RAM
- Process videos in smaller batches

### Database Errors
**Solution:**
- Delete `data/bowling_stats.db` and run `python src/database.py` to recreate
- Check disk space availability
- Ensure you have write permissions in the `data/` directory

## Technical Details

### Ball Tracking Algorithm
1. Background subtraction (MOG2) identifies moving objects
2. Hough Circle Transform finds circular shapes
3. Color validation (optional) confirms ball color
4. Candidates are merged based on proximity
5. Kalman filter predicts next position and validates detections
6. Only detections above minimum confidence threshold are kept

### Calibration System
- **Single-point**: Uses ball diameter for initial calibration
- **Multi-point**: Combines multiple measurements with weighted average
- **Confidence**: Based on measurement consistency (standard deviation)
- **Adaptive**: Improves accuracy as more shots are processed

### Coordinate System
- **X-axis**: Lateral position (boards 1-39, right to left)
- **Y-axis**: Distance down lane (feet from foul line)
- **Board 20**: Center of lane (headpin)
- **Boards 1-10**: Right side (for right-handed bowlers)
- **Boards 30-39**: Left side

## Known Limitations

- **Pin detection**: Not yet implemented (manual entry required in dashboard)
- **Scoring**: Not automated (strikes/spares must be entered manually)
- **Ball speed range**: Accurate for 10-25 mph (typical bowling speeds)
- **Video quality**: Requires clear footage with visible ball throughout shot
- **Camera angle**: Works best when shot from directly behind the bowler
- **Lighting**: Requires consistent lighting (dim alley lighting can reduce accuracy)

## Roadmap

### Completed ✅
- [x] Basic project structure
- [x] Video processing pipeline
- [x] Batch processing for multiple videos
- [x] Hybrid ball tracking (motion + shape + color + Kalman)
- [x] Video stabilization for head-mounted cameras
- [x] Speed calculation with multi-point calibration
- [x] Database integration (SQLite)
- [x] Streamlit dashboard
- [x] Trajectory overlay visualization with board numbers
- [x] Hook point detection and visualization
- [x] Comprehensive error handling and validation

### In Progress 🚧
- [ ] Unit tests and integration tests
- [ ] Calibration from known lane features (foul line, arrows, dots)

### Future 🔮
- [ ] Automated pin detection using computer vision
- [ ] Automatic scoring (strikes, spares, open frames)
- [ ] Ball revolution rate (rev rate) calculation
- [ ] Axis rotation and tilt measurements
- [ ] Session comparison features
- [ ] Export to PDF/CSV for detailed reports
- [ ] Automated video import from Meta Ray-Ban glasses via cloud sync
- [ ] Mobile app for on-lane analysis
- [ ] Web-based dashboard

## Performance Expectations

- **Processing time**: ~1-3 minutes per minute of video (depending on hardware)
- **Detection rate**: 70-90% of frames (varies with video quality and camera movement)
- **Speed accuracy**: ±1-2 mph with proper calibration
- **Memory usage**: ~500MB-2GB depending on video resolution

## Contributing

This is a personal project, but suggestions and bug reports are welcome! Please open an issue on GitHub.

## License

Personal use project. Not licensed for commercial use.

## Acknowledgments

- Built with OpenCV, NumPy, Pandas, Streamlit, and Plotly
- Designed for Meta Ray-Ban smart glasses footage
- Inspired by professional bowling analytics systems

---

**Questions or Issues?** Check the Troubleshooting section or open a GitHub issue.
