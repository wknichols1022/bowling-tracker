# Bowling Tracker - Quick Start Guide

## Setup Instructions

### 1. Install Dependencies

First, create a virtual environment and install required packages:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize Database

Run the database setup script:

```bash
python src/database.py
```

This will create the SQLite database at `data/bowling_stats.db`.

## Usage Workflow

### Step 1: Record Videos with Ray-Ban Glasses

- Wear your Meta Ray-Ban glasses at the bowling alley
- Record videos of your bowling shots
- Export videos from the Meta View app to your computer
- Place video files in `data/raw_videos/` directory

### Step 2: Process Videos

Process individual videos using the command line:

```bash
# Basic usage (creates new session)
python src/video_processor.py --video data/raw_videos/shot1.mp4

# Specify ball color
python src/video_processor.py --video data/raw_videos/shot1.mp4 --color blue

# Specify location
python src/video_processor.py --video data/raw_videos/shot1.mp4 --location "Strike Zone Bowling"

# Save annotated video with tracking visualization
python src/video_processor.py --video data/raw_videos/shot1.mp4 --save-video

# Add to existing session
python src/video_processor.py --video data/raw_videos/shot2.mp4 --session-id 1
```

### Step 3: View Dashboard

Launch the Streamlit dashboard to visualize your performance:

```bash
streamlit run src/dashboard.py
```

The dashboard will open in your web browser at `http://localhost:8501`

## Tips for Best Results

### Camera Positioning
- Position yourself behind the bowler's approach
- Keep the entire lane visible in the frame
- Try to maintain consistent height/angle between shots

### Ball Detection
- The system works best with solid-colored balls
- Adjust the `--color` parameter to match your ball:
  - Red: `--color red` (default)
  - Blue: `--color blue`
  - Green: `--color green`
  - Black: `--color black`
  - Purple: `--color purple`
  - Orange: `--color orange`

### Video Quality
- Higher frame rate videos (60fps) provide more accurate speed calculations
- Good lighting is important for ball detection
- Minimize camera shake for best tracking results

## Understanding the Data

### Speed Metrics
- **Speed (mph)**: Ball speed in miles per hour
- **Speed (ft/s)**: Ball speed in feet per second
- Typical bowling speeds range from 12-22 mph for most bowlers
- Professional bowlers often throw 16-19 mph

### Trajectory Analysis
- **Straight**: Ball travels in relatively straight line
- **Hook Left**: Ball curves to the left
- **Hook Right**: Ball curves to the right
- Lateral deviation indicates amount of hook/curve

### Dashboard Features
- **Overview**: Quick stats and speed distribution
- **Speed Analysis**: Speed trends over time, session comparisons
- **Trajectory**: Visual analysis of ball paths (coming soon)
- **Shot Details**: Detailed information for each shot

## Troubleshooting

### Ball Not Detected
- Check that the ball color setting matches your ball
- Ensure adequate lighting in the video
- Try the alternative detection method (experimental feature)

### Inaccurate Speed
- Ensure camera is relatively stable
- Check that the entire lane is visible
- Speed calibration uses ball size - may need manual adjustment

### Database Issues
- If database errors occur, check that `data/` directory exists
- To reset database, delete `data/bowling_stats.db` and run initialization again

## Next Steps

Once you have the basic system working:

1. **Collect more data**: Process multiple sessions to build your performance history
2. **Analyze trends**: Use the dashboard to identify improvements or patterns
3. **Experiment with settings**: Try different color detection settings if needed
4. **Future enhancements**: Web app and mobile access (Phase 3)

## Project Structure Reference

```
bowling-tracker/
├── data/
│   ├── raw_videos/          # Place your videos here
│   ├── processed/           # Annotated videos saved here
│   └── bowling_stats.db     # Your performance database
├── src/
│   ├── video_processor.py   # Main processing script
│   ├── ball_tracker.py      # Computer vision tracking
│   ├── speed_calculator.py  # Speed/trajectory calculations
│   ├── database.py          # Database operations
│   └── dashboard.py         # Visualization dashboard
└── requirements.txt         # Python dependencies
```

## Common Commands Cheat Sheet

```bash
# Process a video
python src/video_processor.py --video path/to/video.mp4

# View dashboard
streamlit run src/dashboard.py

# Initialize/reset database
python src/database.py

# Process with all options
python src/video_processor.py \
  --video data/raw_videos/shot1.mp4 \
  --color blue \
  --location "My Bowling Alley" \
  --save-video
```

Happy bowling! 🎳
