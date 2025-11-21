"""
Streamlit dashboard for visualizing bowling performance data.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

from database import BowlingDatabase


def create_trajectory_overlay(shots_df):
    """
    Create a bowling lane visualization with overlaid shot trajectories.

    Args:
        shots_df: DataFrame containing shot data with trajectory_data

    Returns:
        Plotly figure object
    """
    # Bowling lane dimensions
    LANE_LENGTH = 60  # foul line to headpin (feet)
    LANE_WIDTH = 3.5  # standard lane width (feet)
    BOARD_WIDTH = 1.0  # inches per board
    NUM_BOARDS = 39  # standard number of boards

    # Convert to pixels for visualization (scale factor)
    SCALE = 10  # pixels per foot
    lane_length_px = LANE_LENGTH * SCALE
    lane_width_px = LANE_LENGTH * SCALE  # Make it square for better visibility

    # Calculate board positions
    board_width_px = lane_width_px / NUM_BOARDS

    # Create figure
    fig = go.Figure()

    # Draw bowling lane background
    fig.add_shape(
        type="rect",
        x0=0, y0=0,
        x1=lane_width_px, y1=lane_length_px,
        fillcolor="wheat",
        line=dict(color="black", width=2),
        layer="below"
    )

    # Draw board lines (every 5th board for clarity)
    for board in range(0, NUM_BOARDS + 1, 5):
        x_pos = board * board_width_px
        fig.add_shape(
            type="line",
            x0=x_pos, y0=0,
            x1=x_pos, y1=lane_length_px,
            line=dict(color="lightgray", width=0.5),
            layer="below"
        )

    # Draw foul line
    fig.add_shape(
        type="line",
        x0=0, y0=10,
        x1=lane_width_px, y1=10,
        line=dict(color="red", width=2, dash="dash"),
        layer="below"
    )

    # Draw approach dots (release area markers) - at boards 10, 20, 30
    approach_y = 5
    for board in [10, 20, 30]:
        x_pos = board * board_width_px
        fig.add_shape(
            type="circle",
            x0=x_pos - 2, y0=approach_y - 2,
            x1=x_pos + 2, y1=approach_y + 2,
            fillcolor="black",
            line=dict(color="black"),
            layer="below"
        )

    # Draw pin positions at the end (head pin is at board 20, the center)
    pin_y = lane_length_px - 10
    pin_positions = [
        (20 * board_width_px, pin_y),  # 1-pin (head pin) - board 20
        (17 * board_width_px, pin_y + 5),  # 2-pin
        (23 * board_width_px, pin_y + 5),  # 3-pin
        (14 * board_width_px, pin_y + 10),  # 4-pin
        (20 * board_width_px, pin_y + 10),  # 5-pin
        (26 * board_width_px, pin_y + 10),  # 6-pin
        (11 * board_width_px, pin_y + 15),  # 7-pin
        (17 * board_width_px, pin_y + 15),  # 8-pin
        (23 * board_width_px, pin_y + 15),  # 9-pin
        (29 * board_width_px, pin_y + 15),  # 10-pin
    ]

    for x, y in pin_positions:
        fig.add_shape(
            type="circle",
            x0=x - 3, y0=y - 3,
            x1=x + 3, y1=y + 3,
            fillcolor="black",
            line=dict(color="black"),
            layer="below"
        )

    # Color schemes for different trajectory types
    color_map = {
        'straight': 'blue',
        'hook': 'orange',
        'curve': 'purple',
        'backup': 'green',
        'default': 'gray'
    }

    # Plot trajectories
    for idx, shot in shots_df.iterrows():
        if shot['trajectory_data']:
            try:
                traj_data = json.loads(shot['trajectory_data'])
                if 'points' in traj_data and traj_data['points']:
                    points = traj_data['points']

                    # Extract x, y coordinates from trajectory points
                    # Points are in format [(x, y, frame), ...] from video coordinates
                    x_coords = []
                    y_coords = []

                    # Get the range of coordinates to map to lane
                    x_values = [p[0] for p in points if len(p) >= 2]
                    y_values = [p[1] for p in points if len(p) >= 2]

                    if not x_values or not y_values:
                        continue

                    x_min, x_max = min(x_values), max(x_values)
                    y_min, y_max = min(y_values), max(y_values)

                    # Prevent division by zero
                    x_range = x_max - x_min if x_max != x_min else 1
                    y_range = y_max - y_min if y_max != y_min else 1

                    for point in points:
                        if len(point) >= 2:
                            # Map video x-coordinates to lane width (boards)
                            # Center the trajectory horizontally around board 20
                            x_normalized = (point[0] - x_min) / x_range  # 0 to 1
                            x_centered = (x_normalized - 0.5)  # -0.5 to 0.5
                            # Map to boards (use about 15 boards range for visibility)
                            x_board = 20 + (x_centered * 15)  # Center around board 20
                            x_norm = x_board * board_width_px

                            # Map video y-coordinates to lane length
                            # Start at foul line (y=10px) and go to pins
                            y_normalized = (point[1] - y_min) / y_range  # 0 to 1
                            y_norm = 10 + (y_normalized * (lane_length_px - 20))

                            x_coords.append(x_norm)
                            y_coords.append(y_norm)

                    # Convert x positions to board numbers for display
                    board_positions = [int(x / board_width_px) for x in x_coords]

                    # Get trajectory type for color
                    traj_type = 'default'
                    if 'analysis' in traj_data and 'trajectory_type' in traj_data['analysis']:
                        traj_type = traj_data['analysis']['trajectory_type']

                    color = color_map.get(traj_type, color_map['default'])

                    # Get speed for hover info
                    speed_info = f"{shot['speed_mph']:.1f} mph" if pd.notna(shot['speed_mph']) else "N/A"

                    # Calculate release and impact boards
                    release_board = board_positions[0] if board_positions else "N/A"
                    impact_board = board_positions[-1] if board_positions else "N/A"

                    # Extract hook point data if available
                    hook_point_info = ""
                    hook_board = "N/A"
                    hook_distance = "N/A"

                    if 'analysis' in traj_data and 'hook_point' in traj_data['analysis']:
                        hook_data = traj_data['analysis']['hook_point']
                        if 'hook_point_x' in hook_data and 'hook_point_y' in hook_data:
                            # Map hook point using same transformation as trajectory
                            hook_x_normalized = (hook_data['hook_point_x'] - x_min) / x_range
                            hook_x_centered = (hook_x_normalized - 0.5)
                            hook_x_board = 20 + (hook_x_centered * 15)
                            hook_x_px = hook_x_board * board_width_px
                            hook_board = int(hook_x_board)

                            hook_y_normalized = (hook_data['hook_point_y'] - y_min) / y_range
                            hook_y_px = 10 + (hook_y_normalized * (lane_length_px - 20))

                            if hook_data.get('distance_from_start_feet'):
                                hook_distance = f"{hook_data['distance_from_start_feet']:.1f} ft"
                            else:
                                hook_distance = "N/A"

                            hook_point_info = f"<br>Hook Point: Board {hook_board} @ {hook_distance}"

                            # Plot hook point marker
                            fig.add_trace(go.Scatter(
                                x=[hook_x_px],
                                y=[hook_y_px],
                                mode='markers',
                                marker=dict(
                                    size=10,
                                    color=color,
                                    symbol='x',
                                    line=dict(width=2, color='white')
                                ),
                                showlegend=False,
                                hovertemplate=f"<b>Hook Point</b><br>" +
                                            f"Shot {shot['shot_number']}<br>" +
                                            f"Board: {hook_board}<br>" +
                                            f"Distance: {hook_distance}<extra></extra>"
                            ))

                    # Plot trajectory
                    fig.add_trace(go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='lines',
                        line=dict(color=color, width=2),
                        opacity=0.6,
                        name=f"Shot {shot['shot_number']} ({traj_type})",
                        hovertemplate=f"<b>Shot {shot['shot_number']}</b><br>" +
                                    f"Speed: {speed_info}<br>" +
                                    f"Type: {traj_type}<br>" +
                                    f"Release: Board {release_board}<br>" +
                                    f"Impact: Board {impact_board}" +
                                    hook_point_info + "<br>" +
                                    f"Date: {shot['session_date']}<extra></extra>"
                    ))

            except Exception as e:
                # Skip shots with invalid trajectory data
                continue

    # Update layout with board numbers
    fig.update_layout(
        title="Bowling Shot Trajectories Overlay",
        xaxis=dict(
            title="Board Number (1 = Right Gutter, 20 = Center, 39 = Left Gutter)",
            range=[0, lane_width_px],
            tickmode='array',
            tickvals=[i * board_width_px for i in [1, 5, 10, 15, 20, 25, 30, 35, 39]],
            ticktext=['1', '5', '10', '15', '20', '25', '30', '35', '39'],
            showgrid=True,
            gridcolor='lightgray',
            side='bottom'
        ),
        yaxis=dict(
            title="Distance from Foul Line (feet)",
            range=[0, lane_length_px],
            tickmode='array',
            tickvals=[10, 20*SCALE, 40*SCALE, lane_length_px],
            ticktext=['0', '20', '40', '60'],
            showgrid=True,
            gridcolor='lightgray'
        ),
        plot_bgcolor='white',
        height=800,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        hovermode='closest'
    )

    return fig


def load_data():
    """Load data from database."""
    db = BowlingDatabase()
    
    sessions = db.get_all_sessions()
    all_shots = []
    
    for session in sessions:
        shots = db.get_session_shots(session['session_id'])
        for shot in shots:
            shot['session_date'] = session['date']
            shot['location'] = session['location']
            all_shots.append(shot)
    
    db.close()
    
    return pd.DataFrame(sessions), pd.DataFrame(all_shots)


def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="Bowling Performance Tracker",
        page_icon="🎳",
        layout="wide"
    )
    
    st.title("🎳 Bowling Performance Tracker")
    st.markdown("---")
    
    # Load data
    sessions_df, shots_df = load_data()
    
    if shots_df.empty:
        st.info("No bowling data found. Process some videos first!")
        st.markdown("""
        To get started:
        1. Place your videos in `data/raw_videos/`
        2. Run: `python src/video_processor.py --video data/raw_videos/your_video.mp4`
        3. Refresh this dashboard
        """)
        return
    
    # Sidebar - Session selector
    st.sidebar.header("Filters")
    
    # Session selection
    session_options = ["All Sessions"] + [
        f"{row['date']} - {row['location'] or 'Unknown'}" 
        for _, row in sessions_df.iterrows()
    ]
    selected_session = st.sidebar.selectbox("Select Session", session_options)
    
    # Filter shots based on selection
    if selected_session != "All Sessions":
        session_idx = session_options.index(selected_session) - 1
        session_id = sessions_df.iloc[session_idx]['session_id']
        filtered_shots = shots_df[shots_df['session_id'] == session_id]
    else:
        filtered_shots = shots_df
    
    # Remove rows with null speeds for statistics
    valid_shots = filtered_shots[filtered_shots['speed_mph'].notna()]
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Shots",
            len(filtered_shots)
        )
    
    with col2:
        if not valid_shots.empty:
            avg_speed = valid_shots['speed_mph'].mean()
            st.metric(
                "Avg Speed",
                f"{avg_speed:.1f} mph"
            )
        else:
            st.metric("Avg Speed", "N/A")
    
    with col3:
        if not valid_shots.empty:
            max_speed = valid_shots['speed_mph'].max()
            st.metric(
                "Max Speed",
                f"{max_speed:.1f} mph"
            )
        else:
            st.metric("Max Speed", "N/A")
    
    with col4:
        if not valid_shots.empty:
            std_speed = valid_shots['speed_mph'].std()
            st.metric(
                "Consistency (σ)",
                f"{std_speed:.2f} mph"
            )
        else:
            st.metric("Consistency", "N/A")
    
    st.markdown("---")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", 
        "📈 Speed Analysis", 
        "🎯 Trajectory", 
        "📋 Shot Details"
    ])
    
    with tab1:
        st.header("Performance Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Speed distribution
            if not valid_shots.empty:
                st.subheader("Speed Distribution")
                fig = px.histogram(
                    valid_shots,
                    x='speed_mph',
                    nbins=20,
                    labels={'speed_mph': 'Speed (mph)'},
                    title="Distribution of Ball Speeds"
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No speed data available")
        
        with col2:
            # Session summary
            st.subheader("Session Summary")
            if not sessions_df.empty:
                session_summary = sessions_df[[
                    'date', 'location', 'total_shots', 'avg_speed'
                ]].copy()
                session_summary['avg_speed'] = session_summary['avg_speed'].round(2)
                st.dataframe(
                    session_summary,
                    column_config={
                        'date': 'Date',
                        'location': 'Location',
                        'total_shots': st.column_config.NumberColumn(
                            'Total Shots',
                            format="%d"
                        ),
                        'avg_speed': st.column_config.NumberColumn(
                            'Avg Speed',
                            format="%.2f mph"
                        )
                    },
                    hide_index=True,
                    use_container_width=True
                )
    
    with tab2:
        st.header("Speed Analysis")
        
        if not valid_shots.empty:
            # Add session date for time series
            valid_shots_sorted = valid_shots.sort_values('shot_id')
            
            # Speed over time
            st.subheader("Speed Progression")
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=valid_shots_sorted['shot_id'],
                y=valid_shots_sorted['speed_mph'],
                mode='lines+markers',
                name='Speed',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ))
            
            # Add average line
            avg_speed = valid_shots_sorted['speed_mph'].mean()
            fig.add_hline(
                y=avg_speed,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Average: {avg_speed:.2f} mph"
            )
            
            fig.update_layout(
                xaxis_title="Shot Number",
                yaxis_title="Speed (mph)",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Speed statistics by session
            st.subheader("Speed Statistics by Session")
            session_stats = valid_shots.groupby('session_date').agg({
                'speed_mph': ['mean', 'min', 'max', 'std', 'count']
            }).round(2)
            session_stats.columns = ['Avg Speed', 'Min Speed', 'Max Speed', 'Std Dev', 'Count']
            st.dataframe(session_stats, use_container_width=True)
        else:
            st.info("No speed data available for analysis")
    
    with tab3:
        st.header("Trajectory Analysis")

        if not filtered_shots.empty:
            # Date range selector for trajectory overlay
            st.subheader("Trajectory Overlay")

            col1, col2 = st.columns(2)
            with col1:
                # Get unique dates
                unique_dates = sorted(filtered_shots['session_date'].unique())
                if len(unique_dates) > 0:
                    start_date = st.selectbox(
                        "Start Date",
                        options=unique_dates,
                        index=0
                    )

            with col2:
                if len(unique_dates) > 0:
                    end_date = st.selectbox(
                        "End Date",
                        options=unique_dates,
                        index=len(unique_dates) - 1
                    )

            # Filter shots by date range
            if len(unique_dates) > 0:
                date_filtered_shots = filtered_shots[
                    (filtered_shots['session_date'] >= start_date) &
                    (filtered_shots['session_date'] <= end_date)
                ]

                # Create trajectory overlay visualization
                fig = create_trajectory_overlay(date_filtered_shots)
                st.plotly_chart(fig, use_container_width=True)

                # Show trajectory statistics
                st.subheader("Trajectory Statistics")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Shots Displayed", len(date_filtered_shots))

                with col2:
                    # Parse trajectory data to get types
                    trajectory_types = []
                    for _, shot in date_filtered_shots.iterrows():
                        if shot['trajectory_data']:
                            try:
                                traj_data = json.loads(shot['trajectory_data'])
                                if 'analysis' in traj_data and 'trajectory_type' in traj_data['analysis']:
                                    trajectory_types.append(traj_data['analysis']['trajectory_type'])
                            except:
                                pass

                    if trajectory_types:
                        from collections import Counter
                        most_common = Counter(trajectory_types).most_common(1)[0][0]
                        st.metric("Most Common Type", most_common.title())

                with col3:
                    if not date_filtered_shots[date_filtered_shots['speed_mph'].notna()].empty:
                        avg_speed = date_filtered_shots[date_filtered_shots['speed_mph'].notna()]['speed_mph'].mean()
                        st.metric("Avg Speed", f"{avg_speed:.1f} mph")
            else:
                st.info("No trajectory data available")
        else:
            st.info("No shots found for trajectory analysis")
    
    with tab4:
        st.header("Shot Details")
        
        # Display detailed shot information
        if not filtered_shots.empty:
            display_df = filtered_shots[[
                'shot_id', 'session_date', 'shot_number', 
                'speed_mph', 'speed_fps', 'pins_hit', 
                'is_strike', 'is_spare'
            ]].copy()
            
            display_df = display_df.sort_values('shot_id', ascending=False)
            
            st.dataframe(
                display_df,
                column_config={
                    'shot_id': 'Shot ID',
                    'session_date': 'Date',
                    'shot_number': 'Shot #',
                    'speed_mph': st.column_config.NumberColumn(
                        'Speed (mph)',
                        format="%.2f"
                    ),
                    'speed_fps': st.column_config.NumberColumn(
                        'Speed (ft/s)',
                        format="%.2f"
                    ),
                    'pins_hit': 'Pins',
                    'is_strike': 'Strike',
                    'is_spare': 'Spare'
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No shots found")
    
    # Footer
    st.markdown("---")
    st.caption("Bowling Performance Tracker - Powered by Computer Vision")


if __name__ == "__main__":
    main()
