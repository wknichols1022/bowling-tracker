"""
Database module for storing bowling shot data and session information.
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple


class BowlingDatabase:
    """Handles all database operations for bowling shot tracking."""
    
    def __init__(self, db_path: str = "data/bowling_stats.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_directory()
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self.cursor = self.conn.cursor()
        self._create_tables()
    
    def _ensure_db_directory(self):
        """Create database directory if it doesn't exist."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        
        # Sessions table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                location TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Shots table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS shots (
                shot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                shot_number INTEGER NOT NULL,
                video_file TEXT NOT NULL,
                speed_mph REAL,
                speed_fps REAL,
                trajectory_data TEXT,
                release_point_x INTEGER,
                release_point_y INTEGER,
                impact_point_x INTEGER,
                impact_point_y INTEGER,
                pins_hit INTEGER,
                is_strike BOOLEAN,
                is_spare BOOLEAN,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """)
        
        # Create indexes for faster queries
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_date 
            ON sessions(date)
        """)
        
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_shots_session 
            ON shots(session_id)
        """)
        
        self.conn.commit()
        print(f"Database initialized at {self.db_path}")
    
    def create_session(self, date: str = None, location: str = None, 
                      notes: str = None) -> int:
        """
        Create a new bowling session.
        
        Args:
            date: Session date (YYYY-MM-DD format), defaults to today
            location: Bowling alley name
            notes: Optional session notes
            
        Returns:
            session_id: ID of created session
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        self.cursor.execute("""
            INSERT INTO sessions (date, location, notes)
            VALUES (?, ?, ?)
        """, (date, location, notes))
        
        self.conn.commit()
        return self.cursor.lastrowid
    
    def add_shot(self, session_id: int, shot_number: int, video_file: str,
                 speed_mph: float = None, speed_fps: float = None,
                 trajectory_data: str = None, release_point: Tuple[int, int] = None,
                 impact_point: Tuple[int, int] = None, pins_hit: int = None,
                 is_strike: bool = False, is_spare: bool = False) -> int:
        """
        Add a shot to the database.
        
        Args:
            session_id: ID of the session
            shot_number: Shot number within the session
            video_file: Path to video file
            speed_mph: Ball speed in miles per hour
            speed_fps: Ball speed in feet per second
            trajectory_data: JSON string of trajectory points
            release_point: (x, y) coordinates of release
            impact_point: (x, y) coordinates of pin impact
            pins_hit: Number of pins knocked down
            is_strike: Whether shot was a strike
            is_spare: Whether shot was a spare
            
        Returns:
            shot_id: ID of created shot
        """
        release_x, release_y = release_point if release_point else (None, None)
        impact_x, impact_y = impact_point if impact_point else (None, None)
        
        self.cursor.execute("""
            INSERT INTO shots (
                session_id, shot_number, video_file, speed_mph, speed_fps,
                trajectory_data, release_point_x, release_point_y,
                impact_point_x, impact_point_y, pins_hit, is_strike, is_spare
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, shot_number, video_file, speed_mph, speed_fps,
              trajectory_data, release_x, release_y, impact_x, impact_y,
              pins_hit, is_strike, is_spare))
        
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_session_shots(self, session_id: int) -> List[Dict]:
        """
        Get all shots for a specific session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            List of shot dictionaries
        """
        self.cursor.execute("""
            SELECT * FROM shots
            WHERE session_id = ?
            ORDER BY shot_number
        """, (session_id,))
        
        return [dict(row) for row in self.cursor.fetchall()]
    
    def get_all_sessions(self) -> List[Dict]:
        """
        Get all bowling sessions.
        
        Returns:
            List of session dictionaries
        """
        self.cursor.execute("""
            SELECT s.*, COUNT(sh.shot_id) as total_shots,
                   AVG(sh.speed_mph) as avg_speed
            FROM sessions s
            LEFT JOIN shots sh ON s.session_id = sh.session_id
            GROUP BY s.session_id
            ORDER BY s.date DESC
        """)
        
        return [dict(row) for row in self.cursor.fetchall()]
    
    def get_speed_stats(self, session_id: int = None) -> Dict:
        """
        Get speed statistics.
        
        Args:
            session_id: Optional session ID to filter by
            
        Returns:
            Dictionary with speed statistics
        """
        query = """
            SELECT 
                AVG(speed_mph) as avg_speed,
                MIN(speed_mph) as min_speed,
                MAX(speed_mph) as max_speed,
                COUNT(*) as total_shots
            FROM shots
            WHERE speed_mph IS NOT NULL
        """
        
        if session_id:
            query += " AND session_id = ?"
            self.cursor.execute(query, (session_id,))
        else:
            self.cursor.execute(query)
        
        row = self.cursor.fetchone()
        return dict(row) if row else {}
    
    def close(self):
        """Close database connection."""
        self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def main():
    """Initialize database and show example usage."""
    db = BowlingDatabase()
    
    print("Database tables created successfully!")
    print("\nExample usage:")
    print("  session_id = db.create_session(location='Strike Zone Bowling')")
    print("  shot_id = db.add_shot(session_id, 1, 'video.mp4', speed_mph=16.5)")
    print("  shots = db.get_session_shots(session_id)")
    print("  stats = db.get_speed_stats()")
    
    db.close()


if __name__ == "__main__":
    main()
