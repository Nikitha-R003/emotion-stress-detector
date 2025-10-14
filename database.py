import sqlite3
import bcrypt
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
import os

class DatabaseManager:
    def __init__(self, db_path: str = "mental_wellness.db"):
        self.db_path = db_path
        self.init_database()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_database(self):
        """Initialize database tables if they don't exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Mood history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS mood_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    text TEXT NOT NULL,
                    emotion TEXT NOT NULL,
                    emotion_conf REAL NOT NULL,
                    sentiment TEXT NOT NULL,
                    stress_level TEXT NOT NULL,
                    wellness_score REAL NOT NULL,
                    polarity REAL NOT NULL,
                    subjectivity REAL NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')

            # Journal entries table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS journal_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    content TEXT NOT NULL,
                    emotion TEXT NOT NULL,
                    sentiment TEXT NOT NULL,
                    insights TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')

            conn.commit()

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def create_user(self, username: str, password: str) -> bool:
        """Create a new user account"""
        try:
            hashed_password = self.hash_password(password)
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (username, hashed_password)
                )
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            # Username already exists
            return False
        except Exception as e:
            print(f"Error creating user: {e}")
            return False

    def authenticate_user(self, username: str, password: str) -> Optional[int]:
        """Authenticate user and return user_id if successful"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, password_hash FROM users WHERE username = ?",
                    (username,)
                )
                result = cursor.fetchone()

                if result and self.verify_password(password, result[1]):
                    return result[0]
                return None
        except Exception as e:
            print(f"Error authenticating user: {e}")
            return None

    def save_mood_entry(self, user_id: int, entry: Dict[str, Any]) -> bool:
        """Save a mood analysis entry to database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO mood_history
                    (user_id, timestamp, text, emotion, emotion_conf, sentiment,
                     stress_level, wellness_score, polarity, subjectivity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id,
                    entry['timestamp'].isoformat() if isinstance(entry['timestamp'], datetime) else entry['timestamp'],
                    entry['text'],
                    entry['emotion'],
                    entry['emotion_conf'],
                    entry['sentiment'],
                    entry['stress_level'],
                    entry['wellness_score'],
                    entry['polarity'],
                    entry['subjectivity']
                ))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error saving mood entry: {e}")
            return False

    def save_journal_entry(self, user_id: int, entry: Dict[str, Any]) -> bool:
        """Save a journal entry to database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO journal_entries
                    (user_id, timestamp, content, emotion, sentiment, insights)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    user_id,
                    entry['timestamp'].isoformat() if isinstance(entry['timestamp'], datetime) else entry['timestamp'],
                    entry['content'],
                    entry['emotion'],
                    entry['sentiment'],
                    json.dumps(entry['insights'])  # Store insights as JSON
                ))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error saving journal entry: {e}")
            return False

    def get_mood_history(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all mood history entries for a user"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT timestamp, text, emotion, emotion_conf, sentiment,
                           stress_level, wellness_score, polarity, subjectivity
                    FROM mood_history
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                ''', (user_id,))

                entries = []
                for row in cursor.fetchall():
                    entry = {
                        'timestamp': datetime.fromisoformat(row[0]),
                        'text': row[1],
                        'emotion': row[2],
                        'emotion_conf': row[3],
                        'sentiment': row[4],
                        'stress_level': row[5],
                        'wellness_score': row[6],
                        'polarity': row[7],
                        'subjectivity': row[8]
                    }
                    entries.append(entry)
                return entries
        except Exception as e:
            print(f"Error getting mood history: {e}")
            return []

    def get_journal_entries(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all journal entries for a user"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT timestamp, content, emotion, sentiment, insights
                    FROM journal_entries
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                ''', (user_id,))

                entries = []
                for row in cursor.fetchall():
                    entry = {
                        'timestamp': datetime.fromisoformat(row[0]),
                        'content': row[1],
                        'emotion': row[2],
                        'sentiment': row[3],
                        'insights': json.loads(row[4]) if row[4] else []
                    }
                    entries.append(entry)
                return entries
        except Exception as e:
            print(f"Error getting journal entries: {e}")
            return []

    def user_exists(self, username: str) -> bool:
        """Check if a username already exists"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
                return cursor.fetchone() is not None
        except Exception as e:
            print(f"Error checking user existence: {e}")
            return False

# Global database instance
db_manager = DatabaseManager()
