"""
db.py — Database layer for BelievFit AI Gym Coach.
Uses standard library sqlite3 for persistent user accounts, sessions, and workout logs.
Enhanced with robust connection handling, input validation, and security safeguards.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

# Allow environment variable override for testing and custom deployment
DB_PATH = os.environ.get("BELIEVFIT_DB_PATH", os.path.join(os.path.dirname(__file__), "gym_coach.db"))

USERNAME_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.]{3,32}$")
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def get_connection() -> sqlite3.Connection:
    """Return a new SQLite connection with Row factory and foreign keys enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Initialize database tables for users, sessions, and workout sessions."""
    conn = get_connection()
    try:
        with conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            # Active user sessions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)

            # Workouts history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workouts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    exercise TEXT NOT NULL,
                    total_reps INTEGER NOT NULL,
                    good_reps INTEGER NOT NULL,
                    avg_score REAL NOT NULL,
                    duration_s REAL NOT NULL,
                    report_json TEXT NOT NULL,
                    video_filename TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)

            # Create indices for fast lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_workouts_user ON workouts(user_id)")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Password hashing & verification
# ---------------------------------------------------------------------------
def _hash_password(password: str, salt: str) -> str:
    """Hash password using PBKDF2-HMAC-SHA256 with 100,000 rounds."""
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    ).hex()


def create_user(username: str, email: str, password: str) -> dict[str, Any]:
    """Register a new user account with strict input validation."""
    username = username.strip()
    email = email.strip().lower()
    
    if not username or not USERNAME_REGEX.match(username):
        raise ValueError("Username must be 3-32 characters (letters, numbers, underscore, hyphen, period).")
    if not email or not EMAIL_REGEX.match(email):
        raise ValueError("A valid email address is required.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    if len(password) > 128:
        raise ValueError("Password is too long (maximum 128 characters).")

    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)
    now = datetime.now(timezone.utc).isoformat()

    conn = get_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (username, email, password_hash, salt, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (username, email, password_hash, salt, now))
            user_id = cursor.lastrowid
            return {"id": user_id, "username": username, "email": email, "created_at": now}
    except sqlite3.IntegrityError as e:
        err_msg = str(e).lower()
        if "username" in err_msg:
            raise ValueError("Username is already taken.")
        elif "email" in err_msg:
            raise ValueError("Email is already registered.")
        else:
            raise ValueError("User already exists.")
    finally:
        conn.close()


def authenticate_user(username_or_email: str, password: str) -> Optional[dict[str, Any]]:
    """Verify credentials and return user info, or None on failure."""
    if not username_or_email or not password:
        return None
        
    identifier = username_or_email.strip().lower()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, username, email, password_hash, salt, created_at
            FROM users
            WHERE LOWER(username) = ? OR LOWER(email) = ?
        """, (identifier, identifier))
        row = cursor.fetchone()
        if not row:
            return None

        computed_hash = _hash_password(password, row["salt"])
        if secrets.compare_digest(computed_hash, row["password_hash"]):
            return {
                "id": row["id"],
                "username": row["username"],
                "email": row["email"],
                "created_at": row["created_at"],
            }
        return None
    finally:
        conn.close()


def create_session(user_id: int, days_valid: int = 7) -> str:
    """Create an active auth session token (32 random urlsafe bytes)."""
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    expires = (now + timedelta(days=days_valid)).isoformat()
    
    conn = get_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (token, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
            """, (token, user_id, now.isoformat(), expires))
        return token
    finally:
        conn.close()


def get_user_by_session(token: str) -> Optional[dict[str, Any]]:
    """Validate session token and return user dictionary if valid and unexpired."""
    if not token or not isinstance(token, str):
        return None
    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT u.id, u.username, u.email, u.created_at
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token = ? AND s.expires_at > ?
        """, (token, now))
        row = cursor.fetchone()
        if row:
            return {
                "id": row["id"],
                "username": row["username"],
                "email": row["email"],
                "created_at": row["created_at"],
            }
        return None
    finally:
        conn.close()


def delete_session(token: str) -> None:
    """Invalidate and remove an active session token."""
    if not token:
        return
    conn = get_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
    finally:
        conn.close()


def cleanup_expired_sessions() -> int:
    """Purge expired sessions from the database."""
    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
            return cursor.rowcount
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Workout recording & analytics
# ---------------------------------------------------------------------------
def save_workout(
    user_id: int,
    exercise: str,
    total_reps: int,
    good_reps: int,
    avg_score: float,
    duration_s: float,
    report_dict: dict[str, Any],
    video_filename: Optional[str] = None,
) -> int:
    """Save completed workout metrics and JSON telemetry to database."""
    now = datetime.now(timezone.utc).isoformat()
    report_json = json.dumps(report_dict)
    conn = get_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO workouts (
                    user_id, exercise, total_reps, good_reps, avg_score,
                    duration_s, report_json, video_filename, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, exercise, total_reps, good_reps, round(avg_score, 1),
                round(duration_s, 2), report_json, video_filename, now
            ))
            workout_id = cursor.lastrowid
            return workout_id
    finally:
        conn.close()


def get_user_workouts(user_id: int, limit: int = 50) -> list[dict[str, Any]]:
    """Retrieve list of user's past workout summaries."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, exercise, total_reps, good_reps, avg_score, duration_s,
                   video_filename, created_at
            FROM workouts
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (user_id, max(1, min(limit, 500))))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_workout_details(workout_id: int, user_id: int) -> Optional[dict[str, Any]]:
    """Retrieve full details including JSON rep report for a specific workout."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, exercise, total_reps, good_reps, avg_score, duration_s,
                   report_json, video_filename, created_at
            FROM workouts
            WHERE id = ? AND user_id = ?
        """, (workout_id, user_id))
        row = cursor.fetchone()
        if not row:
            return None
        data = dict(row)
        data["report"] = json.loads(data["report_json"])
        del data["report_json"]
        return data
    finally:
        conn.close()


def get_user_stats(user_id: int) -> dict[str, Any]:
    """Calculate lifetime summary statistics for a user."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total_workouts,
                COALESCE(SUM(total_reps), 0) as total_reps,
                COALESCE(SUM(good_reps), 0) as total_good_reps,
                COALESCE(AVG(avg_score), 0) as overall_avg_score,
                COALESCE(MAX(avg_score), 0) as best_score
            FROM workouts
            WHERE user_id = ?
        """, (user_id,))
        row = cursor.fetchone()
        total_w = row["total_workouts"]
        tot_reps = row["total_reps"]
        good_reps = row["total_good_reps"]
        acc = round((good_reps / tot_reps * 100) if tot_reps > 0 else 0, 1)
        return {
            "total_workouts": total_w,
            "total_reps": tot_reps,
            "good_reps": good_reps,
            "accuracy_pct": acc,
            "avg_score": round(row["overall_avg_score"], 1),
            "best_score": round(row["best_score"], 1),
        }
    finally:
        conn.close()
