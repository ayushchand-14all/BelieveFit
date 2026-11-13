"""
app.py — Full-stack Mobile & Desktop Web Server for BelievFit AI Gym Coach.

Features:
  • Binds to 0.0.0.0 so any mobile phone on the same Wi-Fi can connect.
  • Auto-detects and displays laptop's LAN IP address (e.g. http://192.168.x.x:8000).
  • Full HTTP 206 Partial Content (Range requests) support for seamless video
    streaming and scrubbing on iOS Safari and Android Chrome.
  • Remote Laptop File Explorer: browse, stream, and analyse videos stored on the laptop.
  • Mobile video capture: record directly from mobile phone camera and analyse on laptop.
  • SQLite user auth, real-time analytics, SVG telemetry, and simulation.
  • Hardened security: Path traversal prevention, rate limiting, secure cookies,
    CSP and security headers, upload extension/size validation, thread-safe job tracking.

Usage:
    python app.py
"""
from __future__ import annotations

import http.cookies
import json
import mimetypes
import os
import re
import socket
import sys
import time
import urllib.parse
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from datetime import datetime
import subprocess
import threading
import uuid

import db
# demo imported lazily

PORT = int(os.environ.get("PORT", 8000))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
# Render's persistent disk is mounted at /var/data. Locally, default to the
# project directory so existing Windows workflows remain unchanged.
DATA_DIR = os.environ.get("BELIEVFIT_DATA_DIR", BASE_DIR)
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

# Initialise SQLite database on startup
db.init_db()

# Global background analysis job storage (thread-safe)
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()

# Rate limiting for auth endpoints (in-memory IP tracking)
LOGIN_ATTEMPTS: dict[str, list[float]] = {}
RATE_LIMIT_LOCK = threading.Lock()
RATE_LIMIT_MAX_ATTEMPTS = 15
RATE_LIMIT_WINDOW_SECONDS = 60


def check_rate_limit(client_ip: str) -> bool:
    """Return True if request is allowed, False if rate limited."""
    now = time.time()
    with RATE_LIMIT_LOCK:
        attempts = LOGIN_ATTEMPTS.get(client_ip, [])
        # Keep only timestamps within the window
        attempts = [t for t in attempts if now - t < RATE_LIMIT_WINDOW_SECONDS]
        if len(attempts) >= RATE_LIMIT_MAX_ATTEMPTS:
            LOGIN_ATTEMPTS[client_ip] = attempts
            return False
        attempts.append(now)
        LOGIN_ATTEMPTS[client_ip] = attempts
        return True


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    """Thread-safe retrieval of job status."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def update_job(job_id: str, **kwargs: Any) -> None:
    """Thread-safe update of job status."""
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)


def _resolve_safe_path(base_dir: str, rel_path: str) -> Optional[str]:
    """
    Safely resolve a relative path within a base directory, strictly preventing
    path traversal attacks (e.g., ../, null bytes, absolute paths).
    """
    if not rel_path or "\0" in rel_path:
        return None
    # Strip any leading slashes or backslashes
    clean_rel = rel_path.lstrip("/\\")
    target = os.path.abspath(os.path.join(base_dir, clean_rel))
    real_base = os.path.abspath(base_dir)
    try:
        if os.path.commonpath([real_base, target]) != real_base:
            return None
    except ValueError:
        return None
    if os.path.exists(target) and os.path.isfile(target):
        return target
    return None


def _is_user_owned_filename(filename: str, user_id: int) -> bool:
    """Only expose files whose server-generated name belongs to this user."""
    return os.path.basename(filename).startswith(f"user{int(user_id)}_")


def _make_browser_video(video_path: str, output_path: str) -> bool:
    """Create a broadly browser-compatible H.264/AAC MP4 for HTML5 playback.

    Uploaded originals may use HEVC, unusual pixel formats, or codecs that
    OpenCV can decode but Chrome/Firefox cannot play. Keep the original intact.
    """
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        command = [
            ffmpeg, "-y", "-i", video_path,
            "-map", "0:v:0", "-map", "0:a?",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", "-sn", "-dn", output_path,
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.returncode == 0 and os.path.isfile(output_path) and os.path.getsize(output_path) > 0
    except Exception as exc:
        print(f"[video] Browser-compatible transcode failed: {exc}")
        return False


def notify_device_login(username: str, client_ip: str, action: str = "login") -> None:
    """Dispatches audible chime, Windows toast, and mobile push notification when a user logs in."""
    def worker():
        timestamp = datetime.now().strftime("%H:%M:%S")
        print("\n" + "=" * 70)
        print(f"  🔔 [BELIEVFIT SIGNAL] Remote user '{username}' {action}ed at {timestamp}!")
        print(f"  📍 Client IP: {client_ip}")
        print("=" * 70 + "\n")

        # 1. Audible Windows Sound Chime
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except Exception:
            pass

        # 2. Windows Desktop Balloon Notification
        try:
            ps_script = f"""
            Add-Type -AssemblyName System.Windows.Forms
            $n = New-Object System.Windows.Forms.NotifyIcon
            $n.Icon = [System.Drawing.SystemIcons]::Information
            $n.Visible = $true
            $n.ShowBalloonTip(5000, 'BelievFit Alert', "Athlete '{username}' {action}ed from {client_ip}", [System.Windows.Forms.ToolTipIcon]::Info)
            """
            subprocess.Popen(["powershell", "-NoProfile", "-Command", ps_script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

        # 3. Mobile Push Notification via ntfy.sh
        try:
            topic = os.environ.get("NTFY_TOPIC", "believfit_alerts")
            req = urllib.request.Request(
                f"https://ntfy.sh/{topic}",
                data=f"Athlete '{username}' just {action}ed from IP {client_ip}.".encode("utf-8"),
                headers={
                    "Title": f"🏋️ BelievFit: {username} {action.capitalize()}!",
                    "Priority": "high",
                    "Tags": "muscle,bell"
                }
            )
            urllib.request.urlopen(req, timeout=3)
        except Exception:
            pass

    threading.Thread(target=worker, daemon=True).start()


def start_analysis_job(video_path: str, exercise: str, user_id: int, safe_name: str) -> str:
    """Run MediaPipe analysis in background worker thread with progress tracking."""
    job_id = str(uuid.uuid4())
    initial_job = {
        "job_id": job_id,
        "user_id": int(user_id),
        "status": "processing",
        "progress": 0.05,
        "message": "Initializing BelievFit MediaPipe neural model...",
        "report": None,
        "error": None,
        "workout_id": None,
        "created_at": time.time(),
    }
    with JOBS_LOCK:
        # Prune old jobs if dict grows large
        if len(JOBS) > 200:
            now = time.time()
            expired = [k for k, v in JOBS.items() if now - v.get("created_at", 0) > 3600]
            for k in expired:
                del JOBS[k]
        JOBS[job_id] = initial_job

    def worker():
        try:
            def on_progress(pct, msg):
                update_job(job_id, progress=round(pct, 2), message=msg)

            from pipeline import run_analysis
            json_path = run_analysis(video_path, exercise=exercise, max_people=2, progress_callback=on_progress)

            with open(json_path, "r", encoding="utf-8") as f_json:
                report = json.load(f_json)

            people = report.get("people", [])
            p1 = people[0] if people else {"reps": [], "average_score": 0}
            reps = p1.get("reps", [])
            good_reps = sum(r["valid"] for r in reps)
            avg_score = p1.get("average_score") or 0.0
            duration_s = report.get("duration_s", 0.0)

            name = os.path.splitext(safe_name)[0]
            # Always prepare a browser-safe H.264 copy. OpenCV may analyse HEVC,
            # AVI, or other codecs that HTML5 browsers cannot decode even though
            # the server-side pipeline succeeds. Keep the original upload intact.
            browser_name = f"{name}_browser.mp4"
            browser_path = os.path.join(OUTPUT_DIR, browser_name)
            browser_ready = _make_browser_video(video_path, browser_path)
            report["video_url"] = (f"/output/{urllib.parse.quote(browser_name)}" if browser_ready
                                   else f"/videos/{urllib.parse.quote(safe_name)}")
            report["original_video_url"] = f"/videos/{urllib.parse.quote(safe_name)}"
            report["browser_video_ready"] = browser_ready
            report["form_video_url"] = f"/output/{name}_form.mp4"
            report["skeleton_video_url"] = f"/output/{name}_skeleton.mp4"
            report["dashboard_video_url"] = f"/output/{name}_dashboard.mp4"

            workout_id = db.save_workout(
                user_id=user_id,
                exercise=exercise,
                total_reps=len(reps),
                good_reps=good_reps,
                avg_score=avg_score,
                duration_s=duration_s,
                report_dict=report,
                video_filename=safe_name,
            )

            update_job(
                job_id,
                status="completed",
                progress=1.0,
                message="Analysis complete!",
                report=report,
                workout_id=workout_id,
            )

        except Exception as e:
            update_job(
                job_id,
                status="failed",
                error=f"Analysis encountered an error: {str(e)}",
                message="Analysis failed",
            )

    threading.Thread(target=worker, daemon=True).start()
    return job_id


def _save_multipart_stream(rfile, content_length: int, boundary_str: str, user_id: int) -> tuple[str, str, str]:
    """Streams multipart data directly to disk to handle video files without excessive memory spikes."""
    if content_length > MAX_UPLOAD_SIZE:
        raise ValueError(f"File size exceeds maximum allowed limit ({MAX_UPLOAD_SIZE // (1024 * 1024)} MB).")

    temp_spool = os.path.join(VIDEOS_DIR, f"_upload_{uuid.uuid4().hex}.tmp")
    bytes_left = content_length
    with open(temp_spool, "wb") as f_spool:
        while bytes_left > 0:
            chunk = rfile.read(min(bytes_left, 65536))
            if not chunk:
                break
            f_spool.write(chunk)
            bytes_left -= len(chunk)

    delimiter = f"--{boundary_str}".encode("utf-8")
    exercise = "squat"
    saved_video_path = None
    safe_name = None

    try:
        with open(temp_spool, "rb") as f_in:
            content = f_in.read()

        parts = content.split(delimiter)
        for part in parts:
            part = part.strip()
            if not part or part == b"--":
                continue
            headers_and_body = part.split(b"\r\n\r\n", 1)
            if len(headers_and_body) != 2:
                continue
            raw_headers, raw_body = headers_and_body
            if raw_body.endswith(b"\r\n"):
                raw_body = raw_body[:-2]

            headers_str = raw_headers.decode("utf-8", errors="replace")
            name_match = re.search(r'name="([^"]+)"', headers_str)
            if not name_match:
                continue
            field_name = name_match.group(1)

            if field_name == "exercise":
                parsed_ex = raw_body.decode("utf-8", errors="replace").strip().lower()
                if parsed_ex in ("squat", "pushup"):
                    exercise = parsed_ex
            elif field_name == "video":
                fn_match = re.search(r'filename="([^"]+)"', headers_str)
                raw_filename = fn_match.group(1) if fn_match else "workout.mp4"
                clean_fn = os.path.basename(raw_filename)
                
                # Sanitize filename to alphanumeric and safe symbols
                name_root, ext = os.path.splitext(clean_fn)
                ext = ext.lower()
                if ext not in ALLOWED_VIDEO_EXTENSIONS:
                    raise ValueError(f"Invalid video format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}")

                safe_root = re.sub(r"[^a-zA-Z0-9_\-]", "_", name_root)[:64]
                safe_name = f"user{user_id}_{int(time.time())}_{safe_root}{ext}"
                saved_video_path = os.path.join(VIDEOS_DIR, safe_name)
                with open(saved_video_path, "wb") as f_out:
                    f_out.write(raw_body)

    finally:
        try:
            if os.path.exists(temp_spool):
                os.remove(temp_spool)
        except Exception:
            pass

    if not saved_video_path or not os.path.exists(saved_video_path):
        raise ValueError("No valid video file found in upload payload.")

    return saved_video_path, exercise, safe_name


def get_local_ip() -> str:
    """Auto-detect laptop LAN IP address for phone connection."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        pass
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "127.0.0.1"


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------
class GymCoachHandler(BaseHTTPRequestHandler):
    server_version = "BelievFit/2.0"

    def _set_security_headers(self) -> None:
        """Inject robust HTTP security headers."""
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self';"
        )

    def _send_json(self, data: Any, status: int = 200, cookie: Optional[str] = None) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._set_security_headers()
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, message: str, status: int = 400) -> None:
        self._send_json({"error": message, "success": False}, status=status)

    def _get_cookie(self, name: str) -> Optional[str]:
        cookie_header = self.headers.get("Cookie")
        if not cookie_header:
            return None
        c = http.cookies.SimpleCookie()
        try:
            c.load(cookie_header)
            return c[name].value if name in c else None
        except Exception:
            return None

    def _get_client_ip(self) -> str:
        forwarded = self.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return self.client_address[0] if self.client_address else "127.0.0.1"

    def _session_cookie(self, token: str, max_age: int = 604800) -> str:
        secure = "; Secure" if os.environ.get("BELIEVFIT_COOKIE_SECURE", "false").lower() in ("1", "true", "yes") else ""
        return f"gym_session={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age={max_age}{secure}"

    def _origin_is_same_site(self) -> bool:
        """Reject browser mutations explicitly marked as cross-site."""
        fetch_site = self.headers.get("Sec-Fetch-Site", "").lower()
        return fetch_site not in ("cross-site",)

    def _get_current_user(self) -> Optional[dict[str, Any]]:
        token = self._get_cookie("gym_session")
        if not token:
            auth_header = self.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:].strip()
        if not token:
            return None
        return db.get_user_by_session(token)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            return {}
        if content_length > 10 * 1024 * 1024:  # 10MB JSON limit
            raise ValueError("Payload too large")
        raw_body = self.rfile.read(content_length)
        return json.loads(raw_body.decode("utf-8"))

    # -----------------------------------------------------------------------
    # GET Handlers
    # -----------------------------------------------------------------------
    def do_GET(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        # Health ping
        if path == "/api/ping":
            self._send_json({"status": "ok", "app": "BelievFit", "version": "2.0"})
            return

        # 1. Main web application
        if path in ("/", "/index.html"):
            self._serve_file(os.path.join(WEB_DIR, "index.html"), "text/html; charset=utf-8")
            return

        # 2. Server LAN info for mobile pairing
        if path == "/api/network/info":
            local_ip = get_local_ip()
            self._send_json({
                "local_ip": local_ip,
                "port": PORT,
                "mobile_url": f"http://{local_ip}:{PORT}",
                "desktop_url": f"http://localhost:{PORT}",
            })
            return

        # 3. Stream or access Laptop Output files (videos, dashboards, reports) with safe path check
        if path.startswith("/output/") or path.startswith("/videos/"):
            user = self._get_current_user()
            if not user:
                self._send_error("Authentication required", status=HTTPStatus.UNAUTHORIZED)
                return
            is_output = path.startswith("/output/")
            rel_path = urllib.parse.unquote(path[8:])
            if not _is_user_owned_filename(rel_path, user["id"]):
                self._send_error("File not found or access denied", status=404)
                return
            base_dir = OUTPUT_DIR if is_output else VIDEOS_DIR
            safe_target = _resolve_safe_path(base_dir, rel_path)
            if safe_target:
                mime, _ = mimetypes.guess_type(safe_target)
                self._serve_file_with_range(safe_target, mime or "application/octet-stream")
                return
            self._send_error("File not found or access denied", status=404)
            return

        # 5. Remote Laptop File Explorer API
        if path == "/api/laptop/files":
            user = self._get_current_user()
            if not user:
                self._send_error("Please log in to browse laptop files", status=HTTPStatus.UNAUTHORIZED)
                return

            videos_list = []
            for f in sorted(os.listdir(VIDEOS_DIR)):
                full = os.path.join(VIDEOS_DIR, f)
                if os.path.isfile(full) and _is_user_owned_filename(f, user["id"]) and f.lower().endswith(tuple(ALLOWED_VIDEO_EXTENSIONS)) and not f.startswith(("_", ".")):
                    stat = os.stat(full)
                    videos_list.append({
                        "name": f,
                        "size": _format_size(stat.st_size),
                        "bytes": stat.st_size,
                        "url": f"/videos/{urllib.parse.quote(f)}",
                        "mtime": stat.st_mtime,
                    })

            outputs_list = []
            for f in sorted(os.listdir(OUTPUT_DIR), reverse=True):
                full = os.path.join(OUTPUT_DIR, f)
                if os.path.isfile(full) and _is_user_owned_filename(f, user["id"]) and not f.startswith(("_", ".")):
                    stat = os.stat(full)
                    file_type = "video" if f.lower().endswith((".mp4", ".mov")) else ("image" if f.lower().endswith((".png", ".jpg")) else "json")
                    outputs_list.append({
                        "name": f,
                        "type": file_type,
                        "size": _format_size(stat.st_size),
                        "bytes": stat.st_size,
                        "url": f"/output/{urllib.parse.quote(f)}",
                        "mtime": stat.st_mtime,
                    })

            self._send_json({
                "laptop_videos": videos_list,
                "laptop_outputs": outputs_list,
                "laptop_ip": get_local_ip(),
            })
            return

        # 6. Current user info
        if path == "/api/auth/me":
            user = self._get_current_user()
            if not user:
                self._send_error("Not authenticated", status=HTTPStatus.UNAUTHORIZED)
            else:
                self._send_json(user)
            return

        # 7. User lifetime stats
        if path == "/api/user/stats":
            user = self._get_current_user()
            if not user:
                self._send_json({
                    "total_workouts": 0, "total_reps": 0, "good_reps": 0,
                    "accuracy_pct": 0, "avg_score": 0, "best_score": 0,
                })
            else:
                self._send_json(db.get_user_stats(user["id"]))
            return

        # 8. User workouts history
        if path == "/api/workouts":
            user = self._get_current_user()
            if not user:
                self._send_json([])
            else:
                self._send_json(db.get_user_workouts(user["id"]))
            return

        # 9. Specific workout details
        if path.startswith("/api/workouts/"):
            user = self._get_current_user()
            if not user:
                self._send_error("Not authenticated", status=HTTPStatus.UNAUTHORIZED)
                return
            try:
                workout_id = int(path.split("/")[-1])
                details = db.get_workout_details(workout_id, user["id"])
                if not details:
                    self._send_error("Workout not found", status=404)
                else:
                    self._send_json(details)
            except ValueError:
                self._send_error("Invalid workout ID", status=400)
            return

        # 10. Background Analysis Job Polling
        if path.startswith("/api/jobs/"):
            user = self._get_current_user()
            if not user:
                self._send_error("Authentication required", status=401)
                return
            job_id = path.replace("/api/jobs/", "").strip()
            job = get_job(job_id)
            if not job or job.get("user_id") != user["id"]:
                self._send_error("Job not found", status=404)
            else:
                public_job = {k: v for k, v in job.items() if k != "user_id"}
                self._send_json(public_job)
            return

        self._send_error("Endpoint not found", status=404)

    # -----------------------------------------------------------------------
    # POST Handlers
    # -----------------------------------------------------------------------
    def do_POST(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        client_ip = self._get_client_ip()
        if not self._origin_is_same_site():
            self._send_error("Cross-origin request rejected.", status=403)
            return

        # 1. Register with rate limiting
        if path == "/api/auth/register":
            if not check_rate_limit(client_ip):
                self._send_error("Too many registration attempts. Please wait a minute.", status=429)
                return
            try:
                body = self._read_json_body()
                username = str(body.get("username", "")).strip()
                email = str(body.get("email", "")).strip()
                password = str(body.get("password", ""))

                user = db.create_user(username, email, password)
                token = db.create_session(user["id"])
                cookie = self._session_cookie(token)
                self._send_json({"user": user}, cookie=cookie)
                notify_device_login(user["username"], client_ip, action="register")
            except ValueError as e:
                self._send_error(str(e), status=400)
            except Exception as e:
                self._send_error("Registration failed. Please verify your details.", status=400)
            return

        # 2. Login with rate limiting
        if path == "/api/auth/login":
            if not check_rate_limit(client_ip):
                self._send_error("Too many login attempts. Please wait a minute.", status=429)
                return
            try:
                body = self._read_json_body()
                identifier = str(body.get("username_or_email", "")).strip()
                password = str(body.get("password", ""))

                user = db.authenticate_user(identifier, password)
                if not user:
                    self._send_error("Invalid username or password", status=401)
                    return
                token = db.create_session(user["id"])
                cookie = self._session_cookie(token)
                self._send_json({"user": user}, cookie=cookie)
                notify_device_login(user["username"], client_ip, action="login")
            except Exception:
                self._send_error("Authentication error. Please try again.", status=400)
            return

        # 3. Logout
        if path == "/api/auth/logout":
            token = self._get_cookie("gym_session")
            if token:
                db.delete_session(token)
            cookie = self._session_cookie("", max_age=0) + "; Expires=Thu, 01 Jan 1970 00:00:00 GMT"
            self._send_json({"success": True}, cookie=cookie)
            return

        # 4. Instant AI Simulation
        if path == "/api/workouts/simulate":
            user = self._get_current_user()
            if not user:
                self._send_error("Please log in to record workout sessions.", status=401)
                return

            try:
                body = self._read_json_body()
                exercise = body.get("exercise", "squat")

                from demo import build_pushup_series, build_squat_series
                if exercise == "pushup":
                    series, reps = build_pushup_series(fps=30.0)
                    angle_name = "Elbow angle"
                else:
                    exercise = "squat"
                    series, reps = build_squat_series(fps=30.0)
                    angle_name = "Knee angle"

                scores = [r["score"] for r in reps]
                avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0
                good_reps = sum(r["valid"] for r in reps)
                total_reps = len(reps)
                duration_s = series[-1][0] if series else 18.0

                report = {
                    "exercise": exercise,
                    "angle_name": angle_name,
                    "video": f"simulated_{exercise}",
                    "fps": 30.0,
                    "duration_s": duration_s,
                    "people": [{
                        "person": 1,
                        "colour_bgr": [80, 200, 120],
                        "reps": reps,
                        "average_score": int(round(avg_score)),
                        "series": series,
                        "horizontal_share": 0.9 if exercise == "pushup" else 0.0,
                    }],
                    "cues": [{"time_s": r["time_s"], "text": r["cue"]} for r in reps],
                }

                # Simulations contain generated telemetry only; never attach another user's media.
                report["form_video_url"] = None
                report["video_url"] = None

                workout_id = db.save_workout(
                    user_id=user["id"],
                    exercise=exercise,
                    total_reps=total_reps,
                    good_reps=good_reps,
                    avg_score=avg_score,
                    duration_s=duration_s,
                    report_dict=report,
                    video_filename=None,
                )

                self._send_json({
                    "success": True,
                    "workout_id": workout_id,
                    "report": report,
                })
            except Exception as e:
                self._send_error(f"Simulation failed: {str(e)}", status=500)
            return

        # 5. Remote Laptop File Analysis
        if path == "/api/laptop/analyze":
            user = self._get_current_user()
            if not user:
                self._send_error("Please log in to run analysis.", status=401)
                return

            try:
                body = self._read_json_body()
                filename = os.path.basename(body.get("filename", ""))
                exercise = body.get("exercise", "squat")
                if exercise not in ("squat", "pushup"):
                    exercise = "squat"

                safe_video = _resolve_safe_path(VIDEOS_DIR, filename)
                if not safe_video or not _is_user_owned_filename(filename, user["id"]):
                    self._send_error("Video file not found or access denied.", status=404)
                    return

                job_id = start_analysis_job(safe_video, exercise, user["id"], filename)
                self._send_json({
                    "success": True,
                    "job_id": job_id,
                    "status": "processing",
                })
            except Exception as e:
                self._send_error(f"Analysis error: {str(e)}", status=500)
            return

        # 6. Video File Upload
        if path == "/api/workouts/upload":
            user = self._get_current_user()
            if not user:
                self._send_error("Please log in to upload workouts.", status=401)
                return

            try:
                content_type = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in content_type:
                    self._send_error("Expected multipart/form-data", status=400)
                    return

                boundary_match = re.search(r'boundary=([^;]+)', content_type)
                if not boundary_match:
                    self._send_error("Invalid multipart boundary", status=400)
                    return

                boundary = boundary_match.group(1).strip('"')
                content_length = int(self.headers.get("Content-Length", 0))

                saved_video_path, exercise, safe_name = _save_multipart_stream(
                    self.rfile, content_length, boundary, user["id"]
                )

                job_id = start_analysis_job(saved_video_path, exercise, user["id"], safe_name)
                self._send_json({
                    "success": True,
                    "job_id": job_id,
                    "status": "processing",
                })
            except ValueError as ve:
                self._send_error(str(ve), status=400)
            except Exception as e:
                self._send_error(f"Upload error: {str(e)}", status=500)
            return

        self._send_error("Endpoint not found", status=404)

    # -----------------------------------------------------------------------
    # Video Streaming & Range Request Handler
    # -----------------------------------------------------------------------
    def _serve_file_with_range(self, filepath: str, content_type: str) -> None:
        """Serve media file with HTTP 206 Partial Content support for mobile video playback."""
        if not os.path.exists(filepath):
            self._send_error("File not found", status=404)
            return

        file_size = os.path.getsize(filepath)
        range_header = self.headers.get("Range")

        if not range_header:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self._set_security_headers()
            self.end_headers()
            with open(filepath, "rb") as f:
                self.wfile.write(f.read())
            return

        m = re.match(r"bytes=(\d+)-(\d+)?", range_header)
        if not m:
            self.send_response(416)  # Range Not Satisfiable
            self.send_header("Content-Range", f"bytes */{file_size}")
            self._set_security_headers()
            self.end_headers()
            return

        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else file_size - 1

        if start >= file_size or end >= file_size or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{file_size}")
            self._set_security_headers()
            self.end_headers()
            return

        chunk_size = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(chunk_size))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self._set_security_headers()
        self.end_headers()

        with open(filepath, "rb") as f:
            f.seek(start)
            remaining = chunk_size
            while remaining > 0:
                read_amount = min(remaining, 65536)
                data = f.read(read_amount)
                if not data:
                    break
                try:
                    self.wfile.write(data)
                    remaining -= len(data)
                except (BrokenPipeError, ConnectionResetError):
                    break

    def _serve_file(self, filepath: str, content_type: str) -> None:
        if not os.path.exists(filepath):
            self._send_error("File not found", status=404)
            return

        with open(filepath, "rb") as f:
            content = f.read()

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self._set_security_headers()
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(f"[BelievFit] {self.address_string()} - {format % args}\n")


def run_server(port: int = PORT) -> None:
    local_ip = get_local_ip()
    server_address = ("0.0.0.0", port)
    httpd = ThreadingHTTPServer(server_address, GymCoachHandler)

    print("\n" + "=" * 68)
    print(f"  🏋️ BelievFit AI Gym Coach — Mobile & Desktop Server Running!")
    print(f"  💻 On this Laptop : http://localhost:{port}")
    print(f"  📱 On your Phone  : http://{local_ip}:{port}")
    print(f"  📂 Laptop Storage : {BASE_DIR}")
    print("=" * 68)
    print(f"  Connect your phone to the SAME Wi-Fi network and open the Phone URL.")
    print("=" * 68 + "\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nBelievFit server shutting down gracefully.")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
