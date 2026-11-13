# 🏋️ BelievFit — AI Gym Coach & Biomechanics Hub

BelievFit is a local, privacy-preserving AI Gym Coach and biomechanical analysis platform. It leverages Google MediaPipe pose estimation, computer vision, and neural text-to-speech to deliver real-time feedback, rep counting, and comprehensive telemetry for squats and push-ups on desktop and mobile devices over local Wi-Fi.

---

## 🚀 Quick Start

### 1. Requirements
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.13)
- Windows / macOS / Linux

### 2. Installation
Navigate to your project root and install dependencies:
```bash
cd D:\BelieveFit
pip install -r requirements.txt
```

### 3. Launching the Local Server
```bash
python app.py
```
Access the application:
- **On this Laptop:** `http://localhost:8000`
- **On your Mobile Phone:** `http://<YOUR_LAN_IP>:8000` (shown on the terminal banner)

### 4. Running the CLI Coach
Analyze a video directly from command line:
```bash
python coach.py videos/squat.mp4 --exercise squat
python coach.py videos/pushup.mp4 --exercise pushup --people 1
```

### 5. Running Automated Tests
```bash
python -m unittest discover -s tests -p "test_*.py" -v
# or with pytest:
pytest -v
```

---

## 📂 Project Architecture

```
D:\BelieveFit\
├── app.py                # Mobile & desktop HTTP server, security layer, Range streaming
├── coach.py              # CLI entry point for full video analysis and report generation
├── pipeline.py           # Core MediaPipe pose pipeline, multi-person tracking & HUD renderer
├── form_scorer.py        # Exercise biomechanics, angle thresholds, and rep state machines
├── pose_utils.py         # Model loading, landmark smoothing, joint geometry & drawing
├── db.py                 # SQLite database layer, user authentication, session management
├── demo.py               # Pure-math simulation engine for instant preview without video
├── requirements.txt      # Dependency specification with pinned version bounds
├── pytest.ini            # Pytest configuration
├── models/               # MediaPipe neural model assets (pose_landmarker_full.task)
├── videos/               # Input workout video storage
├── output/               # Annotated videos, skeletons, JSON telemetry reports
├── web/                  # Mobile-first single-page application (index.html)
├── tests/                # Automated integration, unit, and security test suite
└── documentation/        # Full technical, security, architecture and setup guides
```

---

## 🔒 Security Hardening

- **Private Media:** File-serving routes require authentication and enforce per-user filename ownership plus path traversal protection.
- **Strict Rate Limiting:** In-memory IP tracking prevents brute-force login and registration attacks.
- **Account Sessions:** Individual accounts, HttpOnly/SameSite cookies, and user-scoped workout records/media/jobs. HTTPS deployments set the Secure cookie flag.
- **Content-Security-Policy (CSP):** Full security headers configured on all HTTP responses.
- **XSS Prevention:** Dynamic DOM output in `web/index.html` is escaped via HTML sanitization.
- **File Upload Protection:** Strict extension allowlist (`.mp4`, `.mov`, `.avi`, `.webm`, `.mkv`), 500 MB upload ceiling, and regex filename sanitization.
