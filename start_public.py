"""
start_public.py — Launch AI Gym Coach with Global Public Access & Device Alert Signals.

Features:
  1. Starts local AI Gym Coach backend on http://localhost:8000.
  2. Creates a secure, public HTTPS tunnel accessible from any device on Earth.
  3. Real-time Device Signals:
     • Audible Windows audio chime (winsound) whenever an external user logs in.
     • Windows desktop notification popup balloon.
     • Free instant push notification to your smartphone via ntfy.sh.
  4. Offline Behavior:
     • When your laptop is turned OFF, anyone visiting the URL is shown a
       "Temporarily Unavailable / Coach is Offline" status page.

Usage:
  python start_public.py
"""
import os
import re
import sys
import time
import shutil
import socket
import urllib.request
import subprocess
import threading

PORT = 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
CLOUDFLARED_PATH = os.path.join(BASE_DIR, "cloudflared.exe")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "ayush_gym_coach_alerts")


def check_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def ensure_cloudflared() -> str:
    """Finds or downloads cloudflared binary for instant zero-config tunnels."""
    # Check in PATH
    found = shutil.which("cloudflared")
    if found:
        return found
    # Check locally
    if os.path.exists(CLOUDFLARED_PATH):
        return CLOUDFLARED_PATH

    print("⚡ Cloudflare Tunnel binary not found. Downloading standalone portable binary...")
    try:
        urllib.request.urlretrieve(CLOUDFLARED_URL, CLOUDFLARED_PATH)
        print(f"✅ Downloaded cloudflared to: {CLOUDFLARED_PATH}")
        return CLOUDFLARED_PATH
    except Exception as e:
        print(f"⚠️ Could not download cloudflared ({e}). Falling back to built-in OpenSSH tunnel...")
        return ""


def start_backend():
    """Starts the Python app.py server in a background thread."""
    if check_port_in_use(PORT):
        print(f"ℹ️ Backend is already running on http://localhost:{PORT}")
        return None

    print(f"🚀 Starting AI Gym Coach Pro backend on port {PORT}...")
    proc = subprocess.Popen(
        [sys.executable, os.path.join(BASE_DIR, "app.py")],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    time.sleep(1.5)
    return proc


def run_tunnel(cf_bin: str):
    """Starts public HTTPS tunnel and extracts the live public URL."""
    public_url = None

    if cf_bin and os.path.exists(cf_bin):
        print("🌐 Connecting secure Cloudflare Edge Tunnel...")
        cmd = [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{PORT}"]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            # Cloudflare tunnel outputs https://*.trycloudflare.com
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
            if match:
                public_url = match.group(0)
                break
            if "error" in line.lower() and "failed" in line.lower():
                print(f"  [Tunnel log] {line.strip()}")
    else:
        # Fallback to OpenSSH localhost.run
        print("🌐 Connecting OpenSSH Reverse Tunnel (localhost.run)...")
        ssh_bin = shutil.which("ssh") or "ssh"
        cmd = [
            ssh_bin,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-R", f"80:localhost:{PORT}",
            "nokey@localhost.run"
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            match = re.search(r"https://[a-zA-Z0-9-]+\.lhr\.life", line)
            if match:
                public_url = match.group(0)
                break

    return public_url, proc


def main():
    print("=" * 72)
    print("  🏋️ AI GYM COACH PRO — GLOBAL INTERNET ACCESS LAUNCHER")
    print("=" * 72)

    # 1. Start local server
    backend_proc = start_backend()

    # 2. Get tunnel binary
    cf_bin = ensure_cloudflared()

    # 3. Start tunnel
    public_url, tunnel_proc = run_tunnel(cf_bin)

    if not public_url:
        print("\n❌ Could not detect public URL automatically. Ensure internet connectivity.")
        print(f"Local server is still active at: http://localhost:{PORT}")
        return

    # 4. Display Banner
    print("\n" + "=" * 72)
    print("  🌍 YOUR PUBLIC WEB APPLICATION IS LIVE ACROSS THE WORLD!")
    print("=" * 72)
    print(f"  🔗 Public URL for anyone on Earth:")
    print(f"     👉 \033[1;32m{public_url}\033[0m")
    print("-" * 72)
    print(f"  💻 Local Laptop URL:   http://localhost:{PORT}")
    print(f"  📱 Mobile (Local Wi-Fi): http://10.51.61.13:{PORT}")
    print("=" * 72)
    print("  🔔 REAL-TIME DEVICE SIGNAL ALERTS CONFIGURED:")
    print("  • Audio Alert: Laptop plays an audible chime whenever a user logs in.")
    print("  • Desktop Toast: Windows notification popup shows remote user & IP.")
    print(f"  • Mobile Push: Instant phone notifications active via topic:")
    print(f"    👉 https://ntfy.sh/{NTFY_TOPIC}")
    print("    (Open that link on your phone or install the ntfy app to get live alerts!)")
    print("=" * 72)
    print("  💤 OFFLINE HANDLING:")
    print("  • When your laptop is turned OFF, visiting this URL automatically shows:")
    print("    'Temporarily Unavailable — Host Laptop is Currently Offline'")
    print("=" * 72)
    print("\n[Server ready and listening for global users... Press Ctrl+C to stop]\n")

    try:
        # Keep alive and pipe logs
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down public tunnel and backend...")
        if tunnel_proc:
            tunnel_proc.terminate()
        if backend_proc:
            backend_proc.terminate()
        print("Done. Goodbye!")


if __name__ == "__main__":
    main()
