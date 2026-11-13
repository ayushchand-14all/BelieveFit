"""
coach.py — CLI entry point for BelievFit AI Gym Coach.

Runs the complete local analysis pipeline on a single workout video:
  1. Analysis   — pose estimation, rep counting, form scoring  (pipeline.py)
  2. Voice      — neural TTS coaching track, mixed into audio  (voice_coach.py)
  3. Dashboard  — analytics dashboard rendered to MP4          (dashboard.py)

Usage:
    python coach.py videos/squat.mp4
    python coach.py videos/pushups.mp4 --exercise pushup
    python coach.py videos/squat.mp4   --people 1
    python coach.py videos/squat.mp4   --voice en-GB-RyanNeural
    python coach.py videos/squat.mp4   --no-voice
    python coach.py videos/squat.mp4   --no-music
    python coach.py videos/squat.mp4   --output-dir custom_output/

Exit codes:
    0 = Success
    1 = Invalid arguments
    2 = Video file not found
    3 = No valid reps detected
    4 = Analysis runtime error
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from pipeline import run_analysis
from voice_coach import DEFAULT_VOICE, generate_voice_track
from dashboard import render_dashboard


def main() -> int:
    ap = argparse.ArgumentParser(
        description="BelievFit AI Gym Coach — analyse workout video and produce coached biomechanical dashboard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python coach.py videos/squat.mp4
  python coach.py videos/pushups.mp4 --exercise pushup
  python coach.py videos/squat.mp4   --voice en-GB-RyanNeural
  python coach.py videos/squat.mp4   --no-voice
        """,
    )
    ap.add_argument("video", help="Path to the workout video (e.g. videos/squat.mp4)")
    ap.add_argument("--exercise", choices=["squat", "pushup"], default="squat",
                    help="Exercise type to evaluate (default: squat)")
    ap.add_argument("--people", type=int, default=2,
                    help="Max people to track simultaneously (default: 2)")
    ap.add_argument("--voice", default=DEFAULT_VOICE,
                    help=f"edge-tts neural voice name (default: {DEFAULT_VOICE})")
    ap.add_argument("--no-voice", action="store_true",
                    help="Skip voice generation (silent dashboard)")
    ap.add_argument("--no-music", action="store_true",
                    help="Drop the original video's audio from the voice mix")
    ap.add_argument("--output-dir", default=None,
                    help="Custom directory to store generated outputs (default: ./output)")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}", file=sys.stderr)
        return 2

    print(f"\n{'='*64}")
    print(f"  🏋️ BelievFit AI Gym Coach  |  {args.exercise.upper()}  |  up to {args.people} person(s)")
    print(f"{'='*64}\n")

    # ------------------------------------------------------------------ Step 1
    print("[ 1 / 3 ]  Analysing biomechanics …\n")
    try:
        json_path = run_analysis(args.video, args.exercise, args.people)
    except Exception as e:
        print(f"\nError during analysis: {e}", file=sys.stderr)
        return 4

    # check at least one rep was found before proceeding
    with open(json_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    if not any(p["reps"] for p in report.get("people", [])):
        print(
            f"\nNo {args.exercise} reps were detected.\n"
            "Tips:\n"
            "  • Make sure your whole body is visible in the frame.\n"
            "  • For squats: film from the SIDE or FRONT.\n"
            "  • For push-ups: film from the SIDE at floor level.\n"
            + ("  • Did you mean --exercise pushup?\n" if args.exercise == "squat" else ""),
            file=sys.stderr
        )
        return 3

    # ------------------------------------------------------------------ Step 2
    name = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    form_video = os.path.join(out_dir, f"{name}_form.mp4")

    if not args.no_voice:
        print("\n[ 2 / 3 ]  Synthesizing coaching voice track …\n")
        try:
            coached = generate_voice_track(
                report_path=json_path,
                form_video=form_video,
                source_video=args.video,
                voice=args.voice,
                keep_music=not args.no_music,
            )
            if coached is None:
                print("  (voice generation skipped — offline or edge-tts unavailable)")
        except Exception as e:
            print(f"  (voice generation encountered an error: {e})")
    else:
        print("\n[ 2 / 3 ]  Voice track skipped (--no-voice)\n")

    # ------------------------------------------------------------------ Step 3
    print("\n[ 3 / 3 ]  Rendering analytics dashboard …\n")
    try:
        dashboard = render_dashboard(args.video)
        dashboard_msg = f"Dashboard   →  {os.path.abspath(dashboard)}"
    except Exception as e:
        dashboard_msg = f"Dashboard   →  (skipped: {e})"

    print(f"\n{'='*64}")
    print("  Done!  Your BelievFit outputs:")
    print(f"    Form video  →  {os.path.abspath(form_video)}")
    if not args.no_voice:
        coached_vid = os.path.join(out_dir, f"{name}_coached.mp4")
        if os.path.exists(coached_vid):
            print(f"    With voice  →  {os.path.abspath(coached_vid)}")
    print(f"    {dashboard_msg}")
    print(f"{'='*64}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
