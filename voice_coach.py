"""
voice_coach.py — Generates a natural coaching voice track.

Reads the rep JSON produced by the analysis pipeline and builds a timeline
of spoken lines: mid-rep corrections, rep counts, tip comments, and a
set-summary at the end.  Lines are synthesised with Microsoft edge-tts (free,
neural voices, requires internet) and mixed into an M4A audio file by ffmpeg.

The coach speaks like a trainer, not a text-to-speech robot:
  - Corrections shout the moment a fault appears ("Chest up!")
  - Rep counts land on the lockout, with contextual praise
  - "No rep" is brief when the reason was already shouted
  - The summary names the one thing to work on next set
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from collections import Counter
from typing import Optional

import edge_tts
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

DEFAULT_VOICE = "en-US-AndrewNeural"   # warm, energetic male coach
# en-US-AriaNeural (female), en-GB-RyanNeural (British), en-US-GuyNeural
# en-AU-WilliamNeural, en-US-BrianNeural.  See all: edge-tts --list-voices

GAP_S         = 0.12   # silence between two lines
COUNT_LEAD_S  = 0.10   # counts start this early so they land on the lockout
MUSIC_VOLUME  = 0.35   # original video audio level (ducked under voice)

LATE_OK = {1: 0.25, 2: 0.35, 3: 0.70, 4: 99.0}   # seconds a line may arrive late

STYLES = {
    "shout": {"rate": "+14%", "pitch": "+6Hz",  "volume": "+10%"},
    "count": {"rate": "+6%",  "pitch": "+3Hz",  "volume": "+5%"},
    "say":   {"rate": "+0%",  "pitch": "+0Hz",  "volume": "+0%"},
}

NUMBERS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
]

# ---------------------------------------------------------------------------
# Squat phrase banks
# ---------------------------------------------------------------------------
SHOUT = {
    "lean":  ["Chest up!", "Chest up, chest up!", "Stay tall!"],
    "heels": ["Heels down!", "Stay on your heels!", "Heels planted!"],
    "knees": ["Knees out!", "Push the knees out!", "Knees out, knees out!"],
}
FIX = {
    "depth": ["Go lower.", "Hips below knees.", "Sit deeper."],
    "lean":  ["Chest up.", "Don't fold forward."],
    "heels": ["Heels down.", "Stay on your heels."],
    "knees": ["Knees out.", "Knees over toes."],
    "tempo": ["Too rushed.", "Slow it down."],
    None:    ["Not quite there."],
}
TIP = {
    "depth": ["try to sit a little deeper.", "a bit lower next one."],
    "lean":  ["keep that chest proud.", "stay a little taller."],
    "heels": ["weight in your heels.", "keep those heels planted."],
    "knees": ["keep your knees over your toes.", "drive the knees out."],
    "tempo": ["slow it down on the way down.", "control the descent."],
}
FOCUS_SQUAT = {
    "depth": "getting a little deeper",
    "lean":  "keeping your chest up",
    "heels": "keeping your heels down",
    "knees": "pushing your knees out",
    "tempo": "slowing down on the way down",
}

# ---------------------------------------------------------------------------
# Push-up phrase banks
# ---------------------------------------------------------------------------
PUSHUP_SHOUT = {
    "sag":  ["Hips up!", "Squeeze your glutes!", "Hips up, hips up!"],
    "pike": ["Hips down!", "Straight line!", "Flatten out!"],
    "head": ["Head up!", "Chest first!", "Head in line!"],
}
PUSHUP_FIX = {
    "depth":   ["Chest lower.", "All the way down.", "Go lower."],
    "sag":     ["Hips up.", "Squeeze your glutes."],
    "pike":    ["Hips down.", "Straight line."],
    "head":    ["Chest first, not your head."],
    "tempo":   ["Too rushed.", "Slow it down."],
    "lockout": ["Lock out at the top."],
    None:      ["Not quite there."],
}
PUSHUP_TIP = {
    "depth":   ["a little lower next one.", "get that chest closer to the floor."],
    "sag":     ["keep those hips up.", "squeeze your glutes."],
    "pike":    ["hips down a touch.", "keep a straight line."],
    "head":    ["keep your head in line.", "lead with the chest."],
    "tempo":   ["slow it down on the way down.", "control the descent."],
    "lockout": ["finish with straight arms.", "lock out at the top."],
}
FOCUS_PUSHUP = {
    "depth":   "getting your chest lower",
    "sag":     "keeping your hips up",
    "pike":    "keeping a straight body line",
    "head":    "keeping your head in line",
    "tempo":   "slowing down on the way down",
    "lockout": "locking out at the top",
}

PRAISE   = ["Nice.", "Good.", "That's it.", "Strong.", "Beautiful.", "Yes!"]
NO_REP   = ["That one doesn't count.", "No rep.", "Not that one."]
RULE_OF  = {"sag": "body", "pike": "body"}


def _phrases(report: dict) -> dict:
    if report.get("exercise") == "pushup":
        return {"shout": PUSHUP_SHOUT, "fix": PUSHUP_FIX, "tip": PUSHUP_TIP,
                "focus": FOCUS_PUSHUP, "unit": "push-ups"}
    return {"shout": SHOUT, "fix": FIX, "tip": TIP,
            "focus": FOCUS_SQUAT, "unit": "reps"}


def _pick(options: list, n: int) -> str:
    return options[n % len(options)]


def _say_number(n: int) -> str:
    return NUMBERS[n] if n < len(NUMBERS) else str(n)


def _shout_for(rep: dict, ph: dict) -> Optional[dict]:
    fault = rep["main_fault"]
    if fault not in ph["shout"]:
        return None
    lost = rep["points_lost"].get(RULE_OF.get(fault, fault), 0)
    if rep["valid"] and lost < 5:
        return None
    return next((lf for lf in rep.get("live_faults", []) if lf["fault"] == fault), None)


# ---------------------------------------------------------------------------
# Script builder
# ---------------------------------------------------------------------------
def _lines_for_person(person: dict, many: bool, counter: Counter, ph: dict) -> list[dict]:
    who  = f"Person {person['person']}, " if many else ""
    lines: list[dict] = []
    good = streak = 0
    deepest: Optional[float] = None
    first_descent: Optional[float] = None
    last_tip: dict = {}

    for rep in person["reps"]:
        m     = rep["measured"]
        fault = rep["main_fault"]
        if first_descent is None:
            first_descent = m.get("tempo", 1.0)

        # 1) Mid-rep correction shout
        lf = _shout_for(rep, ph)
        if lf:
            lines.append({
                "time_s": lf["time_s"], "priority": 3, "style": "shout",
                "text": who + _pick(ph["shout"][fault], counter["shout"]),
            })
            counter["shout"] += 1

        # 2) No-rep call
        if not rep["valid"]:
            streak = 0
            text   = "No rep." if lf else f"No rep. {_pick(ph['fix'][fault], counter['fix'])}"
            if not lf:
                counter["fix"] += 1
            lines.append({
                "time_s": rep["time_s"], "priority": 3, "style": "say",
                "text": who + text, "short": who + "No rep.",
            })
            continue

        # 3) Good rep: count + optional comment
        good  += 1
        streak = streak + 1 if rep["score"] >= 85 else 0
        count  = _say_number(good)
        count  = count if who else count.capitalize()
        depth  = m.get("depth", 90.0)
        short  = f"{who}{count}."

        if lf:
            text, priority = short, 2
        elif fault == "tempo" and good > 2 and m.get("tempo", 1) < (first_descent or 1) * 0.7:
            text, priority = f"{who}{count}. Slow it down.", 3
        elif fault in ph["tip"] and good - last_tip.get(fault, -9) >= 3:
            last_tip[fault] = good
            text, priority = f"{who}{count}, {_pick(ph['tip'][fault], counter['tip'])}", 2
            counter["tip"] += 1
        elif deepest is not None and depth < deepest - 5:
            text, priority = f"{who}{count}. Deepest one yet!", 2
        elif streak == 3:
            text, priority = f"{who}{count}. Three clean in a row!", 2
        elif good % 2 == 0:
            text, priority = short, 2
        else:
            text, priority = f"{who}{count}. {_pick(PRAISE, counter['praise'])}", 2
            counter["praise"] += 1

        if many and priority == 2:
            priority = 1
        deepest = depth if deepest is None else min(deepest, depth)
        lines.append({
            "time_s": max(0.0, rep["time_s"] - COUNT_LEAD_S),
            "priority": priority, "style": "count",
            "text": text, "short": short,
        })
    return lines


def _summary_line(report: dict) -> str:
    ph      = _phrases(report)
    people  = report["people"]
    faults  = Counter(r["main_fault"] for p in people for r in p["reps"] if r["main_fault"])
    if len(people) == 1:
        reps = people[0]["reps"]
        good = sum(r["valid"] for r in reps)
        text = f"Set done. {_say_number(good).capitalize()} clean out of {_say_number(len(reps))}."
    else:
        parts = [
            f"Person {p['person']}, {_say_number(sum(r['valid'] for r in p['reps']))} "
            f"out of {_say_number(len(p['reps']))}."
            for p in people
        ]
        text = "Set done. " + " ".join(parts)
    avg = [p["average_score"] for p in people if p.get("average_score") is not None]
    if faults:
        text += f" Next set, focus on {ph['focus'][faults.most_common(1)[0][0]]}."
    elif avg and min(avg) >= 90:
        text += " That was a great set."
    return text


def write_lines(report: dict) -> list[dict]:
    people = [p for p in report["people"] if p["reps"]]
    if not people:
        return []
    many    = len(people) > 1
    counter: Counter = Counter()
    ph      = _phrases(report)
    lines: list[dict] = []
    for p in people:
        lines += _lines_for_person(p, many, counter, ph)

    first = min(r["start_s"] for p in people for r in p["reps"])
    if first >= 1.2:
        lines.append({
            "time_s": max(0.0, first - 1.2),
            "priority": 1, "style": "shout", "text": "Let's go!",
        })
    last = max(r["time_s"] for p in people for r in p["reps"])
    lines.append({
        "time_s": last + 0.6,
        "priority": 4, "style": "say",
        "text": _summary_line(report),
    })
    return sorted(lines, key=lambda l: l["time_s"])


# ---------------------------------------------------------------------------
# TTS synthesis
# ---------------------------------------------------------------------------
def _media_info(path: str) -> dict:
    out = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", path],
        capture_output=True, text=True,
    ).stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", out)
    duration = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 0.0
    return {"duration": duration, "has_audio": "Audio:" in out}


async def _tts_clip(text: str, voice: str, style: str, folder: str, cache: dict) -> tuple:
    key = (text, style)
    if key in cache:
        return cache[key]
    i   = len(cache)
    mp3 = os.path.join(folder, f"raw_{i:03d}.mp3")
    wav = os.path.join(folder, f"line_{i:03d}.wav")
    await edge_tts.Communicate(text, voice, **STYLES[style]).save(mp3)
    trim = (
        "silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.02,"
        "areverse,silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.06,"
        "areverse"
    )
    subprocess.run(
        [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
         "-i", mp3, "-af", trim, "-ar", "48000", "-ac", "1", wav],
        check=True,
    )
    cache[key] = (wav, _media_info(wav)["duration"])
    return cache[key]


async def _synthesise(lines: list[dict], voice: str, folder: str) -> None:
    os.makedirs(folder, exist_ok=True)
    cache: dict = {}
    for line in lines:
        line["audio"], line["duration_s"] = await _tts_clip(
            line["text"], voice, line["style"], folder, cache
        )
        if line.get("short") and line["short"] != line["text"]:
            line["short_audio"], line["short_duration_s"] = await _tts_clip(
                line["short"], voice, line["style"], folder, cache
            )
        print(f"  [{line['time_s']:6.2f}s] {line['text']}")


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------
def _use_short(line: dict) -> None:
    line["text"]       = line["short"]
    line["audio"]      = line["short_audio"]
    line["duration_s"] = line["short_duration_s"]


def schedule_lines(lines: list[dict], video_duration: float) -> list[dict]:
    lines   = sorted(lines, key=lambda l: l["time_s"])
    kept: list[dict] = []
    free_at = 0.0
    for i, line in enumerate(lines):
        start = max(line["time_s"], free_at)
        if start - line["time_s"] > LATE_OK[line["priority"]]:
            print(f"  dropped (would be {start - line['time_s']:.1f}s late): {line['text']}")
            continue
        if start >= video_duration and line["priority"] < 4:
            continue
        later = [o for o in lines[i + 1:] if o["priority"] >= 2]
        nxt   = later[0] if later else None
        if "short_audio" in line and nxt:
            room = 0.5 if nxt["priority"] == 4 else (0.25 if line["priority"] >= 3 else 0.0)
            if start + line["duration_s"] + GAP_S > nxt["time_s"] + room:
                _use_short(line)
        fixes = [o for o in lines[i + 1:] if o["priority"] == 3]
        if line["priority"] < 3 and fixes:
            if start + line["duration_s"] + GAP_S > fixes[0]["time_s"] + 0.1:
                if "short_audio" in line:
                    _use_short(line)
                if start + line["duration_s"] + GAP_S > fixes[0]["time_s"] + 0.1:
                    print(f"  dropped (correction due): {line['text']}")
                    continue
        line["start_s"] = start
        free_at         = start + line["duration_s"] + GAP_S
        kept.append(line)
    return kept


# ---------------------------------------------------------------------------
# Audio mixing
# ---------------------------------------------------------------------------
def mix_voice_track(
    lines: list[dict],
    source_video: str,
    out_path: str,
    keep_music: bool = True,
) -> None:
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error"]
    for line in lines:
        cmd += ["-i", line["audio"]]
    music = keep_music and _media_info(source_video)["has_audio"]
    if music:
        cmd += ["-i", source_video]
    parts: list[str] = []
    for i, line in enumerate(lines):
        ms = int(round(line["start_s"] * 1000))
        parts.append(f"[{i}:a]adelay={ms}:all=1[v{i}]")
    voices = "".join(f"[v{i}]" for i in range(len(lines)))
    parts.append(
        f"{voices}amix=inputs={len(lines)}:normalize=0:duration=longest,"
        "highpass=f=80,acompressor=threshold=-18dB:ratio=3:attack=5:release=150:makeup=2"
        "[vox]"
    )
    if music:
        parts.append("[vox]asplit=2[voxa][voxsc]")
        parts.append(
            f"[{len(lines)}:a]aresample=48000,aformat=channel_layouts=mono,"
            f"volume={MUSIC_VOLUME}[mus]"
        )
        parts.append(
            "[mus][voxsc]sidechaincompress=threshold=0.02:ratio=10:attack=10:release=400[duck]"
        )
        parts.append("[voxa][duck]amix=inputs=2:normalize=0:duration=longest[mix]")
        last = "[mix]"
    else:
        last = "[vox]"
    parts.append(
        f"{last}loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,"
        "aformat=channel_layouts=stereo[aout]"
    )
    cmd += ["-filter_complex", ";".join(parts), "-map", "[aout]",
            "-c:a", "aac", "-b:a", "192k", out_path]
    subprocess.run(cmd, check=True)


def mux_video_audio(video: str, audio: str, out_path: str) -> None:
    extra = max(0.0, _media_info(audio)["duration"] - _media_info(video)["duration"]) + 0.5
    cmd   = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-i", video, "-i", audio,
        "-filter_complex", f"[0:v]tpad=stop_mode=clone:stop_duration={extra:.2f}[v]",
        "-map", "[v]", "-map", "1:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
        "-c:a", "copy", "-shortest", "-movflags", "+faststart", out_path,
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def generate_voice_track(
    report_path: str,
    form_video:  str,
    source_video: str,
    voice:       str = DEFAULT_VOICE,
    keep_music:  bool = True,
) -> Optional[str]:
    """
    Generate a voice track and return the path to the coached video,
    or None if there are no reps to comment on.

    Requires internet access for edge-tts.
    """
    with open(report_path) as f:
        report = json.load(f)

    lines = write_lines(report)
    if not lines:
        print("No reps found — skipping voice generation.")
        return None

    name        = os.path.splitext(os.path.basename(source_video))[0]
    tts_folder  = f"output/{name}_voice"
    voice_path  = f"output/{name}_voice.m4a"
    out_path    = f"output/{name}_coached.mp4"

    print(f"Generating {len(lines)} voice lines with {voice} …")
    try:
        asyncio.run(_synthesise(lines, voice, tts_folder))
    except Exception as exc:
        print(f"Voice generation failed: {exc}\nCheck your internet connection.")
        return None

    video_dur = _media_info(form_video)["duration"]
    lines     = schedule_lines(lines, video_dur)
    print("\nFinal voice timing:")
    for li in lines:
        print(f"  {li['start_s']:6.2f}s  {li['text']}")

    print("Mixing audio …")
    mix_voice_track(lines, source_video, voice_path, keep_music=keep_music)
    mux_video_audio(form_video, voice_path, out_path)
    print(f"Saved coached video → {os.path.abspath(out_path)}")
    return out_path
