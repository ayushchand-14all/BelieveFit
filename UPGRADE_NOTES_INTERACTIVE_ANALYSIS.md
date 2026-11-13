# BelieveFit interactive analysis upgrade

## Modified files
- `pipeline.py`: stores timestamped, normalized pose landmarks and per-region movement states in the workout report (`pose_frames`).
- `app.py`: includes a URL for the original uploaded video in the completed analysis report. The browser can use this when the OpenCV-generated MP4 is not browser-decodable.
- `web/index.html`: adds a synchronized Canvas skeleton overlay, green/red/neutral segment and joint rendering, a moving vertical chart cursor, smooth playback-clock updates, and graph-click-to-video seeking.

## Compatibility notes
- Existing report fields and workout records are retained.
- Old workout reports without `pose_frames` still load; they simply do not show the new browser-side skeleton overlay.
- The dashboard prefers the original uploaded video and falls back to the annotated output if the original cannot be decoded.
- The new overlay uses pose samples generated during analysis; it does not rerun MediaPipe in the browser.

## Local installation
1. Back up the existing `D:\BelieveFit` folder and database.
2. Extract this ZIP to a temporary folder.
3. Copy the modified `app.py`, `pipeline.py`, and `web\index.html` into the existing `D:\BelieveFit` folder, replacing those three files only.
4. Keep your existing `gym_coach.db`, `videos`, `output`, `models`, and `.venv`.
5. Restart the server with `python app.py`.
6. Upload and analyse a new video. Older reports do not contain timestamped pose samples.

## Validation performed in the build environment
- Python compilation: `app.py` and `pipeline.py` passed.
- JavaScript syntax: inline script passed `node --check`.
- Existing pytest suite: 18 passed, 4 skipped, 1 failed. The failure is `TestAppServer.test_simulate_workout_flow` (expected HTTP 200, received 500) in the existing simulation endpoint. This was not changed by the video/overlay work and remains unresolved.
- Real browser playback and MediaPipe analysis on the user's Windows device still need to be verified locally.
