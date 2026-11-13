# GitHub upload instructions

This package is a clean source-only copy for GitHub. It excludes local databases, workout videos, generated output, MediaPipe model binaries, fonts caches, virtual environments, and Python cache files.

1. Extract this ZIP. Open the extracted `BelieveFit` folder.
2. In your private GitHub repository, choose **Add file → Upload files**.
3. Open the extracted `BelieveFit` folder and drag its CONTENTS (not the outer folder) into GitHub.
4. Confirm that `app.py`, `Dockerfile`, `render.yaml`, `vercel.json`, and `web/` appear at the repository root.
5. Commit the upload.

Do not add local `.db`, `.env`, `.venv`, `videos/`, `output/`, `models/`, or personal workout files. Render's persistent disk is intended to hold runtime data and the downloaded MediaPipe model.

Before deploying, replace the placeholder Render hostname in `vercel.json` after Render creates the backend service, then commit and push that change.
