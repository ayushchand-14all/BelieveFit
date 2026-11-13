# BelieveFit deployment: Vercel frontend + Render backend

## Hosting split

- **Vercel:** serves the existing static frontend from `web/`.
- **Render:** runs the Python/MediaPipe HTTP backend in Docker.
- **Render persistent disk:** stores SQLite, uploaded videos, generated outputs and the downloaded MediaPipe model under `/var/data`.

The frontend uses same-origin relative `/api`, `/videos` and `/output` URLs. `vercel.json` proxies those paths to Render so browser requests remain on the Vercel domain. After each successful analysis, the backend also creates an H.264/AAC MP4 browser copy of the uploaded source (when FFmpeg conversion succeeds). The original upload remains unchanged; the browser copy is served from the authenticated `/output/` route. This avoids blank HTML5 video playback for source formats/codecs that OpenCV can analyse but browsers cannot decode.

## Multi-user beta security work included

The application now has account registration and login backed by SQLite sessions. The public beta changes also:
- Require a valid session to stream `/videos/` and `/output/` files.
- Restrict media filenames to server-generated `user<ID>_...` names belonging to the signed-in account.
- Filter the remote file explorer to the signed-in user's files.
- Restrict `/api/laptop/analyze` to the signed-in user's uploaded videos.
- Attach an owner ID to each background analysis job and restrict job polling to that owner.
- Keep workout history and statistics scoped by user ID in SQLite.
- Set the session cookie `HttpOnly` and `SameSite=Strict`; Render sets `Secure` through `BELIEVFIT_COOKIE_SECURE=true`.
- Avoid returning the session token in the JSON response; the browser receives it only as an HttpOnly cookie.
- Remove the one-click demo login and the shared sample-video attachment from simulations.
- Store generated analysis outputs under `BELIEVFIT_DATA_DIR`, matching Render's persistent disk.

These controls are suitable for a **small, single-instance friends beta**, not a security certification or a mature public production launch. Remaining limitations:
1. Registration is open to anyone who has the URL; there is no email verification, invite approval, password reset, or account deletion UI yet.
2. Login rate limits and background jobs are in memory. They reset on restart and are not shared across multiple service instances.
3. The multipart upload parser still reads the spooled upload back into memory; large simultaneous uploads can exhaust RAM. Test with short clips first.
4. SQLite on Render's persistent disk is for a small single-instance beta only. Do not horizontally scale it; a larger release should use managed Postgres, object storage, and a durable job queue.
5. The custom standard-library HTTP server is retained to minimize disruption to the existing project. Before a broad public launch, move to a production-grade application server and arrange an independent security review.
6. Users should upload only workout videos they have permission to share. Add a privacy notice and retention/deletion policy before inviting people outside a trusted group.

Do not treat the app as fully production-ready. Use test accounts and non-sensitive videos while validating deployment.

## Files added/changed

- `Dockerfile`: Render container build.
- `render.yaml`: Render web service blueprint and persistent disk.
- `vercel.json`: static frontend output and reverse-proxy rewrites.
- `.dockerignore`: prevents local data/secrets from entering the Docker build.
- `app.py`: persistent data directory, account authentication, media/job ownership enforcement, secure session cookies.
- `db.py`: per-user accounts, sessions and workout records.
- `pipeline.py`: generated outputs stored on the configured persistent disk.
- `pose_utils.py`: configurable MediaPipe model path.
- `DEPLOYMENT_VERCEL_RENDER.md`: this guide.

The deployment ZIP intentionally excludes your local `gym_coach.db`, `videos/`, `output/`, `.venv/`, and `.pytest_cache/`. Your original ZIP/local project data are not modified.

## Deploy steps (after security fixes)

### 1. Put the project in GitHub
1. Extract this project.
2. Review `.gitignore`; ensure `.env`, database files, videos, outputs, and model binaries are not committed.
3. Create a private GitHub repository and push the project.
4. Do not upload the local database or user videos.

### 2. Create the Render backend
1. In Render, choose **New + → Blueprint** and connect the GitHub repository.
2. Render should detect `render.yaml`. Review the proposed `believefit-api` Docker web service.
3. The blueprint requests a Starter web service and a 10 GB persistent disk. A persistent disk is required because Render's ordinary filesystem is ephemeral. Confirm current pricing before creating it.
4. Deploy and wait for `/api/ping` health check to pass.
5. Copy the service URL, e.g. `https://believefit-api.onrender.com`.

### 3. Connect Vercel to Render
1. Open `vercel.json`.
2. Replace all three occurrences of `https://YOUR-RENDER-SERVICE.onrender.com` with the actual Render service URL.
3. Commit and push the change.
4. In Vercel, import the same GitHub repository.
5. Set the project Root Directory to the repository root (the directory containing `vercel.json`), not `web/`.
6. Deploy. `outputDirectory: web` tells Vercel to publish the frontend files.

### 4. Verify
- Open the Vercel URL and check that the UI loads.
- Open `https://YOUR-RENDER-SERVICE.onrender.com/api/ping` and check for JSON status `ok`.
- Test registration/login with a throwaway account (password must be at least 8 characters).
- Test upload, analysis, playback, report retrieval and logout.
- Register a second throwaway account and confirm it cannot access the first account's files/jobs/workout history.
- Test short videos first; the current upload handler has a 500 MB ceiling but large uploads are RAM-intensive.

## Local Windows usage

The `BELIEVFIT_DATA_DIR` setting defaults to the project folder, so local behaviour remains unchanged. You can continue to run `python app.py` from your Windows environment. The `BELIEVFIT_MODEL_PATH` override is optional.

## Operational limitations

- Render's persistent disk is single-instance storage and prevents scaling the service to multiple instances.
- Render free web services do not provide persistent disks; the provided blueprint deliberately uses a paid Starter service plus disk so uploads and SQLite survive restarts.
- Video analysis is CPU/RAM intensive. Choose a service size that can handle MediaPipe/OpenCV workloads and test with representative videos.
- A cold start or first analysis may download the MediaPipe model to `/var/data/models`.
- Vercel rewrites are documented at https://vercel.com/docs/project-configuration/vercel-json.
- Render persistent disk behaviour is documented at https://render.com/docs/disks.
