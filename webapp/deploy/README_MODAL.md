# Running the site on Modal

Serverless: the web app runs on a small CPU container that scales to zero,
and every uploaded scan runs on an A10G that exists only for those ~17
minutes. There is no instance, disk, DNS or certificate to manage; Modal
gives an HTTPS URL. On the free plan $30 of compute a month is included,
which is roughly 90 scans.

```
 browser ──HTTPS──▶ https://<workspace>--airway.modal.run
                    web()  FastAPI, api/app.py, basic auth
                      │  Volume "airway-jobs" mounted at /jobs
                      └──▶ run_job(job_id).spawn()  A10G, the worker image
```

`modal_app.py` is the whole definition. `api/cloud.py` is the only code that
knows which backend it is on (`AIRWAY_BACKEND=modal`): it commits the Volume
after the upload is written, reloads it before every read, and spawns the
GPU function instead of waking an EC2 instance.

## Costs (list prices, September 2026)

| item | price | note |
|---|---|---|
| A10G while a job runs | ≈ $1.10 / h | 17 min per scan ≈ $0.31 |
| CPU container while pages are open | ≈ $0.13 / h | scales to zero after 5 idle minutes |
| Volume | free tier covers it | `--retain-days` equivalent: delete old job dirs by hand or add a cron |

## One-time setup

1. **Modal account.** modal.com → Continue with GitHub. Workspace name goes
   into the URL. Settings → API Tokens → New Token, then on izmir
   `~/.modal.toml` gets `token_id` / `token_secret` (or `modal token set`).
2. **The worker image in a registry.** Modal cannot build our multi-stage
   Dockerfile, so it pulls the image izmir already built and verified:
   ```bash
   gh auth refresh -h github.com -s write:packages,read:packages     # once
   # gh >= 2.14: TOKEN=$(gh auth token); older gh: read oauth_token from ~/.config/gh/hosts.yml
   echo "$TOKEN" | docker login ghcr.io -u HUmutI --password-stdin
   docker tag airway-worker ghcr.io/humuti/airway-worker:v1
   docker push ghcr.io/humuti/airway-worker:v1                         # 20 GB
   ```
   The package is private by default; keep it so (it contains third-party
   model weights, `models/README.md`).
3. **Secrets** (Modal dashboard → Secrets, or CLI):
   ```bash
   modal secret create ghcr REGISTRY_USERNAME=HUmutI REGISTRY_PASSWORD=<GitHub token with read:packages>
   modal secret create airway-auth AIRWAY_BASIC_AUTH_USER=quinlab AIRWAY_BASIC_AUTH_PASSWORD='<lab password>'
   ```
4. **Deploy:**
   ```bash
   cd webapp && ../webapp/.venv/bin/modal deploy modal_app.py
   ```
   Prints the URL. First deploy pulls the 20 GB image into Modal's cache
   (several minutes); after that a GPU container starts in ~1 min.

## Updating

- Lab password: `modal secret create airway-auth AIRWAY_BASIC_AUTH_USER=quinlab AIRWAY_BASIC_AUTH_PASSWORD='<new>' --force`,
  then `modal app stop airway -y && modal deploy modal_app.py`. A warm web
  container keeps the old secret until it is restarted.

- HTML/JS/CSS/API change: bump `?v=N`, then `modal app stop airway -y && modal deploy modal_app.py`. A plain deploy leaves a warm web container serving the old files.
- Worker/stage change: rebuild on izmir, push a new tag
  (`ghcr.io/humuti/airway-worker:v2`), set `AIRWAY_WORKER_IMAGE` accordingly
  and deploy.
- Logs: `modal app logs airway`, or the dashboard.

## Deployed 2026-09-23

https://humuti--airway.modal.run, workspace `humuti`, login `quinlab`. First
deploy: Modal's import of the 20 GB image timed out once ("could not read
bytes for base index ... timeout"); the second `modal deploy` went through in
seconds because the import had in fact completed. If it recurs, splitting the
19 GB envs layer in `Dockerfile.worker` into one COPY per env is the fix.
Cold start of the GPU container is about 2.5 minutes (image load), then the
pipeline runs at izmir speed.

## Behaviour to know

- Uploads land in the Volume; `run_job` commits every 5 s so the job page
  moves. Reload/commit are cheap.
- Two uploads at once run on two GPUs in parallel; there is no queue.
- A `spawn` whose container failed (e.g. a broken image) is retried by Modal
  once a redeploy fixes the function. On 2026-09-23 that made the first job
  run twice, the two runs overwriting each other's `status.json`. If you
  redeploy after a failed spawn, do not spawn the same job by hand as well.
- The worker image's `ENTRYPOINT` is `worker.py`; `modal_app.py` clears it
  (`.entrypoint([])`), otherwise Modal's runner is passed to `worker.py` as
  arguments and the container exits with "unrecognized arguments".
- A job whose container dies is left `running`; the API's Retry button
  requeues it (the stale-job sweep in `worker.py` is not used here because
  there is no long-lived worker). `AIRWAY_STAGE_TIMEOUT_MIN` still applies.
- No PHI: same rule as everywhere. The Volume holds de-identified uploads
  and results; delete job directories when they are no longer needed
  (`modal volume rm airway-jobs /<job_id> -r`).
