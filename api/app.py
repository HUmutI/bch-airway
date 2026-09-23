"""The web face of the pipeline: upload a scan, watch it run, get the results.

Deliberately thin. It writes an upload and a queued status into a job
directory and reads status back; the worker does everything else. That is
what lets the same API front a local GPU today and a queue of cloud workers
later, by pointing JOBS_DIR somewhere else.

    ./.venv/bin/uvicorn api.app:app --host 0.0.0.0 --port 8000
"""
import json, os, re, secrets, sys
from datetime import datetime, timezone

from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
from api.cloud import wake_worker  # noqa: E402

WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
app = FastAPI(title="Airway segmentation and branch labelling", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=WEB), name="static")

# Anything the packaging and morphometry stages write, nothing else: plain
# names, one of a few extensions, no path separators.
RESULT_FILE = re.compile(r"^[a-z0-9_]+\.(nii\.gz|mz3|json|csv|txt|zip)$")
MEDIA = {".gz": "application/gzip", ".zip": "application/zip", ".json": "application/json",
         ".mz3": "application/octet-stream", ".csv": "text/csv", ".txt": "text/plain"}
JOB_ID = re.compile(r"^[a-z0-9]{16}$")


def job_dir(job_id):
    if not JOB_ID.match(job_id):
        raise HTTPException(404, "no such job")
    d = os.path.join(C.JOBS_DIR, job_id)
    if not os.path.isdir(d):
        raise HTTPException(404, "no such job")
    return d


def read_status(d):
    try:
        return json.load(open(os.path.join(d, "status.json")))
    except FileNotFoundError:
        raise HTTPException(404, "no such job")


@app.get("/", response_class=HTMLResponse)
def index():
    return open(os.path.join(WEB, "index.html")).read()


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(job_id: str):
    job_dir(job_id)
    return open(os.path.join(WEB, "job.html")).read()


@app.post("/api/jobs")
async def create_job(request: Request, file: UploadFile = File(...)):
    name = os.path.basename(file.filename or "")
    if not (name.endswith(".nii.gz") or name.endswith(".nii")):
        raise HTTPException(400, "upload a NIfTI file (.nii or .nii.gz)")
    job_id = secrets.token_hex(8)
    d = os.path.join(C.JOBS_DIR, job_id)
    os.makedirs(d, exist_ok=False)
    dst = os.path.join(d, "input.nii.gz" if name.endswith(".gz") else "input.nii")
    limit = C.MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    with open(dst, "wb") as fh:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > limit:
                fh.close(); os.remove(dst); os.rmdir(d)
                raise HTTPException(413, f"file exceeds {C.MAX_UPLOAD_MB} MB")
            fh.write(chunk)
    if dst.endswith(".nii"):
        # the worker expects input.nii.gz; gzip the plain file once
        import gzip, shutil
        with open(dst, "rb") as fi, gzip.open(dst + ".gz", "wb", compresslevel=1) as fo:
            shutil.copyfileobj(fi, fo)
        os.remove(dst)
    status = {"id": job_id, "state": "queued", "filename": name, "size_bytes": size,
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "client": request.client.host if request.client else None,
              "progress": 0.0, "message": "Waiting for the worker", "error": None}
    json.dump(status, open(os.path.join(d, "status.json"), "w"), indent=1)
    wake_worker()
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    s = read_status(job_dir(job_id))
    s.pop("client", None)
    return JSONResponse(s, headers={"Cache-Control": "no-store"})


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str):
    """Put a failed job back on the queue, from the upload, with a clean slate."""
    import shutil
    d = job_dir(job_id)
    s = read_status(d)
    if s.get("state") not in ("failed", "done"):
        raise HTTPException(409, "job is not finished")
    for sub in ("case", "logs", "results"):
        shutil.rmtree(os.path.join(d, sub), ignore_errors=True)
    for f in ("log.txt",):
        try: os.remove(os.path.join(d, f))
        except FileNotFoundError: pass
    for k in ("stages", "results", "results_ready", "morphometry_error", "started", "ended"):
        s.pop(k, None)
    s.update(state="queued", progress=0.0, error=None, message="Waiting for the worker",
             retried=s.get("retried", 0) + 1)
    json.dump(s, open(os.path.join(d, "status.json"), "w"), indent=1)
    wake_worker()
    return {"id": job_id, "state": "queued"}


@app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
def get_log(job_id: str, lines: int = 60):
    d = job_dir(job_id)
    out = []
    p = os.path.join(d, "log.txt")
    if os.path.exists(p):
        out.append(open(p).read())
    # the live tail of whichever stage logs are still being written
    logs = os.path.join(d, "logs")
    if os.path.isdir(logs):
        for f in sorted(os.listdir(logs)):
            t = open(os.path.join(logs, f)).read().strip().splitlines()
            if t:
                out.append(f"--- {f} ---\n" + "\n".join(t[-12:]))
    text = "\n".join(out)
    return "\n".join(text.splitlines()[-lines:])


@app.get("/api/jobs/{job_id}/files/{name}")
def get_file(job_id: str, name: str):
    if not RESULT_FILE.match(name):
        raise HTTPException(404, "no such file")
    p = os.path.join(job_dir(job_id), "results", name)
    if not os.path.exists(p):
        raise HTTPException(404, "not ready")
    media = MEDIA[os.path.splitext(name)[1]]
    return FileResponse(p, media_type=media, filename=name)


@app.get("/api/jobs")
def list_jobs(limit: int = 30):
    rows = []
    if os.path.isdir(C.JOBS_DIR):
        for d in os.listdir(C.JOBS_DIR):
            try:
                s = json.load(open(os.path.join(C.JOBS_DIR, d, "status.json")))
                rows.append({k: s.get(k) for k in ("id", "state", "filename", "created", "progress")})
            except Exception:
                pass
    rows.sort(key=lambda r: r.get("created") or "", reverse=True)
    return rows[:limit]
