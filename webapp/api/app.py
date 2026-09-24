"""The web face of the pipeline: upload a scan, watch it run, get the results.

Deliberately thin. It writes an upload and a queued status into a job
directory and reads status back; the worker does everything else. That is
what lets the same API front a local GPU today and a queue of cloud workers
later, by pointing JOBS_DIR somewhere else.

    ./.venv/bin/uvicorn api.app:app --host 0.0.0.0 --port 8000
"""
import base64, json, os, re, secrets, sys
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
from api.cloud import after_write, before_read, wake_worker  # noqa: E402

WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
app = FastAPI(title="Airway segmentation and branch labelling", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=WEB), name="static")


if C.BASIC_AUTH_USER and C.BASIC_AUTH_PASSWORD:
    _expected = base64.b64encode(f"{C.BASIC_AUTH_USER}:{C.BASIC_AUTH_PASSWORD}".encode()).decode()

    @app.middleware("http")
    async def basic_auth(request: Request, call_next):
        header = request.headers.get("authorization", "")
        if header.startswith("Basic ") and secrets.compare_digest(header[6:], _expected):
            return await call_next(request)
        return Response("Authentication required", status_code=401,
                        headers={"WWW-Authenticate": 'Basic realm="airway", charset="UTF-8"'})

# Anything the packaging and morphometry stages write, nothing else: plain
# names, one of a few extensions, no path separators.
RESULT_FILE = re.compile(r"^[a-z0-9_]+\.(nii\.gz|mz3|json|csv|txt|zip)$")
MEDIA = {".gz": "application/gzip", ".zip": "application/zip", ".json": "application/json",
         ".mz3": "application/octet-stream", ".csv": "text/csv", ".txt": "text/plain"}
JOB_ID = re.compile(r"^[a-z0-9]{16}$")


def job_dir(job_id):
    if not JOB_ID.match(job_id):
        raise HTTPException(404, "no such job")
    before_read()
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


QUEUED_MSG = ("Starting a GPU for this scan, usually one to three minutes"
              if C.BACKEND == "modal" else "Waiting for the worker")


def parse_age(age):
    """Optional age in years, 0 to 18, or None. Anything else is a 400."""
    if age is None or str(age).strip() == "":
        return None
    try:
        v = float(str(age).replace(",", "."))
    except ValueError:
        raise HTTPException(400, "age must be a number of years")
    if not (0 <= v <= 18):
        raise HTTPException(400, "age must be between 0 and 18 years")
    return round(v, 2)


@app.post("/api/jobs")
async def create_job(request: Request, file: UploadFile = File(...), age_years: str = Form(None)):
    age = parse_age(age_years)
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
              "age_years": age,
              "progress": 0.0, "message": QUEUED_MSG, "error": None}
    json.dump(status, open(os.path.join(d, "status.json"), "w"), indent=1)
    await run_in_threadpool(wake_worker, job_id)      # may block on the volume commit
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
    s.update(state="queued", progress=0.0, error=None, message=QUEUED_MSG,
             retried=s.get("retried", 0) + 1)
    json.dump(s, open(os.path.join(d, "status.json"), "w"), indent=1)
    wake_worker(job_id)
    return {"id": job_id, "state": "queued"}


@app.post("/api/jobs/{job_id}/meta")
async def set_meta(job_id: str, request: Request):
    """Patient age for the percentiles, settable after the upload. Nothing that
    identifies a person is accepted here; the page only ever asks for age."""
    d = job_dir(job_id)
    body = await request.json()
    age = parse_age(body.get("age_years"))
    p = os.path.join(d, "status.json")
    s = json.load(open(p))
    s["age_years"] = age
    json.dump(s, open(p, "w"), indent=1)
    await run_in_threadpool(after_write)
    return {"id": job_id, "age_years": age}


@app.get("/history", response_class=HTMLResponse)
def history_page():
    return open(os.path.join(WEB, "history.html")).read()


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
    before_read()
    rows = []
    if os.path.isdir(C.JOBS_DIR):
        for d in os.listdir(C.JOBS_DIR):
            try:
                s = json.load(open(os.path.join(C.JOBS_DIR, d, "status.json")))
                rows.append({k: s.get(k) for k in ("id", "state", "filename", "created", "progress",
                                                   "age_years", "message", "ended")})
            except Exception:
                pass
    rows.sort(key=lambda r: r.get("created") or "", reverse=True)
    return rows[:limit]
