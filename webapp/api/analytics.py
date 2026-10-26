"""Usage counters: page views, unique visitors and uploads.

One JSON line per event, appended to JOBS_DIR/_analytics/<day>_<container>.jsonl
(one file per container and day, so API replicas never write the same file on
the Modal Volume). A visitor is the SHA-256 of client IP + user agent, truncated;
no cookies, no names. Uploads are counted from the job directories themselves,
so jobs made before this module existed are included.
"""
import hashlib, json, os, socket, threading, time
from collections import defaultdict
from datetime import datetime, timezone

import config as C
from api.cloud import after_write, before_read

DIR = "_analytics"
_lock = threading.Lock()
_container = (os.environ.get("MODAL_TASK_ID") or socket.gethostname())[-12:]
_commit_pending = False
COMMIT_DELAY_S = 30      # page views are committed in batches; uploads at once


def _commit_later():
    """One Volume commit per COMMIT_DELAY_S at most, however many views arrive."""
    global _commit_pending
    with _lock:
        if _commit_pending:
            return
        _commit_pending = True

    def run():
        global _commit_pending
        time.sleep(COMMIT_DELAY_S)
        with _lock:
            _commit_pending = False
        after_write()
    threading.Thread(target=run, daemon=True).start()


def visitor_of(request):
    ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or \
        (request.client.host if request.client else "")
    ua = request.headers.get("user-agent", "")
    return hashlib.sha256(f"{ip}|{ua}".encode()).hexdigest()[:12]


def record(request, event, **extra):
    now = datetime.now(timezone.utc)
    row = {"ts": now.isoformat(timespec="seconds"), "event": event, "path": request.url.path,
           "visitor": visitor_of(request), **extra}
    d = os.path.join(C.JOBS_DIR, DIR)
    with _lock:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{now:%Y-%m-%d}_{_container}.jsonl"), "a") as fh:
            fh.write(json.dumps(row) + "\n")
    if event == "upload":
        after_write()
    else:
        _commit_later()


def _events():
    d = os.path.join(C.JOBS_DIR, DIR)
    if not os.path.isdir(d):
        return
    for f in sorted(os.listdir(d)):
        if not f.endswith(".jsonl"):
            continue
        with open(os.path.join(d, f)) as fh:
            for line in fh:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if isinstance(e, dict) and isinstance(e.get("ts"), str) and e.get("event"):
                    yield e


def stats():
    before_read()
    # jobs still on disk give state and size; upload events keep the count after a purge
    jobs, states, size, first, last = set(), defaultdict(int), 0, None, None
    root = C.JOBS_DIR
    for name in os.listdir(root) if os.path.isdir(root) else []:
        try:
            s = json.load(open(os.path.join(root, name, "status.json")))
        except Exception:
            continue
        jobs.add(name); states[s.get("state") or "unknown"] += 1; size += int(s.get("size_bytes") or 0)
        c = s.get("created") or ""
        first = c if first is None or c < first else first
        last = c if last is None or c > last else last
    views, visitors, by_day = 0, set(), defaultdict(lambda: {"views": 0, "visitors": set(), "uploads": 0})
    for e in _events():
        day = e["ts"][:10]
        if e["event"] == "view":
            views += 1; visitors.add(e["visitor"]); by_day[day]["views"] += 1; by_day[day]["visitors"].add(e["visitor"])
        elif e["event"] == "upload":
            by_day[day]["uploads"] += 1
            if e.get("job") not in jobs:
                jobs.add(e.get("job")); size += int(e.get("size_bytes") or 0); states["purged"] += 1
                first = e["ts"] if first is None or e["ts"] < first else first
                last = e["ts"] if last is None or e["ts"] > last else last
    return {"uploads": {"total": len(jobs), "by_state": dict(states), "bytes": size, "first": first, "last": last},
            "views": views, "unique_visitors": len(visitors),
            "by_day": {d: {"views": v["views"], "visitors": len(v["visitors"]), "uploads": v["uploads"]}
                       for d, v in sorted(by_day.items())}}
