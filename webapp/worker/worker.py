#!/usr/bin/env python
"""Pick up queued jobs one at a time and run them.

One GPU, one job at a time. The queue is the jobs directory: a job is queued
when its status.json says so, and claimed by rewriting that state before any
work starts, so two workers on the same directory cannot both take it.

Run this OUTSIDE the sandbox, on the machine with the GPU:
    ./.venv/bin/python worker/worker.py
    ./.venv/bin/python worker/worker.py --mock   # no GPU, fakes the stages

On AWS this loop becomes an SQS consumer and the jobs directory becomes S3;
nothing in pipeline.py changes.
"""
import argparse, json, os, signal, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C          # noqa: E402
from pipeline import run_job  # noqa: E402


def queued_jobs():
    out = []
    if not os.path.isdir(C.JOBS_DIR):
        return out
    for d in os.listdir(C.JOBS_DIR):
        p = os.path.join(C.JOBS_DIR, d, "status.json")
        try:
            s = json.load(open(p))
        except Exception:
            continue
        if s.get("state") == "queued":
            out.append((s.get("created", ""), os.path.join(C.JOBS_DIR, d)))
    return [d for _, d in sorted(out)]


def claim(job_dir):
    p = os.path.join(job_dir, "status.json")
    s = json.load(open(p))
    if s.get("state") != "queued":
        return False
    s["state"] = "claimed"
    json.dump(s, open(p, "w"), indent=1)
    return True


def purge_old(days):
    """Delete finished jobs older than `days`. Uploads are research scans the
    user already has; keeping them forever on a cloud disk costs money and
    widens exposure for nothing."""
    import shutil
    from datetime import datetime, timezone, timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    for d in os.listdir(C.JOBS_DIR):
        p = os.path.join(C.JOBS_DIR, d)
        try:
            s = json.load(open(os.path.join(p, "status.json")))
            created = datetime.fromisoformat(s.get("created", "").replace("Z", "+00:00"))
        except Exception:
            continue
        if s.get("state") in ("done", "failed") and created < cutoff:
            shutil.rmtree(p, ignore_errors=True)
            print(f"purged {d} from {created.date()}", flush=True)


def recover_stale():
    """A job left claimed/running at startup has no owner: one worker per jobs
    directory, and the previous one died (killed, instance stopped, crash)
    before it could record a failure. Mark it failed so Retry works."""
    for d in os.listdir(C.JOBS_DIR) if os.path.isdir(C.JOBS_DIR) else []:
        p = os.path.join(C.JOBS_DIR, d, "status.json")
        try:
            s = json.load(open(p))
        except Exception:
            continue
        if s.get("state") not in ("claimed", "running"):
            continue
        for st in s.get("stages", []):
            if st.get("state") == "running":
                st["state"] = "cancelled"
        s.update(state="failed", message="Failed",
                 error="The worker was stopped while this job ran. Press Retry.")
        tmp = p + ".tmp"
        json.dump(s, open(tmp, "w"), indent=1)
        os.replace(tmp, p)
        print(f"recovered stale job {d}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--poll", type=float, default=3.0)
    ap.add_argument("--exit-when-idle", type=float, default=0, metavar="MIN",
                    help="exit 0 after this many minutes with nothing queued (cloud: the "
                         "host then stops the instance). 0 = run forever")
    ap.add_argument("--retain-days", type=int, default=0,
                    help="delete finished jobs older than this. 0 = keep everything")
    a = ap.parse_args()
    print(f"worker up, jobs in {C.JOBS_DIR}, mock={a.mock}, "
          f"idle-exit={a.exit_when_idle or 'never'} min, retain={a.retain_days or 'all'}", flush=True)
    # docker stop / systemctl stop send SIGTERM; turn it into SystemExit so the
    # running job is cancelled and recorded instead of silently abandoned.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    recover_stale()
    last_work = time.time()
    last_purge = 0
    while True:
        for job_dir in queued_jobs():
            if not claim(job_dir):
                continue
            print(f"running {os.path.basename(job_dir)}", flush=True)
            try:
                run_job(job_dir, mock=a.mock)
            except Exception:
                traceback.print_exc()
            last_work = time.time()
        if a.retain_days and time.time() - last_purge > 3600:
            purge_old(a.retain_days)
            last_purge = time.time()
        if a.exit_when_idle and time.time() - last_work > a.exit_when_idle * 60:
            print(f"idle for {a.exit_when_idle} min, exiting", flush=True)
            return
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
