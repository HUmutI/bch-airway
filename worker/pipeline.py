"""Run one uploaded scan through the whole pipeline and report progress.

Each stage is a subprocess in the Python environment its model needs, so the
three incompatible environments never meet in one interpreter. Progress is a
JSON file the web API reads, updated after every stage transition and while a
stage runs, so the bar moves during the long ones rather than jumping.

Order:  prepare -> airmorph -> (medpseg | tfenet) -> fuse -> label -> package
        -> morphometry
MEDPSeg and TfeNet run concurrently on the GPU when config.PARALLEL_MEMBERS is
set. AirMorph must come first because TfeNet's crop box comes from its lung
mask, and the fusion is seeded from its airway. The viewer opens as soon as
package is done (status.results_ready); morphometry fills in afterwards and
is not allowed to fail the job.
"""
import json, os, shutil, subprocess, sys, time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402

STAGES = [
    ("prepare",  "Preparing the volume"),
    ("airmorph", "AirMorph segmentation"),
    ("medpseg",  "MEDPSeg segmentation"),
    ("tfenet",   "TfeNet segmentation"),
    ("fuse",     "Fusing the three masks"),
    ("label",    "Naming the branches"),
    ("package",  "Packaging results"),
    ("morphometry", "Measuring the airway"),
]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Job:
    def __init__(self, job_dir):
        self.dir = job_dir
        self.case = os.path.join(job_dir, "case")
        self.logs = os.path.join(job_dir, "logs")
        self.status_path = os.path.join(job_dir, "status.json")
        os.makedirs(self.case, exist_ok=True)
        os.makedirs(self.logs, exist_ok=True)
        self.s = json.load(open(self.status_path))
        self.s.setdefault("stages", [
            {"name": n, "label": l, "state": "pending", "started": None, "ended": None}
            for n, l in STAGES])
        self.save()

    # ---------------------------------------------------------------- status
    def save(self):
        self.s["progress"] = self.progress()
        self.s["updated"] = now()
        tmp = self.status_path + ".tmp"
        json.dump(self.s, open(tmp, "w"), indent=1)
        os.replace(tmp, self.status_path)

    def stage(self, name):
        return next(x for x in self.s["stages"] if x["name"] == name)

    def set(self, name, state, message=None):
        st = self.stage(name)
        st["state"] = state
        if state == "running":
            st["started"] = time.time()
        if state in ("done", "failed", "skipped", "cancelled"):
            st["ended"] = time.time()
        if message:
            self.s["message"] = message
        self.save()

    def progress(self):
        total = sum(C.STAGE_MINUTES.values())
        done = 0.0
        for st in self.s.get("stages", []):
            w = C.STAGE_MINUTES[st["name"]]
            if st["state"] in ("done", "skipped"):
                done += w
            elif st["state"] == "running" and st["started"]:
                frac = (time.time() - st["started"]) / (w * 60)
                done += w * min(frac, 0.92)
        return round(min(done / total, 0.999), 4)

    def log(self, line):
        with open(os.path.join(self.dir, "log.txt"), "a") as fh:
            fh.write(f"[{datetime.now().strftime('%H:%M:%S')}] {line}\n")

    # ------------------------------------------------------------- running
    def spawn(self, name, cmd, env_extra=None):
        """Start a stage subprocess, its output going to logs/<name>.log."""
        self.set(name, "running", self.stage(name)["label"])
        self.log(f"start {name}: {' '.join(cmd)}")
        env = dict(os.environ, OR3_CODE=C.CODE, PYTHONUNBUFFERED="1",
                   CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
        if env_extra:
            env.update(env_extra)
        fh = open(os.path.join(self.logs, f"{name}.log"), "w")
        # Own process group, so a stage that is itself a shell wrapper (MEDPSeg)
        # can be killed together with the child that actually holds the GPU.
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                             cwd=self.case, start_new_session=True)
        p._airway_fh = fh
        return p

    def kill(self, p):
        import signal
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        p._airway_fh.close()

    def wait(self, procs):
        """Wait on {name: Popen}, ticking progress, raising on the first failure."""
        pending = dict(procs)
        while pending:
            for name, p in list(pending.items()):
                rc = p.poll()
                if rc is None:
                    continue
                p._airway_fh.close()
                del pending[name]
                if rc == 0:
                    self.set(name, "done")
                    self.log(f"done {name}")
                else:
                    self.set(name, "failed")
                    tail = open(os.path.join(self.logs, f"{name}.log")).read()[-1500:]
                    self.log(f"FAILED {name} (exit {rc})\n{tail}")
                    for other, q in pending.items():
                        self.kill(q)
                        self.set(other, "cancelled")
                        self.log(f"cancelled {other}")
                    raise RuntimeError(f"{self.stage(name)['label']} failed. "
                                       + last_line(tail))
            if pending:
                self.save()
                time.sleep(2)


    # ------------------------------------------------------------ the tail
    def results_ready(self):
        """The viewer can open now. Morphometry still runs behind it."""
        summ = os.path.join(self.dir, "results", "summary.json")
        if os.path.exists(summ):
            self.s["results"] = json.load(open(summ))
        self.s["results_ready"] = True
        self.s["message"] = "Segmentation ready, measuring the airway"
        self.save()

    def morphometry(self):
        """Measure the tree. A failure is recorded on the stage, not the job."""
        try:
            self.wait({"morphometry": self.spawn("morphometry", [C.ENVS["airmorph"],
                       f"{C.STAGES_DIR}/stage_morphometry.py", "--job-dir", self.dir])})
            self.s.pop("morphometry_error", None)
        except Exception as exc:                                  # noqa: BLE001
            self.s["morphometry_error"] = str(exc)
            self.log(f"morphometry failed, job continues: {exc}")
            self.save()


def last_line(text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1][:300] if lines else ""


def run_job(job_dir, mock=False):
    j = Job(job_dir)
    j.s["state"] = "running"
    j.s["error"] = None                       # a rerun must not inherit a stale one
    j.s["results_ready"] = False
    j.s.pop("results", None)
    j.s.pop("morphometry_error", None)
    j.s["started"] = now()
    for st in j.s["stages"]:
        st.update(state="pending", started=None, ended=None)
    j.save()
    inp = os.path.join(job_dir, "input.nii.gz")
    S = C.STAGES_DIR
    py = C.ENVS
    try:
        if mock:
            _mock(j)
        else:
            j.wait({"prepare": j.spawn("prepare", [py["airmorph"],
                    f"{S}/stage_prepare.py", "--input", inp, "--case-dir", j.case])})
            j.wait({"airmorph": j.spawn("airmorph", [py["airmorph"],
                    f"{S}/stage_airmorph.py", "--case-dir", j.case])})

            medpseg = ["bash", "-c",
                       f'"{C.MEDPSEG_BIN}" -i "{j.case}/medpseg_in" -o "{j.case}/medpseg_out" '
                       f'--disable_lobe && cp "{j.case}/medpseg_out/image_airway.nii.gz" '
                       f'"{j.case}/medpseg_airway.nii.gz"']
            tfenet = [py["tfenet"], f"{S}/stage_tfenet.py", "--case-dir", j.case,
                      "--margin", str(C.TFENET_CROP_MARGIN)]
            if not C.TFENET_CROP:
                tfenet.append("--no-crop")
            if C.PARALLEL_MEMBERS:
                j.wait({"medpseg": j.spawn("medpseg", medpseg),
                        "tfenet": j.spawn("tfenet", tfenet)})
            else:
                j.wait({"medpseg": j.spawn("medpseg", medpseg)})
                j.wait({"tfenet": j.spawn("tfenet", tfenet)})

            j.wait({"fuse": j.spawn("fuse", [py["airmorph"], f"{S}/stage_fuse.py",
                    "--case-dir", j.case, "--fuse-script", C.FUSE_SCRIPT,
                    "--connectivity", str(C.CONNECTIVITY)])})
            j.wait({"label": j.spawn("label", [py["airmorph"], f"{S}/stage_label.py",
                    "--case-dir", j.case, "--class2anno", C.CLASS2ANNO])})
            j.wait({"package": j.spawn("package", [py["airmorph"], f"{S}/stage_package.py",
                    "--job-dir", job_dir, "--class2anno", C.CLASS2ANNO])})
            j.results_ready()
            j.morphometry()

        j.s["state"] = "done"
        j.s["message"] = "Finished"
        j.s["progress"] = 1.0
        j.s["ended"] = now()
        j.save()
        j.log("job done")
    except Exception as exc:
        j.s["state"] = "failed"
        j.s["error"] = str(exc)
        j.s["message"] = "Failed"
        j.s["ended"] = now()
        j.save()
        j.log(f"job failed: {exc}")
        raise


def _mock(j):
    """Pretend to run, using a finished case so the viewer has real output.

    For exercising the web app on a machine with no GPU. Every stage sleeps a
    few seconds; at the end the sample case's files are copied in and packaged
    by the real packaging stage, so what the page shows is what a real run
    would show.
    """
    sample = os.environ.get("AIRWAY_MOCK_CASE")
    if not sample or not os.path.isdir(sample):
        raise RuntimeError("AIRWAY_MOCK_CASE must point at a finished case directory")
    inp = os.path.join(j.dir, "input.nii.gz")
    for name, _ in STAGES[:-2]:
        j.set(name, "running", j.stage(name)["label"])
        for _ in range(3):
            time.sleep(1); j.save()
        j.set(name, "done")
    case_id = os.path.basename(sample.rstrip("/"))
    for f in os.listdir(sample):
        src = os.path.join(sample, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(j.case, f.replace(case_id, "case")))
    # The upload stays as uploaded. The sample case's masks are rotated back to
    # the orientation its orient.json records, so the upload must be a scan in
    # that orientation: the sample's own source file, not its LPS working copy.
    if not os.path.exists(os.path.join(j.case, "orient.json")):
        json.dump({"source_orientation": "LPS"}, open(os.path.join(j.case, "orient.json"), "w"))
    j.wait({"package": j.spawn("package", [C.ENVS["airmorph"],
            f"{C.STAGES_DIR}/stage_package.py", "--job-dir", j.dir,
            "--class2anno", C.CLASS2ANNO])})
    j.results_ready()
    j.morphometry()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    run_job(a.job_dir, mock=a.mock)
