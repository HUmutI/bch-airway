"""The site on Modal: serverless GPU, pay only while a scan runs.

Two functions share one Modal Volume mounted at /jobs, which plays the part
the jobs directory plays on izmir and EFS plays on AWS:

    web()      the FastAPI app (api/app.py), CPU, scales to zero, HTTPS URL
    run_job()  one job on an A10G, spawned by the API when a scan is uploaded

Nothing in api/ or worker/ knows about Modal except api/cloud.py, which
commits the Volume after a write, reloads it before a read, and spawns
run_job instead of starting an EC2 instance when AIRWAY_BACKEND=modal.

Deploy (see deploy/README_MODAL.md):
    modal secret create airway-auth AIRWAY_BASIC_AUTH_USER=quinlab AIRWAY_BASIC_AUTH_PASSWORD=...
    modal secret create ghcr REGISTRY_USERNAME=<github user> REGISTRY_PASSWORD=<token with read:packages>
    modal deploy modal_app.py
"""
import os
import sys

import modal

APP_NAME = "airway"
WORKER_IMAGE = os.environ.get("AIRWAY_WORKER_IMAGE", "ghcr.io/humuti/airway-worker:v3")
GPU = os.environ.get("AIRWAY_GPU", "A10G")          # 24 GB; L4 also works (arch 8.9 compiled)

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("airway-jobs", create_if_missing=True)

HERE = os.path.dirname(os.path.abspath(__file__))

# The verified worker image, built on izmir from deploy/Dockerfile.worker and
# pushed to GHCR. Modal adds its own Python next to it; the stages keep using
# the three venvs under /opt/airway/envs.
worker_image = (
    modal.Image.from_registry(WORKER_IMAGE, secret=modal.Secret.from_name("ghcr"), add_python="3.11")
    .entrypoint([])          # the image's ENTRYPOINT is worker.py; Modal must start its own runner
    .env({
        "AIRWAY_ENVS": "/opt/airway/envs", "AIRWAY_CODE": "/opt/airway/code",
        "AIRWAY_FUSE_SCRIPT": "/opt/airway/fuse_union3.py",
        "AIRWAY_MORPHOMETRY_SCRIPT": "/opt/airway/scripts/09_airmorph_morphometry.py",
        "AIRWAY_AXIS_GEOMETRY_SCRIPT": "/opt/airway/scripts/axis_geometry.py",
        "AIRWAY_JOBS": "/jobs", "PYTHONUNBUFFERED": "1",
    })
)

api_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi>=0.110", "uvicorn[standard]>=0.29", "python-multipart")
    .env({"AIRWAY_JOBS": "/jobs", "AIRWAY_BACKEND": "modal", "AIRWAY_MODAL_APP": APP_NAME})
    .add_local_dir(os.path.join(HERE, "api"), remote_path="/root/app/api")
    .add_local_dir(os.path.join(HERE, "web"), remote_path="/root/app/web")
    .add_local_file(os.path.join(HERE, "config.py"), remote_path="/root/app/config.py")
)


@app.function(image=worker_image, gpu=GPU, volumes={"/jobs": vol},
              timeout=100 * 60, memory=24 * 1024, cpu=4, retries=0)
def run_job(job_id: str):
    """Run one queued job to completion, committing the Volume as it goes so
    the job page sees progress. pipeline.run_job records failure itself."""
    import threading
    import time
    import traceback

    sys.path.insert(0, "/opt/airway/app")
    sys.path.insert(0, "/opt/airway/app/worker")
    import pipeline  # noqa: E402

    vol.reload()
    d = f"/jobs/{job_id}"
    if not os.path.isdir(d):
        raise FileNotFoundError(d)

    def target():
        try:
            pipeline.run_job(d)
        except Exception:                                    # already recorded in status.json
            traceback.print_exc()

    t = threading.Thread(target=target, daemon=True)
    t.start()
    while t.is_alive():
        t.join(5)
        vol.commit()

    # Nothing else is writing now; make sure the final status.json is the one
    # that got committed (re-read after commit+reload, retry if not).
    import json
    status = os.path.join(d, "status.json")
    final = json.load(open(status))
    for attempt in range(3):
        json.dump(final, open(status, "w"), indent=1)
        vol.commit()
        vol.reload()
        if json.load(open(status)).get("state") == final.get("state"):
            break
        print(f"run_job {job_id}: committed status disagrees, retrying", flush=True)
        time.sleep(2)
    print(f"run_job {job_id} finished: {final.get('state')}", flush=True)


@app.function(image=api_image, volumes={"/jobs": vol},
              secrets=[modal.Secret.from_name("airway-auth")],
              scaledown_window=300, timeout=600)
@modal.concurrent(max_inputs=50)
@modal.asgi_app(label="airway-seg")
def web():
    sys.path.insert(0, "/root/app")
    from api.app import app as fastapi_app
    return fastapi_app
