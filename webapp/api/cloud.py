"""Where the GPU is, and how the API's files reach it.

Three backends, chosen by AIRWAY_BACKEND (config.BACKEND):

  local   izmir. A worker process polls the jobs directory; nothing to do.
  aws     the GPU EC2 instance stops itself when idle and the API starts it
          here (deploy/README_AWS.md). Needs boto3 and an instance role with
          ec2:StartInstances / ec2:DescribeInstances on the worker instance.
  modal   the jobs directory is a Modal Volume. Writes must be committed and
          reads preceded by a reload, and a queued job is run by spawning the
          GPU function (modal_app.py) instead of by a polling worker.

api/app.py calls three things: before_read() at the top of every request
that reads job files, after_write() is folded into wake_worker(), and
wake_worker(job_id) after a job has been queued.
"""
import os, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402

_last = 0.0
_vol = None


def _volume():
    global _vol
    if _vol is None:
        import modal
        _vol = modal.Volume.from_name(C.MODAL_VOLUME)
    return _vol


def before_read():
    """See the GPU function's latest commit. Cheap; called per request."""
    if C.BACKEND == "modal":
        try:
            _volume().reload()
        except Exception as exc:                                 # noqa: BLE001
            print(f"cloud: volume reload failed: {exc}", flush=True)


def wake_worker(job_id=None):
    if C.BACKEND == "modal":
        _volume().commit()                       # the upload and status.json
        import modal
        modal.Function.from_name(C.MODAL_APP, "run_job").spawn(job_id)
        print(f"cloud: spawned run_job({job_id})", flush=True)
        return
    if C.BACKEND == "aws" and C.WORKER_INSTANCE_ID:
        global _last
        if time.time() - _last < 30:             # a burst of uploads is one wake-up
            return
        _last = time.time()
        threading.Thread(target=_start_ec2, daemon=True).start()


def _start_ec2():
    try:
        import boto3
        ec2 = boto3.client("ec2")
        state = ec2.describe_instances(InstanceIds=[C.WORKER_INSTANCE_ID]) \
                   ["Reservations"][0]["Instances"][0]["State"]["Name"]
        if state == "stopped":
            ec2.start_instances(InstanceIds=[C.WORKER_INSTANCE_ID])
            print(f"cloud: starting worker {C.WORKER_INSTANCE_ID}", flush=True)
        elif state == "stopping":
            # cannot start until it has stopped; try again shortly
            time.sleep(45)
            ec2.start_instances(InstanceIds=[C.WORKER_INSTANCE_ID])
            print(f"cloud: worker was stopping, restarted {C.WORKER_INSTANCE_ID}", flush=True)
        else:
            print(f"cloud: worker is {state}", flush=True)
    except Exception as exc:                                     # noqa: BLE001
        print(f"cloud: could not start worker: {exc}", flush=True)
