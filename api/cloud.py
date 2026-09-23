"""Wake the GPU worker when there is work.

On izmir a worker process is always running and this module does nothing. In
the cloud the GPU instance stops itself when idle (deploy/airway-worker.service)
and the API starts it again here, so the GPU is billed only while jobs run.
Starting an instance that is already running is a no-op for EC2, so this is
called on every new job without bookkeeping.

Needs: boto3, AWS_DEFAULT_REGION, and an instance role on the web machine with
ec2:StartInstances and ec2:DescribeInstances on the worker instance.
"""
import os, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402

_last = 0.0


def wake_worker():
    if not C.WORKER_INSTANCE_ID:
        return
    global _last
    if time.time() - _last < 30:          # a burst of uploads is one wake-up
        return
    _last = time.time()
    threading.Thread(target=_start, daemon=True).start()


def _start():
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
