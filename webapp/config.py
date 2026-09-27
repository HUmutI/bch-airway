"""Where everything lives. One place, overridable by environment variables.

The worker runs the three segmentation models in three separate Python
environments, because their dependencies conflict. Each stage is therefore a
subprocess in its own interpreter, and this file is what tells the worker which
interpreter is which. Moving to AWS means changing these paths, nothing else.
"""
import os

BASE = os.environ.get("AIRWAY_BASE", "/fileserver/external/quin/ege")
OR3 = os.path.join(BASE, "or_all3_run")

# Every path below has an AIRWAY_* override so the same code runs from this
# file server and from the Docker image (deploy/Dockerfile.worker sets them).
_ENVS_ROOT = os.environ.get("AIRWAY_ENVS", os.path.join(OR3, "envs"))
ENVS = {
    "airmorph": os.path.join(_ENVS_ROOT, "airmorph", "bin", "python"),
    "medpseg":  os.path.join(_ENVS_ROOT, "medpseg",  "bin", "python"),
    "tfenet":   os.path.join(_ENVS_ROOT, "tfenet",   "bin", "python"),
}
MEDPSEG_BIN = os.path.join(_ENVS_ROOT, "medpseg", "bin", "medpseg")
CODE = os.environ.get("AIRWAY_CODE", os.path.join(OR3, "code"))      # AirMorph/, TfeNet/
FUSE_SCRIPT = os.environ.get("AIRWAY_FUSE_SCRIPT", os.path.join(BASE, "all_3_lcc", "fuse_union3.py"))
CLASS2ANNO = os.path.join(CODE, "AirMorph", "configs", "class2anno.json")

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.environ.get("AIRWAY_JOBS", os.path.join(HERE, "jobs"))
STAGES_DIR = os.path.join(HERE, "worker", "stages")

# Seeded component selection, 6-connectivity, the rule the paper's numbers
# were measured with. Do not change without re-reading the paper's Section 3.
CONNECTIVITY = 6

# Run MEDPSeg and TfeNet on the GPU at the same time. TfeNet peaks near 11 GB
# and MEDPSeg near 5 GB, which fits a 24 GB card with room to spare. Set to 0
# on a smaller card and they run one after the other.
PARALLEL_MEMBERS = os.environ.get("AIRWAY_PARALLEL", "1") == "1"

# Restrict TfeNet's sliding window to the box that contains the lungs and the
# airway AirMorph found, plus a margin. The airway cannot lie outside that box,
# and the box is typically a third of the volume, so this is where most of the
# speed comes from. Window placement inside the box differs from a full-volume
# run, so voxels near window edges can differ slightly. Set to 0 for the exact
# cohort recipe.
TFENET_CROP = os.environ.get("AIRWAY_TFENET_CROP", "1") == "1"
TFENET_CROP_MARGIN = 24                                # voxels, each side

# Expected stage durations in minutes on an RTX A5000, used only to place the
# progress bar. Wrong values make the bar lie, not the pipeline.
STAGE_MINUTES = {
    "prepare": 0.15, "airmorph": 1.3, "medpseg": 2.0, "tfenet": 2.6,
    "fuse": 0.3, "label": 1.2, "package": 0.6, "morphometry": 0.6,
}
# A stage still running after this long is killed and the job fails. The
# slowest stage takes ~6 min on an A5000; a T4 with a big scan maybe 3x that.
# This is only a guard against hangs (e.g. MEDPSeg after a DataLoader crash).
STAGE_TIMEOUT_MIN = float(os.environ.get("AIRWAY_STAGE_TIMEOUT_MIN", "45"))
PALETTE = os.path.join(HERE, "web", "palette.json")   # served as /static/palette.json

# Cohort scripts the morphometry stage reuses. Nothing is copied: the web app
# measures with the exact code the cohort tables were produced with.
MORPHOMETRY_SCRIPT = os.environ.get("AIRWAY_MORPHOMETRY_SCRIPT",
                                    os.path.join(OR3, "scripts", "09_airmorph_morphometry.py"))
AXIS_GEOMETRY_SCRIPT = os.environ.get("AIRWAY_AXIS_GEOMETRY_SCRIPT",
                                      os.path.join(BASE, "cohort90_results",
                                                   "reproducibility_scripts", "axis_geometry.py"))

# Where the GPU is: "local" (a worker polls JOBS_DIR, as on izmir), "aws"
# (the API starts a GPU instance per queued job, deploy/README_AWS.md) or
# "modal" (JOBS_DIR is a Modal Volume and the API spawns the GPU function,
# modal_app.py). api/cloud.py implements the three.
BACKEND = os.environ.get("AIRWAY_BACKEND", "local")
WORKER_INSTANCE_ID = os.environ.get("AIRWAY_WORKER_INSTANCE_ID", "")   # aws
MODAL_APP = os.environ.get("AIRWAY_MODAL_APP", "airway")                # modal
MODAL_VOLUME = os.environ.get("AIRWAY_MODAL_VOLUME", "airway-jobs")     # modal

# Shared lab login in front of every page, when both are set. On izmir they
# are not; on AWS Caddy does it; on Modal there is no Caddy, so the API does.
BASIC_AUTH_USER = os.environ.get("AIRWAY_BASIC_AUTH_USER", "")
BASIC_AUTH_PASSWORD = os.environ.get("AIRWAY_BASIC_AUTH_PASSWORD", "")

# The viewer keeps the CT on the GPU at 4 bytes per voxel. A display copy is
# resampled so that it never exceeds this many voxels (60 M = 240 MB).
VIEW_MAX_VOXELS = int(os.environ.get("AIRWAY_VIEW_MAX_VOXELS", str(60_000_000)))
# A coarse copy of the CT (~1.5 mm, a few MB) that the viewer shows within
# seconds while the full display copy (~60 MB) is still downloading.
VIEW_LOWRES_VOXELS = int(os.environ.get("AIRWAY_VIEW_LOWRES_VOXELS", str(6_000_000)))

MAX_UPLOAD_MB = int(os.environ.get("AIRWAY_MAX_UPLOAD_MB", "1500"))
