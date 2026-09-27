# izmir: how the lab machine maps onto this repository

izmir is the lab GPU machine (RTX A5000 24 GB, driver 550, nvcc 12.4, Docker
with the NVIDIA runtime). Everything lives under
`/fileserver/external/quin/ege/`. Paths below are relative to that.

## What is where

| on izmir | in the repo | note |
|---|---|---|
| `bch-airway/` | this repository, checked out | canonical copy; `git pull` here |
| `or_all3_run/scripts/` | `pipeline/scripts/` | symlink into the checkout since 2026-09-23 (`scripts_pre_monorepo_2026-09-23/` is the old copy) |
| `or_all3_run/code/`, `or_all3_run/envs/` | built by `pipeline/scripts/00_setup.sh` | not in git: 3 venvs, extracted model code |
| `or_all3_run/{manifest,qc}.csv`, `cases_*.txt`, logs, tables | outputs | not in git |
| `all_3_lcc/*.tar.gz` | Release `models-v1` | the three model tarballs |
| `all_3_lcc/fuse_union3.py` | `pipeline/fuse_union3.py` | symlink |
| `airway-dataset/` | — | the 849 source scans (de-identified). **Never copied anywhere.** |
| `airway-seg/` | — | results, one directory per scan, mirrors the dataset tree |
| `cohort90_results/reproducibility_scripts/` | `cohort90/` | symlink |
| `cohort90_results/` (rest) | — | cohort tables, report, figures; not in git |
| `airway_web/` | `webapp/` | symlink into the checkout; `jobs/`, `.venv/`, `deploy/vendor/` live inside it, ignored by git |
| `paper/` | `paper/` | only `.tex` and `.bib` are in git |

If you edit a file through one of the symlinked paths you are editing the
checkout; commit from `bch-airway/`.

## Environment variables

Every absolute path has a default that is izmir's and an override:

| variable | default | used by |
|---|---|---|
| `OR3_BASE` | `/fileserver/external/quin/ege` | `config.sh`, morphometry, QC scripts |
| `OR3_RUN`, `OR3_CODE`, `OR3_ENVS`, `OR3_TARBALLS`, `OR3_DATASET`, `OR3_OUT` | derived from `OR3_BASE` | pipeline stages |
| `AIRWAY_BASE`, `AIRWAY_ENVS`, `AIRWAY_CODE`, `AIRWAY_JOBS`, … | see `webapp/config.py` | web app |

On izmir you set nothing.

## The sandbox

Agents run inside `claude-safe.sh`, which hides `/dev/nvidia*` and the Docker
socket. GPU jobs, `docker build`, `./run_worker.sh` all have to run outside it
(a plain terminal). `HOW_TO_ENTER_SANDBOX_EGE.txt` in the base directory
describes it.

## Services

The website on izmir is two processes started from `airway_web/`
(`./run_api.sh`, `./run_worker.sh`), logging to `api.log` and `worker.log`
there. `http://izmir:8000`. The PI has said izmir may not serve the site to
users and no tunnels; the AWS move is in `webapp/deploy/README_AWS.md`.

## Conventions

- Stage scripts are numbered and resumable; re-running skips finished cases.
- Orientation: the dataset is LAS, the models want LPS; stage 1 reorients,
  stage 5 rotates back, stage 7 proves it (`pipeline/README_RUN.md`).
- Label volumes store `class + 1`; 0 is background; read `class2anno.json`
  or `labels.json`, never hardcode the order.
- Commits to this repo are authored as `huseyinumut2004@gmail.com`.
