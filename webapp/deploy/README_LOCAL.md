# Running the airway site on your own Ubuntu machine

The same two containers that run the hosted site, on one computer. Scans and
results never leave the machine. One CT takes about 10 to 15 minutes.

```
 browser ──▶ http://localhost:8000 ──▶ api container ──┐
                                                       ├── webapp/jobs/  (one folder per scan)
                                    worker container ──┘   GPU: AirMorph-R, MEDPSeg-R, TfeNet
```

## What the machine needs

| | |
|---|---|
| OS | Ubuntu 22.04 or 24.04 (other Linux distributions work the same way) |
| GPU | one NVIDIA GPU with at least 16 GB memory, 24 GB recommended (tested: RTX A5000 24 GB, A10G 24 GB) |
| NVIDIA driver | 535 or newer (`nvidia-smi` must work) |
| Disk | about 60 GB free: 20 GB for the worker image, about 1 GB per processed scan |
| Memory | 32 GB RAM |

macOS and Windows without an NVIDIA GPU are not supported.

## One-time setup

1. **Docker and the NVIDIA container toolkit**

   ```bash
   # Docker Engine with the compose plugin
   curl -fsSL https://get.docker.com | sudo sh
   sudo usermod -aG docker $USER          # log out and in again afterwards

   # NVIDIA container toolkit (lets containers use the GPU)
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
     | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
     | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker

   # check: this must print your GPU
   docker run --rm --gpus all ubuntu nvidia-smi
   ```

2. **The code**

   ```bash
   git clone https://github.com/quin-med-harvard-edu/airway-seg-and-labelling.git
   cd airway-seg-and-labelling/webapp
   ```

3. **Access to the worker image.** The worker image (models and their
   environments, about 20 GB) is a private package on the GitHub Container
   Registry. Ask Umut for read access, then create a GitHub personal access
   token (classic) with only the `read:packages` scope and log in once:

   ```bash
   echo <token> | docker login ghcr.io -u <your-github-username> --password-stdin
   ```

## Start, use, stop

```bash
cd airway-seg-and-labelling/webapp
docker compose -f deploy/compose.local.yml up -d      # first time: downloads 20 GB
```

Open **http://localhost:8000**, upload a CT (`.nii.gz`, `.nii`, `.mha`, `.nrrd`
or a zipped DICOM series) and follow the job page. Results, downloads and the
3D viewer are the same as on the hosted site. Earlier scans are under
*Previous uploads*.

```bash
docker compose -f deploy/compose.local.yml logs -f worker   # what the GPU is doing
docker compose -f deploy/compose.local.yml down             # stop (results are kept)
```

The containers restart by themselves after a reboot until you run `down`.

## Options

Set them in front of the `up` command, e.g. `AIRWAY_PORT=8080 docker compose ... up -d`.

| variable | default | meaning |
|---|---|---|
| `AIRWAY_PORT` | `8000` | port of the site |
| `AIRWAY_BIND` | `127.0.0.1` | `0.0.0.0` lets other computers on the network open the site. It has no password, so only do this on a trusted network |
| `AIRWAY_JOBS_HOST` | `../jobs` (= `webapp/jobs`) | where uploads and results are stored |
| `AIRWAY_RETAIN_DAYS` | `0` | delete jobs older than this many days; 0 keeps everything |
| `AIRWAY_WORKER_IMAGE` | `ghcr.io/humuti/airway-worker:v8` | worker image to use |

## Updating

```bash
git pull
docker compose -f deploy/compose.local.yml pull worker
docker compose -f deploy/compose.local.yml up -d --build
```

## Data

Use de-identified scans only, as on the hosted site. Everything is stored in
`webapp/jobs/`; delete a folder there to delete that scan and its results.

## Troubleshooting

- `could not select device driver "nvidia"`: the NVIDIA container toolkit is
  missing or Docker was not restarted (step 1).
- `denied` or `unauthorized` when pulling: the `docker login ghcr.io` step is
  missing, or the token lacks `read:packages`, or your account has no access yet.
- A job fails with `CUDA out of memory`: the GPU has less than 16 GB, or
  another program is using it (`nvidia-smi`).
- `Bus error` in the worker log: the shared-memory setting was removed from
  the compose file; it needs `shm_size: 8gb`.
