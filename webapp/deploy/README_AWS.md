# Running the airway site on AWS

Two machines and a shared disk. The web machine is tiny and always on. The GPU
machine starts when a job is queued and stops itself after fifteen idle
minutes, so the expensive part is billed by the job, not by the month.

```
 browser ──HTTPS──▶ web instance (t3.small)          GPU instance (g5.xlarge, A10G 24 GB)
                    caddy ─▶ api container            worker container
                         │                                 │
                         └────── EFS  /srv/airway/jobs ────┘
                                (one directory per job)
```

The API and the worker only ever talk through that directory, which is why
nothing in the pipeline changes between izmir and AWS.

## Cost, us-east-1, September 2026 list prices

| item | price | note |
|---|---|---|
| g5.xlarge while a job runs | $1.006 / h | 14 min per scan ≈ $0.25 |
| same, spot | ≈ $0.40 / h | can be interrupted, fine for a queue |
| t3.small web instance | ≈ $15 / month | or t3.micro on the free tier |
| EFS | $0.30 / GB-month | ≈ 300 MB per job; `--retain-days 30` keeps it small |
| ECR image storage | $0.10 / GB-month | worker image ≈ 20 GB → $2 |
| domain | $0 | free `*.duckdns.org` subdomain to start; a real domain later is ≈ $12 / year |

Fifty scans a month ≈ $15 GPU + $15 web + $3 storage ≈ **$35 / month**.
A GPU instance left running around the clock would be $730 / month. The idle
shutdown is what makes this affordable, do not disable it.

Google Cloud equivalent: g2-standard-4 (L4 24 GB, ≈ $0.70 / h), e2-small,
Filestore. Everything in `deploy/` is plain Docker and works there too; only
`api/cloud.py` (start the GPU instance) is AWS-specific and is forty lines.

## Before you start

Decisions taken 2026-09-23: a **personal AWS account**, region **us-east-1**,
domain a free **DuckDNS** subdomain until IT provides one.

- AWS account: create an IAM user (e.g. `airway-deploy`) with
  `AdministratorAccess`, make an access key, `aws configure` on izmir
  (`~/.local/bin/aws` is installed there). Longer term a BCH research
  computing account with a HIPAA agreement is the right home; move then.
- **Quota, first thing:** Service Quotas → EC2 → *Running On-Demand G and VT
  instances* → request 4 vCPUs. New accounts start at 0; approval takes one
  to two days and nothing GPU-side can happen before it.
- Apply for [AWS Cloud Credit for Research](https://aws.amazon.com/government-education/research-and-technical-computing/cloud-credit-for-research/)
  in parallel. It routinely covers a year of this.
- Domain: at duckdns.org (GitHub login) create `<name>.duckdns.org`, keep the
  token out of git (it goes in the web instance's `.env` only). Point it at
  the Elastic IP once there is one. Let's Encrypt issues for duckdns.org but
  not for `*.amazonaws.com` names, which is why some name is needed at all.
  Swapping in a real domain later is one line in `.env`.
- Only de-identified NIfTI. Same rule as on izmir. The site says so on every page.

## 1. Build the images (on izmir, once, outside the sandbox)

```bash
cd /fileserver/external/quin/ege/airway_web
./deploy/prepare.sh                                    # copies tarballs + scripts into deploy/vendor
docker build -f deploy/Dockerfile.worker -t airway-worker .   # 30-45 min, ≈ 20 GB
docker build -f deploy/Dockerfile.api    -t airway-api .      # 1 min
```

Test the worker image on izmir's own GPU before it goes anywhere:

```bash
mkdir -p /tmp/jobs_docker
docker run --rm --gpus all --shm-size 8g -v /tmp/jobs_docker:/jobs airway-worker --exit-when-idle 1
# then, while it runs, drop a job in: copy an existing jobs/<id>/ with state "queued"
```

Things the first builds (2026-09-23) ran into, all fixed in the recipe:

- `--shm-size 8g` is mandatory (compose has it). With Docker's default 64 MB
  `/dev/shm`, MEDPSeg's DataLoader workers die and the process then hangs.
- The vendored AirMorph tarball is upstream code; the lab copy has local edits
  that make pycuda optional. `deploy/airmorph_cpu_path.patch` is that diff,
  applied in the Dockerfile. Regenerate it if `or_all3_run/code/AirMorph`
  changes: `diff -ruN <upstream> <lab copy> -x __pycache__ -x '*.pyc' -x '*.pth' -x '*.pt' -x '*.ckpt'`.
- The worker kills any stage running longer than `AIRWAY_STAGE_TIMEOUT_MIN`
  (45) and, on SIGTERM / instance stop, cancels the running stages and marks
  the job failed so Retry works. A job found `running` at worker start is
  marked failed the same way.
- Files the container writes into the jobs directory are root-owned.

Push to ECR (replace ACCOUNT and REGION):

```bash
aws ecr create-repository --repository-name airway-worker
aws ecr create-repository --repository-name airway-api
aws ecr get-login-password | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.REGION.amazonaws.com
for i in airway-worker airway-api; do
  docker tag $i ACCOUNT.dkr.ecr.REGION.amazonaws.com/$i:latest
  docker push ACCOUNT.dkr.ecr.REGION.amazonaws.com/$i:latest
done
```

## 2. Shared disk

EFS console → Create file system → same VPC as the instances → note the
file system id. Security group: allow NFS (2049) from the instances' group.

## 3. GPU instance

- AMI: **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)**. It has
  the driver, Docker and the NVIDIA container toolkit. Nothing to install.
- Type g5.xlarge (A10G, 24 GB). Root disk 60 GB gp3 (the image is 20 GB).
- Shutdown behaviour: **Stop** (not Terminate). Instance settings → Change
  shutdown behavior.
- IAM role with `AmazonEC2ContainerRegistryReadOnly`.

On the instance:

```bash
sudo mkdir -p /srv/airway/jobs && sudo apt-get install -y nfs-common git
echo "FS-ID.efs.REGION.amazonaws.com:/ /srv/airway/jobs nfs4 nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,_netdev 0 0" | sudo tee -a /etc/fstab
sudo mount -a
sudo git clone <the repo> /srv/airway/app        # or scp airway_web/ there
cat <<EOT | sudo tee /srv/airway/airway.env
AIRWAY_WORKER_IMAGE=ACCOUNT.dkr.ecr.REGION.amazonaws.com/airway-worker:latest
AIRWAY_JOBS_HOST=/srv/airway/jobs
EOT
aws ecr get-login-password --region REGION | sudo docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.REGION.amazonaws.com
sudo docker pull ACCOUNT.dkr.ecr.REGION.amazonaws.com/airway-worker:latest
sudo cp /srv/airway/app/deploy/airway-worker.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable airway-worker
```

Boot runs the worker; the worker exits after 15 idle minutes; the unit's
`ExecStopPost` shuts the instance down; shutdown behaviour Stop keeps the disk.
Note the instance id (`i-…`). ECR logins expire after 12 h; add a cron line
or an ECR credential helper if the image will be re-pulled on boot.

## 4. Web instance

- Ubuntu 22.04, t3.small, 20 GB. Security group: 80 and 443 from anywhere,
  22 from your IP.
- Elastic IP, and an A record for your domain pointing at it.
- IAM role with `AmazonEC2ContainerRegistryReadOnly` plus this inline policy,
  so the API may start the worker and nothing else:

```json
{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Action":["ec2:StartInstances"],"Resource":"arn:aws:ec2:REGION:ACCOUNT:instance/i-WORKER"},
 {"Effect":"Allow","Action":["ec2:DescribeInstances"],"Resource":"*"}]}
```

On the instance:

```bash
sudo apt-get install -y docker.io docker-compose-v2 nfs-common git
sudo mkdir -p /srv/airway/jobs
echo "FS-ID.efs.REGION.amazonaws.com:/ /srv/airway/jobs nfs4 nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,_netdev 0 0" | sudo tee -a /etc/fstab
sudo mount -a
sudo git clone <the repo> /srv/airway/app
cd /srv/airway/app/deploy
HASH=$(sudo docker run --rm caddy:2 caddy hash-password --plaintext 'the-lab-password')
cat <<EOT | sudo tee .env
AIRWAY_API_IMAGE=ACCOUNT.dkr.ecr.REGION.amazonaws.com/airway-api:latest
AIRWAY_JOBS_HOST=/srv/airway/jobs
AIRWAY_DOMAIN=airway.example.org
AIRWAY_BASIC_AUTH_USER=lab
AIRWAY_BASIC_AUTH_HASH=$HASH
AIRWAY_WORKER_INSTANCE_ID=i-WORKER
AWS_DEFAULT_REGION=REGION
EOT
aws ecr get-login-password --region REGION | sudo docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.REGION.amazonaws.com
sudo docker compose -f compose.yml up -d api caddy
```

Caddy fetches the certificate on first request. Open `https://airway.example.org`,
log in with `lab` and the password, upload a scan. Within a minute the GPU
instance shows as *running* in the console, the job page starts moving after
two to three minutes (boot + image start), and the whole thing takes about
17 minutes end to end on an A10G.

## 5. Updates

Code changes: rebuild the image that changed, push, then on the machine
`docker compose pull && docker compose up -d api caddy` (web) or just let the
GPU instance pull on its next boot (add `docker compose pull worker` to the
unit's `ExecStartPre` if you want that automatic).

## Alternatives considered

- **One GPU instance running everything.** Simplest, `docker compose up` of
  all three services on the g5, but it cannot stop when idle without taking
  the website down. Fine for a demo week, $730 / month otherwise.
- **Docker on radiologists' computers.** Needs a 24 GB GPU and 20 GB of
  models. Not realistic.
- **Serverless GPU (SageMaker async, Modal, RunPod).** Cheaper per job and
  no instance to babysit, but the worker would have to be rewritten around
  their job APIs and BCH would need to approve the vendor. Worth revisiting
  once the site has users.
- **Login.** Basic auth is a shared lab password. When IT provides SSO,
  swap the `basic_auth` block in `Caddyfile` for their OIDC provider.
