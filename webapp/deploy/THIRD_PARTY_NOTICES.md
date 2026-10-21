# Third-party components in the airway worker image

The image bundles the following software and model weights. Each keeps its own
license; the full license texts are inside the image at the paths given.

| Component | What is included | License | Source |
|---|---|---|---|
| AirMorph (Zhang et al., arXiv 2412.11039) | inference code: networks, segmentator, tree parsing, `configs/class2anno.json` (`/opt/airway/code/AirMorph`) | used and redistributed with the authors' permission | https://github.com/EndoluminalSurgicalVision-IMR/AirMorph |
| AirMorph-R weights | the five segmentation networks in `/opt/airway/code/AirMorph/checkpoints`, trained by QUIN Lab on ATM'26 Task-1 training data | research use; cite the ATM papers | this project |
| MEDPSeg (Carmo et al.) | code and lung model (`/opt/airway/code/medpseg`, `LICENSE`) | MIT | https://github.com/MICLab-Unicamp/medpseg |
| MEDPSeg-R weights | `poly_medseg_25d_fix.ckpt` in the medpseg environment, retrained by QUIN Lab on ATM'26 Task-1 training data | research use; cite the ATM papers | this project |
| TfeNet (Wu et al.) | code and BAS checkpoint (`/opt/airway/code/TfeNet`, `LICENSE`) | MIT | https://github.com/QibiaoWu/TfeNet |
| lungmask (Hofmanninger et al.) | R231 lung model weights (`unet_r231-d5d2fc3d.pth`) | Apache-2.0 | https://github.com/JoHof/lungmask |
| PyTorch, MONAI, SimpleITK, NumPy, SciPy and other Python packages | the three Python environments under `/opt/airway/envs` | their own licenses, see each package's `dist-info` | PyPI |

Models trained on ATM'22/ATM'26 data: the challenge rules ask every user of the
data to cite its papers, in particular Zhang et al., "Multi-site, multi-domain
airway tree modeling", Medical Image Analysis 2023, doi:10.1016/j.media.2023.102957.

Research use only. Not a medical device; do not use the output for diagnosis or treatment.
