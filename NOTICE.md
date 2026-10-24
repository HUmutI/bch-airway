# Third-party code and data in this repository

AirMorph (Zhang et al., arXiv:2412.11039, https://github.com/EndoluminalSurgicalVision-IMR/AirMorph).
Its repository has no licence file; the code is used and redistributed here with the
authors' permission. Files that contain AirMorph code:

- `evaluation/benchmark_adult/shim/util/airwayatlas_eval.py`: unchanged copy of `util/airwayatlas_eval.py`.
- `evaluation/benchmark_adult/shim/training/eval_patch.py` and
  `evaluation/airmorph_reconstruction/eval_patch.py`: a clDice function with a
  one-character fix and three scipy tree-parsing helpers from `data_process/tree_parse_skelbased.py`.
- `evaluation/airmorph_reconstruction/nets/wingsnet_train.py`: training-mode
  subclass of `networks/airway_network_wingsnet.py`.
- `webapp/deploy/airmorph_cpu_path.patch`: a unified diff against AirMorph's
  `data_process/` files (CPU fallback); applied inside the worker image.
- The worker image (`webapp/deploy/THIRD_PARTY_NOTICES.md`) bundles AirMorph's inference code.

MEDPSeg (Carmo et al., arXiv:2312.02365): MIT, https://github.com/MICLab-Unicamp/medpseg.
TfeNet (Wu et al., arXiv:2507.06581): MIT, https://github.com/QibiaoWu/TfeNet.
lungmask (Hofmanninger et al.): Apache-2.0, https://github.com/JoHof/lungmask.
These are imported as packages or bundled in the worker image, not copied into this tree.

ATM'22 / ATM'26 challenge data: not redistributed. The weights trained on them
(`airmorph_seg.tar.gz`, `medpseg_airway.tar.gz` in the `models-v1` Release) are
CC BY-NC 4.0, see `LICENSE-WEIGHTS.md`; users must cite Zhang et al.,
Medical Image Analysis 2023, doi:10.1016/j.media.2023.102957.

Everything else in this repository is MIT licensed (`LICENSE`).
