# GS-Surrogate: Deformable Gaussian Splatting for Parameter Space Exploration of Ensemble Simulations

[![arXiv](https://img.shields.io/badge/arXiv-2604.06358-b31b1b.svg)](https://arxiv.org/abs/2604.06358)

**GS-Surrogate** is a Gaussian Splatting surrogate model that allows interactive visual exploration of ensemble simulation data across a continuous parameter space. We train a canonical 3D Gaussian Splatting (3DGS) field on a reference simulation condition, then learn a deformation MLP that maps arbitrary simulation parameters to per-Gaussian position, orientation, scale, opacity, and color adjustments, allowing real-time rendering of any unseen parameter combination.

With this framework, users can:

- Train a deformable Gaussian surrogate on any ensemble simulation dataset with a shared camera setup.
- Explore the full parameter space of an ensemble interactively, rendering novel conditions not seen during training.
- Evaluate simulation results across both seen and unseen viewpoints and simulation parameters.

## Table of Contents

- [Installation](#installation)
- [Dataset Format](#dataset-format)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## 🛠️ Installation

We recommend creating a new Conda environment:

```bash
conda create -n gs python=3.10
conda activate gs
```

A C++ compiler (GCC 9+) is required to build the CUDA extensions. If you get compilation errors, make sure `gcc` is on your PATH.

Install PyTorch (adjust for your CUDA version):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Install remaining dependencies:

```bash
pip install -r requirements.txt
```

## 🗂️ Dataset Format

Each dataset should be a directory with the following structure:

```
MyDataset/
├── names.txt                   # Simulation parameter file
├── sparse/
│   └── 0/
│       ├── cameras.bin         # COLMAP camera intrinsics (shared across all conditions)
│       ├── images.bin          # COLMAP camera poses (shared across all conditions)
│       └── points3D.bin        # SfM point cloud (for Gaussian initialization)
├── p001/                       # Condition 0 (1-indexed folder names)
│   ├── frame_0001.jpg
│   └── ...
├── p002/                       # Condition 1
│   └── ...
└── ...
```

**`names.txt` format** — one line per simulation condition:

```
0001_0.5_1.2_3.7        # index_param1_param2_param3
0002_0.6_1.3_3.8
...
```


## 🚀 Training

All training is run from the `examples/` directory:

```bash
cd examples/
```
```bash
bash benchmarks/basic_xcompact_sfm_3dgs_mlp_all.sh
```

## 📊 Evaluation

Evaluation runs automatically at the end of training, reporting a 2×2 matrix of metrics:

| | Seen Cameras | Unseen Cameras |
|---|---|---|
| **Train Conditions** | ✓ | ✓ |
| **Test Conditions (holdout)** | ✓ | ✓ |

Metrics reported: **PSNR**, **SSIM**, **LPIPS**.


```

## 📄 Citation

If you find GS-Surrogate useful for your research, please cite:

```bibtex
@article{li2026gs,
  title={GS-Surrogate: Deformable Gaussian Splatting for Parameter Space Exploration of Ensemble Simulations},
  author={Li, Ziwei and Perera, Rumali and Forbes, Angus and Moreland, Ken and Pugmire, Dave and Klasky, Scott and Chao, Wei-Lun and Shen, Han-Wei},
  journal={arXiv preprint arXiv:2604.06358},
  year={2026}
}
```

## 🏆 Acknowledgements

This work was supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research’s Computer Science Competitive Portfolios program under Contract No. DEAC05-00OR22725. This research used resources of the Oak Ridge Leadership Computing Facility at the Oak Ridge National Laboratory, which is supported by the Advanced Scientific Computing Research programs in the Office of Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725.


