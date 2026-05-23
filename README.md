# GS-Surrogate: Deformable Gaussian Splatting for Parameter Space Exploration of Ensemble Simulations

[![arXiv](https://img.shields.io/badge/arXiv-2604.06358-b31b1b.svg)](https://arxiv.org/abs/2604.06358)

**GS-Surrogate** is a neural rendering surrogate that enables interactive visual exploration of ensemble simulation data across a continuous parameter space. We train a canonical 3D Gaussian Splatting (3DGS) field on a reference simulation condition, then learn a deformation MLP that maps arbitrary simulation parameters to per-Gaussian position, orientation, scale, opacity, and color adjustments — enabling real-time rendering of any unseen parameter combination.

With this framework, users can:

- Train a deformable Gaussian surrogate on any ensemble simulation dataset with a shared camera setup.
- Explore the full parameter space of an ensemble interactively, rendering novel conditions not seen during training.
- Evaluate interpolation and extrapolation quality across both seen and unseen viewpoints and simulation parameters.

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

Install PyTorch (adjust for your CUDA version):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Install the gsplat CUDA backend (our modified version is included in this repo):

```bash
pip install -e gsplat/
```

Install remaining dependencies:

```bash
pip install tyro imageio tqdm tensorboard viser nerfview \
    torchmetrics[image] fused-ssim pillow scikit-learn \
    pyyaml numpy
```

## 🗂️ Dataset Format

Each dataset is a directory with the following structure:

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

The index is 1-based (matching the `p001/`, `p002/` folder names). Parameter values are parsed automatically as the condition vector; dimensionality is detected from the first entry. The COLMAP reconstruction is shared across all conditions.

## 🚀 Training

All training is run from the `examples/` directory:

```bash
cd examples/
```

**XCompact3D dataset** (edit `DATASET` path in the script first):

```bash
bash benchmarks/basic_xcompact_sfm_3dgs_mlp_all.sh
```

Or run directly:

```bash
python simple_trainer.py default \
    --data_dir /path/to/XCompact_Dataset \
    --use_deformation \
    --holdout_conditions "9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121" \
    --reference_condition 64 \
    --deform_feature_dim 128 \
    --deform_hidden_dim 512 \
    --deform_start_step 30000 \
    --max_steps 110000 \
    --test_every 2 \
    --stage1_grow_grad2d 0.0002 --stage1_refine_every 100 --force_split \
    --hard_mining_exponent 1.5 \
    --learn_deform_sh --learn_deform_alpha \
    --white_bkgd \
    --deform_scale 1.0 \
    --deform_lr 0.0001 \
    --no-freeze_canonical_in_stage2 --stage2_canonical_lr_scale 0.001 \
    --result_dir results/xcompact/run1/ \
    --disable_viewer
```

### Key Training Arguments

| Argument | Description | Default |
|---|---|---|
| `--data_dir` | Path to dataset directory | required |
| `--use_deformation` | Enable deformation field (required for surrogate mode) | `False` |
| `--reference_condition` | Condition index for Stage 1 canonical training | `0` |
| `--deform_start_step` | Step at which Stage 2 (deformation) training begins | `5000` |
| `--max_steps` | Total training steps | `30000` |
| `--holdout_conditions` | Comma-separated condition indices held out for testing | `None` |
| `--deform_feature_dim` | Feature dimension for spatial/condition encoders | `64` |
| `--deform_hidden_dim` | Hidden dimension for deformation MLP | `128` |
| `--deform_lr` | Learning rate for deformation network | `1e-3` |
| `--deform_scale` | Scale factor applied to predicted deformation magnitudes | `0.1` |
| `--learn_deform_alpha` | Learn per-Gaussian opacity adjustments | `False` |
| `--learn_deform_sh` | Learn per-Gaussian SH (color) adjustments | `False` |
| `--force_split` | Error-guided force-splitting during Stage 1 | `False` |
| `--hard_mining_exponent` | Exponent for loss-weighted sampling in Stage 2 | `1.0` |
| `--test_every` | Every N cameras is a held-out validation view | `8` |

### Two-Stage Training

**Stage 1** (`step 0` → `--deform_start_step`): Standard 3DGS on the reference condition. Builds a dense canonical Gaussian field with aggressive densification and optional error-guided force-splitting.

**Stage 2** (`--deform_start_step` → `--max_steps`): Trains the deformation MLP on all ensemble conditions. Canonical Gaussians are jointly fine-tuned at a reduced learning rate. Loss-weighted hard mining focuses compute on difficult conditions.

## 📊 Evaluation

Evaluation runs automatically at the end of training, reporting a 2×2 matrix of metrics:

| | Seen Cameras | Unseen Cameras |
|---|---|---|
| **Train Conditions** | ✓ | ✓ |
| **Test Conditions (holdout)** | ✓ | ✓ |

Metrics reported: **PSNR**, **SSIM**, **LPIPS**.

Results are saved to `<result_dir>/stats/` as JSON and to TensorBoard at `<result_dir>/tb/`.

To run evaluation only from a saved checkpoint:

```bash
python simple_trainer.py default \
    --data_dir /path/to/dataset \
    --use_deformation \
    --ckpt results/xcompact/run1/ckpts/ckpt_109999_rank0.pt \
    --result_dir results/xcompact/eval/
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


