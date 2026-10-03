# Transfer Function Adapter

Extends GS-Surrogate to support multiple transfer functions (TFs). A volume deformation model is trained first on the base TF. A TF appearance adapter is then trained on top of it, conditioned on both the simulation parameters and a 4-D TF vector, to predict per-Gaussian opacity and color (SH) adjustments. Geometry is left unchanged by the adapter.

## Pipeline

1. Train the volume deformation model with the main pipeline (../examples/) on the base TF.
2. Train the TF adapter with `simple_trainer.py --use_tf_adapter`. The volume model is loaded with `--pretrained_volume_ckpt` and kept frozen. Only the adapter is optimized.
3. Render any simulation parameter and TF combination with `inference_tf.py`.

## TF Format

Each TF is named `s1_<v>_o1_<v>_s2_<v>_o2_<v>` and is parsed into the 4-D vector `[s1, o1, s2, o2]`, for example:
s1_+0.000_o1_+0.000_s2_+0.000_o2_+0.125


The base TF is the all-zero vector.

## Dataset Format

The dataset follows the main layout (`names.txt`, `sparse/0/`, one `pXXX` folder per condition), with an additional `transfer_functions.txt` listing one TF name per line. Each condition folder contains one subfolder per TF, named after the TF.

```
MyDataset/
├── names.txt
├── transfer_functions.txt
├── sparse/0/
├── p001/
│   ├── s1_+0.000_o1_+0.000_s2_+0.000_o2_+0.000/
│   │   └── frame_0001.jpg ...
│   └── s1_+0.000_o1_+0.000_s2_+0.000_o2_+0.125/
│       └── ...
└── p002/
    └── ...
```

The `data/` folder contains the Nyx files for this layout: `names.txt`, `transfer_functions.txt` (77 TFs), and `nyx_holdout_tfs.txt` (indices of the TFs held out for testing, usable with `--holdout_tfs`).

## Training

Edit `DATASET` and `VOL_CKPT` in the script, then run it from this directory:
cd tf_adapter/
bash benchmarks/basic_nyx_sfm_3dgs_mlp_all.sh


Relevant options:

| Option | Description |
|---|---|
| `--use_tf_adapter` | Enable TF adapter training |
| `--tf_file` | Path to `transfer_functions.txt` |
| `--pretrained_volume_ckpt` | Trained volume model checkpoint |
| `--tf_member_fraction` | Fraction of ensemble members used for each non-base TF |
| `--num_holdout_tfs` / `--holdout_tfs` | Number or indices of TFs held out for testing |
| `--tf_adapter_reg` | Regularization weight on the adapter output |

## Inference

Edit the paths in `inference.sh`, then run:
bash inference.sh


`--sim_params` sets the simulation parameters and `--tf_residual` sets the 4-D TF vector to render.
