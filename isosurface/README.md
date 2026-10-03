# Isosurface Training

Isosurface extraction.

## Dataset Format

The dataset follows the main layout (`names.txt`, `sparse/0/`, one `pXXX` folder per condition), with an additional `isovalues.txt` listing one isovalue per line.

```
MyDataset/
├── names.txt
├── isovalues.txt
├── sparse/0/
├── p001/
└── ...
```

The `data/` folder contains the XCompact files for this layout: `names.txt` and `isovalues.txt`.

## Training

Edit `DATASET` in the script, then run it from this directory:

```bash
cd isosurface/
bash benchmarks/basic_xcompact_sfm_3dgs_mlp_all.sh
```

Relevant options:

| Option | Description |
|---|---|
| `--use_surface` | Enable isosurface mode |
| `--isovalues_file` | Path to `isovalues.txt` (default `isovalues.txt`) |
| `--holdout_conditions` | Simulation conditions held out for testing |
| `--holdout_isovalues` / `--num_holdout_isovalues` | Isovalue indices or number of isovalues held out for testing |
| `--reference_condition` | Condition used to train the canonical field in stage 1 |
| `--init_ckpt` | Optional canonical checkpoint. If given, stage 1 is skipped |
