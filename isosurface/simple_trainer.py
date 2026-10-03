import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from fused_ssim import fused_ssim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap


# Conditional 4DGS imports
from conditional_dataset import ConditionalParser, ConditionalDataset
from conditional_dataset import SurfaceConditionalParser, SurfaceConditionalDataset
from conditional_dataset import GroupedSurfaceDataset
from deformation_model import create_surface_deformation_field
from deformation_model_volume import create_deformation_field as create_volume_deformation_field
CONDITIONAL_AVAILABLE = True


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 1
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Initialization strategy: "sfm" or "random"
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Stage-1 aggressive densification overrides (DefaultStrategy only)
    # Lower grow_grad2d → more splits in high-gradient regions
    stage1_grow_grad2d: float = 0.0001
    # Densify every N steps (default 100 is conservative)
    stage1_refine_every: int = 50
    # Disable pruning in stage 1 (only duplicate/split, never remove).
    # Useful when training canonical GS on diverse images (e.g. surface mode)
    # where Gaussians get low opacity because no single shape fits all conditions.
    stage1_disable_pruning: bool = False
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False
    # Use white background (for GT images rendered with white background)
    white_bkgd: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    # Whether use fused-bilateral grid
    use_fused_bilagrid: bool = False

    # ========================================================================
    # Deformable 4DGS Settings
    # ========================================================================
    # Enable conditional deformation field
    use_deformation: bool = False
    # Feature dimension for deformation network
    deform_feature_dim: int = 128
    # Hidden dimension for deformation MLP
    deform_hidden_dim: int = 512
    # Learning rate for deformation network
    deform_lr: float = 1e-3
    # Deformation magnitude scaling (smaller = more stable training)
    deform_scale: float = 0.1
    # Regularization weight for small deformations (0 = disabled)
    deform_reg: float = 0.0
    # Path to names.txt file containing condition vectors
    names_file: str = "names.txt"

    # ========================================================================
    # Surface / Isosurface Deformation Settings
    # ========================================================================
    # Enable surface (isosurface) mode with separate isovalue condition
    use_surface: bool = False
    # Path to isovalues.txt file
    isovalues_file: str = "isovalues.txt"
    # Number of isovalues to hold out for testing (0 = use all for training)
    num_holdout_isovalues: int = 0
    # Specific isovalue indices to hold out (comma-separated, overrides num_holdout_isovalues)
    holdout_isovalues: Optional[str] = None
    # Random seed for selecting holdout isovalues
    holdout_iso_seed: int = 42
    # Exponent for loss-weighted hard-example mining in stage 2
    # 0.0 = uniform, 0.5 = mild (default for volume), 1.0 = proportional, 2.0 = aggressive
    hard_mining_exponent: float = 1.0
    # Reference isovalue index for surface stage 1 (-1 = auto-pick middle isovalue)
    # Stage 1 trains canonical 3DGS on this single isovalue across all train params.
    reference_isovalue: int = -1
    # Enable multi-isovalue batched training in Stage 2 (surface mode only).
    # Renders all K isovalues per (param, camera) step, sharing the expensive
    # spatial + condition encoding via apply_deformation_multi_iso().
    multi_iso_batch: bool = True
    # Number of isovalues to render per step (0 or -1 = all K).
    # Randomly samples a subset each step to cut rasterisation cost.
    # Recommended: 3-4 for K=10 isovalues (~3× faster per step).
    multi_iso_K_sub: int = 0

    # ========================================================================
    # Chained Deformation (Volume → Surface)
    # ========================================================================
    # Path to pretrained volume model checkpoint (canonical splats + volume deformation).
    # When set with use_surface=True, enables chained deformation:
    #   Canonical → [Frozen Volume Deform (params)] → [Trainable Surface Deform (params+iso)]
    # Stage 1 is skipped entirely; canonical splats come from this checkpoint.
    pretrained_volume_ckpt: Optional[str] = None
    # Feature dim for the surface deformation field (can differ from volume)
    surface_deform_feature_dim: int = 128
    # Hidden dim for the surface deformation field
    surface_deform_hidden_dim: int = 512
    # Deformation scale for the surface deformation field
    surface_deform_scale: float = 0.1
    # Learning rate for the surface deformation field
    surface_deform_lr: float = 1e-3
    # Freeze volume deformation in stage 2 (chained mode only).
    # False = jointly fine-tune volume + surface deformation fields.
    freeze_volume_in_stage2: bool = True
    # Learning rate for volume deformation fine-tuning (when not frozen).
    # Typically much smaller than surface_deform_lr since it's already pretrained.
    volume_deform_lr: float = 1e-5

    # condition_dim: int = 3
    
    # ========================================================================
    # Two-Stage Training
    # ========================================================================
    # Reference condition index for stage 1 (canonical field training)
    # If set to -1, then use all conditions from the start (single-stage)
    reference_condition: int = 0
    # Step to start stage 2 (deformation training). Before this, only trains on reference.
    # If set to 0, then skip stage 1 (train deformation from the start)
    deform_start_step: int = 5000
    # Whether to freeze canonical Gaussians in stage 2
    freeze_canonical_in_stage2: bool = False
    # Learning rate multiplier for canonical Gaussians in stage 2 (1.0 = same as stage 1, 0.1 = 10x smaller)
    stage2_canonical_lr_scale: float = 0.1
    # Whether to learn per-Gaussian alpha (opacity adjustment) in deformation
    learn_deform_alpha: bool = False
    # Whether to learn per-Gaussian SH (color) adjustment in deformation
    learn_deform_sh: bool = False
    # Load checkpoint for splats only (to initialize canonical field from previous training)
    init_ckpt: Optional[str] = None
    
    # ========================================================================
    # Condition Train/Test Split
    # ========================================================================
    # Number of conditions to hold out for testing (0 = use all for training)
    num_holdout_conditions: int = 0
    # Specific condition indices to hold out (override num_holdout_conditions)
    # Example: "80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99"
    holdout_conditions: Optional[str] = None
    # Random seed for selecting holdout conditions (if num_holdout_conditions > 0)
    holdout_seed: int = 42

    # ========================================================================
    # Error-Guided Force Splitting (Stage 1)
    # ========================================================================
    # Enable periodic force-splitting of large Gaussians in high-error regions
    force_split: bool = False
    # Run force-split every N steps during stage 1
    force_split_every: int = 500
    # Don't force-split before this step (let normal densification warm up)
    force_split_start: int = 500
    # Number of views to render for error estimation
    force_split_num_views: int = 8
    # Percentile threshold for per-Gaussian error (higher = only split the worst)
    force_split_error_percentile: float = 85.0
    # Percentile threshold for Gaussian scale (higher = only split the largest)
    force_split_scale_percentile: float = 50.0
    # Maximum number of Gaussians to split per round
    force_split_max: int = 2000

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0, #0.1, #1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    # ------------------------------------------------------------------
    # Force-split helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
        """Quaternion (w,x,y,z) to rotation matrix.  q: (..., 4) → (..., 3, 3)."""
        q = F.normalize(q, dim=-1)
        w, x, y, z = q.unbind(-1)
        R = torch.stack([
            1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
            2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
            2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
        ], dim=-1).reshape(*q.shape[:-1], 3, 3)
        return R

    @torch.no_grad()
    def _append_gaussians(self, new_params: Dict[str, torch.Tensor]):
        """Append new Gaussians to splats and extend Adam optimizer states with zeros."""
        num_new = next(iter(new_params.values())).shape[0]
        if num_new == 0:
            return

        for name in list(new_params.keys()):
            old_param = self.splats[name]
            new_data = new_params[name].to(old_param.device)

            cat_data = torch.cat([old_param.data, new_data], dim=0)
            new_param = torch.nn.Parameter(cat_data)

            # Extend Adam optimizer state
            opt = self.optimizers[name]
            old_state = opt.state.get(old_param, {})
            new_state = {}
            for key, val in old_state.items():
                if (
                    isinstance(val, torch.Tensor)
                    and val.dim() > 0
                    and val.shape[0] == old_param.shape[0]
                ):
                    padding = torch.zeros(
                        num_new, *val.shape[1:], device=val.device, dtype=val.dtype
                    )
                    new_state[key] = torch.cat([val, padding], dim=0)
                else:
                    new_state[key] = val  # scalar tensor (step), int, etc.

            # Swap references
            if old_param in opt.state:
                del opt.state[old_param]
            opt.state[new_param] = new_state
            opt.param_groups[0]["params"] = [new_param]
            self.splats[name] = new_param

    @torch.no_grad()
    def _force_split_high_error(self, step: int, trainset, cfg):
 
        device = self.device
        means = self.splats["means"]
        N = means.shape[0]

        gauss_max_error = torch.zeros(N, device=device)

        # --- accumulate per-Gaussian error across random views ----------- #
        view_indices = torch.randperm(len(trainset))[: cfg.force_split_num_views]

        for idx in view_indices:
            data = trainset[int(idx)]
            c2w = data["camtoworld"].unsqueeze(0).to(device)
            K   = data["K"].unsqueeze(0).to(device)
            gt  = data["image"].unsqueeze(0).to(device) / 255.0
            h, w = gt.shape[1:3]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=c2w, Ks=K, width=w, height=h,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane, far_plane=cfg.far_plane,
                condition_vector=None,
            )
            error_map = (renders[..., 0:3] - gt).pow(2).mean(dim=-1)  # [1,H,W]

            # Project Gaussian centres → pixel coords
            w2c = torch.linalg.inv(c2w)          # [1,4,4]
            R_cam = w2c[0, :3, :3]                # [3,3]
            t_cam = w2c[0, :3, 3]                 # [3]
            pts_cam = means @ R_cam.T + t_cam     # [N,3]

            z = pts_cam[:, 2].clamp(min=1e-6)
            fx, fy = K[0, 0, 0], K[0, 1, 1]
            cx, cy = K[0, 0, 2], K[0, 1, 2]
            px = (pts_cam[:, 0] / z * fx + cx).long()
            py = (pts_cam[:, 1] / z * fy + cy).long()

            valid = (px >= 0) & (px < w) & (py >= 0) & (py < h) & (z > 0)
            vi = valid.nonzero(as_tuple=True)[0]

            err = torch.zeros(N, device=device)
            err[vi] = error_map[0, py[vi], px[vi]]
            gauss_max_error = torch.max(gauss_max_error, err)

        # --- select candidates ------------------------------------------- #
        active = gauss_max_error > 0
        if active.sum() == 0:
            return 0

        error_thresh = torch.quantile(
            gauss_max_error[active], cfg.force_split_error_percentile / 100.0
        )
        scales_val = torch.exp(self.splats["scales"].detach())
        scale_mag = scales_val.max(dim=-1).values
        scale_thresh = torch.quantile(
            scale_mag, cfg.force_split_scale_percentile / 100.0
        )

        split_mask = (gauss_max_error > error_thresh) & (scale_mag > scale_thresh)
        split_idx = split_mask.nonzero(as_tuple=True)[0]
        if len(split_idx) == 0:
            return 0

        # Prioritise by error * scale if too many
        if len(split_idx) > cfg.force_split_max:
            scores = gauss_max_error[split_idx] * scale_mag[split_idx]
            top_k = scores.topk(cfg.force_split_max).indices
            split_idx = split_idx[top_k]

        S = len(split_idx)

        # --- compute split geometry -------------------------------------- #
        scales_log = self.splats["scales"][split_idx].clone()       # [S,3]
        quats      = self.splats["quats"][split_idx].clone()        # [S,4]
        parent_xyz = self.splats["means"][split_idx].clone()        # [S,3]

        major_dim = scales_val[split_idx].argmax(dim=-1)            # [S]
        offset_mag = scales_val[split_idx].max(dim=-1).values * 0.5 # [S]

        R = self._quat_to_rotmat(quats)                             # [S,3,3]

        offset_local = torch.zeros(S, 3, device=device)
        offset_local[torch.arange(S, device=device), major_dim] = offset_mag
        offset_world = torch.bmm(R, offset_local.unsqueeze(-1)).squeeze(-1)

        child1_means = parent_xyz + offset_world
        child2_means = parent_xyz - offset_world
        child_scales = scales_log - math.log(1.6)

        # --- child 1: modify parent in-place ----------------------------- #
        self.splats["means"].data[split_idx]  = child1_means
        self.splats["scales"].data[split_idx] = child_scales

        # --- child 2: append new Gaussians ------------------------------- #
        new_params = {
            "means":     child2_means,
            "scales":    child_scales.clone(),
            "quats":     self.splats["quats"][split_idx].clone(),
            "opacities": self.splats["opacities"][split_idx].clone(),
            "sh0":       self.splats["sh0"][split_idx].clone(),
            "shN":       self.splats["shN"][split_idx].clone(),
        }
        self._append_gaussians(new_params)

        # --- reset strategy state so accumulators match new count -------- #
        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()

        # Inform strategy of new Gaussian count
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        return S

    def _parse_holdout_conditions(self, cfg: Config, num_conditions: int) -> List[int]:
        # Parse holdout conditions from config
        holdout_indices = []
        total_conds = num_conditions
        
        if cfg.holdout_conditions is not None:
            try:
                holdout_indices = [int(x.strip()) for x in cfg.holdout_conditions.split(",")] # Parse comma-separated list
                print(f"Using specified holdout conditions: {holdout_indices}")
            except ValueError:
                print(f"Warning: Could not parse holdout_conditions '{cfg.holdout_conditions}'. Using none.")
                holdout_indices = []
        elif cfg.num_holdout_conditions > 0:
            # Randomly select conditions to hold out
            import random
            rng = random.Random(cfg.holdout_seed)
            all_conditions = list(range(total_conds))  
            
            # Don't include reference condition in random selection
            selectable = [c for c in all_conditions if c != cfg.reference_condition]
            holdout_indices = sorted(rng.sample(selectable, min(cfg.num_holdout_conditions, len(selectable))))
            print(f"Randomly selected {len(holdout_indices)} holdout conditions (seed={cfg.holdout_seed})")
        
        return holdout_indices

    def _parse_holdout_isovalues(self, cfg: Config, num_isovalues: int) -> List[int]:
        """Parse holdout isovalue indices from config."""
        holdout_indices = []
        
        if cfg.holdout_isovalues is not None:
            try:
                holdout_indices = [int(x.strip()) for x in cfg.holdout_isovalues.split(",")]
                print(f"Using specified holdout isovalues: {holdout_indices}")
            except ValueError:
                print(f"Warning: Could not parse holdout_isovalues '{cfg.holdout_isovalues}'. Using none.")
                holdout_indices = []
        elif cfg.num_holdout_isovalues > 0:
            import random
            rng = random.Random(cfg.holdout_iso_seed)
            all_isos = list(range(num_isovalues))
            holdout_indices = sorted(rng.sample(all_isos, min(cfg.num_holdout_isovalues, len(all_isos))))
            print(f"Randomly selected {len(holdout_indices)} holdout isovalues (seed={cfg.holdout_iso_seed})")
        
        return holdout_indices

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # ====================================================================
        # Load dataset (conditional or standard on scene)
        # ====================================================================
        self.condition_dim = 0  # Default: no conditions
        self.isovalue_dim = 0   # Default: no isovalues
        self.train_conditions = None  # Conditions used for training
        self.test_conditions = None   # Conditions held out for testing
        self.train_isovalues = None   # Isovalues used for training
        self.test_isovalues = None    # Isovalues held out for testing
        
        if cfg.use_deformation and cfg.use_surface and CONDITIONAL_AVAILABLE:
            # ============================================================
            # Surface mode: dual condition (sim params + isovalue)
            # ============================================================
            print("Loading surface conditional dataset...")
            
            self.parser = SurfaceConditionalParser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=cfg.normalize_world_space,
                test_every=cfg.test_every,
                names_file=cfg.names_file,
                isovalues_file=cfg.isovalues_file,
            )
            
            # Parse holdout conditions (sim params)
            holdout_param_indices = self._parse_holdout_conditions(cfg, self.parser.num_conditions)
            
            # Parse holdout isovalues
            holdout_iso_indices = self._parse_holdout_isovalues(cfg, self.parser.num_isovalues)
            
            # Determine train/test splits for BOTH axes
            all_params = list(range(self.parser.num_conditions))
            all_isos = list(range(self.parser.num_isovalues))
            
            if holdout_param_indices:
                self.test_conditions = sorted(holdout_param_indices)
                self.train_conditions = sorted(set(all_params) - set(holdout_param_indices))
                # Ensure reference condition is in training
                if cfg.reference_condition >= 0 and cfg.reference_condition not in self.train_conditions:
                    self.train_conditions.append(cfg.reference_condition)
                    self.train_conditions = sorted(self.train_conditions)
                    self.test_conditions = [c for c in self.test_conditions if c != cfg.reference_condition]
            else:
                self.train_conditions = all_params
                self.test_conditions = []
            
            if holdout_iso_indices:
                self.test_isovalues = sorted(holdout_iso_indices)
                self.train_isovalues = sorted(set(all_isos) - set(holdout_iso_indices))
            else:
                self.train_isovalues = all_isos
                self.test_isovalues = []
            
            print(f"Param split:    {len(self.train_conditions)} train, {len(self.test_conditions)} test")
            print(f"Isovalue split: {len(self.train_isovalues)} train, {len(self.test_isovalues)} test")
            if self.test_conditions:
                print(f"  Holdout params: {self.test_conditions}")
            if self.test_isovalues:
                print(f"  Holdout isos:   {self.test_isovalues}")
            
            # Training set: train params × train isovalues
            self.trainset = SurfaceConditionalDataset(
                self.parser, split="train", patch_size=cfg.patch_size,
                param_indices=self.train_conditions,
                isovalue_indices=self.train_isovalues,
            )
            # Validation set: train params × train isos, unseen views
            self.valset = SurfaceConditionalDataset(
                self.parser, split="val",
                param_indices=self.train_conditions,
                isovalue_indices=self.train_isovalues,
            )
            
            # Train conditions on unseen views
            self.testset_train_unseen = self.valset
            # Train conditions on seen views
            self.testset_train_seen = SurfaceConditionalDataset(
                self.parser, split="train",
                param_indices=self.train_conditions,
                isovalue_indices=self.train_isovalues,
            )
            
            # --- Holdout params × Train isos ---
            if self.test_conditions:
                self.testset_holdout_param_unseen = SurfaceConditionalDataset(
                    self.parser, split="val",
                    param_indices=self.test_conditions,
                    isovalue_indices=self.train_isovalues,
                )
                self.testset_holdout_param_seen = SurfaceConditionalDataset(
                    self.parser, split="train",
                    param_indices=self.test_conditions,
                    isovalue_indices=self.train_isovalues,
                )
            else:
                self.testset_holdout_param_unseen = None
                self.testset_holdout_param_seen = None
            
            # --- Train params × Holdout isos ---
            if self.test_isovalues:
                self.testset_holdout_iso_unseen = SurfaceConditionalDataset(
                    self.parser, split="val",
                    param_indices=self.train_conditions,
                    isovalue_indices=self.test_isovalues,
                )
                self.testset_holdout_iso_seen = SurfaceConditionalDataset(
                    self.parser, split="train",
                    param_indices=self.train_conditions,
                    isovalue_indices=self.test_isovalues,
                )
            else:
                self.testset_holdout_iso_unseen = None
                self.testset_holdout_iso_seen = None
            
            # --- Holdout params × Holdout isos (hardest) ---
            if self.test_conditions and self.test_isovalues:
                self.testset_holdout_both_unseen = SurfaceConditionalDataset(
                    self.parser, split="val",
                    param_indices=self.test_conditions,
                    isovalue_indices=self.test_isovalues,
                )
                self.testset_holdout_both_seen = SurfaceConditionalDataset(
                    self.parser, split="train",
                    param_indices=self.test_conditions,
                    isovalue_indices=self.test_isovalues,
                )
            else:
                self.testset_holdout_both_unseen = None
                self.testset_holdout_both_seen = None
            
            # For backward compat with existing eval code
            self.testset_holdout_unseen = self.testset_holdout_param_unseen
            self.testset_holdout_seen = self.testset_holdout_param_seen
            
            self.condition_dim = self.parser.condition_dim
            self.isovalue_dim = 1  # Always 1 for surface mode
            print(f"Surface mode: {self.parser.num_conditions} params (dim={self.condition_dim}), "
                  f"{self.parser.num_isovalues} isovalues")
            print(f"  Training samples: {len(self.trainset)}")
            print(f"  Validation samples: {len(self.valset)}")

        elif cfg.use_deformation and CONDITIONAL_AVAILABLE:
            print("Loading conditional dataset...")
            
            self.parser = ConditionalParser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=cfg.normalize_world_space,
                test_every=cfg.test_every,
                names_file=cfg.names_file,
                # condition_dim=cfg.condition_dim,
            )
            
            # Parse holdout conditions (after parser so we know num_conditions)
            holdout_condition_indices = self._parse_holdout_conditions(cfg, self.parser.num_conditions)
            
            # Determine train/test condition split
            all_conditions = list(range(self.parser.num_conditions))
            if holdout_condition_indices:
                self.test_conditions = sorted(holdout_condition_indices)
                self.train_conditions = sorted(set(all_conditions) - set(holdout_condition_indices))
                
                # Ensure reference condition is in training set
                if cfg.reference_condition not in self.train_conditions:
                    print(f"WARNING: Reference condition {cfg.reference_condition} is in holdout set!")
                    print(f"         Moving it to training set.")
                    self.train_conditions.append(cfg.reference_condition)
                    self.train_conditions = sorted(self.train_conditions)
                    self.test_conditions = [c for c in self.test_conditions if c != cfg.reference_condition]
            else:   # w/o test
                self.train_conditions = all_conditions
                self.test_conditions = []
            
            print(f"Condition split: {len(self.train_conditions)} train, {len(self.test_conditions)} test")
            if self.test_conditions:
                print(f"  Train conditions: {self.train_conditions[:5]}...{self.train_conditions[-5:] if len(self.train_conditions) > 10 else ''}")
                print(f"  Test conditions (holdout): {self.test_conditions}")
            
            # Training set
            self.trainset = ConditionalDataset(
                self.parser,
                split="train",
                patch_size=cfg.patch_size, # ignore 
                condition_indices=self.train_conditions,  # Only train conditions
            )
            # Validation set = train conditions, unseen views (for eval during training)
            self.valset = ConditionalDataset(
                self.parser, 
                split="val",
                condition_indices=self.train_conditions,  # Val on train conditions
            )
            
            # Train conditions on SEEN views
            self.testset_train_seen = ConditionalDataset(
                self.parser,
                split="train",  # Same cameras as training
                condition_indices=self.train_conditions,
            )
            # Train conditions on UNSEEN views
            self.testset_train_unseen = self.valset  # Same as valset
            
            print(f"  Train condition test sets:")
            print(f"    - Seen views (train cameras): {len(self.testset_train_seen)} samples")
            print(f"    - Unseen views (val cameras): {len(self.testset_train_unseen)} samples")
            
            # Create test datasets for holdout conditions (if we have)
            if self.test_conditions:
                # Holdout with UNSEEN camera views (val split - same as training eval)
                self.testset_holdout_unseen = ConditionalDataset(
                    self.parser,
                    split="val",  # held-out cameras (unseen during training)
                    condition_indices=self.test_conditions,
                )
                # Holdout with SEEN camera views (train split - cameras used in training)
                self.testset_holdout_seen = ConditionalDataset(
                    self.parser,
                    split="train",  # train cameras (seen during training)
                    condition_indices=self.test_conditions,
                )
                print(f"  Holdout condition test sets:")
                print(f"    - Seen views (train cameras): {len(self.testset_holdout_seen)} samples")
                print(f"    - Unseen views (val cameras): {len(self.testset_holdout_unseen)} samples")
            else:
                self.testset_holdout_unseen = None
                self.testset_holdout_seen = None
            
            self.condition_dim = self.parser.condition_dim
            print(f"Conditional mode: {self.parser.num_conditions} total conditions, dim={self.condition_dim}")
            print(f"  Training samples: {len(self.trainset)}")
            print(f"  Validation samples: {len(self.valset)}")
        else:
            # Standard single-scene loading
            if cfg.use_deformation and not CONDITIONAL_AVAILABLE:
                print("Warning: use_deformation=True but conditional modules not found. Using standard mode.")
            
            self.parser = Parser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=cfg.normalize_world_space,
                test_every=cfg.test_every,
            )
            self.trainset = Dataset(
                self.parser,
                split="train",
                patch_size=cfg.patch_size,
                load_depths=cfg.depth_loss,
            )
            self.valset = Dataset(self.parser, split="val")
        
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # ====================================================================
        # Initialize Deformation Field(s)
        # ====================================================================
        self.deform_field = None
        self.deform_optimizers = []
        
        if cfg.use_deformation and CONDITIONAL_AVAILABLE and self.condition_dim > 0:

            if cfg.use_surface:
                # ── Surface mode: canonical splats + surface deformation ──
                # If a pretrained checkpoint is provided, load canonical splats
                # from it and skip stage 1. Otherwise, stage 1 will build the
                # canonical field and stage 2 trains the surface deformation.
                _canon_ckpt_path = cfg.pretrained_volume_ckpt or cfg.init_ckpt
                if _canon_ckpt_path is not None:
                    print(f"\n{'='*60}")
                    print("SURFACE DEFORMATION: Loading canonical checkpoint")
                    print(f"{'='*60}")
                    
                    print(f"Loading canonical ckpt: {_canon_ckpt_path}")
                    canon_ckpt = torch.load(_canon_ckpt_path, map_location=self.device, weights_only=True)
                    
                    # Load canonical splats
                    if "splats" in canon_ckpt:
                        for k in self.splats.keys():
                            if k in canon_ckpt["splats"]:
                                self.splats[k].data = canon_ckpt["splats"][k]
                        print(f"  Loaded canonical Gaussians: {len(self.splats['means'])} points")
                    else:
                        print(f"  WARNING: No splats in checkpoint!")
                    
                    # Freeze or fine-tune canonical Gaussians
                    if cfg.freeze_canonical_in_stage2:
                        for p in self.splats.values():
                            p.requires_grad = False
                        print(f"  Canonical Gaussians: FROZEN ({len(self.splats['means'])} points)")
                    else:
                        print(f"  Canonical Gaussians: FINE-TUNING (lr_scale={cfg.stage2_canonical_lr_scale}, {len(self.splats['means'])} points)")
                
                # Create surface deformation field (params + isovalue)
                self.deform_field = create_surface_deformation_field(
                    condition_dim=self.condition_dim,
                    isovalue_dim=self.isovalue_dim,
                    feature_dim=cfg.surface_deform_feature_dim,
                    hidden_dim=cfg.surface_deform_hidden_dim,
                    deform_scale=cfg.surface_deform_scale,
                    scene_scale=self.scene_scale,
                    learn_alpha=cfg.learn_deform_alpha,
                    learn_sh=cfg.learn_deform_sh,
                    sh_dim=((cfg.sh_degree + 1) ** 2) * 3,
                ).to(self.device)
                
                self.deform_optimizers = [
                    torch.optim.Adam(
                        self.deform_field.parameters(),
                        lr=cfg.surface_deform_lr * math.sqrt(cfg.batch_size),
                    )
                ]
                
                surf_params = sum(p.numel() for p in self.deform_field.parameters())
                print(f"  Surface deform: {surf_params:,} params (TRAINABLE)")
                print(f"  Condition dim: {self.condition_dim}, Isovalue dim: {self.isovalue_dim}")
                if _canon_ckpt_path is not None:
                    print(f"{'='*60}\n")

            else:
                # ── Volume-only mode: single deformation field (no isovalue) ──
                self.deform_field = create_volume_deformation_field(
                    condition_dim=self.condition_dim,
                    feature_dim=cfg.deform_feature_dim,
                    hidden_dim=cfg.deform_hidden_dim,
                    deform_scale=cfg.deform_scale,
                    scene_scale=self.scene_scale,
                    learn_alpha=cfg.learn_deform_alpha,
                    learn_sh=cfg.learn_deform_sh,
                    sh_dim=((cfg.sh_degree + 1) ** 2) * 3,
                ).to(self.device)
                
                self.deform_optimizers = [
                    torch.optim.Adam(
                        self.deform_field.parameters(),
                        lr=cfg.deform_lr * math.sqrt(cfg.batch_size),
                    )
                ]
                
                num_params = sum(p.numel() for p in self.deform_field.parameters())
                print(f"Volume deformation field: {num_params:,} parameters")
                print(f"  Condition dim: {self.condition_dim}")

        # ====================================================================
        # Load initial checkpoint (for canonical Gaussians)
        # This allows starting stage 2 with pre-trained reference/canonical field
        # Skipped in surface mode (already loaded above) and chained mode
        # ====================================================================
        if cfg.init_ckpt is not None and not cfg.use_surface and cfg.pretrained_volume_ckpt is None:
            print(f"\nLoading initial checkpoint from: {cfg.init_ckpt}")
            ckpt = torch.load(cfg.init_ckpt, map_location=self.device, weights_only=True)
            
            # Load splats (canonical Gaussians)
            if "splats" in ckpt:
                for k in self.splats.keys():
                    if k in ckpt["splats"]:
                        self.splats[k].data = ckpt["splats"][k]
                print(f"  Loaded canonical Gaussians: {len(self.splats['means'])} points")
            
            # Optionally: load deformation field if it exists
            if cfg.use_deformation and self.deform_field is not None:
                if "deform_field" in ckpt:
                    self.deform_field.load_state_dict(ckpt["deform_field"])
                    print("  Loaded deformation field")
                else:
                    print("  No deformation field in checkpoint (will train from scratch)")
            
            print("")

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

        # ====================================================================
        # Report Model Statistics
        # ====================================================================
        self._report_model_stats()

    def _report_model_stats(self):
        """Report number of parameters, model size, and memory usage."""
        cfg = self.cfg
        
        print(f"\n{'='*70}")
        print("MODEL STATISTICS")
        print(f"{'='*70}")
        
        # Canonical Gaussians
        num_gaussians = len(self.splats["means"])
        gaussian_params = sum(p.numel() for p in self.splats.values())
        gaussian_size_mb = sum(p.numel() * p.element_size() for p in self.splats.values()) / (1024**2)
        
        print(f"\n[Canonical Gaussians]")
        print(f"  Number of Gaussians: {num_gaussians:,}")
        print(f"  Parameters: {gaussian_params:,}")
        print(f"  Model size: {gaussian_size_mb:.2f} MB")
        
        # Per-Gaussian breakdown
        print(f"  Breakdown:")
        for name, param in self.splats.items():
            param_count = param.numel()
            param_size = param.numel() * param.element_size() / (1024**2)
            print(f"    {name}: {list(param.shape)} = {param_count:,} params ({param_size:.2f} MB)")
        
        # Deformation Field
        if self.deform_field is not None:
            deform_params = sum(p.numel() for p in self.deform_field.parameters())
            deform_trainable = sum(p.numel() for p in self.deform_field.parameters() if p.requires_grad)
            deform_size_mb = sum(p.numel() * p.element_size() for p in self.deform_field.parameters()) / (1024**2)
            
            label = "Surface Deformation" if cfg.use_surface else "Deformation Model"
            print(f"\n[{label}]")
            print(f"  Total parameters: {deform_params:,}")
            print(f"  Trainable parameters: {deform_trainable:,}")
            print(f"  Model size: {deform_size_mb:.2f} MB")
            print(f"  Learn alpha: {cfg.learn_deform_alpha}")
            print(f"  Learn SH: {cfg.learn_deform_sh}")
        
        # Total:
        total_params = gaussian_params
        total_trainable = sum(p.numel() for p in self.splats.values() if p.requires_grad)
        total_size_mb = gaussian_size_mb
        
        if self.deform_field is not None:
            total_params += deform_params
            total_trainable += deform_trainable
            total_size_mb += deform_size_mb
        
        print(f"\n[Total]")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Total trainable: {total_trainable:,}")
        print(f"  Total model size: {total_size_mb:.2f} MB")
        
        # GPU memory:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
            
            print(f"\n[GPU Memory]")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
            print(f"  Max allocated: {max_allocated:.2f} GB")
        
        print(f"{'='*70}\n")

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        condition_vector: Optional[Tensor] = None,  # Simulation parameters
        isovalue: Optional[Tensor] = None,           # Isovalue (surface mode)
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = self.splats["scales"]  # [N, 3]
        opacities_logit = self.splats["opacities"]  # [N,]

        # ====================================================================
        # Apply deformation (if enabled)
        # ====================================================================
        deformed_sh = None
        if self.deform_field is not None and condition_vector is not None:
            # Use first condition in batch (assume batch has same condition)
            cond = condition_vector[0] if condition_vector.dim() > 1 else condition_vector
            iso = None
            if isovalue is not None:
                iso = isovalue[0] if isovalue.dim() > 1 else isovalue
            sh_coeffs = torch.cat([self.splats["sh0"], self.splats["shN"]], 1) if self.cfg.learn_deform_sh else None
            
            # Apply deformation (surface or volume — single field)
            if iso is not None:
                # Surface mode: pass isovalue
                means, quats, scales, opacities_logit, deformed_sh = self.deform_field.apply_deformation(
                    means, quats, scales, opacities_logit.unsqueeze(-1), cond, isovalue=iso, sh=sh_coeffs
                )
            else:
                # Volume mode: no isovalue
                means, quats, scales, opacities_logit, deformed_sh = self.deform_field.apply_deformation(
                    means, quats, scales, opacities_logit.unsqueeze(-1), cond, sh=sh_coeffs
                )
            opacities_logit = opacities_logit.squeeze(-1)  # [N,]
        
        # Convert from log/logit space AFTER deformation
        scales = torch.exp(scales)  # [N, 3]
        opacities = torch.sigmoid(opacities_logit)  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            if deformed_sh is not None:
                colors = deformed_sh  # [N, K, 3]
            else:
                colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info


    def train(self):
        cfg = self.cfg  
        # Dump cfg
        if self.world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        # Choose training mode
        if cfg.use_deformation and CONDITIONAL_AVAILABLE and (
            cfg.reference_condition >= 0 or cfg.use_surface
        ):
            # Two-stage training for conditional 4DGS
            # Surface mode: stage 1 trains on ALL images (reference_condition irrelevant)
            # Non-surface mode: stage 1 trains on reference_condition only
            self._train_two_stage()
        else:
            # Single-stage training (original 3DGS or deformation from start)
            self._train_single_stage()

    def _train_two_stage(self):
        """
        Two-stage training for deformable 3DGS:
        - Stage 1: Standard 3DGS on reference condition only (or multi-ref merge in surface mode)
        - Stage 2: Train deformation on all conditions
        
        If a pretrained canonical checkpoint was loaded (via pretrained_volume_ckpt
        or init_ckpt in surface mode), stage 1 is skipped.
        """
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        
        print(f"\n{'='*70}")
        
        _canon_ckpt_path = cfg.pretrained_volume_ckpt or (cfg.init_ckpt if cfg.use_surface else None)
        if _canon_ckpt_path is not None and cfg.use_surface:
            # ── Surface mode with pretrained canonical: skip stage 1 ──
            print("SURFACE DEFORMATION TRAINING (canonical from checkpoint)")
            print(f"{'='*70}")
            print(f"Stage 1: SKIPPED (canonical splats from pretrained ckpt)")
            print(f"Stage 2: Train SURFACE deformation (params + isovalue)")
            print(f"         {cfg.max_steps} steps")
            if cfg.freeze_canonical_in_stage2:
                print(f"         Canonical GS:  FROZEN ({len(self.splats['means'])} points)")
            else:
                print(f"         Canonical GS:  FINE-TUNING (lr_scale={cfg.stage2_canonical_lr_scale}, {len(self.splats['means'])} points)")
            print(f"{'='*70}\n")
            
            # Override: all steps go to stage 2
            cfg.deform_start_step = 0
            
            self._run_stage2()
        else:
            # ── Normal two-stage ──
            print("TWO-STAGE CONDITIONAL 4DGS TRAINING")
            print(f"{'='*70}")
            if cfg.use_surface:
                print(f"Stage 1: Multi-reference merge (surface mode)")
                print(f"         {len(self.train_isovalues)} isos × {len(self.train_conditions)} params, independent 3DGS each")
            else:
                print(f"Stage 1: Standard 3DGS on condition {cfg.reference_condition}")
            print(f"         Steps 0 to {cfg.deform_start_step - 1}")
            print(f"Stage 2: Deformation training on all {self.parser.num_conditions} conditions")
            print(f"         Steps {cfg.deform_start_step} to {cfg.max_steps - 1}")
            if cfg.freeze_canonical_in_stage2:
                print("         (Canonical Gaussians FROZEN in Stage 2)")
            print(f"{'='*70}\n")
            
            # ==================== Stage 1 ====================
            if cfg.use_surface:
                self._run_surface_stage1()
            else:
                self._run_stage1()
            
            # ==================== Stage 2 ====================
            self._run_stage2()

    def _run_stage1(self):
        """
        Stage 1: Train canonical 3DGS on reference condition only (standard 3DGS training).
        """
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        
        stage1_steps = cfg.deform_start_step
        
        if stage1_steps == 0:
            print(f"\n{'='*60}")
            print(f"STAGE 1: Skipped (deform_start_step=0)")
            print(f"{'='*60}\n")
            return
        
        print(f"\n{'='*60}")
        print(f"STAGE 1: Canonical 3DGS on Condition {cfg.reference_condition}")
        print(f"         {stage1_steps} steps")
        print(f"{'='*60}\n")
        
        # Override densification strategy for aggressive stage-1 splitting
        if isinstance(cfg.strategy, DefaultStrategy):
            # Save original values to restore after stage 1
            _orig_grow_grad2d = cfg.strategy.grow_grad2d
            _orig_refine_every = cfg.strategy.refine_every
            _orig_refine_stop_iter = cfg.strategy.refine_stop_iter
            _orig_prune_opa = None
            _orig_prune_scale3d = None
            
            cfg.strategy.grow_grad2d = cfg.stage1_grow_grad2d
            cfg.strategy.refine_every = cfg.stage1_refine_every
            # Ensure densification covers ~80% of stage 1
            cfg.strategy.refine_stop_iter = int(stage1_steps * 0.8)
            
            if cfg.stage1_disable_pruning:
                _orig_prune_opa = cfg.strategy.prune_opa
                _orig_prune_scale3d = cfg.strategy.prune_scale3d
                cfg.strategy.prune_opa = 0.0
                cfg.strategy.prune_scale3d = 1e10
            
            print(f"Stage 1 densification overrides:")
            print(f"  grow_grad2d:    {_orig_grow_grad2d} -> {cfg.strategy.grow_grad2d}")
            print(f"  refine_every:   {_orig_refine_every} -> {cfg.strategy.refine_every}")
            print(f"  refine_stop_iter: {_orig_refine_stop_iter} -> {cfg.strategy.refine_stop_iter}")
            if cfg.stage1_disable_pruning:
                print(f"  prune_opa:      {_orig_prune_opa} -> {cfg.strategy.prune_opa} (DISABLED)")
                print(f"  prune_scale3d:  {_orig_prune_scale3d} -> {cfg.strategy.prune_scale3d} (DISABLED)")
            print()

        stage1_trainset = ConditionalDataset(
            self.parser, split="train", patch_size=cfg.patch_size,
            condition_indices=self.train_conditions, 
        )
        
        # stage1_trainset = ConditionalDataset(
        #     self.parser, split="train", patch_size=cfg.patch_size,
        #     condition_indices=[97,183,132,219,156],
        # )
 
        # stage1_trainset = ConditionalDataset(
        #     self.parser, split="train", patch_size=cfg.patch_size,
        #     # condition_indices=[cfg.reference_condition],
        #     condition_filter=cfg.reference_condition,
        # )


        ####### Validation datasets ######

        stage1_valset = ConditionalDataset(
            self.parser, split="val",
            condition_filter=cfg.reference_condition,
        )

        print(f"Stage 1 train: {len(stage1_trainset)} samples")
        print(f"Stage 1 val: {len(stage1_valset)} samples")
        
        trainloader = torch.utils.data.DataLoader(
            stage1_trainset, batch_size=cfg.batch_size, shuffle=True,
            num_workers=4, persistent_workers=True, pin_memory=True,
        )
        trainloader_iter = iter(trainloader)
        
        # Schedulers for stage-1
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / stage1_steps)
            ),
        ]
        if cfg.pose_opt:
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / stage1_steps)
                )
            )
        
        # Training loop
        pbar = tqdm.tqdm(range(stage1_steps), desc="Stage 1")
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            num_train_rays_per_step = pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # Forward - NO deformation in Stage-1
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds, Ks=Ks, width=width, height=height,
                sh_degree=sh_degree_to_use, near_plane=cfg.near_plane,
                far_plane=cfg.far_plane, image_ids=image_ids,
                condition_vector=None,  # No deformation!!!
                render_mode="RGB", masks=masks,
            )
            colors = renders[..., 0:3]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)
            elif cfg.white_bkgd:
                colors = colors + (1.0 - alphas)

            self.cfg.strategy.step_pre_backward(
                params=self.splats, optimizers=self.optimizers,
                state=self.strategy_state, step=step, info=info,
            )

            # Loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            if cfg.opacity_reg > 0.0:
                loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            if cfg.scale_reg > 0.0:
                loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

            loss.backward()
            pbar.set_description(f"Stage 1 | loss={loss.item():.4f} | GS={len(self.splats['means'])}")

            # Logging
            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                self.writer.add_scalar("stage1/loss", loss.item(), step)
                self.writer.add_scalar("stage1/l1loss", l1loss.item(), step)
                self.writer.add_scalar("stage1/num_GS", len(self.splats["means"]), step)
                self.writer.flush()

            # Optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Densification
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats, optimizers=self.optimizers,
                    state=self.strategy_state, step=step, info=info, packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats, optimizers=self.optimizers,
                    state=self.strategy_state, step=step, info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )

            # Error-guided force splitting
            if (
                cfg.force_split
                and step >= cfg.force_split_start
                and step % cfg.force_split_every == 0
                and step < int(stage1_steps * 0.8)  # stop before end
            ):
                n_split = self._force_split_high_error(
                    step, stage1_trainset, cfg
                )
                if n_split > 0:
                    print(f"\n  [Force-split @ step {step}] "
                          f"Split {n_split} Gaussians → "
                          f"total {len(self.splats['means'])}\n")

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                try:
                    num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                    num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
                    self.viewer.render_tab_state.num_train_rays_per_sec = num_train_rays_per_sec
                    self.viewer.update(step, num_train_rays_per_step)
                except Exception as e:
                    pass  # Viewer update is optional

        # Stage 1 evaluation
        print(f"\n--- Stage 1 Complete: Evaluating on reference condition ---")
        self._eval_stage1(stage1_steps - 1, stage1_valset)
        
        # Report updated stats after densification
        self._report_model_stats()
        
        # Save Stage 1 checkpoint
        ckpt_data = {"step": stage1_steps - 1, "splats": self.splats.state_dict(), "stage": 1}
        torch.save(ckpt_data, f"{self.ckpt_dir}/ckpt_stage1_rank{self.world_rank}.pt")
        print(f"Stage 1 checkpoint saved\n")
        
        # Restore original strategy params (densification is off in stage 2 anyway,
        # but keeps state clean)
        if isinstance(cfg.strategy, DefaultStrategy):
            cfg.strategy.grow_grad2d = _orig_grow_grad2d
            cfg.strategy.refine_every = _orig_refine_every
            cfg.strategy.refine_stop_iter = _orig_refine_stop_iter
            if _orig_prune_opa is not None:
                cfg.strategy.prune_opa = _orig_prune_opa
                cfg.strategy.prune_scale3d = _orig_prune_scale3d

    def _rebuild_splats_and_optimizers(self, tensor_dict: Dict[str, torch.Tensor]):

        cfg = self.cfg
        device = self.device
        
        lr_map = {
            "means": cfg.means_lr * self.scene_scale,
            "scales": cfg.scales_lr,
            "quats": cfg.quats_lr,
            "opacities": cfg.opacities_lr,
            "sh0": cfg.sh0_lr,
            "shN": cfg.shN_lr,
        }
        
        # Build new ParameterDict
        new_splats = torch.nn.ParameterDict()
        for name, tensor in tensor_dict.items():
            new_splats[name] = torch.nn.Parameter(tensor.to(device))
        new_splats = new_splats.to(device)
        
        # Build new optimizers
        BS = cfg.batch_size * self.world_size
        if cfg.sparse_grad:
            optimizer_class = torch.optim.SparseAdam
        elif cfg.visible_adam:
            optimizer_class = SelectiveAdam
        else:
            optimizer_class = torch.optim.Adam
        
        new_optimizers = {}
        for name in new_splats.keys():
            lr = lr_map.get(name, cfg.sh0_lr)  # fallback for unknown keys
            new_optimizers[name] = optimizer_class(
                [{"params": new_splats[name], "lr": lr * math.sqrt(BS), "name": name}],
                eps=1e-15 / math.sqrt(BS),
                betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            )
        
        self.splats = new_splats
        self.optimizers = new_optimizers
        
        # Reset strategy state
        cfg.strategy.check_sanity(self.splats, self.optimizers)
        if isinstance(cfg.strategy, DefaultStrategy):
            self.strategy_state = cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(cfg.strategy, MCMCStrategy):
            self.strategy_state = cfg.strategy.initialize_state()

    def _run_surface_stage1(self):
        
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size
        
        total_steps = cfg.deform_start_step
        
        if total_steps == 0:
            print(f"\n{'='*60}")
            print(f"STAGE 1 (Surface): Skipped (deform_start_step=0)")
            print(f"{'='*60}\n")
            return
        
        num_isos = len(self.train_isovalues)
        per_iso_steps = total_steps // num_isos
        
        iso_values = [
            self.parser.isovalues[i] if hasattr(self.parser, 'isovalues') else i
            for i in self.train_isovalues
        ]
        
        print(f"\n{'='*60}")
        print(f"STAGE 1 (Surface): Multi-Reference Merge")
        print(f"  {num_isos} isovalues × {per_iso_steps} steps each = {num_isos * per_iso_steps} total")
        print(f"  {len(self.train_conditions)} training params per isovalue")
        print(f"  Isovalues: {[f'{v:.4f}' for v in iso_values]}")
        print(f"{'='*60}\n")
        
        # Collect Gaussians from each isovalue run
        collected_splats: List[Dict[str, torch.Tensor]] = []
        
        for iso_run, iso_idx in enumerate(self.train_isovalues):
            iso_val = iso_values[iso_run]
            
            # ── Get per-isovalue points or fall back to default ──
            has_per_iso = (hasattr(self.parser, 'points_per_iso') 
                           and iso_idx in self.parser.points_per_iso)
            if has_per_iso:
                pts_np, rgbs_np = self.parser.points_per_iso[iso_idx]
                pts_source = f"points3D_{iso_val}.bin ({len(pts_np)} pts)"
            else:
                pts_source = f"random init ({cfg.init_num_pts} pts)"
            
            print(f"\n{'─'*50}")
            print(f"  Isovalue {iso_run+1}/{num_isos}: index={iso_idx}, value={iso_val:.4f}")
            print(f"  Init from: {pts_source}")
            print(f"  Training {per_iso_steps} steps")
            print(f"{'─'*50}")
            
            # ── Create fresh splats from this isovalue's points ──
            feature_dim = 32 if cfg.app_opt else None
            
            if has_per_iso:
                points = torch.from_numpy(pts_np).float()
                rgbs = torch.from_numpy(rgbs_np / 255.0).float()
                
                dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
                dist_avg = torch.sqrt(dist2_avg)
                scales = torch.log(dist_avg * cfg.init_scale).unsqueeze(-1).repeat(1, 3)
                
                # Distribute across ranks
                points = points[self.world_rank::self.world_size]
                rgbs = rgbs[self.world_rank::self.world_size]
                scales = scales[self.world_rank::self.world_size]
                
                N = points.shape[0]
                quats = torch.rand((N, 4))
                opacities = torch.logit(torch.full((N,), cfg.init_opa))
                colors = torch.zeros((N, (cfg.sh_degree + 1) ** 2, 3))
                colors[:, 0, :] = rgb_to_sh(rgbs)
                
                # Rebuild splats dict with per-iso points
                iso_tensor_dict = {
                    "means": points,
                    "scales": scales,
                    "quats": quats,
                    "opacities": opacities,
                    "sh0": colors[:, :1, :],
                    "shN": colors[:, 1:, :],
                }
                self._rebuild_splats_and_optimizers(iso_tensor_dict)
            else:
                # No per-iso points — use random initialization
                random_splats, random_optimizers = create_splats_with_optimizers(
                    self.parser,
                    init_type="random",
                    init_num_pts=cfg.init_num_pts,
                    init_extent=cfg.init_extent,
                    init_opacity=cfg.init_opa,
                    init_scale=cfg.init_scale,
                    means_lr=cfg.means_lr,
                    scales_lr=cfg.scales_lr,
                    opacities_lr=cfg.opacities_lr,
                    quats_lr=cfg.quats_lr,
                    sh0_lr=cfg.sh0_lr,
                    shN_lr=cfg.shN_lr,
                    scene_scale=self.scene_scale,
                    sh_degree=cfg.sh_degree,
                    sparse_grad=cfg.sparse_grad,
                    visible_adam=cfg.visible_adam,
                    batch_size=cfg.batch_size,
                    feature_dim=feature_dim,
                    device=self.device,
                    world_rank=self.world_rank,
                    world_size=self.world_size,
                )
                self.splats = random_splats
                self.optimizers = random_optimizers
                # Reset strategy state
                cfg.strategy.check_sanity(self.splats, self.optimizers)
                if isinstance(cfg.strategy, DefaultStrategy):
                    self.strategy_state = cfg.strategy.initialize_state(
                        scene_scale=self.scene_scale
                    )
                elif isinstance(cfg.strategy, MCMCStrategy):
                    self.strategy_state = cfg.strategy.initialize_state()
            
            print(f"  Initialized {len(self.splats['means'])} Gaussians")
            
            # ── Override densification for this sub-run ──
            _orig_grow_grad2d = None
            _orig_refine_every = None
            _orig_refine_stop_iter = None
            if isinstance(cfg.strategy, DefaultStrategy):
                _orig_grow_grad2d = cfg.strategy.grow_grad2d
                _orig_refine_every = cfg.strategy.refine_every
                _orig_refine_stop_iter = cfg.strategy.refine_stop_iter
                
                cfg.strategy.grow_grad2d = cfg.stage1_grow_grad2d
                cfg.strategy.refine_every = cfg.stage1_refine_every
                cfg.strategy.refine_stop_iter = int(per_iso_steps * 0.8)
            
            # ── Dataset: all train params × this single isovalue ──
            iso_trainset = SurfaceConditionalDataset(
                self.parser, split="train", patch_size=cfg.patch_size,
                param_indices=self.train_conditions,
                isovalue_indices=[iso_idx],
            )
            trainloader = torch.utils.data.DataLoader(
                iso_trainset, batch_size=cfg.batch_size, shuffle=True,
                num_workers=4, persistent_workers=True, pin_memory=True,
            )
            trainloader_iter = iter(trainloader)
            
            # ── Scheduler ──
            schedulers = [
                torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizers["means"], gamma=0.01 ** (1.0 / max(per_iso_steps, 1))
                ),
            ]
            
            # ── Training loop for this isovalue ──
            pbar = tqdm.tqdm(range(per_iso_steps),
                             desc=f"Stage 1 iso {iso_run+1}/{num_isos}")
            for step in pbar:
                try:
                    data = next(trainloader_iter)
                except StopIteration:
                    trainloader_iter = iter(trainloader)
                    data = next(trainloader_iter)

                camtoworlds = data["camtoworld"].to(device)
                Ks = data["K"].to(device)
                pixels = data["image"].to(device) / 255.0
                image_ids = data["image_id"].to(device)
                masks = data["mask"].to(device) if "mask" in data else None
                height, width = pixels.shape[1:3]

                sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

                renders, alphas, info = self.rasterize_splats(
                    camtoworlds=camtoworlds, Ks=Ks, width=width, height=height,
                    sh_degree=sh_degree_to_use, near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane, image_ids=image_ids,
                    condition_vector=None, render_mode="RGB", masks=masks,
                )
                colors = renders[..., 0:3]

                if cfg.random_bkgd:
                    bkgd = torch.rand(1, 3, device=device)
                    colors = colors + bkgd * (1.0 - alphas)
                elif cfg.white_bkgd:
                    colors = colors + (1.0 - alphas)

                self.cfg.strategy.step_pre_backward(
                    params=self.splats, optimizers=self.optimizers,
                    state=self.strategy_state, step=step, info=info,
                )

                l1loss = F.l1_loss(colors, pixels)
                ssimloss = 1.0 - fused_ssim(
                    colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2),
                    padding="valid",
                )
                loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

                if cfg.opacity_reg > 0.0:
                    loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
                if cfg.scale_reg > 0.0:
                    loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

                loss.backward()
                pbar.set_description(
                    f"Stage 1 iso {iso_run+1}/{num_isos} | "
                    f"loss={loss.item():.4f} | GS={len(self.splats['means'])}"
                )

                for optimizer in self.optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for scheduler in schedulers:
                    scheduler.step()

                if isinstance(self.cfg.strategy, DefaultStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.splats, optimizers=self.optimizers,
                        state=self.strategy_state, step=step, info=info,
                        packed=cfg.packed,
                    )
                elif isinstance(self.cfg.strategy, MCMCStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.splats, optimizers=self.optimizers,
                        state=self.strategy_state, step=step, info=info,
                        lr=schedulers[0].get_last_lr()[0],
                    )

                if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                    global_step = iso_run * per_iso_steps + step
                    self.writer.add_scalar("stage1/loss", loss.item(), global_step)
                    self.writer.add_scalar("stage1/num_GS", len(self.splats["means"]), global_step)
                    self.writer.flush()

            # ── Collect this isovalue's Gaussians ──
            n_gs = len(self.splats["means"])
            collected = {k: v.data.detach().clone() for k, v in self.splats.items()}
            collected_splats.append(collected)
            print(f"  → Collected {n_gs} Gaussians from isovalue {iso_idx} (value={iso_val:.4f})")
            
            # Restore strategy params for next iso
            if isinstance(cfg.strategy, DefaultStrategy) and _orig_grow_grad2d is not None:
                cfg.strategy.grow_grad2d = _orig_grow_grad2d
                cfg.strategy.refine_every = _orig_refine_every
                cfg.strategy.refine_stop_iter = _orig_refine_stop_iter
        
        # ══════════════════════════════════════════════════════════════
        # Merge all collected Gaussians
        # ══════════════════════════════════════════════════════════════
        print(f"\n{'='*60}")
        print(f"MERGING Gaussians from {num_isos} isovalue runs")
        
        per_iso_counts = [len(c["means"]) for c in collected_splats]
        for i, (iso_idx, count) in enumerate(zip(self.train_isovalues, per_iso_counts)):
            iso_val = iso_values[i]
            print(f"  Iso {iso_idx} (value={iso_val:.4f}): {count} Gaussians")
        
        merged = {}
        for key in collected_splats[0].keys():
            merged[key] = torch.cat([c[key] for c in collected_splats], dim=0)
        
        total_gs = len(merged["means"])
        print(f"  Total merged: {total_gs} Gaussians")
        
        # ── Deduplicate nearby Gaussians ──
        # Remove Gaussians that are very close in 3D space (keep higher opacity)
        dedup_threshold = 1e-4 * self.scene_scale
        means = merged["means"]
        if total_gs > 0 and total_gs < 500_000:  # skip dedup if too many (slow)
            try:
                dists = knn(means.cpu(), 2)[:, 1]  # dist to nearest neighbor
                opacities = torch.sigmoid(merged["opacities"].cpu())
                
                # For each pair closer than threshold, mark lower-opacity one
                close_mask = dists < dedup_threshold
                if close_mask.any():
                    # Simple approach: sort by opacity descending, mark duplicates
                    # (not perfect but fast and good enough)
                    keep = torch.ones(total_gs, dtype=torch.bool)
                    sorted_idx = torch.argsort(opacities, descending=True)
                    seen_positions = set()
                    
                    # Quantize positions for fast lookup
                    quantized = (means.cpu() / dedup_threshold).long()
                    for idx in sorted_idx:
                        pos_key = tuple(quantized[idx].tolist())
                        if pos_key in seen_positions:
                            keep[idx] = False
                        else:
                            seen_positions.add(pos_key)
                    
                    n_removed = (~keep).sum().item()
                    if n_removed > 0:
                        for key in merged:
                            merged[key] = merged[key][keep]
                        print(f"  Deduplicated: removed {n_removed} near-duplicates "
                              f"→ {len(merged['means'])} Gaussians")
            except Exception as e:
                print(f"  Dedup skipped (error: {e})")
        
        # ── Install merged Gaussians ──
        self._rebuild_splats_and_optimizers(merged)
        
        print(f"  Final canonical field: {len(self.splats['means'])} Gaussians")
        print(f"{'='*60}\n")
        
        # Report stats
        self._report_model_stats()
        
        # Save Stage 1 checkpoint
        ckpt_data = {
            "step": total_steps - 1,
            "splats": self.splats.state_dict(),
            "stage": 1,
            "per_iso_counts": per_iso_counts,
        }
        torch.save(ckpt_data, f"{self.ckpt_dir}/ckpt_stage1_rank{self.world_rank}.pt")
        print(f"Stage 1 checkpoint saved\n")

    def _run_stage2(self):
        """
        Stage 2: Train deformation network on all conditions.
        """
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        
        stage2_steps = cfg.max_steps - cfg.deform_start_step
        
        print(f"\n{'='*60}")
        print(f"STAGE 2: Deformation Training on All Conditions")
        print(f"         {stage2_steps} steps")
        if cfg.freeze_canonical_in_stage2:
            print("         Canonical Gaussians: FROZEN")
        else:
            print("         Canonical Gaussians: Joint fine-tuning")
        print(f"{'='*60}\n")
        
        # Freeze canonical (if requested)
        if cfg.freeze_canonical_in_stage2:
            for param in self.splats.values():
                param.requires_grad = False
            print("Canonical Gaussians frozen.\n")
        
        # Use full dataset (all conditions) with loss-weighted sampling
        resample_every = 2000
        stage2_dataset = self.trainset
        stage2_batch_size = cfg.batch_size

        # Per-isovalue loss tracking
        iso_loss_sums = defaultdict(float)
        iso_loss_counts = defaultdict(int)
        
        num_samples = len(stage2_dataset)
        # Per-(condition, isovalue) pair loss tracking for hard mining.
        # With 316K samples but only 1,400 (cond, iso) pairs, every pair
        # gets visited often → weights actually reflect difficulty.
        pair_loss_avg = {}     # (cond_idx, iso_idx) -> running avg loss
        pair_loss_alpha = 0.1  # EMA smoothing

        # Build sample_idx -> (cond_idx, iso_idx) mapping
        sample_to_pair = {}
        for sidx, s in enumerate(stage2_dataset.samples):
            sample_to_pair[sidx] = (s["param_idx"], s["isovalue_idx"])

        sample_weights = torch.ones(num_samples)  # uniform start

        def _make_loader(weights):
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, num_samples=num_samples, replacement=True,
            )
            return torch.utils.data.DataLoader(
                stage2_dataset, batch_size=stage2_batch_size, sampler=sampler,
                num_workers=2, persistent_workers=True, pin_memory=True,
            )

        trainloader = _make_loader(sample_weights)
        trainloader_iter = iter(trainloader)
        
        # Schedulers for stage-2
        schedulers = []
        if not cfg.freeze_canonical_in_stage2:
            # Reduce LR for canonical Gaussians in stage-2 using "stage2_canonical_lr_scale"
            lr_scale = cfg.stage2_canonical_lr_scale
            for param_group in self.optimizers["means"].param_groups:
                param_group['lr'] = cfg.means_lr * lr_scale * math.sqrt(cfg.batch_size)
            print(f"Stage 2 canonical LR scale: {lr_scale} (means_lr = {cfg.means_lr * lr_scale * math.sqrt(cfg.batch_size):.6f})")
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizers["means"], gamma=0.1 ** (1.0 / stage2_steps)
                )
            )
        
        # Deformation scheduler(s) — cosine annealing with floor at 10% of peak
        for deform_opt in self.deform_optimizers:
            schedulers.append(
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    deform_opt, T_max=stage2_steps, eta_min=deform_opt.defaults['lr'] * 0.1,
                )
            )
        
        # Training loop
        pbar = tqdm.tqdm(range(stage2_steps), desc="Stage 2")
        for step_offset in pbar:
            step = cfg.deform_start_step + step_offset
            
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)

            # ── Single-sample step ──
            pixels = data["image"].to(device) / 255.0
            
            # Flag (near-)black GT images — still render (DDP needs all ranks in sync)
            is_black_gt = pixels.mean() < 0.005

            num_train_rays_per_step = pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]
            
            # Get condition
            condition_vector = data["condition_vector"].to(device)
            condition_idx = data["condition_idx"]
            if isinstance(condition_idx, torch.Tensor):
                condition_idx = condition_idx.item()

            # Get isovalue (surface mode)
            isovalue_tensor = None
            isovalue_idx = None
            if "isovalue" in data:
                isovalue_tensor = data["isovalue"].to(device)
                isovalue_idx = data["isovalue_idx"]
                if isinstance(isovalue_idx, torch.Tensor):
                    isovalue_idx = isovalue_idx.item()

            # Forward WITH deformation (bf16 autocast — cuts deformation MLP
            # + rasterization cost on tensor-core GPUs with minimal accuracy
            # loss; weights/optimizer state stay fp32. Ported from
            # gsplat_surrogate_unified_SH/examples/simple_trainer.py's Stage 2
            # loop, paired with BF16LayerNorm in deformation_model.py.)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                renders, alphas, info = self.rasterize_splats(
                    camtoworlds=camtoworlds, Ks=Ks, width=width, height=height,
                    sh_degree=cfg.sh_degree, near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane, image_ids=image_ids,
                    condition_vector=condition_vector,
                    isovalue=isovalue_tensor,
                    render_mode="RGB", masks=masks,
                )
                colors = renders[..., 0:3]

                if cfg.random_bkgd:
                    bkgd = torch.rand(1, 3, device=device)
                    colors = colors + bkgd * (1.0 - alphas)
                elif cfg.white_bkgd:
                    colors = colors + (1.0 - alphas)

                # Loss (RGB)
                l1loss = F.l1_loss(colors, pixels)
                ssimloss = 1.0 - fused_ssim(
                    colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
                )
                loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

                # Zero loss for black GT — forward ran for DDP sync, but don't learn from it
                if is_black_gt:
                    loss = loss * 0.0

            loss.backward()
            loss_val = loss.item()

            if not is_black_gt:
                iso_str = f" iso={isovalue_idx}" if isovalue_idx is not None else ""
                pbar.set_description(f"Stage 2 | loss={loss_val:.4f} | cond={condition_idx}{iso_str}")

            # Track per-(cond, iso) pair loss (EMA) — skip black GT
            if "sample_idx" in data and not is_black_gt:
                sidx = data["sample_idx"]
                if isinstance(sidx, torch.Tensor):
                    sidx = sidx.item()
                pair_key = sample_to_pair.get(sidx)
                if pair_key is not None:
                    old = pair_loss_avg.get(pair_key, loss_val)
                    pair_loss_avg[pair_key] = (1 - pair_loss_alpha) * old + pair_loss_alpha * loss_val

            # Track per-isovalue loss — skip black GT
            if isovalue_idx is not None and not is_black_gt:
                iso_loss_sums[isovalue_idx] += loss_val
                iso_loss_counts[isovalue_idx] += 1

            # Rebuild sampler: propagate pair-level loss to all samples in that pair
            if step_offset > 0 and step_offset % resample_every == 0:
                for sidx, pair_key in sample_to_pair.items():
                    if pair_key in pair_loss_avg:
                        sample_weights[sidx] = pair_loss_avg[pair_key]
                weights = sample_weights ** cfg.hard_mining_exponent
                trainloader = _make_loader(weights)
                trainloader_iter = iter(trainloader)
                # Summary
                n_visited = len(pair_loss_avg)
                n_total = len(set(sample_to_pair.values()))
                top5_pairs = sorted(pair_loss_avg.items(), key=lambda x: -x[1])[:5]
                top5_str = [f"c{k[0]}i{k[1]}={v:.4f}" for k, v in top5_pairs]
                iso_summary = {k: iso_loss_sums[k] / max(iso_loss_counts[k], 1)
                               for k in sorted(iso_loss_sums.keys())}
                iso_str_summary = ", ".join(f"iso{k}={v:.4f}" for k, v in iso_summary.items())
                print(f"\n  [Resample] step {step}: {n_visited}/{n_total} pairs visited, "
                      f"exp={cfg.hard_mining_exponent}")
                print(f"  [Hardest pairs] {top5_str}")
                print(f"  [Per-iso avg] {iso_str_summary}")
                iso_loss_sums.clear()
                iso_loss_counts.clear()

            # Logging — skip black GT
            if world_rank == 0 and cfg.tb_every > 0 and step_offset % cfg.tb_every == 0 and not is_black_gt:
                self.writer.add_scalar("stage2/loss", loss_val, step)
                if isovalue_idx is not None:
                    self.writer.add_scalar(f"stage2/loss_iso{isovalue_idx}", loss_val, step)
                self.writer.flush()

            # Optimize
            if not cfg.freeze_canonical_in_stage2:
                for optimizer in self.optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            
            for optimizer in self.deform_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            for scheduler in schedulers:
                scheduler.step()

            # Evaluation at specified steps
            if step in [i - 1 for i in cfg.eval_steps]:
                print(f"\n--- Evaluating at step {step} ---")
                self.eval(step)

            # Checkpoints
            if step in [i - 1 for i in cfg.save_steps] or step_offset == stage2_steps - 1:
                ckpt_data = {"step": step, "splats": self.splats.state_dict(), "stage": 2}
                if self.deform_field is not None:
                    ckpt_data["deform_field"] = self.deform_field.state_dict()
                    ckpt_data["condition_min"] = self.parser.condition_min.tolist()
                    ckpt_data["condition_max"] = self.parser.condition_max.tolist()
                    if hasattr(self.parser, 'isovalue_min'):
                        ckpt_data["isovalue_min"] = self.parser.isovalue_min
                        ckpt_data["isovalue_max"] = self.parser.isovalue_max
                        ckpt_data["isovalue_range"] = self.parser.isovalue_range
                torch.save(ckpt_data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt")
                print(f"Checkpoint saved: step {step}")

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                try:
                    num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                    num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
                    self.viewer.render_tab_state.num_train_rays_per_sec = num_train_rays_per_sec
                    self.viewer.update(step, num_train_rays_per_step)
                except Exception as e:
                    pass  # Viewer update is optional

        # ====================================================================
        print(f"\n{'='*70}")
        print(f"FINAL EVALUATION")
        print(f"{'='*70}")
        
        eval_sets = []
        
        # # (1) Train params × Train isos × Unseen views (standard val)
        # eval_sets.append(("train_param__train_iso__unseen_view", self.testset_train_unseen))
        
        # # (2) Train params × Train isos × Seen views
        # eval_sets.append(("train_param__train_iso__seen_view", self.testset_train_seen))
        
        if cfg.use_surface:
            # --- Holdout params × Train isos (seen isos, unseen params, unseen views) ---
            hp = getattr(self, 'testset_holdout_param_unseen', None)
            # hps = getattr(self, 'testset_holdout_param_seen', None)
            eval_sets.append(("holdout_param__train_iso__unseen_view", hp))
            # eval_sets.append(("holdout_param__train_iso__seen_view", hps))
            
            # --- Holdout params × Holdout isos (unseen isos, unseen params, unseen views) ---
            hb = getattr(self, 'testset_holdout_both_unseen', None)
            # hbs = getattr(self, 'testset_holdout_both_seen', None)
            eval_sets.append(("holdout_param__holdout_iso__unseen_view", hb))
            # eval_sets.append(("holdout_param__holdout_iso__seen_view", hbs))
        else:
            # Volume mode: just param holdout
            if self.testset_holdout_unseen is not None:
                eval_sets.append(("holdout_param__unseen_view", self.testset_holdout_unseen))
            # if self.testset_holdout_seen is not None:
            #     eval_sets.append(("holdout_param__seen_view", self.testset_holdout_seen))
        
        for name, dataset in eval_sets:
            if dataset is not None and len(dataset) > 0:
                print(f"\n--- {name} ({len(dataset)} samples) ---")
                self.eval_holdout(cfg.max_steps - 1, dataset, name)
        
        # Print summary table
        self._print_surface_evaluation_summary(cfg.max_steps - 1) if cfg.use_surface else self._print_evaluation_summary(cfg.max_steps - 1)
        
        # Save grid: train+test params × view 0 × all isovalues
        if cfg.use_surface:
            self.save_params_view0_all_isovalues(cfg.max_steps - 1)
        
        self.render_traj(cfg.max_steps - 1)
        
        # Report final stats
        self._report_model_stats()

    def _multi_iso_step(self, data, cfg, device):
    
        # Unpack grouped data (batch_size=1, so squeeze batch dim)
        all_pixels = data["images"][0].to(device) / 255.0        # [K, H, W, 3]
        camtoworld = data["camtoworld"][0].to(device)             # [4, 4]
        K_mat = data["K"][0].to(device)                           # [3, 3]
        condition_vector = data["condition_vector"][0].to(device)  # [cond_dim]
        isovalues = data["isovalues"][0].to(device)               # [K]
        valid_mask = data["valid_mask"][0]                         # [K] bool
        condition_idx = data["condition_idx"]
        if isinstance(condition_idx, torch.Tensor):
            condition_idx = condition_idx.item()

        K_iso = len(isovalues)
        height, width = all_pixels.shape[1], all_pixels.shape[2]

        # ── Isovalue subsampling: randomly pick K_sub of K isovalues ──
        K_sub = cfg.multi_iso_K_sub
        if K_sub > 0 and K_sub < K_iso:
            # Only subsample from valid (existing-image) indices
            valid_indices = [i for i in range(K_iso) if valid_mask[i]]
            if len(valid_indices) > K_sub:
                chosen = sorted(
                    torch.randperm(len(valid_indices))[:K_sub].tolist()
                )
                sub_indices = [valid_indices[c] for c in chosen]
            else:
                sub_indices = valid_indices  # fewer valid than K_sub, use all
            # Slice everything to the subset
            all_pixels = all_pixels[sub_indices]
            isovalues = isovalues[sub_indices]
            valid_mask = valid_mask[sub_indices]
            K_iso = len(sub_indices)

        num_train_rays = int(valid_mask.sum().item()) * height * width

        # ── Get canonical Gaussian attributes ──
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = self.splats["scales"]
        opacities_logit = self.splats["opacities"]
        sh_coeffs = torch.cat([self.splats["sh0"], self.splats["shN"]], 1) if cfg.learn_deform_sh else None

        # ── Multi-iso deformation (shared encoding!) ──
        iso_tensors = [isovalues[k:k+1] for k in range(K_iso)]

        all_deformed = self.deform_field.apply_deformation_multi_iso(
            means, quats, scales, opacities_logit.unsqueeze(-1),
            condition_vector, iso_tensors,
            sh=sh_coeffs,
        )

        # ── Render each isovalue and accumulate loss ──
        total_loss = torch.tensor(0.0, device=device)
        num_valid = 0

        viewmat = torch.linalg.inv(camtoworld).unsqueeze(0)       # [1, 4, 4]
        K_cam = K_mat.unsqueeze(0)                                  # [1, 3, 3]

        for k in range(K_iso):
            if not valid_mask[k]:
                continue

            d_means, d_quats, d_scales_log, d_opacities_logit, d_sh = all_deformed[k]
            d_opacities_logit = d_opacities_logit.squeeze(-1)

            d_scales = torch.exp(d_scales_log)
            d_opacities = torch.sigmoid(d_opacities_logit)

            if d_sh is not None:
                colors = d_sh
            else:
                colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)

            render_colors, render_alphas, info = rasterization(
                means=d_means,
                quats=d_quats,
                scales=d_scales,
                opacities=d_opacities,
                colors=colors,
                viewmats=viewmat,
                Ks=K_cam,
                width=width,
                height=height,
                packed=cfg.packed,
                rasterize_mode="antialiased" if cfg.antialiased else "classic",
                camera_model=cfg.camera_model,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
            )

            rendered = render_colors[0, ..., 0:3].clamp(0, 1)      # [H, W, 3]
            gt = all_pixels[k]                                       # [H, W, 3]

            if cfg.random_bkgd:
                bkgd = torch.rand(3, device=device)
                rendered = rendered + bkgd * (1.0 - render_alphas[0])
            elif cfg.white_bkgd:
                rendered = rendered + (1.0 - render_alphas[0])

            # L1 + SSIM loss
            l1 = F.l1_loss(rendered, gt)
            ssim = 1.0 - fused_ssim(
                rendered.unsqueeze(0).permute(0, 3, 1, 2),
                gt.unsqueeze(0).permute(0, 3, 1, 2),
                padding="valid",
            )
            total_loss = total_loss + l1 * (1 - cfg.ssim_lambda) + ssim * cfg.ssim_lambda

            num_valid += 1

        if num_valid > 0:
            total_loss = total_loss / num_valid

        # Regularisation (use first isovalue)
        if cfg.deform_reg > 0.0 and self.deform_field is not None:
            reg_means = self.splats["means"].detach()
            deform_reg_loss = self.deform_field.get_regularization_loss(
                reg_means, condition_vector,
                isovalue=isovalues[0:1],
            )
            total_loss = total_loss + cfg.deform_reg * deform_reg_loss

        total_loss.backward()
        return total_loss, total_loss.item(), condition_idx, num_train_rays

    @torch.no_grad()
    def _eval_stage1(self, step: int, valset):
        """Evaluate Stage 1 on reference condition only."""
        cfg = self.cfg
        device = self.device
        
        valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=1)
        metrics = defaultdict(list)
        
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            colors, alphas, _ = self.rasterize_splats(
                camtoworlds=camtoworlds, Ks=Ks, width=width, height=height,
                sh_degree=cfg.sh_degree, near_plane=cfg.near_plane,
                far_plane=cfg.far_plane, masks=masks, condition_vector=None,
            )
            if cfg.white_bkgd:
                colors = colors[..., 0:3] + (1.0 - alphas)
            colors = torch.clamp(colors, 0.0, 1.0)

            # Save image
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            imageio.imwrite(f"{self.render_dir}/stage1_step{step}_{i:04d}.png", canvas)

            pixels_p = pixels.permute(0, 3, 1, 2)
            colors_p = colors.permute(0, 3, 1, 2)
            metrics["psnr"].append(self.psnr(colors_p, pixels_p))
            metrics["ssim"].append(self.ssim(colors_p, pixels_p))
            metrics["lpips"].append(self.lpips(colors_p, pixels_p))

        stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
        stats["num_GS"] = len(self.splats["means"])
        
        print(f"Stage 1 Results: PSNR={stats['psnr']:.3f}, SSIM={stats['ssim']:.4f}, "
              f"LPIPS={stats['lpips']:.3f}, GS={stats['num_GS']}")
        
        with open(f"{self.stats_dir}/stage1_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        
        self.writer.add_scalar("stage1/psnr", stats['psnr'], step)
        self.writer.add_scalar("stage1/ssim", stats['ssim'], step)
        self.writer.flush()

    def _train_single_stage(self):
        """Original single-stage training (for non-conditional or deformation from start)."""
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None

            # Get condition if available
            condition_vector = None
            isovalue_tensor = None
            if cfg.use_deformation and "condition_vector" in data:
                condition_vector = data["condition_vector"].to(device)
            if cfg.use_surface and "isovalue" in data:
                isovalue_tensor = data["isovalue"].to(device)

            if cfg.depth_loss:
                points = data["points"].to(device)
                depths_gt = data["depths"].to(device)

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                condition_vector=condition_vector,
                isovalue=isovalue_tensor,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(
                    self.bil_grids,
                    grid_xy.expand(colors.shape[0], -1, -1, -1),
                    colors,
                    image_ids.unsqueeze(-1),
                )["rgb"]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)
            elif cfg.white_bkgd:
                colors = colors + (1.0 - alphas)

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.depth_loss:
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )
                grid = points.unsqueeze(2)
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )
                depths = depths.squeeze(3).squeeze(1)
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            if cfg.opacity_reg > 0.0:
                loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            if cfg.scale_reg > 0.0:
                loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

            # Deformation regularization
            if cfg.use_deformation and cfg.deform_reg > 0.0 and condition_vector is not None:
                iso_for_reg = isovalue_tensor[0] if isovalue_tensor is not None and isovalue_tensor.dim() > 1 else isovalue_tensor
                deform_reg_loss = self.deform_field.get_regularization_loss(
                    self.splats["means"].detach(),
                    condition_vector[0] if condition_vector.dim() > 1 else condition_vector,
                    isovalue=iso_for_reg,
                )
                loss = loss + cfg.deform_reg * deform_reg_loss

            loss.backward()

            desc = f"loss={loss.item():.3f}| sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            pbar.set_description(desc)

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                self.writer.flush()

            # save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json", "w") as f:
                    json.dump(stats, f)
                data_ckpt = {"step": step, "splats": self.splats.state_dict()}
                if cfg.use_deformation and self.deform_field is not None:
                    data_ckpt["deform_field"] = self.deform_field.state_dict()
                    if hasattr(self.parser, 'condition_min'):
                        data_ckpt["condition_min"] = self.parser.condition_min.tolist()
                        data_ckpt["condition_max"] = self.parser.condition_max.tolist()
                    if hasattr(self.parser, 'isovalue_min'):
                        data_ckpt["isovalue_min"] = self.parser.isovalue_min
                        data_ckpt["isovalue_max"] = self.parser.isovalue_max
                        data_ckpt["isovalue_range"] = self.parser.isovalue_range
                torch.save(data_ckpt, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt")

            # optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.deform_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # densification
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )

            # eval
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)

            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
                self.viewer.render_tab_state.num_train_rays_per_sec = num_train_rays_per_sec
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation - with per-condition metrics for conditional 4DGS."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        
        # Track per-condition metrics
        per_condition_metrics = defaultdict(lambda: defaultdict(list))
        
        # Track per-condition camera index for better naming
        per_condition_cam_count = defaultdict(int)
        
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            
            # Flag (near-)black GT — still render (DDP needs all ranks in sync) but skip metrics
            is_black_gt = pixels.mean() < 0.005

            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            # Get condition for evaluation
            condition_vector = None
            condition_idx = None
            camera_idx = None
            isovalue_tensor = None
            isovalue_idx = None
            if cfg.use_deformation and "condition_vector" in data:
                condition_vector = data["condition_vector"].to(device)
                condition_idx = data["condition_idx"]
                if isinstance(condition_idx, torch.Tensor):
                    condition_idx = condition_idx.item()
                # Get camera index if available
                if "camera_idx" in data:
                    camera_idx = data["camera_idx"]
                    if isinstance(camera_idx, torch.Tensor):
                        camera_idx = camera_idx.item()
                # Get isovalue if available (surface mode)
                if "isovalue" in data:
                    isovalue_tensor = data["isovalue"].to(device)
                    isovalue_idx = data["isovalue_idx"]
                    if isinstance(isovalue_idx, torch.Tensor):
                        isovalue_idx = isovalue_idx.item()

            torch.cuda.synchronize()
            tic = time.time()
            colors, alphas, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
                condition_vector=condition_vector,
                isovalue=isovalue_tensor,
            )
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            if cfg.white_bkgd:
                colors = colors[..., 0:3] + (1.0 - alphas)
            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0 and not is_black_gt:
                # write images with informative naming
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                
                # Use condition-aware naming for deformation mode
                if cfg.use_deformation and condition_idx is not None:
                    # Track which camera we're on for this condition
                    cam_num = per_condition_cam_count[condition_idx]
                    per_condition_cam_count[condition_idx] += 1
                    
                    # Format: val_step49999_cond003_cam0012.png
                    filename = f"{stage}_step{step}_cond{condition_idx:03d}_cam{cam_num:04d}.png"
                else:
                    # Standard naming for non-deformation mode
                    filename = f"{stage}_step{step}_{i:04d}.png"
                
                imageio.imwrite(f"{self.render_dir}/{filename}", canvas)

                pixels_p = pixels.permute(0, 3, 1, 2)
                colors_p = colors.permute(0, 3, 1, 2)
                
                psnr_val = self.psnr(colors_p, pixels_p)
                ssim_val = self.ssim(colors_p, pixels_p)
                lpips_val = self.lpips(colors_p, pixels_p)
                
                # Detect empty GT (all black) — PSNR will be inf if both are black
                is_empty_gt = pixels.max() < 1e-6
                
                metrics["psnr"].append(psnr_val)
                metrics["ssim"].append(ssim_val)
                metrics["lpips"].append(lpips_val)
                
                # Track per-condition metrics
                if condition_idx is not None:
                    per_condition_metrics[condition_idx]["psnr"].append(psnr_val)
                    per_condition_metrics[condition_idx]["ssim"].append(ssim_val)
                    per_condition_metrics[condition_idx]["lpips"].append(lpips_val)
                    per_condition_metrics[condition_idx]["is_empty"].append(is_empty_gt)
                
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            
            # Compute PSNR excluding infinite values (from empty GT images)
            PSNR_CAP = 50.0
            finite_psnrs = [p for p in metrics["psnr"] if torch.isfinite(p)]
            if finite_psnrs:
                stats["psnr"] = min(torch.stack(finite_psnrs).mean().item(), PSNR_CAP)
            else:
                stats["psnr"] = PSNR_CAP
            
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                    "num_images": len(valloader),
                }
            )
            
            # Print overall metrics
            print(f"\n{'='*70}")
            print(f"EVALUATION RESULTS (Step {step})")
            print(f"{'='*70}")
            print(f"Overall: PSNR={stats['psnr']:.3f}, SSIM={stats['ssim']:.4f}, LPIPS={stats['lpips']:.3f}")
            print(f"         Time={stats['ellipse_time']:.3f}s/image, GS={stats['num_GS']}")
            
            # Print per-condition metrics if available
            if cfg.use_deformation and len(per_condition_metrics) > 0:
                print(f"\n--- Per-Condition PSNR ({len(per_condition_metrics)} conditions) ---")
                
                # Compute per-condition averages
                PSNR_CAP = 50.0  # Cap for empty GT (avoids JSON Infinity)
                condition_stats = {}
                empty_conditions = []
                for cond_idx in sorted(per_condition_metrics.keys()):
                    cond_metrics = per_condition_metrics[cond_idx]
                    cond_psnr = torch.stack(cond_metrics["psnr"]).mean().item()
                    cond_ssim = torch.stack(cond_metrics["ssim"]).mean().item()
                    cond_lpips = torch.stack(cond_metrics["lpips"]).mean().item()
                    is_empty = all(cond_metrics["is_empty"])
                    
                    condition_stats[cond_idx] = {
                        "psnr": min(cond_psnr, PSNR_CAP),
                        "ssim": cond_ssim,
                        "lpips": cond_lpips,
                        "empty_gt": is_empty,
                    }
                    if is_empty:
                        empty_conditions.append(cond_idx)
                
                # Print all conditions in a compact table format
                print(f"{'Cond':<6} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8} {'Note'}")
                print("-" * 42)
                for cond_idx in sorted(condition_stats.keys()):
                    cs = condition_stats[cond_idx]
                    marker = "*" if cond_idx == cfg.reference_condition else " "
                    note = " (empty GT)" if cs["empty_gt"] else ""
                    if cs["empty_gt"]:
                        print(f"{cond_idx:<5}{marker} {'inf':<8} {cs['ssim']:<8.4f} {cs['lpips']:<8.3f}{note}")
                    else:
                        print(f"{cond_idx:<5}{marker} {cs['psnr']:<8.3f} {cs['ssim']:<8.4f} {cs['lpips']:<8.3f}{note}")
                
                # Summary statistics — exclude empty GT conditions
                nonempty_stats = {k: v for k, v in condition_stats.items() if not v["empty_gt"]}
                all_psnrs = [cs["psnr"] for cs in nonempty_stats.values()]
                print("-" * 42)
                if empty_conditions:
                    print(f"Empty GT conditions: {empty_conditions} (excluded from PSNR stats)")
                if all_psnrs:
                    print(f"{'Mean':<6} {sum(all_psnrs)/len(all_psnrs):<8.3f} ({len(all_psnrs)} conditions)")
                    print(f"{'Min':<6} {min(all_psnrs):<8.3f} (cond {min(nonempty_stats, key=lambda x: nonempty_stats[x]['psnr'])})")
                    print(f"{'Max':<6} {max(all_psnrs):<8.3f} (cond {max(nonempty_stats, key=lambda x: nonempty_stats[x]['psnr'])})")
                
                if cfg.reference_condition in condition_stats:
                    print(f"{'Ref':<6} {condition_stats[cfg.reference_condition]['psnr']:<8.3f} (cond {cfg.reference_condition})")
                
                # Save per-condition stats
                stats["per_condition"] = condition_stats
                stats["num_empty_conditions"] = len(empty_conditions)
                stats["empty_conditions"] = empty_conditions
            
            print(f"{'='*70}\n")
            
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f, indent=2)
            
            # save stats to tensorboard
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def eval_holdout(self, step: int, dataset, name: str = "holdout"):
        """Evaluate on test/holdout conditions (unseen during training).

        Args:
            step: Current training step
            dataset: Dataset to evaluate on
            name: Name for output files (e.g., "holdout_unseen", "holdout_seen")
        """
        if dataset is None:
            print(f"No {name} dataset to evaluate.")
            return
        
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank

        valloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1
        )
        
        ellipse_time = 0
        metrics = defaultdict(list)
        per_condition_metrics = defaultdict(lambda: defaultdict(list))
        per_condition_cam_count = defaultdict(int)
        
        for i, data in enumerate(tqdm.tqdm(valloader, desc=f"Evaluating {name}")):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            
            # Flag (near-)black GT — still render (DDP needs all ranks in sync) but skip metrics
            is_black_gt = pixels.mean() < 0.005

            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            condition_vector = data["condition_vector"].to(device)
            condition_idx = data["condition_idx"]
            if isinstance(condition_idx, torch.Tensor):
                condition_idx = condition_idx.item()

            # Get isovalue if available (surface mode)
            isovalue_tensor = None
            isovalue_idx = None
            if "isovalue" in data:
                isovalue_tensor = data["isovalue"].to(device)
                isovalue_idx = data["isovalue_idx"]
                if isinstance(isovalue_idx, torch.Tensor):
                    isovalue_idx = isovalue_idx.item()

            # Debug: verify GT path matches rendered condition
            if i < 5:
                sidx = data["sample_idx"]
                if isinstance(sidx, torch.Tensor):
                    sidx = sidx.item()
                sample = dataset.samples[sidx]
                iso_norm_str = f", iso_val_norm={isovalue_tensor.item():.4f}" if isovalue_tensor is not None else ""
                iso_raw_str = f", iso_val_raw={sample.get('isovalue', 'N/A')}" if 'isovalue' in sample else ""
                print(f"  [DEBUG] sample {i}: param={condition_idx}, iso_idx={isovalue_idx}, "
                      f"cam={data.get('camera_idx', 'N/A')}, "
                      f"gt_path={sample['image_path']}"
                      f"{iso_raw_str}{iso_norm_str}")

            torch.cuda.synchronize()
            tic = time.time()
            colors, alphas, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
                condition_vector=condition_vector,
                isovalue=isovalue_tensor,
            )
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            if cfg.white_bkgd:
                colors = colors[..., 0:3] + (1.0 - alphas)
            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0 and not is_black_gt:
                # Write images
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                
                cam_num = per_condition_cam_count[condition_idx]
                per_condition_cam_count[condition_idx] += 1
                if isovalue_idx is not None:
                    filename = f"{name}_step{step}_cond{condition_idx:03d}_iso{isovalue_idx:02d}_cam{cam_num:04d}.png"
                else:
                    filename = f"{name}_step{step}_cond{condition_idx:03d}_cam{cam_num:04d}.png"
                imageio.imwrite(f"{self.render_dir}/{filename}", canvas)

                pixels_p = pixels.permute(0, 3, 1, 2)
                colors_p = colors.permute(0, 3, 1, 2)
                
                psnr_val = self.psnr(colors_p, pixels_p)
                ssim_val = self.ssim(colors_p, pixels_p)
                lpips_val = self.lpips(colors_p, pixels_p)
                
                is_empty_gt = pixels.max() < 1e-6
                
                metrics["psnr"].append(psnr_val)
                metrics["ssim"].append(ssim_val)
                metrics["lpips"].append(lpips_val)
                
                per_condition_metrics[condition_idx]["psnr"].append(psnr_val)
                per_condition_metrics[condition_idx]["ssim"].append(ssim_val)
                per_condition_metrics[condition_idx]["lpips"].append(lpips_val)
                per_condition_metrics[condition_idx]["is_empty"].append(is_empty_gt)

        if world_rank == 0:
            ellipse_time /= len(valloader)
            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            
            # Compute PSNR excluding infinite values (from empty GT images)
            PSNR_CAP = 50.0
            finite_psnrs = [p for p in metrics["psnr"] if torch.isfinite(p)]
            if finite_psnrs:
                stats["psnr"] = min(torch.stack(finite_psnrs).mean().item(), PSNR_CAP)
            else:
                stats["psnr"] = PSNR_CAP
            
            stats["ellipse_time"] = ellipse_time
            stats["num_GS"] = len(self.splats["means"])
            stats["num_images"] = len(dataset)
            
            # Determine view type from name
            view_type = "unseen views" if "unseen" in name else "seen views"
            
            print(f"\n{'='*70}")
            print(f"HOLDOUT EVALUATION: {name.upper()} (Step {step})")
            print(f"  Evaluating {len(per_condition_metrics)} holdout conditions on {view_type}")
            print(f"  Total images: {len(dataset)}")
            print(f"{'='*70}")
            print(f"Overall: PSNR={stats['psnr']:.3f}, SSIM={stats['ssim']:.4f}, LPIPS={stats['lpips']:.3f}")
            print(f"         Time={stats['ellipse_time']:.3f}s/image, GS={stats['num_GS']}")
            
            # Per-condition breakdown (all views)
            print(f"\n--- Per-Condition PSNR ({len(per_condition_metrics)} TEST conditions) ---")
            print(f"{'Cond':<6} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8} {'Note'}")
            print("-" * 42)
            
            condition_stats = {}
            empty_conditions = []
            for cond_idx in sorted(per_condition_metrics.keys()):
                cond_metrics = per_condition_metrics[cond_idx]
                cond_psnr = torch.stack(cond_metrics["psnr"]).mean().item()
                cond_ssim = torch.stack(cond_metrics["ssim"]).mean().item()
                cond_lpips = torch.stack(cond_metrics["lpips"]).mean().item()
                is_empty = all(cond_metrics["is_empty"])
                
                condition_stats[cond_idx] = {
                    "psnr": cond_psnr,
                    "ssim": cond_ssim,
                    "lpips": cond_lpips,
                    "empty_gt": is_empty,
                }
                if is_empty:
                    empty_conditions.append(cond_idx)
                
                note = " (empty GT)" if is_empty else ""
                if is_empty:
                    print(f"{cond_idx:<6} {'inf':<8} {cond_ssim:<8.4f} {cond_lpips:<8.3f}{note}")
                else:
                    print(f"{cond_idx:<6} {cond_psnr:<8.3f} {cond_ssim:<8.4f} {cond_lpips:<8.3f}{note}")
            
            nonempty_stats = {k: v for k, v in condition_stats.items() if not v["empty_gt"]}
            psnr_values = [cs["psnr"] for cs in nonempty_stats.values()]
            print("-" * 42)
            if empty_conditions:
                print(f"Empty GT conditions: {empty_conditions} (excluded from PSNR stats)")
            if psnr_values:
                print(f"Mean   {np.mean(psnr_values):.3f} ({len(psnr_values)} conditions)")
                min_cond = min(nonempty_stats, key=lambda x: nonempty_stats[x]["psnr"])
                max_cond = max(nonempty_stats, key=lambda x: nonempty_stats[x]["psnr"])
                print(f"Min    {np.min(psnr_values):.3f}   (cond {min_cond})")
                print(f"Max    {np.max(psnr_values):.3f}   (cond {max_cond})")
            print(f"{'='*70}\n")
            
            stats["per_condition"] = condition_stats
            
            # Save stats
            with open(f"{self.stats_dir}/{name}_step{step:04d}.json", "w") as f:
                json.dump(stats, f, indent=2)
            
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"{name}/{k}", v, step)
            self.writer.flush()

    def _print_evaluation_summary(self, step: int):
        """Print a summary table of all evaluations."""
        print(f"\n{'='*70}")
        print(f"EVALUATION SUMMARY (Step {step})")
        print(f"{'='*70}")
        
        # Load all evaluation results
        results = {}
        for name in ["train_seen", "train_unseen", "holdout_seen", "holdout_unseen"]:
            json_path = f"{self.stats_dir}/{name}_step{step:04d}.json"
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    results[name] = json.load(f)
        
        if not results:
            print("No evaluation results found.")
            return
        
        # Print 2x2 table
        print(f"\n{'─'*70}")
        print(f"                      │   SEEN Views        │   UNSEEN Views")
        print(f"                      │   (train cameras)   │   (val cameras)")
        print(f"{'─'*70}")
        
        # Train conditions row
        train_seen = results.get("train_seen", {})
        train_unseen = results.get("train_unseen", {})
        print(f"  TRAIN Conditions    │   {train_seen.get('psnr', 0):>6.2f} dB        │   {train_unseen.get('psnr', 0):>6.2f} dB")
        print(f"  ({len(self.train_conditions):>3} conditions)     │   SSIM: {train_seen.get('ssim', 0):.4f}     │   SSIM: {train_unseen.get('ssim', 0):.4f}")
        print(f"                      │   LPIPS: {train_seen.get('lpips', 0):.4f}    │   LPIPS: {train_unseen.get('lpips', 0):.4f}")
        print(f"{'─'*70}")
        
        # Holdout conditions row (if available)
        if self.test_conditions:
            holdout_seen = results.get("holdout_seen", {})
            holdout_unseen = results.get("holdout_unseen", {})
            print(f"  HOLDOUT Conditions  │   {holdout_seen.get('psnr', 0):>6.2f} dB        │   {holdout_unseen.get('psnr', 0):>6.2f} dB")
            print(f"  ({len(self.test_conditions):>3} conditions)     │   SSIM: {holdout_seen.get('ssim', 0):.4f}     │   SSIM: {holdout_unseen.get('ssim', 0):.4f}")
            print(f"                      │   LPIPS: {holdout_seen.get('lpips', 0):.4f}    │   LPIPS: {holdout_unseen.get('lpips', 0):.4f}")
            print(f"{'─'*70}")
        
        print(f"{'='*70}\n")

    def _print_surface_evaluation_summary(self, step: int):
        """Print summary for surface mode: unseen params × unseen views × seen/unseen isos."""
        print(f"\n{'='*70}")
        print(f"SURFACE EVALUATION SUMMARY (Step {step})")
        print(f"{'='*70}")
        
        results = {}
        for name in [
            "holdout_param__train_iso__unseen_view",
            "holdout_param__holdout_iso__unseen_view",
        ]:
            json_path = f"{self.stats_dir}/{name}_step{step:04d}.json"
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    results[name] = json.load(f)
        
        if not results:
            print("No evaluation results found.")
            return
        
        def _fmt(key):
            r = results.get(key, {})
            psnr = r.get('psnr', 0)
            ssim = r.get('ssim', 0)
            lpips = r.get('lpips', 0)
            return f"{psnr:>6.2f} dB  SSIM {ssim:.4f}  LPIPS {lpips:.4f}" if psnr > 0 else "   ---"
        
        n_hp = len(self.test_conditions) if self.test_conditions else 0
        n_ti = len(self.train_isovalues) if self.train_isovalues else 0
        n_hi = len(self.test_isovalues) if self.test_isovalues else 0
        
        print(f"\n  All results: Holdout Params ({n_hp}) × Unseen Views")
        print(f"{'─'*70}")
        print(f"  × Seen Isos    ({n_ti:>2})  │  {_fmt('holdout_param__train_iso__unseen_view')}")
        print(f"  × Unseen Isos  ({n_hi:>2})  │  {_fmt('holdout_param__holdout_iso__unseen_view')}")
        print(f"{'='*70}\n")

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        # Get camera trajectory
        camtoworlds_all = self.parser.camtoworlds[::8]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
            )
        else:
            raise ValueError(f"Unknown trajectory type: {cfg.render_traj_path}")

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        
        # Handle both dict (original Parser) and array (ConditionalParser) formats
        if isinstance(self.parser.Ks, dict):
            K = torch.from_numpy(list(self.parser.Ks.values())[0]).float().to(device)
        else:
            K = torch.from_numpy(self.parser.Ks[0]).float().to(device)
        
        if isinstance(self.parser.heights, dict):
            height = list(self.parser.heights.values())[0]
        elif isinstance(self.parser.heights, (list, np.ndarray)):
            height = self.parser.heights[0]
        else:
            height = self.parser.heights
            
        if isinstance(self.parser.widths, dict):
            width = list(self.parser.widths.values())[0]
        elif isinstance(self.parser.widths, (list, np.ndarray)):
            width = self.parser.widths[0]
        else:
            width = self.parser.widths

        # Use midpoint condition for trajectory rendering
        condition_vector = None
        isovalue = None
        if cfg.use_deformation and hasattr(self.parser, 'condition_min'):
            # Use midpoint (0.5 in min-max normalized space)
            condition_vector = 0.5 * torch.ones(1, self.condition_dim, device=device)
        if cfg.use_surface and self.isovalue_dim > 0:
            isovalue = 0.5 * torch.ones(1, self.isovalue_dim, device=device)

        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=24)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, alphas_traj, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                condition_vector=condition_vector,  # NEW
                isovalue=isovalue,                   # NEW (surface)
            )  # [1, H, W, 4]
            colors = renders[..., 0:3]
            if cfg.white_bkgd:
                colors = colors + (1.0 - alphas_traj)
            colors = torch.clamp(colors, 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    # ========================================================================
    # Render at specific condition
    # ========================================================================
    @torch.no_grad()
    def render_at_condition(
        self,
        condition_vector: np.ndarray,
        output_dir: str,
        use_val_cameras: bool = True,
        isovalue: Optional[float] = None,
    ):
       
        os.makedirs(output_dir, exist_ok=True)
        
        # Normalize condition (min-max to [0, 1])
        if hasattr(self.parser, 'condition_min'):
            cond_normalized = (condition_vector - self.parser.condition_min) / self.parser.condition_range
        else:
            cond_normalized = condition_vector
        
        cond_tensor = torch.from_numpy(cond_normalized).float().to(self.device).unsqueeze(0)
        
        # Normalize isovalue
        iso_tensor = None
        if isovalue is not None and hasattr(self.parser, 'isovalue_min'):
            iso_normalized = (isovalue - self.parser.isovalue_min) / self.parser.isovalue_range
            iso_tensor = torch.tensor([[iso_normalized]], dtype=torch.float32, device=self.device)
            print(f"Rendering at condition: {condition_vector}, isovalue: {isovalue}")
            print(f"  Normalized cond: {cond_normalized}, iso: {iso_normalized:.4f}")
        else:
            print(f"Rendering at condition: {condition_vector}")
            print(f"  Normalized: {cond_normalized}")
        
        dataset = self.valset if use_val_cameras else self.trainset
        
        for idx in tqdm.trange(len(dataset), desc="Rendering"):
            data = dataset[idx]
            
            camtoworld = data["camtoworld"].unsqueeze(0).to(self.device)
            K = data["K"].unsqueeze(0).to(self.device)
            h, w = data["image"].shape[:2]
            
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworld,
                Ks=K,
                width=w,
                height=h,
                sh_degree=self.cfg.sh_degree,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                condition_vector=cond_tensor,
                isovalue=iso_tensor,
            )
            
            img = (renders[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            imageio.imwrite(f"{output_dir}/view_{idx:04d}.png", img)
        
        print(f"Saved {len(dataset)} images to {output_dir}")

    @torch.no_grad()
    def save_params_view0_all_isovalues(self, step: int):
       
        from PIL import Image
        from collections import defaultdict

        cfg = self.cfg
        device = self.device

        if not (cfg.use_surface and cfg.use_deformation):
            print("save_params_view0_all_isovalues: skipped (not surface mode)")
            return
        if self.train_conditions is None or not hasattr(self.parser, 'isovalues'):
            print("save_params_view0_all_isovalues: skipped (no conditions/isovalues)")
            return

        base_dir = f"{cfg.result_dir}/renders_grid/step{step}"
        os.makedirs(base_dir, exist_ok=True)

        # --- First val camera (camera_idx 0, since 0 % test_every == 0) ---
        cam_idx = 0
        camtoworld = torch.from_numpy(self.parser.camtoworlds[cam_idx]).float().to(device).unsqueeze(0)
        K = torch.from_numpy(self.parser.Ks[cam_idx]).float().to(device).unsqueeze(0)
        h = self.parser.heights[cam_idx]
        w = self.parser.widths[cam_idx]

        # All isovalues (train + test, sorted)
        all_iso_indices = sorted(
            set((self.train_isovalues or []) + (self.test_isovalues or []))
        )

        # Build GT lookup: (param_idx, iso_idx, cam_idx) -> image_path
        gt_lookup = {}
        for s in self.parser.all_samples:
            key = (s["param_idx"], s["isovalue_idx"], s["camera_idx"])
            gt_lookup[key] = s["image_path"]

        # Build list of (label, param_indices) groups to render
        groups = [("train_params", self.train_conditions)]
        if self.test_conditions:
            groups.append(("test_params", self.test_conditions))

        # Collect all metrics for the final JSON
        all_metrics = {}          # per-image results
        per_param_psnr = defaultdict(list)
        per_param_ssim = defaultdict(list)
        per_iso_psnr = defaultdict(list)
        per_iso_ssim = defaultdict(list)

        for group_name, param_indices in groups:
            output_dir = f"{base_dir}/{group_name}"
            os.makedirs(output_dir, exist_ok=True)

            total = len(param_indices) * len(all_iso_indices)
            print(f"\nSaving grid [{group_name}]: {len(param_indices)} params × "
                  f"{len(all_iso_indices)} isovalues × view 0  ({total} images)")

            group_psnrs = []
            group_ssims = []
            count = 0

            for param_idx in param_indices:
                cond_norm = self.parser.get_condition_vector(param_idx, normalize=True)
                cond_tensor = torch.from_numpy(cond_norm).float().to(device).unsqueeze(0)

                for iso_idx in all_iso_indices:
                    iso_norm = self.parser.get_isovalue(iso_idx, normalize=True)
                    iso_tensor = torch.tensor([[iso_norm]], dtype=torch.float32, device=device)

                    # ── Render predicted image ──
                    renders, alphas_grid, _ = self.rasterize_splats(
                        camtoworlds=camtoworld,
                        Ks=K,
                        width=w,
                        height=h,
                        sh_degree=cfg.sh_degree,
                        near_plane=cfg.near_plane,
                        far_plane=cfg.far_plane,
                        condition_vector=cond_tensor,
                        isovalue=iso_tensor,
                    )
                    pred = renders[0, ..., 0:3]
                    if cfg.white_bkgd:
                        pred = pred + (1.0 - alphas_grid[0])
                    pred = pred.clamp(0, 1)  # [H, W, 3]

                    # Save rendered image
                    img_np = (pred.cpu().numpy() * 255).astype(np.uint8)
                    fname = f"cond{param_idx:03d}_iso{iso_idx:02d}.png"
                    imageio.imwrite(f"{output_dir}/{fname}", img_np)
                    count += 1

                    # ── Load GT and compute metrics if available ──
                    gt_path = gt_lookup.get((param_idx, iso_idx, cam_idx))
                    if gt_path is not None and os.path.exists(gt_path):
                        gt_pil = Image.open(gt_path).convert("RGB")
                        if self.parser.factor > 1:
                            gw, gh = gt_pil.size
                            gt_pil = gt_pil.resize(
                                (gw // self.parser.factor, gh // self.parser.factor),
                                Image.BILINEAR,
                            )
                        gt_np = np.array(gt_pil, dtype=np.float32) / 255.0
                        gt = torch.from_numpy(gt_np).to(device)  # [H, W, 3]

                        # [1, C, H, W] for torchmetrics
                        pred_p = pred.unsqueeze(0).permute(0, 3, 1, 2)
                        gt_p = gt.unsqueeze(0).permute(0, 3, 1, 2)

                        psnr_val = self.psnr(pred_p, gt_p).item()
                        ssim_val = self.ssim(pred_p, gt_p).item()

                        group_psnrs.append(psnr_val)
                        group_ssims.append(ssim_val)

                        key = f"{group_name}/cond{param_idx:03d}_iso{iso_idx:02d}"
                        all_metrics[key] = {"psnr": psnr_val, "ssim": ssim_val}

                        per_param_psnr[f"{group_name}/cond{param_idx:03d}"].append(psnr_val)
                        per_param_ssim[f"{group_name}/cond{param_idx:03d}"].append(ssim_val)
                        per_iso_psnr[f"{group_name}/iso{iso_idx:02d}"].append(psnr_val)
                        per_iso_ssim[f"{group_name}/iso{iso_idx:02d}"].append(ssim_val)

            print(f"  Saved {count} images to {output_dir}")
            if group_psnrs:
                mean_psnr = np.mean(group_psnrs)
                mean_ssim = np.mean(group_ssims)
                print(f"  [{group_name}] PSNR={mean_psnr:.3f}  SSIM={mean_ssim:.4f}  "
                      f"({len(group_psnrs)} images with GT)")
                all_metrics[f"{group_name}/__overall__"] = {
                    "psnr": float(mean_psnr), "ssim": float(mean_ssim),
                    "num_images": len(group_psnrs),
                }

        # ── Per-param and per-iso averages ──
        summary = {"per_image": all_metrics}
        summary["per_param"] = {
            k: {"psnr": float(np.mean(v)), "ssim": float(np.mean(per_param_ssim[k]))}
            for k, v in per_param_psnr.items()
        }
        summary["per_iso"] = {
            k: {"psnr": float(np.mean(v)), "ssim": float(np.mean(per_iso_ssim[k]))}
            for k, v in per_iso_psnr.items()
        }

        # Print compact tables
        for group_name, _ in groups:
            param_keys = sorted(k for k in summary["per_param"] if k.startswith(group_name))
            iso_keys = sorted(k for k in summary["per_iso"] if k.startswith(group_name))

            if param_keys:
                print(f"\n  [{group_name}] Per-Param averages:")
                print(f"  {'Param':<20} {'PSNR':<8} {'SSIM':<8}")
                print(f"  {'-'*36}")
                for k in param_keys:
                    s = summary["per_param"][k]
                    print(f"  {k.split('/')[-1]:<20} {s['psnr']:<8.3f} {s['ssim']:<8.4f}")

            if iso_keys:
                print(f"\n  [{group_name}] Per-Iso averages:")
                print(f"  {'Iso':<20} {'PSNR':<8} {'SSIM':<8}")
                print(f"  {'-'*36}")
                for k in iso_keys:
                    s = summary["per_iso"][k]
                    print(f"  {k.split('/')[-1]:<20} {s['psnr']:<8.3f} {s['ssim']:<8.4f}")

        # Save JSON
        json_path = f"{base_dir}/metrics.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Metrics saved to {json_path}")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        # Use midpoint condition for viewer
        condition_vector = None
        isovalue = None
        if self.cfg.use_deformation and self.condition_dim > 0:
            condition_vector = 0.5 * torch.ones(1, self.condition_dim, device=self.device)
        if self.cfg.use_surface and self.isovalue_dim > 0:
            isovalue = 0.5 * torch.ones(1, self.isovalue_dim, device=self.device)

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
            condition_vector=condition_vector,  # NEW
            isovalue=isovalue,                   # NEW (surface)
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        
        # Load deformation field
        if cfg.use_deformation and runner.deform_field is not None:
            if "deform_field" in ckpts[0]:
                runner.deform_field.load_state_dict(ckpts[0]["deform_field"])
                print("Loaded deformation field from checkpoint")
        
        step = ckpts[0]["step"]
        runner.eval(step=step)
        
        # Run holdout evaluations (same as end of _run_stage2)
        if cfg.use_surface:
            eval_sets = []
            hp = getattr(runner, 'testset_holdout_param_unseen', None)
            eval_sets.append(("holdout_param__train_iso__unseen_view", hp))
            hb = getattr(runner, 'testset_holdout_both_unseen', None)
            eval_sets.append(("holdout_param__holdout_iso__unseen_view", hb))
            for eval_name, dataset in eval_sets:
                if dataset is not None and len(dataset) > 0:
                    print(f"\n--- {eval_name} ({len(dataset)} samples) ---")
                    runner.eval_holdout(step, dataset, eval_name)
        else:
            if getattr(runner, 'testset_holdout_unseen', None) is not None:
                runner.eval_holdout(step, runner.testset_holdout_unseen, "holdout_param__unseen_view")
        
        runner.render_traj(step=step)
        if cfg.use_surface:
            runner.save_params_view0_all_isovalues(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)



if __name__ == "__main__":

    # Config objects we can choose between.
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # Import BilateralGrid and related functions based on configuration
    if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
        if cfg.use_fused_bilagrid:
            cfg.use_bilateral_grid = True
            from fused_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )
        else:
            cfg.use_bilateral_grid = True
            from lib_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut:
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d` flag."

    cli(main, cfg, verbose=True)