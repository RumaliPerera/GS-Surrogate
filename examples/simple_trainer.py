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
from deformation_model import create_deformation_field #, create_hexplane_deformation_field
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
    port: int = 8081

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

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 2000 #100_000
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

    hard_mining_exponent: float = 1.0

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Stage-1 aggressive densification overrides (DefaultStrategy only)
    # Lower grow_grad2d → more splits in high-gradient regions
    stage1_grow_grad2d: float = 0.0001
    # Densify every N steps (default 100 is conservative)
    stage1_refine_every: int = 50

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
    deform_feature_dim: int = 64
    # Hidden dimension for deformation MLP
    deform_hidden_dim: int = 128
    # Learning rate for deformation network
    deform_lr: float = 1e-3
    # Deformation magnitude scaling (smaller = more stable training)
    deform_scale: float = 0.1
    # Regularization weight for small deformations (0 = disabled)
    deform_reg: float = 0.0
    # Path to names.txt file containing condition vectors
    names_file: str = "names.txt"

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
    stage2_canonical_lr_scale: float = 0.3
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
                    new_state[key] = val  

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
        self.train_conditions = None  # Conditions used for training
        self.test_conditions = None   # Conditions held out for testing
        
        if cfg.use_deformation and CONDITIONAL_AVAILABLE:
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
            
            # Train conditions on seen views
            self.testset_train_seen = ConditionalDataset(
                self.parser,
                split="train",  # Same cameras as training
                condition_indices=self.train_conditions,
            )
            # Train conditions on unseen views
            self.testset_train_unseen = self.valset  # Same as valset

            
            # Create test datasets for holdout conditions 
            if self.test_conditions:
                self.testset_holdout_unseen = ConditionalDataset(
                    self.parser,
                    split="val",  # held-out cameras (unseen during training)
                    condition_indices=self.test_conditions,
                )
                self.testset_holdout_seen = ConditionalDataset(
                    self.parser,
                    split="train",  # train cameras (seen during training)
                    condition_indices=self.test_conditions,
                )
            else:
                self.testset_holdout_unseen = None
                self.testset_holdout_seen = None
            
            self.condition_dim = self.parser.condition_dim
            print(f"Conditional mode: {self.parser.num_conditions} total conditions, dim={self.condition_dim}")
            print(f" - Training samples: {len(self.trainset)}")
            print(f" - Validation samples: {len(self.valset)}")
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
        # Initialize Deformation Field
        # ====================================================================
        self.deform_field = None
        self.deform_optimizers = []
        
        if cfg.use_deformation and CONDITIONAL_AVAILABLE and self.condition_dim > 0:

            self.deform_field = create_deformation_field(
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
            print(f"Deformation field initialized: {num_params:,} parameters")
            if cfg.learn_deform_alpha:
                print("  (with learnable alpha/opacity)")
            if cfg.learn_deform_sh:
                print("  (with learnable SH/color)")

        # ====================================================================
        # Load initial checkpoint (for canonical Gaussians)
        # ====================================================================
        if cfg.init_ckpt is not None:
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
            # Since batch_size=1 (usually), so each batch has only one image, so one condition !!!!
            cond = condition_vector[0] if condition_vector.dim() > 1 else condition_vector
            sh_coeffs = torch.cat([self.splats["sh0"], self.splats["shN"]], 1) if cfg.learn_deform_sh else None
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
        if cfg.use_deformation and cfg.reference_condition >= 0 and CONDITIONAL_AVAILABLE:
            # Two-stage training for conditional 4DGS
            self._train_two_stage()
        else:
            # Single-stage training (original 3DGS or deformation from start)
            self._train_single_stage()

    def _train_two_stage(self):
        """
        Two-stage training for deformable 3DGS:
        - Stage 1: Standard 3DGS on reference condition only
        - Stage 2: Train deformation on all conditions
        """
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        
        print(f"\n{'='*70}")
        print("TWO-STAGE CONDITIONAL 4DGS TRAINING")
        print(f"{'='*70}")
        print(f"Stage 1: Standard 3DGS on condition {cfg.reference_condition}")
        print(f"         Steps 0 to {cfg.deform_start_step - 1}")
        print(f"Stage 2: Deformation training on all {self.parser.num_conditions} conditions")
        print(f"         Steps {cfg.deform_start_step} to {cfg.max_steps - 1}")
        if cfg.freeze_canonical_in_stage2:
            print("         (Canonical Gaussians FROZEN in Stage 2)")
        print(f"{'='*70}\n")
        
        # ==================== Stage 1 ====================
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
        # Decay to 10% (not 1%) — stage 1 is short, aggressive decay starves positions
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.1 ** (1.0 / stage1_steps)
            ),
        ]
        if cfg.pose_opt:
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.1 ** (1.0 / stage1_steps)
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
        num_samples = len(self.trainset)
        sample_losses = torch.ones(num_samples)

        def _make_loader(weights):
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, num_samples=num_samples, replacement=True,
            )
            return torch.utils.data.DataLoader(
                self.trainset, batch_size=cfg.batch_size, sampler=sampler,
                num_workers=4, persistent_workers=True, pin_memory=True,
            )

        trainloader = _make_loader(sample_losses)
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
        
        # Deformation scheduler
        if self.deform_optimizers:
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.deform_optimizers[0], gamma=0.1 ** (1.0 / stage2_steps)
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
            pixels = data["image"].to(device) / 255.0
            num_train_rays_per_step = pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]
            
            # Get condition
            condition_vector = data["condition_vector"].to(device)
            condition_idx = data["condition_idx"]
            if isinstance(condition_idx, torch.Tensor):
                condition_idx = condition_idx.item()

            # Forward WITH deformation
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds, Ks=Ks, width=width, height=height,
                sh_degree=cfg.sh_degree, near_plane=cfg.near_plane,
                far_plane=cfg.far_plane, image_ids=image_ids,
                condition_vector=condition_vector,  # Apply deformation!
                render_mode="RGB", masks=masks,
            )
            colors = renders[..., 0:3]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)
                
            elif cfg.white_bkgd:
                colors = colors + (1.0 - alphas)

            # Loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            # Deformation regularization
            if cfg.deform_reg > 0.0 and self.deform_field is not None:
                deform_reg_loss = self.deform_field.get_regularization_loss(
                    self.splats["means"].detach(), condition_vector[0]
                )
                loss = loss + cfg.deform_reg * deform_reg_loss

            loss.backward()
            pbar.set_description(f"Stage 2 | loss={loss.item():.4f} | cond={condition_idx}")

            # Track per-sample loss for weighted resampling
            if "sample_idx" in data:
                sidx = data["sample_idx"]
                if isinstance(sidx, torch.Tensor):
                    sidx = sidx.item()
                sample_losses[sidx] = loss.item()

            # Rebuild sampler so high-error samples get drawn more often
            if step_offset > 0 and step_offset % resample_every == 0:
                weights = sample_losses ** cfg.hard_mining_exponent
                trainloader = _make_loader(weights)
                trainloader_iter = iter(trainloader)
                top5 = sample_losses.topk(5).values.tolist()
                print(f"\n  [Resample] step {step}: rebuilt sampler, "
                      f"top-5 losses: {[f'{v:.4f}' for v in top5]}")

            # Logging
            if world_rank == 0 and cfg.tb_every > 0 and step_offset % cfg.tb_every == 0:
                self.writer.add_scalar("stage2/loss", loss.item(), step)
                self.writer.add_scalar("stage2/l1loss", l1loss.item(), step)
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
        # Final Evaluation: 2x2 matrix (train/test conditions × seen/unseen views)
        # ====================================================================
        print(f"{'='*70}")
        
        # Training conditions: unseen views (this is the standard "val" eval)
        print(f"\n--- TRAIN Conditions x UNSEEN Views ---")
        self.eval_holdout(cfg.max_steps - 1, self.testset_train_unseen, "train_unseen")
        
        
        # Holdout/test conditions (if we have)
        if self.testset_holdout_unseen is not None:
            print(f"\n--- TEST Conditions x UNSEEN Views ---")
            self.eval_holdout(cfg.max_steps - 1, self.testset_holdout_unseen, "holdout_unseen")
            
        # Print summary table
        self._print_evaluation_summary(cfg.max_steps - 1)
        



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
            if cfg.use_deformation and "condition_vector" in data:
                condition_vector = data["condition_vector"].to(device)

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
                deform_reg_loss = self.deform_field.get_regularization_loss(
                    self.splats["means"].detach(),
                    condition_vector[0] if condition_vector.dim() > 1 else condition_vector,
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
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            # Get condition for evaluation
            condition_vector = None
            condition_idx = None
            camera_idx = None
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
            )
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            if cfg.white_bkgd:
                colors = colors[..., 0:3] + (1.0 - alphas)
            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
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
                
                metrics["psnr"].append(psnr_val)
                metrics["ssim"].append(ssim_val)
                metrics["lpips"].append(lpips_val)
                
                # Track per-condition metrics
                if condition_idx is not None:
                    per_condition_metrics[condition_idx]["psnr"].append(psnr_val)
                    per_condition_metrics[condition_idx]["ssim"].append(ssim_val)
                    per_condition_metrics[condition_idx]["lpips"].append(lpips_val)
                
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
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
                condition_stats = {}
                for cond_idx in sorted(per_condition_metrics.keys()):
                    cond_metrics = per_condition_metrics[cond_idx]
                    cond_psnr = torch.stack(cond_metrics["psnr"]).mean().item()
                    cond_ssim = torch.stack(cond_metrics["ssim"]).mean().item()
                    cond_lpips = torch.stack(cond_metrics["lpips"]).mean().item()
                    condition_stats[cond_idx] = {
                        "psnr": cond_psnr,
                        "ssim": cond_ssim,
                        "lpips": cond_lpips,
                    }
                
                # Print all conditions in a compact table format
                print(f"{'Cond':<6} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8}")
                print("-" * 32)
                for cond_idx in sorted(condition_stats.keys()):
                    cs = condition_stats[cond_idx]
                    # Mark reference condition with *
                    marker = "*" if cond_idx == cfg.reference_condition else " "
                    print(f"{cond_idx:<5}{marker} {cs['psnr']:<8.3f} {cs['ssim']:<8.4f} {cs['lpips']:<8.3f}")
                
                # Summary statistics
                all_psnrs = [cs["psnr"] for cs in condition_stats.values()]
                print("-" * 32)
                print(f"{'Mean':<6} {sum(all_psnrs)/len(all_psnrs):<8.3f}")
                print(f"{'Min':<6} {min(all_psnrs):<8.3f} (cond {min(condition_stats, key=lambda x: condition_stats[x]['psnr'])})")
                print(f"{'Max':<6} {max(all_psnrs):<8.3f} (cond {max(condition_stats, key=lambda x: condition_stats[x]['psnr'])})")
                
                if cfg.reference_condition in condition_stats:
                    print(f"{'Ref':<6} {condition_stats[cfg.reference_condition]['psnr']:<8.3f} (cond {cfg.reference_condition})")
                
                # Save per-condition stats
                stats["per_condition"] = condition_stats
            
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
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            condition_vector = data["condition_vector"].to(device)
            condition_idx = data["condition_idx"]
            if isinstance(condition_idx, torch.Tensor):
                condition_idx = condition_idx.item()

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
            )
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            if cfg.white_bkgd:
                colors = colors[..., 0:3] + (1.0 - alphas)
            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
                # Write images
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                
                cam_num = per_condition_cam_count[condition_idx]
                per_condition_cam_count[condition_idx] += 1
                filename = f"{name}_step{step}_cond{condition_idx:03d}_cam{cam_num:04d}.png"
                imageio.imwrite(f"{self.render_dir}/{filename}", canvas)

                pixels_p = pixels.permute(0, 3, 1, 2)
                colors_p = colors.permute(0, 3, 1, 2)
                
                psnr_val = self.psnr(colors_p, pixels_p)
                ssim_val = self.ssim(colors_p, pixels_p)
                lpips_val = self.lpips(colors_p, pixels_p)
                
                metrics["psnr"].append(psnr_val)
                metrics["ssim"].append(ssim_val)
                metrics["lpips"].append(lpips_val)
                
                per_condition_metrics[condition_idx]["psnr"].append(psnr_val)
                per_condition_metrics[condition_idx]["ssim"].append(ssim_val)
                per_condition_metrics[condition_idx]["lpips"].append(lpips_val)

        if world_rank == 0:
            ellipse_time /= len(valloader)
            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
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
            print(f"{'Cond':<6} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8}")
            print("-" * 32)
            
            condition_stats = {}
            psnr_values = []
            for cond_idx in sorted(per_condition_metrics.keys()):
                cond_metrics = per_condition_metrics[cond_idx]
                cond_psnr = torch.stack(cond_metrics["psnr"]).mean().item()
                cond_ssim = torch.stack(cond_metrics["ssim"]).mean().item()
                cond_lpips = torch.stack(cond_metrics["lpips"]).mean().item()
                condition_stats[cond_idx] = {
                    "psnr": cond_psnr,
                    "ssim": cond_ssim,
                    "lpips": cond_lpips,
                }
                psnr_values.append(cond_psnr)
                print(f"{cond_idx:<6} {cond_psnr:<8.3f} {cond_ssim:<8.4f} {cond_lpips:<8.3f}")
            
            print("-" * 32)
            print(f"Mean   {np.mean(psnr_values):.3f}")
            print(f"Min    {np.min(psnr_values):.3f}   (cond {list(per_condition_metrics.keys())[np.argmin(psnr_values)]})")
            print(f"Max    {np.max(psnr_values):.3f}   (cond {list(per_condition_metrics.keys())[np.argmax(psnr_values)]})")
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
        if cfg.use_deformation and hasattr(self.parser, 'condition_min'):
            # Use midpoint (0.5 in min-max normalized space)
            condition_vector = 0.5 * torch.ones(1, self.condition_dim, device=device)

        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=24)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, alphas, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                condition_vector=condition_vector,  # NEW
            )  # [1, H, W, 4]
            colors = renders[..., 0:3]
            if cfg.white_bkgd:
                colors = colors + (1.0 - alphas)
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
    ):
        """
        Render all views at a specific condition.
        
        Args:
            condition_vector: Raw [3] condition vector (will be normalized internally)
            output_dir: Path to save rendered images
            use_val_cameras: Use validation cameras (True) or all cameras (False)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Normalize condition (min-max to [0, 1])
        if hasattr(self.parser, 'condition_min'):
            cond_normalized = (condition_vector - self.parser.condition_min) / self.parser.condition_range
        else:
            cond_normalized = condition_vector
        
        cond_tensor = torch.from_numpy(cond_normalized).float().to(self.device).unsqueeze(0)
        
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
            )
            
            img = (renders[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            imageio.imwrite(f"{output_dir}/view_{idx:04d}.png", img)
        
        print(f"Saved {len(dataset)} images to {output_dir}")

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
        if self.cfg.use_deformation and self.condition_dim > 0:
            condition_vector = 0.5 * torch.ones(1, self.condition_dim, device=self.device)

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
        runner.render_traj(step=step)
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