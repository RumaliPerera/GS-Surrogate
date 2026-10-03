#!/usr/bin/env python3
"""
GS-Surrogate Inference Script (TF Adapter Mode)

Renders an image given:
    - view_id:      camera index (0-based, matching COLMAP image order)
    - sim_params:   3 raw simulation parameter values (e.g., OmM, OmB, h)
    - tf_residual:  4 TF control-point offsets [s1, o1, s2, o2]

Requires TWO checkpoints:
    - vol_ckpt:  stage-2 volume model  (canonical splats + Fsim deformation)
    - tf_ckpt:   stage-3 TF adapter    (Fvis appearance adapter)

Pipeline:
    Canonical Gaussians (from vol_ckpt, frozen)
      -> Fsim: VolumeDeformation(sim_params)  (from vol_ckpt, frozen)
      -> Fvis: TFAppearanceAdapter(sim_params, tf_residual)  (from tf_ckpt, frozen)
      -> gsplat rasterization -> RGB image

Usage:
    python inference_tf.py \\
        --data_dir /path/to/dataset \\
        --vol_ckpt /path/to/volume_stage2.pt \\
        --tf_ckpt  /path/to/tf_adapter.pt \\
        --view_id 42 \\
        --sim_params 0.135 0.0225 0.70 \\
        --tf_residual 0.0 0.0 0.0 0.0 \\
        --output rendered.png

    # Diagnostic: test parameter sensitivity (3 sim x 3 TF = 9 images)
    python inference_tf.py \\
        --data_dir /path/to/dataset \\
        --vol_ckpt /path/to/volume_stage2.pt \\
        --tf_ckpt  /path/to/tf_adapter.pt \\
        --view_id 6 \\
        --sim_params 0.135 0.0225 0.70 \\
        --tf_residual 0.016 0.011 0.484 0.125 \\
        --diag --output_dir ./diag_output/
"""

import argparse
import os
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn.functional as F

from deformation_model_volume import create_deformation_field as create_volume_deformation_field
from deformation_model_tf import create_tf_appearance_adapter
from gsplat.rendering import rasterization
from conditional_dataset import read_cameras_binary, read_images_binary, qvec2rotmat


# =========================================================================
# Camera loader (lightweight -- poses + intrinsics only, no images)
# =========================================================================

def load_cameras(data_dir: str, factor: int = 1, normalize: bool = True):
    """
    Parse COLMAP sparse reconstruction for camera poses and intrinsics.

    Returns
    -------
    camtoworlds : ndarray (V, 4, 4)
    Ks           : ndarray (V, 3, 3)
    heights      : list[int]
    widths       : list[int]
    image_names  : list[str]
    scene_scale  : float
    """
    data_dir = Path(data_dir)
    sparse_dir = data_dir / "sparse" / "0"
    if not sparse_dir.exists():
        sparse_dir = data_dir / "sparse"
    assert sparse_dir.exists(), f"COLMAP sparse dir not found in {data_dir}"

    cameras = read_cameras_binary(str(sparse_dir / "cameras.bin"))
    images = read_images_binary(str(sparse_dir / "images.bin"))

    _c2ws, _Ks, _names, _heights, _widths = [], [], [], [], []

    for img_id in images:
        img_data = images[img_id]
        cam_data = cameras[img_data["camera_id"]]
        params = cam_data["params"]
        w, h = cam_data["width"], cam_data["height"]
        model_id = cam_data["model_id"]

        if model_id == 0:                        # SIMPLE_PINHOLE
            fx = fy = params[0]; cx, cy = params[1], params[2]
        elif model_id == 1:                      # PINHOLE
            fx, fy = params[0], params[1]; cx, cy = params[2], params[3]
        elif model_id in (2, 3, 4, 5, 6):       # RADIAL / OPENCV variants
            fx, fy = params[0], params[1]; cx, cy = params[2], params[3]
        else:
            fx = fy = params[0]; cx, cy = w / 2, h / 2

        if factor > 1:
            fx /= factor; fy /= factor; cx /= factor; cy /= factor
            w //= factor; h //= factor

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        R = qvec2rotmat(img_data["qvec"])
        t = img_data["tvec"]
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t

        _c2ws.append(c2w); _Ks.append(K); _names.append(img_data["name"])
        _heights.append(int(h)); _widths.append(int(w))

    # Sort by name (same order the trainer uses)
    inds = np.argsort(_names)
    camtoworlds = np.stack([_c2ws[i] for i in inds])
    Ks = np.stack([_Ks[i] for i in inds])
    image_names = [_names[i] for i in inds]
    heights = [_heights[i] for i in inds]
    widths = [_widths[i] for i in inds]

    if normalize:
        cam_centers = camtoworlds[:, :3, 3]
        center = cam_centers.mean(axis=0)
        norm_factor = np.linalg.norm(cam_centers - center, axis=1).max()
        camtoworlds[:, :3, 3] = (camtoworlds[:, :3, 3] - center) / norm_factor
        scene_scale = np.linalg.norm(
            camtoworlds[:, :3, 3] - camtoworlds[:, :3, 3].mean(axis=0),
            axis=1,
        ).max()
    else:
        scene_scale = 1.0

    return camtoworlds, Ks, heights, widths, image_names, scene_scale


# =========================================================================
# Model loading
# =========================================================================

def load_models(
    vol_ckpt_path: str,
    tf_ckpt_path: str,
    scene_scale: float,
    device: torch.device,
    *,
    condition_dim: int = 3,
    deform_feature_dim: int = 128,
    deform_hidden_dim: int = 512,
    deform_scale: float = 0.1,
    learn_deform_alpha: bool = False,
    learn_deform_sh: bool = False,
    sh_degree: int = 3,
    tf_dim: int = 4,
    tf_feature_dim: int = 64,
    tf_hidden_dim: int = 256,
    tf_alpha_scale: float = 0.1,
    tf_sh_scale: float = 0.1,
    tf_num_layers: int = 3,
    global_scale: float = 1.0,
):
    """
    Load from TWO separate checkpoints:
        vol_ckpt:  stage-2 volume model  (canonical splats + Fsim)
        tf_ckpt:   stage-3 TF adapter    (Fvis)
    """
    sh_dim = ((sh_degree + 1) ** 2) * 3
    eff_scene_scale = scene_scale * 1.1 * global_scale

    # ── Load volume checkpoint (canonical splats + Fsim) ─────────────
    print(f"Loading volume ckpt: {vol_ckpt_path}")
    vol_ckpt = torch.load(vol_ckpt_path, map_location=device, weights_only=True)
    print(f"  Keys: {list(vol_ckpt.keys())}")

    # Canonical Gaussians come from volume checkpoint
    splats = {k: v.to(device) for k, v in vol_ckpt["splats"].items()}
    print(f"  Canonical Gaussians: {splats['means'].shape[0]:,} points")

    # Auto-detect learn_alpha / learn_sh from volume checkpoint
    vol_sd_key = "deform_field"
    if vol_sd_key in vol_ckpt:
        vol_keys = set(vol_ckpt[vol_sd_key].keys())
        if any("delta_alpha" in k for k in vol_keys):
            learn_deform_alpha = True
        if any("delta_sh" in k for k in vol_keys):
            learn_deform_sh = True
    print(f"  Auto-detected: learn_alpha={learn_deform_alpha}, learn_sh={learn_deform_sh}")

    # Build and load Fsim
    vol_deform = create_volume_deformation_field(
        condition_dim=condition_dim,
        feature_dim=deform_feature_dim,
        hidden_dim=deform_hidden_dim,
        deform_scale=deform_scale,
        scene_scale=eff_scene_scale,
        learn_alpha=learn_deform_alpha,
        learn_sh=learn_deform_sh,
        sh_dim=sh_dim,
    ).to(device)
    vol_deform.load_state_dict(vol_ckpt[vol_sd_key])
    vol_deform.eval()
    for p in vol_deform.parameters():
        p.requires_grad = False
    print(f"  Fsim loaded: {sum(p.numel() for p in vol_deform.parameters()):,} params (frozen)")

    # Normalization stats (from whichever checkpoint has them)
    if "condition_min" in vol_ckpt:
        cond_min = np.array(vol_ckpt["condition_min"], dtype=np.float32)
        cond_max = np.array(vol_ckpt["condition_max"], dtype=np.float32)
    else:
        cond_min = cond_max = None  # will try tf_ckpt below

    # ── Load TF adapter checkpoint (Fvis) ────────────────────────────
    print(f"\nLoading TF adapter ckpt: {tf_ckpt_path}")
    tf_ckpt = torch.load(tf_ckpt_path, map_location=device, weights_only=True)
    print(f"  Keys: {list(tf_ckpt.keys())}")

    tf_adapter = create_tf_appearance_adapter(
        condition_dim=condition_dim,
        tf_dim=tf_dim,
        feature_dim=tf_feature_dim,
        hidden_dim=tf_hidden_dim,
        alpha_scale=tf_alpha_scale,
        sh_scale=tf_sh_scale,
        scene_scale=eff_scene_scale,
        sh_dim=sh_dim,
        num_mlp_layers=tf_num_layers,
    ).to(device)
    tf_adapter.load_state_dict(tf_ckpt["tf_adapter"])
    tf_adapter.eval()
    for p in tf_adapter.parameters():
        p.requires_grad = False
    print(f"  Fvis loaded: {sum(p.numel() for p in tf_adapter.parameters()):,} params (frozen)")

    # Normalization stats — prefer tf_ckpt (it should always have them)
    if "condition_min" in tf_ckpt:
        cond_min = np.array(tf_ckpt["condition_min"], dtype=np.float32)
        cond_max = np.array(tf_ckpt["condition_max"], dtype=np.float32)
    assert cond_min is not None, "Neither checkpoint contains condition_min/max!"
    cond_range = cond_max - cond_min + 1e-8
    tf_max_abs = np.array(tf_ckpt["tf_max_abs"], dtype=np.float32)

    print(f"\n  Condition min: {cond_min}")
    print(f"  Condition max: {cond_max}")
    print(f"  TF max|val|:   {tf_max_abs}")

    return (splats, vol_deform, tf_adapter,
            cond_min, cond_max, cond_range, tf_max_abs, learn_deform_sh)


# =========================================================================
# Core render function
# =========================================================================

@torch.no_grad()
def render_image(
    splats: dict,
    vol_deform,
    tf_adapter,
    camtoworld: np.ndarray,        # (4, 4)
    K: np.ndarray,                 # (3, 3)
    height: int,
    width: int,
    condition_vec: torch.Tensor,   # (D,) normalized [0,1]
    tf_vec: torch.Tensor,          # (4,) normalized [-1,1]
    device: torch.device,
    sh_degree: int = 3,
    learn_deform_sh: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """
    Full pipeline: canonical -> Fsim -> Fvis -> rasterize.
    Returns (H, W, 3) uint8 image.
    """
    c2w = torch.from_numpy(camtoworld).float().to(device).unsqueeze(0)
    K_t = torch.from_numpy(K).float().to(device).unsqueeze(0)
    viewmat = torch.linalg.inv(c2w)

    means        = splats["means"]
    quats        = splats["quats"]
    scales       = splats["scales"]
    opacities    = splats["opacities"]
    sh_coeffs    = torch.cat([splats["sh0"], splats["shN"]], dim=1)

    if verbose:
        print(f"\n  [diag] condition_vec = {condition_vec}")
        print(f"  [diag] tf_vec        = {tf_vec}")
        print(f"  [diag] canonical means:  mean={means.mean().item():.6f}, std={means.std().item():.6f}")
        print(f"  [diag] canonical opac:   mean={opacities.mean().item():.4f}")

    # Stage 1: Fsim (volume deformation)
    vol_m, vol_q, vol_s, vol_o, vol_sh = vol_deform.apply_deformation(
        means, quats, scales, opacities.unsqueeze(-1),
        condition_vec,
        sh=sh_coeffs if learn_deform_sh else None,
    )
    if not learn_deform_sh:
        vol_sh = sh_coeffs

    if verbose:
        d_xyz = (vol_m - means).abs()
        d_opa = (vol_o.squeeze(-1) - opacities).abs()
        print(f"  [diag] Fsim |delta_xyz|: mean={d_xyz.mean().item():.6f}, max={d_xyz.max().item():.6f}")
        print(f"  [diag] Fsim |delta_opa|: mean={d_opa.mean().item():.6f}, max={d_opa.max().item():.6f}")
        if learn_deform_sh:
            d_sh = (vol_sh - sh_coeffs).abs()
            print(f"  [diag] Fsim |delta_sh|:  mean={d_sh.mean().item():.6f}, max={d_sh.max().item():.6f}")

    # Stage 2: Fvis (TF appearance adapter)
    fin_m, fin_q, fin_s, fin_o, fin_sh = tf_adapter.apply_deformation(
        vol_m, vol_q, vol_s, vol_o,
        condition_vec, tf_vector=tf_vec, sh=vol_sh,
    )

    if verbose:
        d_opa2 = (fin_o.squeeze(-1) - vol_o.squeeze(-1)).abs()
        d_sh2 = (fin_sh - vol_sh).abs() if fin_sh is not None and vol_sh is not None else None
        print(f"  [diag] Fvis |delta_opa|: mean={d_opa2.mean().item():.6f}, max={d_opa2.max().item():.6f}")
        if d_sh2 is not None:
            print(f"  [diag] Fvis |delta_sh|:  mean={d_sh2.mean().item():.6f}, max={d_sh2.max().item():.6f}")
        print(f"  [diag] final opac (sigmoid): mean={torch.sigmoid(fin_o.squeeze(-1)).mean().item():.4f}")

    # Log / logit -> real space
    fin_s_exp = torch.exp(fin_s)
    fin_o_sig = torch.sigmoid(fin_o.squeeze(-1))

    # Rasterize
    colors, alphas, _ = rasterization(
        means=fin_m,
        quats=fin_q,
        scales=fin_s_exp,
        opacities=fin_o_sig,
        colors=fin_sh,
        viewmats=viewmat,
        Ks=K_t,
        width=width,
        height=height,
        packed=False,
        sh_degree=sh_degree,
    )

    img = colors[0, ..., :3].clamp(0, 1).cpu().numpy()
    return (img * 255).astype(np.uint8)


# =========================================================================
# CLI
# =========================================================================

def main():
    p = argparse.ArgumentParser(description="GS-Surrogate TF inference")

    # -- Paths --
    p.add_argument("--data_dir",  type=str, required=True)
    p.add_argument("--vol_ckpt",  type=str, required=True,
                   help="Path to volume-stage checkpoint (canonical splats + Fsim)")
    p.add_argument("--tf_ckpt",   type=str, required=True,
                   help="Path to TF adapter checkpoint (Fvis)")
    p.add_argument("--output",    type=str, default="rendered.png")
    p.add_argument("--output_dir", type=str, default=None)

    # -- Inputs --
    p.add_argument("--view_id",      type=int,   default=0,
                   help="Camera index (0-based); ignored with --all_views")
    p.add_argument("--sim_params",   type=float, nargs="+", required=True)
    p.add_argument("--tf_residual",  type=float, nargs=4,   required=True)

    # -- Model config (must match training) --
    p.add_argument("--condition_dim",      type=int,   default=3)
    p.add_argument("--deform_feature_dim", type=int,   default=128)
    p.add_argument("--deform_hidden_dim",  type=int,   default=512)
    p.add_argument("--deform_scale",       type=float, default=0.1)
    p.add_argument("--learn_deform_alpha", action="store_true")
    p.add_argument("--learn_deform_sh",    action="store_true")
    p.add_argument("--sh_degree",          type=int,   default=3)
    p.add_argument("--tf_feature_dim",     type=int,   default=64)
    p.add_argument("--tf_hidden_dim",      type=int,   default=256)
    p.add_argument("--tf_alpha_scale",     type=float, default=0.1)
    p.add_argument("--tf_sh_scale",        type=float, default=0.1)
    p.add_argument("--tf_num_layers",      type=int,   default=3)
    p.add_argument("--global_scale",       type=float, default=1.0)

    # -- Data config --
    p.add_argument("--data_factor", type=int, default=1)
    p.add_argument("--normalize",   action="store_true", default=True)
    p.add_argument("--no_normalize", dest="normalize", action="store_false")

    # -- Sweep mode --
    p.add_argument("--all_views",   action="store_true",
                   help="Render all views and report per-frame timing")
    p.add_argument("--sweep_tf",    action="store_true")
    p.add_argument("--sweep_steps", type=int, default=10)
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print deformation diagnostics")
    p.add_argument("--diag", action="store_true",
                   help="Diagnostic: render same view with 3 different sim params + 3 TFs")

    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load cameras ────────────────────────────────────────────────
    print("Loading cameras ...")
    c2ws, Ks, heights, widths, names, scene_scale = load_cameras(
        args.data_dir, factor=args.data_factor, normalize=args.normalize,
    )
    V = len(c2ws)
    print(f"  {V} views, scene_scale={scene_scale:.4f}")
    assert args.all_views or 0 <= args.view_id < V, \
        f"view_id {args.view_id} out of range [0, {V})"

    # ── Load models ─────────────────────────────────────────────────
    print("\nLoading models ...")
    splats, vol_deform, tf_adapter, cond_min, cond_max, cond_range, tf_max_abs, \
        learn_deform_sh = load_models(
            vol_ckpt_path=args.vol_ckpt,
            tf_ckpt_path=args.tf_ckpt,
            scene_scale=scene_scale,
            device=device,
            condition_dim=args.condition_dim,
            deform_feature_dim=args.deform_feature_dim,
            deform_hidden_dim=args.deform_hidden_dim,
            deform_scale=args.deform_scale,
            learn_deform_alpha=args.learn_deform_alpha,
            learn_deform_sh=args.learn_deform_sh,
            sh_degree=args.sh_degree,
            tf_feature_dim=args.tf_feature_dim,
            tf_hidden_dim=args.tf_hidden_dim,
            tf_alpha_scale=args.tf_alpha_scale,
            tf_sh_scale=args.tf_sh_scale,
            tf_num_layers=args.tf_num_layers,
            global_scale=args.global_scale,
        )

    # ── Normalize inputs ────────────────────────────────────────────
    sim_raw = np.array(args.sim_params, dtype=np.float32)
    assert len(sim_raw) == args.condition_dim, \
        f"Expected {args.condition_dim} sim params, got {len(sim_raw)}"

    sim_norm = (sim_raw - cond_min) / cond_range               # [0, 1]
    cond_t = torch.from_numpy(sim_norm).float().to(device)

    tf_raw = np.array(args.tf_residual, dtype=np.float32)
    tf_norm = tf_raw / (tf_max_abs + 1e-8)                     # [-1, 1]
    tf_t = torch.from_numpy(tf_norm).float().to(device)

    print(f"\nInputs:")
    if args.all_views:
        print(f"  Views: ALL ({V})")
    else:
        print(f"  View:  {args.view_id} ({names[args.view_id]})")
    print(f"  Sim:   {sim_raw}  -> normalized {sim_norm}")
    print(f"  TF:    {tf_raw}  -> normalized {tf_norm}")

    # ── Render ──────────────────────────────────────────────────────
    import time

    if args.diag:
        # ── Diagnostic mode: test parameter sensitivity ─────────────
        c2w, K = c2ws[args.view_id], Ks[args.view_id]
        h, w = heights[args.view_id], widths[args.view_id]
        out_dir = args.output_dir or "diag_output"
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print("DIAGNOSTIC MODE: Testing parameter sensitivity")
        print(f"{'='*60}")

        # Test 3 sim param extremes (min, user-specified, max)
        sim_tests = [
            ("sim_min",  cond_min.copy()),
            ("sim_user", sim_raw.copy()),
            ("sim_max",  cond_max.copy()),
        ]
        # Test 3 TF settings
        tf_tests = [
            ("tf_zero",  np.zeros(4, dtype=np.float32)),
            ("tf_user",  tf_raw.copy()),
            ("tf_pos",   tf_max_abs.copy() * 0.5),
        ]

        prev_img = None
        for sim_label, sim_val in sim_tests:
            for tf_label, tf_val in tf_tests:
                sn = (sim_val - cond_min) / cond_range
                tn = tf_val / (tf_max_abs + 1e-8)
                ct = torch.from_numpy(sn.astype(np.float32)).float().to(device)
                tt = torch.from_numpy(tn.astype(np.float32)).float().to(device)

                print(f"\n--- {sim_label} + {tf_label} ---")
                print(f"  sim_raw={sim_val} -> norm={sn}")
                print(f"  tf_raw ={tf_val}  -> norm={tn}")

                img = render_image(
                    splats, vol_deform, tf_adapter,
                    c2w, K, h, w, ct, tt, device,
                    sh_degree=args.sh_degree,
                    learn_deform_sh=learn_deform_sh,
                    verbose=True,
                )

                fname = f"{sim_label}_{tf_label}.png"
                imageio.imwrite(os.path.join(out_dir, fname), img)
                print(f"  -> {fname}")

                # Compare with previous image
                if prev_img is not None:
                    diff = np.abs(img.astype(float) - prev_img.astype(float))
                    print(f"  pixel diff vs prev: mean={diff.mean():.2f}, "
                          f"max={diff.max():.0f}, "
                          f"nonzero={np.count_nonzero(diff > 1)}/{diff.size}")
                prev_img = img

        print(f"\n{'='*60}")
        print(f"Diagnostic images saved to {out_dir}/")
        print(f"{'='*60}")

    elif args.all_views:
        # ── Render ALL views with timing ────────────────────────────
        out_dir = args.output_dir or "renders_all_views"
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nRendering all {V} views -> {out_dir}/")

        # Warmup
        print("Warmup (1 frame) ...")
        _ = render_image(
            splats, vol_deform, tf_adapter,
            c2ws[0], Ks[0], heights[0], widths[0],
            cond_t, tf_t, device,
            sh_degree=args.sh_degree,
            learn_deform_sh=learn_deform_sh,
        )
        torch.cuda.synchronize()
        print("Warmup done.\n")

        times = []
        for vid in range(V):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            img = render_image(
                splats, vol_deform, tf_adapter,
                c2ws[vid], Ks[vid], heights[vid], widths[vid],
                cond_t, tf_t, device,
                sh_degree=args.sh_degree,
                learn_deform_sh=learn_deform_sh,
            )

            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            times.append(elapsed_ms)

            out_path = os.path.join(out_dir, f"view_{vid:03d}.png")
            imageio.imwrite(out_path, img)

            if vid % 25 == 0 or vid == V - 1:
                print(f"  [{vid+1:3d}/{V}] {elapsed_ms:6.1f} ms  -> {out_path}")

        times_arr = np.array(times)
        print(f"\n{'='*50}")
        print(f"Rendering Statistics ({V} views)")
        print(f"{'='*50}")
        print(f"  Mean:   {times_arr.mean():.2f} ms/frame")
        print(f"  Median: {np.median(times_arr):.2f} ms/frame")
        print(f"  Min:    {times_arr.min():.2f} ms/frame")
        print(f"  Max:    {times_arr.max():.2f} ms/frame")
        print(f"  Std:    {times_arr.std():.2f} ms")
        print(f"  FPS:    {1000.0 / times_arr.mean():.1f}")
        print(f"  Total:  {times_arr.sum() / 1000:.2f} s")
        print(f"{'='*50}")

    elif args.sweep_tf and args.output_dir:
        # ── Sweep TF ────────────────────────────────────────────────
        c2w, K = c2ws[args.view_id], Ks[args.view_id]
        h, w = heights[args.view_id], widths[args.view_id]
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\nSweeping TF in {args.sweep_steps} steps -> {args.output_dir}/")
        for i in range(args.sweep_steps + 1):
            t = i / args.sweep_steps
            tf_interp = torch.from_numpy(tf_norm * t).float().to(device)
            img = render_image(
                splats, vol_deform, tf_adapter,
                c2w, K, h, w, cond_t, tf_interp, device,
                sh_degree=args.sh_degree,
                learn_deform_sh=learn_deform_sh,
                verbose=(i == 0 or i == args.sweep_steps),
            )
            out = os.path.join(args.output_dir, f"tf_sweep_{i:03d}.png")
            imageio.imwrite(out, img)
            print(f"  [{i:3d}/{args.sweep_steps}] t={t:.2f} -> {out}")

    else:
        # ── Single view ─────────────────────────────────────────────
        c2w, K = c2ws[args.view_id], Ks[args.view_id]
        h, w = heights[args.view_id], widths[args.view_id]

        # Warmup
        _ = render_image(
            splats, vol_deform, tf_adapter,
            c2w, K, h, w, cond_t, tf_t, device,
            sh_degree=args.sh_degree,
            learn_deform_sh=learn_deform_sh,
        )
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        img = render_image(
            splats, vol_deform, tf_adapter,
            c2w, K, h, w, cond_t, tf_t, device,
            sh_degree=args.sh_degree,
            learn_deform_sh=learn_deform_sh,
            verbose=args.verbose,
        )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        imageio.imwrite(args.output, img)
        print(f"Saved: {args.output} ({w}x{h})")
        print(f"Render time: {elapsed_ms:.2f} ms ({1000/elapsed_ms:.1f} FPS)")


if __name__ == "__main__":
    main()