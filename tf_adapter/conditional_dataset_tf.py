import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image

from conditional_dataset import (
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    qvec2rotmat,
)


# ============================================================================
# TF name parsing
# ============================================================================

def parse_tf_name(name: str) -> Optional[np.ndarray]:

    parts = name.strip().split('_')
    if len(parts) != 8:
        return None
    try:
        s1 = float(parts[1])
        o1 = float(parts[3])
        s2 = float(parts[5])
        o2 = float(parts[7])
        return np.array([s1, o1, s2, o2], dtype=np.float32)
    except (ValueError, IndexError):
        return None


def is_base_tf(tf_vector: np.ndarray, tol: float = 1e-6) -> bool:
    
    return np.all(np.abs(tf_vector) < tol)


# ============================================================================
# TF Conditional Parser
# ============================================================================

class TFConditionalParser:

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = True,
        test_every: int = 8,
        names_file: str = "names.txt",
        tf_file: str = "transfer_functions.txt",
    ):
        self.data_dir = Path(data_dir)
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        # Parse simulation parameters
        self._parse_conditions(names_file)

        # Parse transfer functions
        self._parse_transfer_functions(tf_file)

        self._find_tf_folders()

        self._parse_colmap()

        self._build_samples()


    # ------------------------------------------------------------------ #
    # Parse simulation parameters (same as ConditionalParser)
    # ------------------------------------------------------------------ #
    def _parse_conditions(self, names_file: str):
        names_path = self.data_dir / names_file
        if not names_path.exists():
            raise FileNotFoundError(f"Conditions file not found: {names_path}")

        self.condition_vectors = {}
        self.condition_names = {}

        with open(names_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('_')
                try:
                    idx = int(parts[0])
                except ValueError:
                    continue
                try:
                    vec = [float(p) for p in parts[1:]]
                except ValueError:
                    continue
                if len(vec) == 0:
                    continue
                self.condition_vectors[idx] = np.array(vec, dtype=np.float32)
                self.condition_names[idx] = line

        self.num_conditions = len(self.condition_vectors)
        first_vec = next(iter(self.condition_vectors.values()))
        self.condition_dim = len(first_vec)

        # Verify dimensionality
        for idx, vec in self.condition_vectors.items():
            if len(vec) != self.condition_dim:
                raise ValueError(
                    f"Condition {idx} has dim {len(vec)}, expected {self.condition_dim}"
                )

        # Min-max normalisation stats
        all_vecs = np.stack(list(self.condition_vectors.values()))
        self.condition_min = all_vecs.min(axis=0)
        self.condition_max = all_vecs.max(axis=0)
        self.condition_range = self.condition_max - self.condition_min + 1e-8

        print(f"Loaded {self.num_conditions} simulation conditions from {names_file}")
        print(f"  Sim param dim: {self.condition_dim}")
        print(f"  Sim param min: {self.condition_min}")
        print(f"  Sim param max: {self.condition_max}")

    # ------------------------------------------------------------------ #
    # Parse transfer functions
    # ------------------------------------------------------------------ #
    def _parse_transfer_functions(self, tf_file: str):
        tf_path = self.data_dir / tf_file
        if not tf_path.exists():
            raise FileNotFoundError(f"TF file not found: {tf_path}")

        self.tf_names: List[str] = []       
        self.tf_vectors: List[np.ndarray] = []  # tf_idx → 4D vector
        self.base_tf_idx: Optional[int] = None

        with open(tf_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vec = parse_tf_name(line)
                if vec is None:
                    print(f"  Warning: skipping malformed TF line: {line}")
                    continue
                idx = len(self.tf_names)
                self.tf_names.append(line)
                self.tf_vectors.append(vec)
                if is_base_tf(vec):
                    self.base_tf_idx = idx

        self.num_tfs = len(self.tf_names)
        self.tf_dim = 4  # always [s1, o1, s2, o2]

        # Build name → index lookup for folder matching
        self._tf_name_to_idx: Dict[str, int] = {}
        for i, name in enumerate(self.tf_names):
            self._tf_name_to_idx[name] = i

        # Min-max normalisation stats for TF vectors
        all_tf_vecs = np.stack(self.tf_vectors)
        self.tf_min = all_tf_vecs.min(axis=0)
        self.tf_max = all_tf_vecs.max(axis=0)
        self.tf_range = self.tf_max - self.tf_min + 1e-8
   
        self.tf_max_abs = np.maximum(np.abs(self.tf_min), np.abs(self.tf_max))

        print(f"Loaded {self.num_tfs} transfer functions from {tf_file}")
        print(f"  TF dim: {self.tf_dim}")
        print(f"  TF min: {self.tf_min}")
        print(f"  TF max: {self.tf_max}")
        print(f"  TF max|val| (for normalisation): {self.tf_max_abs}")
        if self.base_tf_idx is not None:
            print(f"  Base TF: index {self.base_tf_idx} ({self.tf_names[self.base_tf_idx]})")
        else:
            print(f"  Warning: No base TF [0,0,0,0] found!")

    # ------------------------------------------------------------------ #
    # Match TF folder names
    # ------------------------------------------------------------------ #
    def _match_tf_folder(self, folder_name: str) -> Optional[int]:
        # Direct string match
        if folder_name in self._tf_name_to_idx:
            return self._tf_name_to_idx[folder_name]

        # Try parsing and finding closest by vector distance
        vec = parse_tf_name(folder_name)
        if vec is None:
            return None

        all_vecs = np.stack(self.tf_vectors)
        diffs = np.linalg.norm(all_vecs - vec, axis=1)
        best_idx = int(np.argmin(diffs))
        if diffs[best_idx] < 1e-4:
            return best_idx
        return None

    # ------------------------------------------------------------------ #
    # Find pXXX/tf_name/ folders
    # ------------------------------------------------------------------ #
    def _find_tf_folders(self):
        self.tf_folders: Dict[Tuple[int, int], Path] = {}

        num_param_folders = 0
        for folder in sorted(self.data_dir.iterdir()):
            if not folder.is_dir() or not folder.name.startswith("p"):
                continue
            try:
                folder_num = int(folder.name[1:])   
                param_idx = folder_num - 1          # 1-indexed → 0-indexed
            except ValueError:
                continue

            if param_idx not in self.condition_vectors:
                continue

            num_param_folders += 1

            # Scan TF subfolders
            for subfolder in sorted(folder.iterdir()):
                if not subfolder.is_dir():
                    continue
                tf_idx = self._match_tf_folder(subfolder.name)
                if tf_idx is not None:
                    self.tf_folders[(param_idx, tf_idx)] = subfolder

        expected = self.num_conditions * self.num_tfs
        print(f"Found {num_param_folders} param folders, "
              f"{len(self.tf_folders)} (param, TF) pairs "
              f"(expected {expected})")

        if len(self.tf_folders) < expected:
            missing_count = expected - len(self.tf_folders)
            print(f"  Warning: {missing_count} folders missing")

    # ------------------------------------------------------------------ #
    # Parse COLMAP (same as SurfaceConditionalParser)
    # ------------------------------------------------------------------ #
    def _parse_colmap(self):
        sparse_dir = self.data_dir / "sparse" / "0"
        if not sparse_dir.exists():
            sparse_dir = self.data_dir / "sparse"
            if not sparse_dir.exists():
                raise FileNotFoundError(
                    f"COLMAP sparse dir not found in {self.data_dir}"
                )

        print(f"Loading COLMAP from: {sparse_dir}")
        cameras = read_cameras_binary(str(sparse_dir / "cameras.bin"))
        images = read_images_binary(str(sparse_dir / "images.bin"))

        points3d_path = sparse_dir / "points3D.bin"
        if points3d_path.exists():
            self.points, self.points_rgb = read_points3D_binary(str(points3d_path))
            print(f"  Loaded {len(self.points)} 3D points")
        else:
            print("  Warning: points3D.bin not found, using random init")
            self.points = np.random.randn(10000, 3).astype(np.float32) * 0.5
            self.points_rgb = np.random.randint(0, 255, (10000, 3)).astype(np.uint8)

        _camtoworlds, _Ks, _image_names, _heights, _widths = [], [], [], [], []

        for img_id in images.keys():
            img_data = images[img_id]
            cam_data = cameras[img_data["camera_id"]]
            params = cam_data["params"]
            width = cam_data["width"]
            height = cam_data["height"]
            model_id = cam_data["model_id"]

            if model_id == 0:
                fx = fy = params[0]; cx, cy = params[1], params[2]
            elif model_id == 1:
                fx, fy = params[0], params[1]; cx, cy = params[2], params[3]
            elif model_id in [2, 3, 4, 5, 6]:
                fx, fy = params[0], params[1]; cx, cy = params[2], params[3]
            else:
                fx = fy = params[0]; cx, cy = width / 2, height / 2

            if self.factor > 1:
                fx /= self.factor; fy /= self.factor
                cx /= self.factor; cy /= self.factor
                width //= self.factor; height //= self.factor

            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            R = qvec2rotmat(img_data["qvec"])
            t = img_data["tvec"]
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = -R.T @ t

            _camtoworlds.append(c2w)
            _Ks.append(K)
            _image_names.append(img_data["name"])
            _heights.append(int(height))
            _widths.append(int(width))

        inds = np.argsort(_image_names)
        self.camtoworlds = np.stack([_camtoworlds[i] for i in inds])
        self.Ks = np.stack([_Ks[i] for i in inds])
        self.image_names = [_image_names[i] for i in inds]
        self.heights = [_heights[i] for i in inds]
        self.widths = [_widths[i] for i in inds]

        print(f"  Loaded {len(self.camtoworlds)} camera poses")

        if self.normalize:
            self._normalize_scene()
        else:
            self.scene_scale = 1.0

    def _normalize_scene(self):
        cam_centers = self.camtoworlds[:, :3, 3]
        center = cam_centers.mean(axis=0)
        dists = np.linalg.norm(cam_centers - center, axis=1)
        norm_factor = dists.max()
        print(f"  Normalization factor: {norm_factor:.4f}")

        self.camtoworlds[:, :3, 3] = (self.camtoworlds[:, :3, 3] - center) / norm_factor
        self.points = (self.points - center) / norm_factor

        camera_locations = self.camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        self.scene_scale = np.linalg.norm(
            camera_locations - scene_center, axis=1
        ).max()
        print(f"  Scene scale (post-norm): {self.scene_scale:.4f}")

    # ------------------------------------------------------------------ #
    # Build samples
    # ------------------------------------------------------------------ #
    def _build_samples(self):
        self.all_samples = []

        name_to_cam_idx = {}
        for i, name in enumerate(self.image_names):
            stem = Path(name).stem
            name_to_cam_idx[stem] = i

        matched = 0
        unmatched = 0
        for (param_idx, tf_idx), folder in sorted(self.tf_folders.items()):
            images = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
            images = sorted(images, key=lambda p: p.stem)

            for img_path in images:
                stem = img_path.stem
                if stem in name_to_cam_idx:
                    cam_idx = name_to_cam_idx[stem]
                    self.all_samples.append({
                        "param_idx": param_idx,
                        "tf_idx": tf_idx,
                        "camera_idx": cam_idx,
                        "image_path": img_path,
                        "condition_vector": self.condition_vectors[param_idx],
                        "tf_name": self.tf_names[tf_idx],
                        "tf_vector": self.tf_vectors[tf_idx],
                    })
                    matched += 1
                else:
                    unmatched += 1

        if unmatched > 0:
            print(f"  Warning: {unmatched} images had no matching camera "
                  f"(matched {matched})")
        print(f"Total samples: {len(self.all_samples)}")


    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    def get_condition_vector(self, param_idx: int, normalize: bool = True) -> np.ndarray:
        vec = self.condition_vectors[param_idx]
        if normalize:
            vec = (vec - self.condition_min) / self.condition_range
        return vec

    def get_tf_vector(self, tf_idx: int, normalize: bool = True) -> np.ndarray:
   
        vec = self.tf_vectors[tf_idx].copy()
        if normalize:
            vec = vec / (self.tf_max_abs + 1e-8)
        return vec


# ============================================================================
# TF Conditional Dataset
# ============================================================================

class TFConditionalDataset(TorchDataset):

    def __init__(
        self,
        parser: TFConditionalParser,
        split: str = "train",
        patch_size: Optional[int] = None,
        normalize_conditions: bool = True,
        # Condition filtering
        param_indices: Optional[List[int]] = None,
        tf_indices: Optional[List[int]] = None,
        # Member subsampling for data efficiency
        tf_member_fraction: float = 1.0,   # 1.0 = all, 0.2 = 20%
        tf_member_seed: int = 42,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.normalize_conditions = normalize_conditions

        allowed_params = set(param_indices) if param_indices is not None else None
        allowed_tfs = set(tf_indices) if tf_indices is not None else None

        # Determine which members to use per TF (subsampling)
        all_param_list = sorted(
            param_indices if param_indices is not None
            else parser.condition_vectors.keys()
        )
        self._allowed_per_tf = self._compute_member_subsampling(
            parser, all_param_list, tf_member_fraction, tf_member_seed,
        )

        # Filter samples
        filtered = parser.all_samples
        if allowed_params is not None:
            filtered = [s for s in filtered if s["param_idx"] in allowed_params]
        if allowed_tfs is not None:
            filtered = [s for s in filtered if s["tf_idx"] in allowed_tfs]

        # Apply per-TF member subsampling
        if tf_member_fraction < 1.0:
            filtered = [
                s for s in filtered
                if s["param_idx"] in self._allowed_per_tf[s["tf_idx"]]
            ]

        # Split by camera viewpoint
        if split == "train":
            self.samples = [
                s for s in filtered
                if s["camera_idx"] % parser.test_every != 0
            ]
        else:
            self.samples = [
                s for s in filtered
                if s["camera_idx"] % parser.test_every == 0
            ]

        unique_params = set(s["param_idx"] for s in self.samples)
        unique_tfs = set(s["tf_idx"] for s in self.samples)
        print(f"{split} split: {len(self.samples)} samples  "
              f"({len(unique_params)} params × {len(unique_tfs)} TFs)")
        if tf_member_fraction < 1.0:
            base_count = len(self._allowed_per_tf.get(parser.base_tf_idx, []))
            other_count = np.mean([
                len(v) for k, v in self._allowed_per_tf.items()
                if k != parser.base_tf_idx
            ]) if len(self._allowed_per_tf) > 1 else 0
            print(f"  Member subsampling: base TF → {base_count} members, "
                  f"others → ~{other_count:.0f} members ({tf_member_fraction*100:.0f}%)")

    @staticmethod
    def _compute_member_subsampling(
        parser: TFConditionalParser,
        all_params: List[int],
        fraction: float,
        seed: int,
    ) -> Dict[int, set]:
        """
        For each TF, compute which param members are allowed.
        Base TF always uses all members.
        Other TFs use fraction of members, with different random subsets per TF.
        """
        allowed = {}
        n_select = max(1, int(len(all_params) * fraction))
        all_set = set(all_params)

        for tf_idx in range(parser.num_tfs):
            if tf_idx == parser.base_tf_idx or fraction >= 1.0:
                allowed[tf_idx] = all_set
            else:
                rng = random.Random(seed + tf_idx)
                selected = sorted(rng.sample(all_params, n_select))
                allowed[tf_idx] = set(selected)

        return allowed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        param_idx = sample["param_idx"]
        tf_idx = sample["tf_idx"]
        cam_idx = sample["camera_idx"]
        img_path = sample["image_path"]

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.parser.factor > 1:
            w, h = image.size
            image = image.resize(
                (w // self.parser.factor, h // self.parser.factor),
                Image.BILINEAR,
            )
        image = np.array(image, dtype=np.float32)

        # Camera
        camtoworld = self.parser.camtoworlds[cam_idx].copy()
        K = self.parser.Ks[cam_idx].copy()

        # Simulation parameters
        cond_vec = self.parser.get_condition_vector(
            param_idx, normalize=self.normalize_conditions
        )

        # Transfer function vector
        tf_vec = self.parser.get_tf_vector(
            tf_idx, normalize=self.normalize_conditions
        )

        # Random patch (training)
        if self.patch_size is not None and self.split == "train":
            h, w = image.shape[:2]
            if h > self.patch_size and w > self.patch_size:
                x = np.random.randint(0, w - self.patch_size)
                y = np.random.randint(0, h - self.patch_size)
                image = image[y:y + self.patch_size, x:x + self.patch_size]
                K[0, 2] -= x
                K[1, 2] -= y

        return {
            "image": torch.from_numpy(image),                        # [H, W, 3]
            "camtoworld": torch.from_numpy(camtoworld),              # [4, 4]
            "K": torch.from_numpy(K),                                # [3, 3]
            "condition_idx": param_idx,                               # int
            "condition_vector": torch.from_numpy(cond_vec),          # [cond_dim]
            "tf_idx": tf_idx,                                         # int
            "tf_vector": torch.from_numpy(tf_vec),                   # [4]
            "camera_idx": cam_idx,
            "image_id": cam_idx,
            "sample_idx": idx,
        }