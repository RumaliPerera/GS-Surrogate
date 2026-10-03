import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import struct
import collections


# ============================================================================
# COLMAP Parsing Utilities
# ============================================================================

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)

CAMERA_MODELS = {
    CameraModel(0, "SIMPLE_PINHOLE", 3),
    CameraModel(1, "PINHOLE", 4),
    CameraModel(2, "SIMPLE_RADIAL", 4),
    CameraModel(3, "RADIAL", 5),
    CameraModel(4, "OPENCV", 8),
    CameraModel(5, "OPENCV_FISHEYE", 8),
    CameraModel(6, "FULL_OPENCV", 12),
    CameraModel(7, "FOV", 5),
    CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    CameraModel(9, "RADIAL_FISHEYE", 5),
    CameraModel(10, "THIN_PRISM_FISHEYE", 12),
}

CAMERA_MODEL_IDS = {cm.model_id: cm for cm in CAMERA_MODELS}


def read_cameras_binary(path: str) -> Dict:
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": np.array(params),
            }
    return cameras


def read_images_binary(path: str) -> Dict:
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name = ""
            while True:
                char = f.read(1).decode("utf-8")
                if char == "\x00":
                    break
                name += char
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * num_points2D)  # Skip points2D
            images[image_id] = {
                "qvec": np.array(qvec),
                "tvec": np.array(tvec),
                "camera_id": camera_id,
                "name": name,
            }
    return images


def read_points3D_binary(path: str) -> Tuple[np.ndarray, np.ndarray]:
    points = []
    colors = []
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            _ = struct.unpack("<Q", f.read(8))[0]  # point3D_id
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            _ = struct.unpack("<d", f.read(8))[0]  # error
            track_length = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * track_length)  # Skip track
            points.append(xyz)
            colors.append(rgb)
    return np.array(points, dtype=np.float32), np.array(colors, dtype=np.uint8)


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    q = qvec / np.linalg.norm(qvec)
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ], dtype=np.float32)


# ============================================================================
# Generic Conditional Parser
# ============================================================================

class ConditionalParser:
    
    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = True,
        test_every: int = 8,
        names_file: str = "names.txt",
    ):
        self.data_dir = Path(data_dir)
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        
        # Parse condition vectors from names.txt (auto-detect dim)
        self._parse_conditions(names_file)
         
        self._find_condition_folders()
         
        self._parse_colmap()
         
        self._build_samples()
        
    
    def _parse_conditions(self, names_file: str):
        
        names_path = self.data_dir / names_file
        
        if not names_path.exists():
            raise FileNotFoundError(f"Conditions file not found: {names_path}")
        
        self.condition_vectors = {}  # condition_idx -> vector
        self.condition_names = {}    # condition_idx -> original name string
        
        with open(names_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                 
                parts = line.split('_')
                try:
                    idx = int(parts[0])
                except ValueError:
                    print(f"  Warning: skipping malformed line: {line[:50]}...")
                    continue
                
                # All remaining parts are condition vector components
                try:
                    vec = [float(p) for p in parts[1:]]
                except ValueError:
                    print(f"  Warning: skipping line with non-numeric values: {line[:50]}...")
                    continue
                
                if len(vec) == 0:
                    print(f"  Warning: skipping line with no condition values: {line[:50]}...")
                    continue
                
                self.condition_vectors[idx] = np.array(vec, dtype=np.float32)
                self.condition_names[idx] = line
        
        self.num_conditions = len(self.condition_vectors)
        
        # Auto-detect condition dimensionality from first entry
        first_vec = next(iter(self.condition_vectors.values()))
        self.condition_dim = len(first_vec)
        
        # Verify all vectors have same dimensionality
        for idx, vec in self.condition_vectors.items():
            if len(vec) != self.condition_dim:
                raise ValueError(
                    f"Condition {idx} has dim {len(vec)}, expected {self.condition_dim}. "
                    f"All conditions must have the same dimensionality."
                )
        
        # Compute min-max normalization stats for conditions
        all_vecs = np.stack(list(self.condition_vectors.values()))
        self.condition_min = all_vecs.min(axis=0)
        self.condition_max = all_vecs.max(axis=0)
        self.condition_range = self.condition_max - self.condition_min + 1e-8
        
        print(f"Loaded {self.num_conditions} conditions from {names_file}")
        print(f"  Condition dim: {self.condition_dim}")
        print(f"  Condition min: {self.condition_min}")
        print(f"  Condition max: {self.condition_max}")
    
    def _find_condition_folders(self):
        """Find condition folders. Supports 'p' prefix (0-indexed) and 'p' prefix (1-indexed)."""
        self.condition_folders = {}  

        if len(self.condition_folders) == 0:
            for folder in sorted(self.data_dir.iterdir()):
                if folder.is_dir() and folder.name.startswith('p'):
                    try:
                        folder_num = int(folder.name[1:])  # "001" -> 1
                        condition_idx = folder_num - 1       # 1-indexed to 0-indexed
                        
                        if condition_idx in self.condition_vectors:
                            self.condition_folders[condition_idx] = folder
                    except ValueError:
                        continue
        
        print(f"Found {len(self.condition_folders)} condition folders")
        
        # Verify all conditions have folders
        missing = set(self.condition_vectors.keys()) - set(self.condition_folders.keys())
        if missing:
            print(f"  Warning: Missing folders for conditions: {sorted(missing)[:5]}...")
    
    def _parse_colmap(self):
        """Parse shared COLMAP reconstruction."""
        sparse_dir = self.data_dir / "sparse" / "0"
        
        if not sparse_dir.exists():
            sparse_dir = self.data_dir / "sparse"
            if not sparse_dir.exists():
                raise FileNotFoundError(f"COLMAP sparse dir not found in {self.data_dir}")
        
        print(f"Loading COLMAP from: {sparse_dir}")
        
        # Read COLMAP files
        cameras = read_cameras_binary(str(sparse_dir / "cameras.bin"))
        images = read_images_binary(str(sparse_dir / "images.bin"))
        
        points3d_path = sparse_dir / "points3D.bin"
        if points3d_path.exists():
            self.points, self.points_rgb = read_points3D_binary(str(points3d_path))
            print(f"  Loaded {len(self.points)} 3D points")
        else:
            print("  Warning: points3D.bin not found, using random initialization")
            self.points = np.random.randn(10000, 3).astype(np.float32) * 0.5
            self.points_rgb = np.random.randint(0, 255, (10000, 3)).astype(np.uint8)
        
        # Build camera matrices — collect first, then sort by image name
        # (gsplat sorts by np.argsort(image_names) for consistent ordering)
        _camtoworlds = []
        _Ks = []
        _image_names = []
        _heights = []
        _widths = []
        
        for img_id in images.keys():
            img_data = images[img_id]
            cam_data = cameras[img_data["camera_id"]]
            
            # Camera intrinsics
            params = cam_data["params"]
            width = cam_data["width"]
            height = cam_data["height"]
            
            # Handle different camera models
            model_id = cam_data["model_id"]
            if model_id == 0:  # SIMPLE_PINHOLE
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model_id == 1:  # PINHOLE
                fx, fy = params[0], params[1]
                cx, cy = params[2], params[3]
            elif model_id in [2, 3, 4, 5, 6]:  # RADIAL, OPENCV variants
                fx, fy = params[0], params[1]
                cx, cy = params[2], params[3]
            else:
                fx = fy = params[0]
                cx, cy = width / 2, height / 2
            
            # Apply downsample factor
            if self.factor > 1:
                fx /= self.factor
                fy /= self.factor
                cx /= self.factor
                cy /= self.factor
                width //= self.factor
                height //= self.factor
            
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Camera extrinsics
            R = qvec2rotmat(img_data["qvec"])
            t = img_data["tvec"]
            
            # World-to-camera -> Camera-to-world
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = -R.T @ t
            
            _camtoworlds.append(c2w)
            _Ks.append(K)
            _image_names.append(img_data["name"])
            _heights.append(int(height))
            _widths.append(int(width))
        
        # Sort by image name (alphabetical) to match gsplat's Parser
        inds = np.argsort(_image_names)
        self.camtoworlds = np.stack([_camtoworlds[i] for i in inds])
        self.Ks = np.stack([_Ks[i] for i in inds])
        self.image_names = [_image_names[i] for i in inds]
        self.heights = [_heights[i] for i in inds]
        self.widths = [_widths[i] for i in inds]
        
        print(f"  Loaded {len(self.camtoworlds)} camera poses")
        
        # Normalize scene
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
        
        # Normalize cameras and points
        self.camtoworlds[:, :3, 3] = (self.camtoworlds[:, :3, 3] - center) / norm_factor
        self.points = (self.points - center) / norm_factor
        
        # Compute scene_scale AFTER normalization (matches gsplat's Parser)
        camera_locations = self.camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        self.scene_scale = np.linalg.norm(camera_locations - scene_center, axis=1).max()
        
        print(f"  Scene scale (post-norm): {self.scene_scale:.4f}")
    
    def _build_samples(self):
        """Build list of (condition_idx, camera_idx, image_path) samples.
        
        Images are matched to cameras by filename, not positional index.
        This is robust even if image naming conventions differ slightly
        between condition folders and COLMAP.
        """
        self.all_samples = []
        
        # Build name-stem -> camera_idx lookup from the sorted camera list
        name_to_cam_idx = {}
        for i, name in enumerate(self.image_names):
            stem = Path(name).stem
            name_to_cam_idx[stem] = i
        
        matched = 0
        unmatched = 0
        for cond_idx, folder in sorted(self.condition_folders.items()):
            # Get all images in this condition folder
            images = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
            images = sorted(images, key=lambda p: p.stem)
            
            for img_path in images:
                stem = img_path.stem
                if stem in name_to_cam_idx:
                    cam_idx = name_to_cam_idx[stem]
                    self.all_samples.append({
                        'condition_idx': cond_idx,
                        'camera_idx': cam_idx,
                        'image_path': img_path,
                        'condition_vector': self.condition_vectors[cond_idx],
                    })
                    matched += 1
                else:
                    unmatched += 1
        
        if unmatched > 0:
            print(f"  Warning: {unmatched} images had no matching camera (matched {matched})")
        print(f"Total samples: {len(self.all_samples)}")

    
    def get_condition_vector(self, condition_idx: int, normalize: bool = True) -> np.ndarray:
        vec = self.condition_vectors[condition_idx]
        if normalize:
            vec = (vec - self.condition_min) / self.condition_range
        return vec


# ============================================================================
# Generic Conditional Dataset
# ============================================================================

class ConditionalDataset(TorchDataset):    
    def __init__(
        self,
        parser: ConditionalParser,
        split: str = "train",
        patch_size: Optional[int] = None,
        normalize_conditions: bool = True,
        condition_indices: Optional[List[int]] = None,
        condition_filter: Optional[int] = None,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.normalize_conditions = normalize_conditions
        
        # Determine which conditions to include
        if condition_filter is not None:
            allowed_conditions = {condition_filter}
        elif condition_indices is not None:
            allowed_conditions = set(condition_indices)
        else:
            allowed_conditions = None
        
        # Filter and split samples
        all_samples = parser.all_samples
        
        if allowed_conditions is not None:
            all_samples = [s for s in all_samples if s['condition_idx'] in allowed_conditions]
        
        # Split by camera viewpoint (train/val)
        if split == "train":
            self.samples = [
                s for s in all_samples
                if s['camera_idx'] % parser.test_every != 0
            ]
        else:  # val
            self.samples = [
                s for s in all_samples
                if s['camera_idx'] % parser.test_every == 0
            ]
        
        unique_conditions = set(s['condition_idx'] for s in self.samples)
        print(f"{split} split: {len(self.samples)} samples from {len(unique_conditions)} conditions")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        cond_idx = sample['condition_idx']
        cam_idx = sample['camera_idx']
        img_path = sample['image_path']
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        if self.parser.factor > 1:
            w, h = image.size
            image = image.resize(
                (w // self.parser.factor, h // self.parser.factor),
                Image.BILINEAR
            )
        
        image = np.array(image, dtype=np.float32)
        
        # Get camera params
        camtoworld = self.parser.camtoworlds[cam_idx].copy()
        K = self.parser.Ks[cam_idx].copy()
        
        # Get condition vector
        cond_vec = self.parser.get_condition_vector(
            cond_idx, normalize=self.normalize_conditions
        )
        
        # Random patch for training
        if self.patch_size is not None and self.split == "train":
            h, w = image.shape[:2]
            if h > self.patch_size and w > self.patch_size:
                x = np.random.randint(0, w - self.patch_size)
                y = np.random.randint(0, h - self.patch_size)
                image = image[y:y+self.patch_size, x:x+self.patch_size]
                K[0, 2] -= x
                K[1, 2] -= y
        
        return {
            "image": torch.from_numpy(image),                    # [H, W, 3]
            "camtoworld": torch.from_numpy(camtoworld),          # [4, 4]
            "K": torch.from_numpy(K),                            # [3, 3]
            "condition_idx": cond_idx,                           # int
            "condition_vector": torch.from_numpy(cond_vec),      # [condition_dim]
            "camera_idx": cam_idx,                               # int (for image_ids)
            "image_id": cam_idx,                                 # alias for compatibility
            "sample_idx": idx,                                   # for loss-weighted sampling
        }


# ============================================================================
# Backward-compatible aliases (drop-in replacement for nyx_dataset.py)
# ============================================================================
NyxParser = ConditionalParser
NyxDataset = ConditionalDataset


# ============================================================================
# Surface (Isosurface) Conditional Parser & Dataset
# ============================================================================

class SurfaceConditionalParser:

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = True,
        test_every: int = 8,
        names_file: str = "names.txt",
        isovalues_file: str = "isovalues.txt",
    ):
        self.data_dir = Path(data_dir)
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        self._parse_conditions(names_file)
 
        self._parse_isovalues(isovalues_file)
 
        self._find_surface_folders()
 
        self._parse_colmap()
 
        self._build_surface_samples()


    # ------------------------------------------------------------------ #
    # Reuse condition parsing from ConditionalParser (copied to be self-contained)
    # ------------------------------------------------------------------ #
    def _parse_conditions(self, names_file: str):
        """Parse names.txt to extract simulation parameter vectors."""
        names_path = self.data_dir / names_file
        if not names_path.exists():
            raise FileNotFoundError(f"Conditions file not found: {names_path}")

        self.condition_vectors = {}   # param_idx → np.ndarray
        self.condition_names = {}     # param_idx → original line

        with open(names_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("_")
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

        # Verify all vectors have same dim
        for idx, vec in self.condition_vectors.items():
            if len(vec) != self.condition_dim:
                raise ValueError(
                    f"Condition {idx} has dim {len(vec)}, expected {self.condition_dim}"
                )

        # Min-max stats for normalisation
        all_vecs = np.stack(list(self.condition_vectors.values()))
        self.condition_min = all_vecs.min(axis=0)
        self.condition_max = all_vecs.max(axis=0)
        self.condition_range = self.condition_max - self.condition_min + 1e-8

        print(f"Loaded {self.num_conditions} simulation conditions from {names_file}")
        print(f"  Sim param dim: {self.condition_dim}")
        print(f"  Sim param min: {self.condition_min}")
        print(f"  Sim param max: {self.condition_max}")

    # ------------------------------------------------------------------ #
    def _parse_isovalues(self, isovalues_file: str):
        """Parse isovalues.txt — one float per line."""
        iso_path = self.data_dir / isovalues_file
        if not iso_path.exists():
            raise FileNotFoundError(f"Isovalues file not found: {iso_path}")

        self.isovalues: List[float] = []
        with open(iso_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.isovalues.append(float(line))

        self.num_isovalues = len(self.isovalues)
        self.isovalues_array = np.array(self.isovalues, dtype=np.float32)

        # Min-max stats for normalisation
        self.isovalue_min = float(self.isovalues_array.min())
        self.isovalue_max = float(self.isovalues_array.max())
        self.isovalue_range = self.isovalue_max - self.isovalue_min + 1e-8

        # Lookup: rounded float → index (for folder matching)
        self._iso_str_to_idx: Dict[str, int] = {}
        for i, v in enumerate(self.isovalues):
            # Store multiple representations for robust matching
            self._iso_str_to_idx[f"{v}"] = i
            self._iso_str_to_idx[f"{v:.4f}"] = i
            self._iso_str_to_idx[f"{v:.6f}"] = i

        print(f"Loaded {self.num_isovalues} isovalues from {isovalues_file}")
        print(f"  Range: [{self.isovalue_min}, {self.isovalue_max}]")
        print(f"  Values: {self.isovalues}")

    # ------------------------------------------------------------------ #
    def _match_isovalue_folder(self, folder_name: str) -> Optional[int]:
        """Match a folder name (string) to an isovalue index."""
        # Direct string match
        if folder_name in self._iso_str_to_idx:
            return self._iso_str_to_idx[folder_name]

        # Try parsing as float and finding closest
        try:
            val = float(folder_name)
        except ValueError:
            return None

        # Find closest isovalue (within tolerance)
        diffs = np.abs(self.isovalues_array - val)
        best_idx = int(np.argmin(diffs))
        if diffs[best_idx] < 1e-4:
            return best_idx
        return None

    # ------------------------------------------------------------------ #
    def _find_surface_folders(self):
        """Find pXXX/isovalue/ folder pairs."""
        self.surface_folders: Dict[Tuple[int, int], Path] = {}
        # surface_folders[(param_idx, iso_idx)] → Path to image folder

        num_param_folders = 0
        for folder in sorted(self.data_dir.iterdir()):
            if not folder.is_dir() or not folder.name.startswith("p"):
                continue
            try:
                folder_num = int(folder.name[1:])     # "001" → 1
                param_idx = folder_num - 1             # 1-indexed → 0-indexed
            except ValueError:
                continue

            if param_idx not in self.condition_vectors:
                continue

            num_param_folders += 1

            # Scan isovalue subfolders
            for subfolder in sorted(folder.iterdir()):
                if not subfolder.is_dir():
                    continue
                iso_idx = self._match_isovalue_folder(subfolder.name)
                if iso_idx is not None:
                    self.surface_folders[(param_idx, iso_idx)] = subfolder

        expected = self.num_conditions * self.num_isovalues
        print(f"Found {num_param_folders} param folders, "
              f"{len(self.surface_folders)} (param, iso) pairs "
              f"(expected {expected})")

        if len(self.surface_folders) < expected:
            # Report missing
            missing_count = expected - len(self.surface_folders)
            print(f"  Warning: {missing_count} folders missing")

    # ------------------------------------------------------------------ #
    def _parse_colmap(self):
        """Parse shared COLMAP reconstruction (same logic as ConditionalParser)."""
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
    def _build_surface_samples(self):
        """Build flat list of (param_idx, iso_idx, cam_idx, image_path) samples."""
        self.all_samples = []

        name_to_cam_idx = {}
        for i, name in enumerate(self.image_names):
            stem = Path(name).stem
            name_to_cam_idx[stem] = i

        matched = 0
        unmatched = 0
        for (param_idx, iso_idx), folder in sorted(self.surface_folders.items()):
            images = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
            images = sorted(images, key=lambda p: p.stem)

            for img_path in images:
                stem = img_path.stem
                if stem in name_to_cam_idx:
                    cam_idx = name_to_cam_idx[stem]
                    self.all_samples.append({
                        "param_idx": param_idx,
                        "isovalue_idx": iso_idx,
                        "camera_idx": cam_idx,
                        "image_path": img_path,
                        "condition_vector": self.condition_vectors[param_idx],
                        "isovalue": self.isovalues[iso_idx],
                    })
                    matched += 1
                else:
                    unmatched += 1

        if unmatched > 0:
            print(f"  Warning: {unmatched} images had no matching camera "
                  f"(matched {matched})")
        print(f"Total samples: {len(self.all_samples)}")


    # ------------------------------------------------------------------ #
    def get_condition_vector(
        self, param_idx: int, normalize: bool = True
    ) -> np.ndarray:
        """Return simulation parameter vector (optionally normalised to [0,1])."""
        vec = self.condition_vectors[param_idx]
        if normalize:
            vec = (vec - self.condition_min) / self.condition_range
        return vec

    def get_isovalue(
        self, iso_idx: int, normalize: bool = True
    ) -> float:
        """Return isovalue scalar (optionally normalised to [0,1])."""
        val = self.isovalues[iso_idx]
        if normalize:
            val = (val - self.isovalue_min) / self.isovalue_range
        return float(val)


class SurfaceConditionalDataset(TorchDataset):

    def __init__(
        self,
        parser: SurfaceConditionalParser,
        split: str = "train",
        patch_size: Optional[int] = None,
        normalize_conditions: bool = True,
        # Filtering: sets of allowed param / isovalue indices
        param_indices: Optional[List[int]] = None,
        isovalue_indices: Optional[List[int]] = None,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.normalize_conditions = normalize_conditions

        allowed_params = set(param_indices) if param_indices is not None else None
        allowed_isos = set(isovalue_indices) if isovalue_indices is not None else None

        # Filter samples
        filtered = parser.all_samples
        if allowed_params is not None:
            filtered = [s for s in filtered if s["param_idx"] in allowed_params]
        if allowed_isos is not None:
            filtered = [s for s in filtered if s["isovalue_idx"] in allowed_isos]

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
        unique_isos = set(s["isovalue_idx"] for s in self.samples)
        print(f"{split} split: {len(self.samples)} samples  "
              f"({len(unique_params)} params × {len(unique_isos)} isos)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        param_idx = sample["param_idx"]
        iso_idx = sample["isovalue_idx"]
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

        # Isovalue
        isovalue = self.parser.get_isovalue(
            iso_idx, normalize=self.normalize_conditions
        )

        # Random patch (training)
        if self.patch_size is not None and self.split == "train":
            h, w = image.shape[:2]
            if h > self.patch_size and w > self.patch_size:
                x = np.random.randint(0, w - self.patch_size)
                y = np.random.randint(0, h - self.patch_size)
                image = image[y : y + self.patch_size, x : x + self.patch_size]
                K[0, 2] -= x
                K[1, 2] -= y

        return {
            "image": torch.from_numpy(image),                       # [H, W, 3]
            "camtoworld": torch.from_numpy(camtoworld),             # [4, 4]
            "K": torch.from_numpy(K),                               # [3, 3]
            "condition_idx": param_idx,                              # int
            "condition_vector": torch.from_numpy(cond_vec),         # [condition_dim]
            "isovalue_idx": iso_idx,                                # int
            "isovalue": torch.tensor(isovalue, dtype=torch.float32),  # scalar
            "camera_idx": cam_idx,
            "image_id": cam_idx,
            "sample_idx": idx,
        }


# ============================================================================
# Grouped Surface Dataset (multi-iso batching)
# ============================================================================

class GroupedSurfaceDataset(TorchDataset):

    def __init__(
        self,
        parser: 'SurfaceConditionalParser',
        split: str = "train",
        patch_size: Optional[int] = None,
        normalize_conditions: bool = True,
        param_indices: Optional[List[int]] = None,
        isovalue_indices: Optional[List[int]] = None,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.normalize_conditions = normalize_conditions

        # Determine which params and isos to include
        if param_indices is not None:
            self.param_list = sorted(param_indices)
        else:
            self.param_list = sorted(parser.condition_vectors.keys())

        if isovalue_indices is not None:
            self.iso_list = sorted(isovalue_indices)
        else:
            self.iso_list = list(range(parser.num_isovalues))

        self.num_isovalues = len(self.iso_list)

        # Build camera list for this split
        all_cam_indices = list(range(len(parser.camtoworlds)))
        if split == "train":
            self.cam_list = set(
                c for c in all_cam_indices if c % parser.test_every != 0
            )
        else:
            self.cam_list = set(
                c for c in all_cam_indices if c % parser.test_every == 0
            )

        # Build lookup: (param_idx, iso_idx, cam_idx) → image_path
        self._image_lookup: Dict[Tuple[int, int, int], Path] = {}
        name_to_cam_idx = {}
        for i, name in enumerate(parser.image_names):
            stem = Path(name).stem
            name_to_cam_idx[stem] = i

        for (p_idx, iso_idx), folder in parser.surface_folders.items():
            if p_idx not in set(self.param_list):
                continue
            if iso_idx not in set(self.iso_list):
                continue
            images = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
            for img_path in sorted(images, key=lambda p: p.stem):
                stem = img_path.stem
                if stem in name_to_cam_idx:
                    cam_idx = name_to_cam_idx[stem]
                    self._image_lookup[(p_idx, iso_idx, cam_idx)] = img_path

        # Build flat list of (param_idx, camera_idx) pairs
        pair_set = set()
        for (p_idx, iso_idx, cam_idx) in self._image_lookup:
            if cam_idx in self.cam_list:
                pair_set.add((p_idx, cam_idx))

        self.pairs = sorted(pair_set)

        # Stats
        total_found = sum(
            1 for (p, c) in self.pairs for iso in self.iso_list
            if (p, iso, c) in self._image_lookup
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        param_idx, cam_idx = self.pairs[idx]

        # Camera
        camtoworld = self.parser.camtoworlds[cam_idx].copy()
        K = self.parser.Ks[cam_idx].copy()
        height = self.parser.heights[cam_idx]
        width = self.parser.widths[cam_idx]

        # Condition vector
        cond_vec = self.parser.get_condition_vector(
            param_idx, normalize=self.normalize_conditions
        )

        # Random patch offset (shared across ALL isovalues for consistency)
        crop_x, crop_y = 0, 0
        do_crop = False
        eff_h, eff_w = height, width
        if self.parser.factor > 1:
            eff_h = height // self.parser.factor
            eff_w = width // self.parser.factor
        if self.patch_size is not None and self.split == "train":
            if eff_h > self.patch_size and eff_w > self.patch_size:
                crop_x = np.random.randint(0, eff_w - self.patch_size)
                crop_y = np.random.randint(0, eff_h - self.patch_size)
                do_crop = True

        # Load all isovalue images
        images = []
        isovalues = []
        isovalue_indices = []
        valid_mask = []

        for iso_idx in self.iso_list:
            iso_val = self.parser.get_isovalue(
                iso_idx, normalize=self.normalize_conditions
            )
            isovalues.append(iso_val)
            isovalue_indices.append(iso_idx)

            img_path = self._image_lookup.get((param_idx, iso_idx, cam_idx))
            if img_path is not None and img_path.exists():
                image = Image.open(img_path).convert("RGB")
                if self.parser.factor > 1:
                    w, h = image.size
                    image = image.resize(
                        (w // self.parser.factor, h // self.parser.factor),
                        Image.BILINEAR,
                    )
                image = np.array(image, dtype=np.float32)

                if do_crop:
                    image = image[crop_y:crop_y + self.patch_size,
                                  crop_x:crop_x + self.patch_size]

                images.append(image)
                valid_mask.append(True)
            else:
                # Placeholder — masked out in loss
                ph = self.patch_size if do_crop else eff_h
                pw = self.patch_size if do_crop else eff_w
                images.append(np.zeros((ph, pw, 3), dtype=np.float32))
                valid_mask.append(False)

        # Adjust K for crop
        K_adj = K.copy()
        if do_crop:
            K_adj[0, 2] -= crop_x
            K_adj[1, 2] -= crop_y

        return {
            "images": torch.from_numpy(np.stack(images)),              # [K, H, W, 3]
            "camtoworld": torch.from_numpy(camtoworld),                # [4, 4]
            "K": torch.from_numpy(K_adj),                              # [3, 3]
            "condition_idx": param_idx,                                # int
            "condition_vector": torch.from_numpy(cond_vec),            # [condition_dim]
            "camera_idx": cam_idx,
            "image_id": cam_idx,
            "isovalues": torch.tensor(isovalues, dtype=torch.float32), # [K]
            "isovalue_indices": torch.tensor(isovalue_indices, dtype=torch.long),
            "valid_mask": torch.tensor(valid_mask, dtype=torch.bool),  # [K]
            "sample_idx": idx,
        }