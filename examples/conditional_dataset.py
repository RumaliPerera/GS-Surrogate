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
    """
    Generic parser for conditional multi-view datasets.
    
    Expected structure:
        data_dir/
            m0000/ or p001/    # Condition folders
            m0001/ or p002/
            ...
            sparse/0/           # Shared COLMAP
            names.txt           # Condition vectors
    
    Folder naming conventions (auto-detected):
        - 'm' prefix: 0-indexed (m0000 -> condition 0)
        - 'p' prefix: 1-indexed (p001  -> condition 0)
    
    Condition dimensionality is auto-detected from names.txt.
    
    Args:
        data_dir: Root directory
        factor: Downsample factor for images
        normalize: Whether to normalize scene coordinates
        test_every: Every Nth image is used for testing
        names_file: Name of the conditions file (default: "names.txt")
    """
    
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
        
        # Find condition folders (auto-detect 'm' vs 'p' prefix)
        self._find_condition_folders()
        
        # Parse COLMAP data
        self._parse_colmap()
        
        # Build sample list
        self._build_samples()
        
        # Print summary
        self._print_summary()
    
    def _parse_conditions(self, names_file: str):
        """Parse names.txt to extract condition vectors. Auto-detects dimensionality."""
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
                
                # Parse: "XXXX_val1_val2_..._valN"
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
        # Maps conditions to [0, 1] — better for HexPlane line indexing,
        # positional encoding, and generalization to unseen conditions.
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
        self.condition_folders = {}  # condition_idx -> Path
        
        
        # Fallback to 'p' prefix (1-indexed: p001 -> condition 0)
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
    
    def _print_summary(self):
        dataset_name = self.data_dir.name
        print("\n" + "="*60)
        print(f"Conditional Dataset Summary ({dataset_name})")
        print("="*60)
        print(f"  Data directory: {self.data_dir}")
        print(f"  Conditions: {self.num_conditions}")
        print(f"  Condition dimension: {self.condition_dim}")
        print(f"  Cameras per condition: {len(self.camtoworlds)}")
        print(f"  Total samples: {len(self.all_samples)}")
        print(f"  Image size: {self.widths[0]}x{self.heights[0]} (after factor={self.factor})")
        print(f"  3D points: {len(self.points)}")
        print("="*60 + "\n")
    
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



NyxParser = ConditionalParser
NyxDataset = ConditionalDataset


# ============================================================================
# In-memory preloading
# ============================================================================

def preload_to_memory(dataset: "ConditionalDataset") -> Dict[str, torch.Tensor]:
    """Eagerly decode every sample once and stack into contiguous CPU tensors.

    ConditionalDataset.__getitem__ re-opens and re-decodes an image from disk
    on every call, which DataLoader workers otherwise have to redo on every
    single training step. A full training split here is only a few GB, so
    it's cheaper to decode it once up front and index into RAM afterwards.
    Assumes every sample has the same image resolution (true whenever
    patch_size is None and parser.factor is fixed, as in this dataset).
    """
    n = len(dataset)
    assert n > 0, "Cannot preload an empty dataset."

    first = dataset[0]
    h, w = first["image"].shape[:2]
    cond_dim = first["condition_vector"].shape[0]

    images = torch.empty((n, h, w, 3), dtype=torch.uint8)
    camtoworlds = torch.empty((n, 4, 4), dtype=torch.float32)
    Ks = torch.empty((n, 3, 3), dtype=torch.float32)
    condition_vectors = torch.empty((n, cond_dim), dtype=torch.float32)
    condition_idx = torch.empty((n,), dtype=torch.long)
    camera_idx = torch.empty((n,), dtype=torch.long)

    def _store(i: int, sample: Dict[str, Any]) -> None:
        images[i] = sample["image"].to(torch.uint8)
        camtoworlds[i] = sample["camtoworld"]
        Ks[i] = sample["K"]
        condition_vectors[i] = sample["condition_vector"]
        condition_idx[i] = int(sample["condition_idx"])
        camera_idx[i] = int(sample["camera_idx"])

    _store(0, first)
    for i in range(1, n):
        _store(i, dataset[i])

    return {
        "images": images,
        "camtoworlds": camtoworlds,
        "Ks": Ks,
        "condition_vectors": condition_vectors,
        "condition_idx": condition_idx,
        "camera_idx": camera_idx,
    }