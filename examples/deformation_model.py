import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.checkpoint import checkpoint  # unused now — checkpointing reverted, see DeformationMLP.forward
from typing import Optional, Tuple
import math


class BF16LayerNorm(nn.LayerNorm):
    """LayerNorm that computes directly in its input's dtype instead of
    being forced to fp32 by `torch.autocast`. Under bf16 autocast, every
    plain nn.LayerNorm call round-trips its (N, hidden_dim) activation
    fp32<->bf16 on top of the norm itself — profiling showed this cast
    traffic alone eating ~30% of step time at N ~= 1e6 gaussians, more
    than the actual matmuls. It's dead weight: PyTorch's native CUDA
    layer_norm kernel already accumulates mean/var in a float32
    accumulator internally regardless of the input/output dtype (that's
    why fp16 needs the forced promotion for range/overflow safety, but
    bf16 — with fp32's exponent range — doesn't). Running on the tensor's
    existing dtype skips the redundant copies with no change in the
    reduction's actual precision.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            weight = self.weight.to(x.dtype) if self.weight is not None else None
            bias = self.bias.to(x.dtype) if self.bias is not None else None
            return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)


class PositionalEncoding(nn.Module):

    def __init__(self, input_dim: int, num_freqs: int = 6, include_input: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        self.include_input = include_input
        
        freqs = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer('freqs', freqs)
        
        self.output_dim = input_dim * num_freqs * 2
        if include_input:
            self.output_dim += input_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = []
        if self.include_input:
            encoded.append(x)
        for freq in self.freqs:
            encoded.append(torch.sin(x * freq * math.pi))
            encoded.append(torch.cos(x * freq * math.pi))
        return torch.cat(encoded, dim=-1)


class ConditionEncoder(nn.Module):
    
    def __init__(
        self,
        condition_dim: int = 3,
        output_dim: int = 32,
        hidden_dim: int = 64,
        num_freqs: int = 2,           # LOW — smooth for generalisation
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding(condition_dim, num_freqs=num_freqs)
        self.mlp = nn.Sequential(
            nn.Linear(self.pos_enc.output_dim, hidden_dim),
            BF16LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        encoded = self.pos_enc(condition)
        return self.mlp(encoded)


class SpatialEncoder(nn.Module):
    
    def __init__(self, output_dim: int = 32, hidden_dim: int = 64, num_freqs: int = 10):
        super().__init__()
        self.pos_enc = PositionalEncoding(3, num_freqs=num_freqs)
        self.mlp = nn.Sequential(
            nn.Linear(self.pos_enc.output_dim, hidden_dim),
            BF16LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            BF16LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        encoded = self.pos_enc(xyz)
        return self.mlp(encoded)


class FiLMModulation(nn.Module):
    """
    Feature-wise Linear Modulation: condition modulates spatial features
    via learned per-channel scale (gamma) and shift (beta).

        out = gamma(condition) * spatial_feat + beta(condition)
    """

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.fc = nn.Linear(cond_dim, feat_dim * 2)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.fc.bias.data[:feat_dim] = 1.0   # gamma=1, beta=0 at init

    def forward(self, cond: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.fc(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma * feat + beta


class BottleneckAdapter(nn.Module):
    """
    Bottleneck MLP adapter: condition features are projected through a
    low-rank bottleneck to produce delta features that are added to
    spatial features.

        out = spatial_feat + adapter(cond)
    
    Zero-initialized output so the model starts as identity (no shift).
    """

    def __init__(self, cond_dim: int, feat_dim: int, bottleneck_dim: int = 16):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(cond_dim, bottleneck_dim),
            BF16LayerNorm(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, feat_dim),
        )
        # Zero-init output layer → starts as identity (no delta)
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, cond: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        return feat + self.adapter(cond)


class DeformationMLP(nn.Module):
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, num_layers: int = 4, learn_alpha: bool = False, learn_sh: bool = False, sh_dim: int = 48):
        super().__init__()
        self.learn_alpha = learn_alpha
        self.learn_sh = learn_sh
        
        layers = [nn.Linear(input_dim, hidden_dim), BF16LayerNorm(hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), BF16LayerNorm(hidden_dim), nn.ReLU(inplace=True)])
        self.backbone = nn.Sequential(*layers)
        
        # self.delta_xyz = nn.Linear(hidden_dim, 3)
        self.delta_xyz = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            BF16LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, 3)
        )

        # self.delta_rot = nn.Linear(hidden_dim, 4)
        self.delta_rot = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            BF16LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, 4)
        )

        # self.delta_scale = nn.Linear(hidden_dim, 3)
        self.delta_scale = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            BF16LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, 3)
        )
        # self.delta_alpha = nn.Linear(hidden_dim, 1) if learn_alpha else None  
        if learn_alpha:
            self.delta_alpha = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                BF16LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),               # nn.ReLU(inplace=True), nn.GELU()
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.delta_alpha = None
        
        # self.delta_sh = nn.Linear(hidden_dim, sh_dim) if learn_sh else None
        if learn_sh:
            self.delta_sh = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                BF16LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),  
                nn.Linear(hidden_dim, sh_dim)
            )
        else:
            self.delta_sh = None
        

        nn.init.zeros_(self.delta_xyz[-1].weight)
        nn.init.zeros_(self.delta_xyz[-1].bias)
        nn.init.zeros_(self.delta_rot[-1].weight)
        nn.init.zeros_(self.delta_rot[-1].bias)
        nn.init.zeros_(self.delta_scale[-1].weight)
        nn.init.zeros_(self.delta_scale[-1].bias)


        if self.delta_alpha is not None:
            nn.init.zeros_(self.delta_alpha[-1].weight)
            nn.init.zeros_(self.delta_alpha[-1].bias)
        

        if self.delta_sh is not None:
            nn.init.zeros_(self.delta_sh[-1].weight)
            nn.init.zeros_(self.delta_sh[-1].bias)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        h = self.backbone(features)
        delta_alpha = self.delta_alpha(h) if self.learn_alpha else None
        delta_sh = self.delta_sh(h) if self.learn_sh else None
        delta_rot = self.delta_rot(h)
        return self.delta_xyz(h), delta_rot, self.delta_scale(h), delta_alpha, delta_sh


class DeformationField(nn.Module):
    """
    Condition-driven deformation field for Gaussian splatting.
    
    Concatenates spatial encoding (from Gaussian positions) with condition
    encoding (from simulation parameters) and predicts per-Gaussian deltas
    for position, rotation, scale, and optionally opacity and SH coefficients.
    
    Works with any condition_dim (3 for Nyx, 6 for Cloverleaf, etc.).
    """
    
    def __init__(
        self,
        condition_dim: int = 3,
        feature_dim: int = 128,
        hidden_dim: int = 256,
        deform_scale: float = 0.1,
        scene_scale: float = 1.0,
        learn_alpha: bool = False,
        learn_sh: bool = False,
        sh_dim: int = 48,
        use_adapter: bool = True,
    ):
        super().__init__()

        self.condition_dim = condition_dim
        self.deform_scale = deform_scale
        self.alpha_scale = deform_scale #* 10
        self.sh_scale = deform_scale #* 10

        self.scene_scale = scene_scale
        self.learn_alpha = learn_alpha
        self.learn_sh = learn_sh
        self.sh_dim = sh_dim
        self.use_adapter = use_adapter

        spatial_out = feature_dim
        cond_out = feature_dim
        bottleneck_dim = feature_dim #// 4

        self.spatial_encoder = SpatialEncoder(
            output_dim=spatial_out,
            hidden_dim=hidden_dim
        )

        self.condition_encoder = ConditionEncoder(
            condition_dim=condition_dim,
            output_dim=cond_out,
            hidden_dim=hidden_dim,
        )


        if self.use_adapter:
            self.adapter = BottleneckAdapter(
                cond_dim=cond_out, feat_dim=spatial_out, bottleneck_dim=bottleneck_dim
            )

        self.mlp = DeformationMLP(
            input_dim=spatial_out,
            hidden_dim=hidden_dim,
            learn_alpha=learn_alpha,
            learn_sh=learn_sh,
            sh_dim=sh_dim,
        )
    
    def forward(self, xyz: torch.Tensor, condition_vector: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        N = xyz.shape[0]
        
        if condition_vector.dim() == 1:
            condition_vector = condition_vector.unsqueeze(0)
        if condition_vector.shape[0] == 1 and N > 1:
            condition_vector = condition_vector.expand(N, -1)
        
        xyz_norm = xyz / self.scene_scale
        
        spatial_feat = self.spatial_encoder(xyz_norm)
        cond_feat = self.condition_encoder(condition_vector)
        if self.use_adapter:
            features = self.adapter(cond_feat, spatial_feat)
        else:
            # ABLATION: no adapter -- direct elementwise fusion.
            features = spatial_feat + cond_feat

        return self.mlp(features)
    
    def apply_deformation(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        condition_vector: torch.Tensor,
        sh: Optional[torch.Tensor] = None,
        detach_xyz_for_encoding: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        xyz_for_encoding = means.detach() if detach_xyz_for_encoding else means
        delta_xyz, delta_rot, delta_scale, delta_alpha, delta_sh = self.forward(xyz_for_encoding, condition_vector)
        
        deformed_means = means + delta_xyz * self.deform_scale
        deformed_quats = F.normalize(quats + delta_rot * self.deform_scale, dim=-1)
        deformed_scales = scales + delta_scale * self.deform_scale

        if delta_alpha is not None:
            delta_alpha = torch.tanh(delta_alpha)          # [-1, 1]
            deformed_opacities = opacities + delta_alpha * self.alpha_scale
        else:
            deformed_opacities = opacities
        
        if delta_sh is not None and sh is not None:
            deformed_sh = sh + delta_sh.view_as(sh) * self.sh_scale
        else:
            deformed_sh = sh
        
        return deformed_means, deformed_quats, deformed_scales, deformed_opacities, deformed_sh
    
    def get_regularization_loss(self, xyz: torch.Tensor, condition_vector: torch.Tensor) -> torch.Tensor:
        delta_xyz, delta_rot, delta_scale, delta_alpha, delta_sh = self.forward(xyz, condition_vector)
        reg = delta_xyz.pow(2).mean() + delta_rot.pow(2).mean() + delta_scale.pow(2).mean()
        if delta_alpha is not None:
            reg = reg + delta_alpha.pow(2).mean()
        if delta_sh is not None:
            reg = reg + delta_sh.pow(2).mean()
        return reg


def create_deformation_field(
    condition_dim: int = 3,
    feature_dim: int = 128,
    hidden_dim: int = 256,
    deform_scale: float = 0.1,
    scene_scale: float = 1.0,
    learn_alpha: bool = False,
    learn_sh: bool = False,
    sh_dim: int = 48,
    use_adapter: bool = True,
) -> DeformationField:
    return DeformationField(
        condition_dim=condition_dim,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        deform_scale=deform_scale,
        scene_scale=scene_scale,
        learn_alpha=learn_alpha,
        learn_sh=learn_sh,
        sh_dim=sh_dim,
        use_adapter=use_adapter,
    )



NyxDeformationField = DeformationField
create_nyx_deformation_field = create_deformation_field