import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

 
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


class SpatialEncoder(nn.Module):

    def __init__(self, output_dim: int = 128, hidden_dim: int = 512, num_freqs: int = 10):
        super().__init__()
        self.pos_enc = PositionalEncoding(3, num_freqs=num_freqs)
        self.mlp = nn.Sequential(
            nn.Linear(self.pos_enc.output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pos_enc(xyz))


class ConditionEncoder(nn.Module):

    def __init__(self, condition_dim: int = 6, output_dim: int = 128,
                 hidden_dim: int = 512, num_freqs: int = 2):
        super().__init__()
        self.pos_enc = PositionalEncoding(condition_dim, num_freqs=num_freqs)
        self.mlp = nn.Sequential(
            nn.Linear(self.pos_enc.output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pos_enc(condition))


class IsovalueEncoder(nn.Module):

    def __init__(self, isovalue_dim: int = 1, output_dim: int = 128,
                 hidden_dim: int = 256, num_freqs: int = 2):
        super().__init__()
        self.pos_enc = PositionalEncoding(isovalue_dim, num_freqs=num_freqs)
        self.mlp = nn.Sequential(
            nn.Linear(self.pos_enc.output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, isovalue: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pos_enc(isovalue))


class BottleneckAdapter(nn.Module):

    def __init__(self, cond_dim: int, feat_dim: int, bottleneck_dim: int = 128):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(cond_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, feat_dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, cond: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        return feat + self.adapter(cond)


class DeformationMLP(nn.Module):

    def __init__(self, input_dim: int = 128, hidden_dim: int = 512,
                 num_layers: int = 4, learn_alpha: bool = True,
                 learn_sh: bool = True, sh_dim: int = 48):
        super().__init__()
        self.learn_alpha = learn_alpha
        self.learn_sh = learn_sh

        layers = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True)])
        self.backbone = nn.Sequential(*layers)

        self.delta_xyz = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        self.delta_rot = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )
        self.delta_scale = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

        if learn_alpha:
            self.delta_alpha = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.delta_alpha = None

        if learn_sh:
            self.delta_sh = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, sh_dim),
            )
        else:
            self.delta_sh = None

        # Zero-init all head outputs
        for head in [self.delta_xyz, self.delta_rot, self.delta_scale]:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        if self.delta_alpha is not None:
            nn.init.zeros_(self.delta_alpha[-1].weight)
            nn.init.zeros_(self.delta_alpha[-1].bias)
            # Start at zero (identity) — let the network learn which Gaussians
            # to make opaque/transparent rather than starting all-opaque.
        if self.delta_sh is not None:
            nn.init.zeros_(self.delta_sh[-1].weight)
            nn.init.zeros_(self.delta_sh[-1].bias)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        h = self.backbone(features)
        d_alpha = self.delta_alpha(h) if self.learn_alpha else None
        d_sh = self.delta_sh(h) if self.learn_sh else None
        return self.delta_xyz(h), self.delta_rot(h), self.delta_scale(h), d_alpha, d_sh


# ============================================================================
# Surface Deformation Field 
# ============================================================================

class SurfaceDeformationField(nn.Module):


    def __init__(
        self,
        condition_dim: int = 6,
        isovalue_dim: int = 1,
        feature_dim: int = 128,
        hidden_dim: int = 512,
        deform_scale: float = 0.1,
        scene_scale: float = 1.0,
        learn_alpha: bool = True,
        learn_sh: bool = True,
        sh_dim: int = 48,
    ):
        super().__init__()

        self.condition_dim = condition_dim
        self.isovalue_dim = isovalue_dim
        self.deform_scale = deform_scale
        self.alpha_scale = deform_scale
        self.sh_scale = deform_scale
        self.scene_scale = scene_scale
        self.learn_alpha = learn_alpha
        self.learn_sh = learn_sh
        self.sh_dim = sh_dim

        spatial_out = feature_dim
        cond_out = feature_dim
        iso_out = feature_dim
        bottleneck_dim = feature_dim

        self.spatial_encoder = SpatialEncoder(
            output_dim=spatial_out, hidden_dim=hidden_dim,
        )

        self.condition_encoder = ConditionEncoder(
            condition_dim=condition_dim,
            output_dim=cond_out, hidden_dim=hidden_dim,
        )

        self.isovalue_encoder = IsovalueEncoder(
            isovalue_dim=isovalue_dim,
            output_dim=iso_out, hidden_dim=hidden_dim,
        )

        # Adapter: concat(cond_feat, iso_feat) shifts spatial features
        self.adapter = BottleneckAdapter(
            cond_dim=cond_out + iso_out,
            feat_dim=spatial_out,
            bottleneck_dim=bottleneck_dim,
        )

        self.mlp = DeformationMLP(
            input_dim=spatial_out,
            hidden_dim=hidden_dim,
            learn_alpha=learn_alpha,
            learn_sh=learn_sh,
            sh_dim=sh_dim,
        )




    def forward(
        self, xyz: torch.Tensor, condition_vector: torch.Tensor, isovalue: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        N = xyz.shape[0]

        # Expand condition
        if condition_vector.dim() == 1:
            condition_vector = condition_vector.unsqueeze(0)
        if condition_vector.shape[0] == 1 and N > 1:
            condition_vector = condition_vector.expand(N, -1)

        # Expand isovalue
        if isovalue.dim() == 0:
            isovalue = isovalue.unsqueeze(0).unsqueeze(-1)
        if isovalue.dim() == 1:
            isovalue = isovalue.unsqueeze(-1)
        if isovalue.shape[0] == 1 and N > 1:
            isovalue = isovalue.expand(N, -1)

        xyz_norm = xyz / self.scene_scale

        spatial_feat = self.spatial_encoder(xyz_norm)         # (N, D)
        cond_feat = self.condition_encoder(condition_vector)  # (N, D)
        iso_feat = self.isovalue_encoder(isovalue)            # (N, D)

        # Adapter: spatial + shift(concat(cond, iso))
        cond_iso = torch.cat([cond_feat, iso_feat], dim=-1)   # (N, 2D)
        features = self.adapter(cond_iso, spatial_feat)        # (N, D)

        return self.mlp(features)

    # ------------------------------------------------------------------ #
    # Apply deformation
    # ------------------------------------------------------------------ #

    def apply_deformation(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        condition_vector: torch.Tensor,
        isovalue: Optional[torch.Tensor] = None,
        sh: Optional[torch.Tensor] = None,
        vol_spatial_feat: Optional[torch.Tensor] = None,   # unused, API compat
        vol_cond_feat: Optional[torch.Tensor] = None,      # unused, API compat
        detach_xyz_for_encoding: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
   
        assert isovalue is not None, "SurfaceDeformationField requires an isovalue"

        xyz_for_encoding = means.detach() if detach_xyz_for_encoding else means

        d_xyz, d_rot, d_scale, d_alpha, d_sh = self.forward(
            xyz_for_encoding, condition_vector, isovalue,
        )

        deformed_means = means + d_xyz * self.deform_scale
        deformed_quats = F.normalize(quats + d_rot * self.deform_scale, dim=-1)
        deformed_scales = scales + d_scale * self.deform_scale

        if d_alpha is not None:
            d_alpha = torch.tanh(d_alpha)
            deformed_opacities = opacities + d_alpha * self.alpha_scale
        else:
            deformed_opacities = opacities

        if d_sh is not None and sh is not None:
            deformed_sh = sh + d_sh.view_as(sh) * self.sh_scale
        else:
            deformed_sh = sh

        return deformed_means, deformed_quats, deformed_scales, deformed_opacities, deformed_sh


# ============================================================================
# Factory
# ============================================================================

def create_surface_deformation_field(
    condition_dim: int = 6,
    isovalue_dim: int = 1,
    feature_dim: int = 128,
    hidden_dim: int = 512,
    deform_scale: float = 0.1,
    scene_scale: float = 1.0,
    learn_alpha: bool = True,
    learn_sh: bool = True,
    sh_dim: int = 48,
    # Accepted but unused — keeps trainer call site compatible
    vol_feature_dim: int = 128,
    decoder_depth: int = 4,
    **kwargs,
) -> SurfaceDeformationField:
    return SurfaceDeformationField(
        condition_dim=condition_dim,
        isovalue_dim=isovalue_dim,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        deform_scale=deform_scale,
        scene_scale=scene_scale,
        learn_alpha=learn_alpha,
        learn_sh=learn_sh,
        sh_dim=sh_dim,
    )


# Backward-compatible alias
create_deformation_field_surface = create_surface_deformation_field