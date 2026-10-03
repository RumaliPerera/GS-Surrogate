import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
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

    def __init__(self, output_dim: int = 64, hidden_dim: int = 256, num_freqs: int = 10):
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

    def __init__(self, condition_dim: int = 3, output_dim: int = 64,
                 hidden_dim: int = 256, num_freqs: int = 2):
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


class TFEncoder(nn.Module):

    def __init__(self, tf_dim: int = 4, output_dim: int = 64,
                 hidden_dim: int = 256, num_freqs: int = 4):
        super().__init__()
        self.pos_enc = PositionalEncoding(tf_dim, num_freqs=num_freqs)
        self.mlp = nn.Sequential(
            nn.Linear(self.pos_enc.output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, tf_vector: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pos_enc(tf_vector))


class BottleneckAdapter(nn.Module):

    def __init__(self, cond_dim: int, feat_dim: int, bottleneck_dim: int = 64):
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


# ============================================================================
# Appearance-Only MLP (no geometry heads)
# ============================================================================

class AppearanceMLP(nn.Module):

    def __init__(self, input_dim: int = 64, hidden_dim: int = 256,
                 num_layers: int = 3, sh_dim: int = 48):
        super().__init__()
        self.sh_dim = sh_dim

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
            ])
        self.backbone = nn.Sequential(*layers)

        # Opacity head
        self.delta_alpha = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # SH / color head
        self.delta_sh = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, sh_dim),
        )

        # Zero-init outputs
        nn.init.zeros_(self.delta_alpha[-1].weight)
        nn.init.zeros_(self.delta_alpha[-1].bias)
        nn.init.zeros_(self.delta_sh[-1].weight)
        nn.init.zeros_(self.delta_sh[-1].bias)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(features)
        return self.delta_alpha(h), self.delta_sh(h)


# ============================================================================
# TF Appearance Adapter 
# ============================================================================

class TFAppearanceAdapter(nn.Module):

    def __init__(
        self,
        condition_dim: int = 3,
        tf_dim: int = 4,
        feature_dim: int = 64,
        hidden_dim: int = 256,
        alpha_scale: float = 0.1,
        sh_scale: float = 0.1,
        scene_scale: float = 1.0,
        sh_dim: int = 48,
        num_mlp_layers: int = 3,
    ):
        super().__init__()

        self.condition_dim = condition_dim
        self.tf_dim = tf_dim
        self.alpha_scale = alpha_scale
        self.sh_scale = sh_scale
        self.scene_scale = scene_scale
        self.sh_dim = sh_dim

        spatial_out = feature_dim
        cond_out = feature_dim
        tf_out = feature_dim
        bottleneck_dim = feature_dim

        self.spatial_encoder = SpatialEncoder(
            output_dim=spatial_out, hidden_dim=hidden_dim,
        )
        self.condition_encoder = ConditionEncoder(
            condition_dim=condition_dim,
            output_dim=cond_out, hidden_dim=hidden_dim,
        )
        self.tf_encoder = TFEncoder(
            tf_dim=tf_dim,
            output_dim=tf_out, hidden_dim=hidden_dim,
        )

        # Adapter: concat(cond_feat, tf_feat) shifts spatial features
        self.adapter = BottleneckAdapter(
            cond_dim=cond_out + tf_out,
            feat_dim=spatial_out,
            bottleneck_dim=bottleneck_dim,
        )

        self.mlp = AppearanceMLP(
            input_dim=spatial_out,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            sh_dim=sh_dim,
        )

        # total = sum(p.numel() for p in self.parameters())


    def forward(
        self,
        xyz: torch.Tensor,
        condition_vector: torch.Tensor,
        tf_vector: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
   
        N = xyz.shape[0]

        # Expand condition vector
        if condition_vector.dim() == 1:
            condition_vector = condition_vector.unsqueeze(0)
        if condition_vector.shape[0] == 1 and N > 1:
            condition_vector = condition_vector.expand(N, -1)

        # Expand TF vector
        if tf_vector.dim() == 1:
            tf_vector = tf_vector.unsqueeze(0)
        if tf_vector.shape[0] == 1 and N > 1:
            tf_vector = tf_vector.expand(N, -1)

        xyz_norm = xyz / self.scene_scale

        spatial_feat = self.spatial_encoder(xyz_norm)          # [N, D]
        cond_feat = self.condition_encoder(condition_vector)   # [N, D]
        tf_feat = self.tf_encoder(tf_vector)                   # [N, D]

        # Adapter: spatial + shift(concat(cond, tf))
        cond_tf = torch.cat([cond_feat, tf_feat], dim=-1)      # [N, 2D]
        features = self.adapter(cond_tf, spatial_feat)          # [N, D]

        return self.mlp(features)

    # ------------------------------------------------------------------ #
    # Apply deformation (appearance only)
    # ------------------------------------------------------------------ #

    def apply_deformation(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        condition_vector: torch.Tensor,
        tf_vector: torch.Tensor,
        sh: Optional[torch.Tensor] = None,
        detach_xyz_for_encoding: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
        xyz_for_encoding = means.detach() if detach_xyz_for_encoding else means

        d_alpha, d_sh = self.forward(xyz_for_encoding, condition_vector, tf_vector)

        # Opacity: add delta in logit space (before sigmoid)
        d_alpha = torch.tanh(d_alpha)  # bound to [-1, 1]
        adapted_opacities = opacities + d_alpha * self.alpha_scale

        # SH: add delta
        if sh is not None:
            adapted_sh = sh + d_sh.view_as(sh) * self.sh_scale
        else:
            adapted_sh = sh

        # Geometry: pass through unchanged
        return means, quats, scales, adapted_opacities, adapted_sh

    # ------------------------------------------------------------------ #
    # Regularization 
    # ------------------------------------------------------------------ #

    def get_regularization_loss(
        self,
        xyz: torch.Tensor,
        condition_vector: torch.Tensor,
        tf_vector: torch.Tensor,
    ) -> torch.Tensor:
        d_alpha, d_sh = self.forward(xyz, condition_vector, tf_vector)
        return d_alpha.pow(2).mean() + d_sh.pow(2).mean()




def create_tf_appearance_adapter(
    condition_dim: int = 3,
    tf_dim: int = 4,
    feature_dim: int = 64,
    hidden_dim: int = 256,
    alpha_scale: float = 0.1,
    sh_scale: float = 0.1,
    scene_scale: float = 1.0,
    sh_dim: int = 48,
    num_mlp_layers: int = 3,
    **kwargs,   # absorb unused args for compatibility
) -> TFAppearanceAdapter:
    return TFAppearanceAdapter(
        condition_dim=condition_dim,
        tf_dim=tf_dim,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        alpha_scale=alpha_scale,
        sh_scale=sh_scale,
        scene_scale=scene_scale,
        sh_dim=sh_dim,
        num_mlp_layers=num_mlp_layers,
    )