"""Relative position categories used only by inter-cell attention."""
import torch
from torch import nn


def spatial_kwargs(config):
    return dict(cell_spatial_bias=getattr(config, 'cell_spatial_bias', False),
                cell_angle_bins=getattr(config, 'cell_angle_bins', 16),
                cell_distance_edges=getattr(config, 'cell_distance_edges', [10., 20., 40., 80., 160., 320.]),
                cell_coordinate_scale=getattr(config, 'cell_coordinate_scale', 1.))


class RelativeCellSpatialBias(nn.Module):
    """Per-head angle + distance embeddings for receiver i -> sender j."""
    def __init__(self, heads, angle_bins=16, distance_edges=(10., 20., 40., 80., 160., 320.), coordinate_scale=1.):
        super().__init__()
        edges = torch.tensor(distance_edges, dtype=torch.float32)
        if angle_bins < 1 or edges.ndim != 1 or not torch.isfinite(edges).all() or (edges <= 0).any() or (edges.diff() <= 0).any():
            raise ValueError('Spatial bins require positive increasing distance edges and angle_bins >= 1')
        if not 0 < coordinate_scale < float('inf'):
            raise ValueError('coordinate_scale must be finite and positive')
        self.angle_bins, self.coordinate_scale = angle_bins, coordinate_scale
        self.register_buffer('distance_edges', edges)
        self.angle_embedding = nn.Embedding(angle_bins, heads)
        self.distance_embedding = nn.Embedding(len(edges) + 1, heads)
        nn.init.zeros_(self.angle_embedding.weight)
        nn.init.zeros_(self.distance_embedding.weight)

    def categories(self, coords):
        if coords.ndim != 3 or coords.shape[-1] != 2 or not torch.isfinite(coords).all():
            raise ValueError('Spatial attention requires finite coords [batch, cells, 2]')
        delta = (coords.float()[:, None, :, :] - coords.float()[:, :, None, :]) * self.coordinate_scale
        distance = torch.linalg.vector_norm(delta, dim=-1)
        theta = torch.remainder(torch.atan2(delta[..., 1], delta[..., 0]), 2 * torch.pi)
        angle = (theta * (self.angle_bins / (2 * torch.pi))).floor().long().clamp_max(self.angle_bins - 1)
        radial = torch.bucketize(distance.contiguous(), self.distance_edges.float(), right=True)
        return angle, radial, distance > 0

    def forward(self, coords):
        angle, radial, directional = self.categories(coords)
        # Coincident cells have distance but no defined direction.
        bias = self.angle_embedding(angle) * directional[..., None] + self.distance_embedding(radial)
        return bias.permute(0, 3, 1, 2)
