import torch
import torch.nn as nn



class PointcloudEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        # backbone arguments
        layer_size = [64, 128, 256],
        use_layer_norm: bool = True,
    ):
        super().__init__()
        assert len(layer_size) == 3, "len(layer_size) must be 3"

        # DP3 backbone specification
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, layer_size[0]),
            nn.LayerNorm(layer_size[0]) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(layer_size[0], layer_size[1]),
            nn.LayerNorm(layer_size[1]) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(layer_size[1], layer_size[2]),
            nn.LayerNorm(layer_size[2]) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
        )
        self.projector = nn.Sequential(
            nn.Linear(layer_size[2], out_channels),
            nn.LayerNorm(out_channels) if use_layer_norm else nn.Identity()
        )

    def forward(self, 
        particles: torch.Tensor,        # (B, N, T, 3)
        action: torch.Tensor,           # (B, N, 3)
        particle_types: torch.Tensor,   # (B, N)
    ) -> torch.Tensor:
        B, N, T, _ = particles.size()
        points = torch.cat([
            particles.view(B, N, T * 3),
            action,
            particle_types.view(B, N, 1)
        ], dim=-1)
        print(points.shape)
        pred_points = self.projector(self.encoder(points))  # (B, N, 3)
        pred_points[particle_types == 1] = particles[particle_types == 1, :, -1]
        return pred_points


if __name__ == '__main__':
    B = 4
    N = 32
    T = 3
    encoder = PointcloudEncoder(3 * T + 3 + 1, 3)
    points = torch.randn(B, N, T, 3)
    action = torch.randn(B, N, 3)
    particle_types = torch.randint(0, 2, (B, N))
    output = encoder(points, action, particle_types)
    print(output.shape)
