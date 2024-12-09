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

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(points))


if __name__ == '__main__':
    encoder = Dp3PointcloudEncoder(3 + 1 + 3, 3)
    points = torch.rand(5, 32, 3 + 1 + 3)
    print(encoder(points).shape)
