import torch
import torch.nn as nn
import torch.nn.functional as F

def gs_activate_head(inputs, residual):
    """
    Apply scaled residual updates to packed Gaussian attributes.

    Args:
        inputs: Base Gaussian attributes with shape (B, N, 14).
        residual: Predicted updates with the same shape as inputs.

    Returns:
        Packed means, colors, opacity, scales and rotations, shape (B, N, 14).
    """
    fmap = inputs #B,N,C
    means =  fmap[ :, :, :3]
    color = fmap[:, :, 3:6]
    opacity = fmap[:, :, 6:7]
    scale = fmap[:,  :, 7:10]
    rotation = fmap[:, :, 10:14]

    fmap = residual #B,N,C
    means = means + fmap[ :, :, :3] * 0.1
    color = color + fmap[:, :, 3:6] * 0.1
    opacity = opacity + fmap[:, :, 6:7] * 0.1
    scale = scale + fmap[:,  :, 7:10] * 0.01
    rotation = F.normalize(rotation + fmap[:, :, 10:14] * 0.01)

    #color = torch.sigmoid(color)
    #opacity = torch.sigmoid(opacity)
    #scale =  F.softplus(scale)
    #rotation = F.normalize(rotation, dim=-1)

    pts3d = torch.concat([means,color,opacity,scale,rotation],dim=-1)


    return pts3d


class PointNetSetAbstraction(nn.Module):
    def __init__(self, in_channels, mlp_channels):
        super().__init__()
        layers = []
        last_ch = in_channels
        for out_ch in mlp_channels:
            layers.append(nn.Conv1d(last_ch, out_ch, 1))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            last_ch = out_ch
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C, N]
        return self.mlp(x)

class PointNetGSFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 14  # 3+3+1+3+4
        self.abstraction1 = PointNetSetAbstraction(self.input_dim, [64, 128])
        self.abstraction2 = PointNetSetAbstraction(128, [128, 256])

        # Add global feature concatenation
        # Input to fusion_mlp will be local (256) + global (256) = 512
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 14, 1)  # Predict refined Gaussian parameters
        )

        # IMPORTANT: Initialize the last layer to 0 so that the initial residual is 0
        # This prevents the fusion module from destroying the pre-trained priors at the start of training
        nn.init.zeros_(self.fusion_mlp[-1].weight)
        nn.init.zeros_(self.fusion_mlp[-1].bias)

    def forward(self, inputs):
        # x: [B, N, 14]
        x = inputs.permute(0, 2, 1)  # [B, 14, N]
        x = self.abstraction1(x)  # [B, 128, N]
        x = self.abstraction2(x)  # [B, 256, N]

        # Global Max Pooling
        global_feat = torch.max(x, 2, keepdim=True)[0] # [B, 256, 1]
        global_feat = global_feat.repeat(1, 1, x.shape[2]) # [B, 256, N]

        # Concatenate local and global features
        x = torch.cat([x, global_feat], dim=1) # [B, 512, N]

        x = self.fusion_mlp(x)    # [B, 14, N]
        x = x.permute(0, 2, 1)    # [B, N, 14]

        # Residual connection
        x = gs_activate_head(inputs, x)

        return x
