"""FRUC causal occlusion fields and cross-agent latent residual denoising."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightControlNet(nn.Module):
    """CALRD residual branch with occlusion and ego-reference conditioning."""

    def __init__(self, embed_dim):
        super().__init__()
        # Ensure channel dimensions match
        feat_channels = embed_dim

        # Encode the causal occlusion field and optional ego-only reference.
        # Input channels = 1 (m_occ) + feat_channels (f_ego)
        self.control_encoder = nn.Sequential(
            nn.Conv2d(1 + feat_channels, feat_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 4, feat_channels // 2, kernel_size=3, padding=1, stride=2), # Downsample
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 2, feat_channels, kernel_size=3, padding=1, stride=2) # Downsample to H/4, W/4
        )

        # Feature processing branch for f_mixed
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1)
        )

        # Zero-convolutions for controlled injection
        self.zero_conv_prior = nn.Conv2d(feat_channels, feat_channels, kernel_size=1)
        self.zero_conv_out = nn.Conv2d(feat_channels, feat_channels, kernel_size=1)

        nn.init.zeros_(self.zero_conv_prior.weight)
        nn.init.zeros_(self.zero_conv_prior.bias)
        nn.init.zeros_(self.zero_conv_out.weight)
        nn.init.zeros_(self.zero_conv_out.bias)

    def forward(self, f_mixed, m_occ, f_ego=None):
        """
        f_mixed: [B*S, C, H, W]
        m_occ: [B*S, 1, H, W]
        f_ego: Optional [B*S, C, H, W] ego-only reference for residual denoising.
        """
        if f_ego is None:
            f_ego = torch.zeros_like(f_mixed)

        control_in = torch.cat([m_occ, f_ego], dim=1)

        # Encode prior and align spatial dimensions
        c_prior = self.control_encoder(control_in)
        if c_prior.shape[-2:] != f_mixed.shape[-2:]:
            c_prior = F.interpolate(c_prior, size=f_mixed.shape[-2:], mode='bilinear', align_corners=False)

        c_prior = self.zero_conv_prior(c_prior)

        # Process feature and inject prior
        f_feat = self.feature_encoder(f_mixed)
        f_fused = f_feat + c_prior

        # Output zero-conv residual
        f_out = self.zero_conv_out(f_fused)

        # Denoised feature = original + residual
        f_denoised = f_mixed + f_out
        return f_denoised

class PriorModulatedDenoising(nn.Module):
    """Infer agent-wise causal occlusion priors and apply FRUC CALRD."""

    def __init__(self, embed_dim):
        super().__init__()
        # 1. Time-Agent Embedding
        # The released cooperative path uses two timestamps and two agents.
        self.time_embed = nn.Parameter(torch.zeros(1, 2, 1, embed_dim))
        self.agent_embed = nn.Parameter(torch.zeros(1, 2, 1, embed_dim))
        nn.init.normal_(self.time_embed, std=0.02)
        nn.init.normal_(self.agent_embed, std=0.02)

        # 2. Agent-wise Causal Masked Attention (for Z_kin)
        self.kinematic_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)

        # 3. Temporal motion decoder.
        self.phi_temporal = nn.Sequential(
            nn.Conv2d(embed_dim + 1, embed_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )
        # Zero-initialize the final layer of phi_temporal
        nn.init.zeros_(self.phi_temporal[2].weight)
        nn.init.zeros_(self.phi_temporal[2].bias)

        # 4. Cross-agent latent residual denoising with zero-initialized injection.
        self.denoiser = LightweightControlNet(embed_dim)

    def get_occlusion_prior(self, z_kin, p_dyn):
        decoder_in = torch.cat([z_kin, p_dyn], dim=1)
        delta_p_kin = self.phi_temporal(decoder_in)
        m_occ = torch.clamp(p_dyn + delta_p_kin, min=0.0, max=1.0)
        return m_occ, delta_p_kin

    def forward(self, f_mixed, p_dyn_list, f_ego_ref=None):
        """
        f_mixed: [B, S, P, C], features across timestamps and agents.
        p_dyn_list: list of [B, 1, H, W] for each frame
        f_ego_ref: Optional ego-only reference features for latent denoising.
        """
        B, S, P, C = f_mixed.shape
        _, _, H, W = p_dyn_list[0].shape

        # 1. Add timestamp and agent embeddings to the mixed features.
        f_embedded = f_mixed.clone()
        if S == 4:
            # Sequence order: [Ego_t0, Collab_t0, Ego_t1, Collab_t1].
            f_embedded = f_embedded.view(B, 2, 2, P, C) # [B, Time, Agent, P, C]
            f_embedded = f_embedded + self.time_embed.unsqueeze(3) + self.agent_embed.unsqueeze(1)
            f_embedded = f_embedded.view(B, S * P, C)
        elif S == 2:
            # Single-timestamp order: [Ego_t1, Collab_t1].
            f_embedded = f_embedded.view(B, 1, 2, P, C)
            f_embedded = f_embedded + self.time_embed[:, 1:2].unsqueeze(3) + self.agent_embed.unsqueeze(1)
            f_embedded = f_embedded.view(B, S * P, C)
        else:
            f_embedded = f_embedded.view(B, S * P, C)

        # 2. Agent-wise Causal Masked Attention (for Z_kin)
        # M_causal[i, j] = 1 if agent(i) == agent(j) and tau(i) >= tau(j)
        agent_mask = torch.zeros(S, S, device=f_mixed.device)
        if S == 4:
            # agent: [0, 1, 0, 1], tau: [0, 0, 1, 1]
            # i=0(Ego0) sees j=0
            agent_mask[0, 0] = 1
            # i=1(Col0) sees j=1
            agent_mask[1, 1] = 1
            # i=2(Ego1) sees j=0, 2
            agent_mask[2, 0] = 1; agent_mask[2, 2] = 1
            # i=3(Col1) sees j=1, 3
            agent_mask[3, 1] = 1; agent_mask[3, 3] = 1
        elif S == 2:
            # i=0(Ego1) sees j=0
            agent_mask[0, 0] = 1
            # i=1(Col1) sees j=1
            agent_mask[1, 1] = 1
        else:
            agent_mask = torch.eye(S, device=f_mixed.device)

        agent_mask_expanded = agent_mask.unsqueeze(1).unsqueeze(3).expand(S, P, S, P).reshape(S*P, S*P)
        agent_mask_inf = torch.zeros_like(agent_mask_expanded, dtype=f_mixed.dtype)
        agent_mask_inf[agent_mask_expanded == 0] = float('-inf')

        z_kin_flat, _ = self.kinematic_attn(
            query=f_embedded,
            key=f_embedded,
            value=f_embedded,
            is_causal=False,
            attn_mask=agent_mask_inf
        )

        # Reshape Z_kin back to spatial maps
        z_kin_spatial = z_kin_flat.view(B, S, P, C)

        # 3. Infer the occlusion prior for every frame.
        m_occ_spatial_list = []
        for i in range(S):
            z_kin_i = z_kin_spatial[:, i].transpose(1, 2).contiguous().view(B, C, H, W)
            m_occ_spatial, _ = self.get_occlusion_prior(z_kin_i, p_dyn_list[i])
            m_occ_spatial_list.append(m_occ_spatial)

        m_occ_all = torch.stack(m_occ_spatial_list, dim=1) # [B, S, 1, H, W]
        m_occ_flat = m_occ_all.view(B, S * P, 1) # [B, S*P, 1]

        # 4. Condition CALRD on the occlusion prior and ego-only reference.
        # Restore spatial maps for the convolutional residual branch.
        f_spatial = f_embedded.view(B, S, P, C).transpose(2, 3).contiguous().view(B * S, C, H, W)
        m_occ_spatial = m_occ_all.view(B * S, 1, H, W)

        f_ego_spatial = None
        if f_ego_ref is not None:
            f_ego_spatial = f_ego_ref.view(B, S, P, C).transpose(2, 3).contiguous().view(B * S, C, H, W)

        f_denoised_spatial = self.denoiser(f_spatial, m_occ_spatial, f_ego=f_ego_spatial)

        f_denoised = f_denoised_spatial.view(B, S, C, P).transpose(2, 3).contiguous()

        return f_denoised, m_occ_all
