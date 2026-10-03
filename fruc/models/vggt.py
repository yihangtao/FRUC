# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from fruc.models.aggregator import Aggregator
from fruc.heads.camera_head import CameraHead
from fruc.heads.dpt_head import DPTHead, GaussianHead
from fruc.heads.track_head import TrackHead
from fruc.models.sky import SkyGaussian
from fruc.models.prior_denoising import PriorModulatedDenoising


class VGGT(nn.Module, PyTorchModelHubMixin):
    """FRUC reconstruction model with a VGGT backbone and Gaussian decoders.

    The inherited class name is retained for existing imports.
    """

    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, semantic_num = 10):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")# ,down_ratio=2)
        #self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")# ,down_ratio=2)
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="sigmoid")

        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)

        #GS attributes
        self.gs_head = GaussianHead(dim_in= 3 * embed_dim, output_dim=3 + 1 + 3 + 4 + 1 , activation="sigmoid")# ,down_ratio=2)#RGB
        self.instance_head = DPTHead(dim_in= embed_dim, output_dim = 1 + 1, activation="linear") # ,down_ratio=2)#RGB
        self.semantic_head = DPTHead(dim_in= embed_dim, output_dim = semantic_num + 1, activation="linear")# ,down_ratio=2)#RGB
        # Color, opacity, scale, rotation
        self.sky_model = SkyGaussian()

        # FRUC cross-agent latent residual denoising (CALRD).
        # Aggregated features have 2 * embed_dim channels.
        self.prior_denoising = PriorModulatedDenoising(embed_dim=2 * embed_dim)

    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
        collab_images: torch.Tensor = None,
        apply_fruc: bool = False,
    ):
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, image_tokens_list, dino_token_list, image_feature, patch_start_idx = self.aggregator(images)

        predictions = {}

        predictions["image_feature"] = image_feature

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

                # Compute sky_model rendering directly here to avoid DDP multi-forward errors
                if hasattr(self, 'sky_model') and self.sky_model is not None:
                    from fruc.utils.pose_enc import pose_encoding_to_extri_intri
                    H, W = images.shape[-2:]
                    extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], (H, W))
                    extrinsic = extrinsics[0]
                    bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsic.device, dtype=extrinsic.dtype).view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)
                    extrinsic = torch.cat([extrinsic, bottom], dim=1)
                    intrinsic = intrinsics[0]

                    bg_render = self.sky_model(images, extrinsic, intrinsic)
                    predictions["bg_render"] = bg_render

            if self.instance_head is not None:
                dynamic_conf, _ = self.instance_head(dino_token_list, images, patch_start_idx)
                predictions["dynamic_conf"] = dynamic_conf

            # FRUC cooperative reconstruction with two timestamps and two agents.
            if apply_fruc and hasattr(self, 'prior_denoising') and self.prior_denoising is not None and images.shape[1] == 4:
                B, S, P, C = aggregated_tokens_list[-1].shape

                seq_tokens = aggregated_tokens_list[-1]

                # We need to process the entire sequence (S=4: Ego_t0, Collab_t0, Ego_t1, Collab_t1)
                f_mixed_spatial = seq_tokens[:, :, patch_start_idx:] # [B, S, P_spatial, C]

                # Get H_patch, W_patch
                _, _, _, H, W = images.shape
                H_patch, W_patch = H // self.aggregator.patch_size, W // self.aggregator.patch_size

                p_mov = predictions.get("dynamic_conf")
                if p_mov is not None:
                    p_mov = torch.sigmoid(p_mov[..., 0:1]).permute(0, 1, 4, 2, 3) # [B, S, 1, H_feat, W_feat]
                else:
                    p_mov = torch.zeros(B, S, 1, H_patch, W_patch, device=seq_tokens.device)

                p_dyn_list = []
                for i in range(S):
                    p_dyn_i = p_mov[:, i]
                    if p_dyn_i.shape[-2:] != (H_patch, W_patch):
                        p_dyn_i = F.interpolate(p_dyn_i, size=(H_patch, W_patch), mode='bilinear', align_corners=False)
                    p_dyn_list.append(p_dyn_i)

                # Run an ego-only pass to obtain the geometric reference.
                # Ego frames are at indices 0 and 2.
                ego_images = images[:, [0, 2]] # [B, 2, C, H, W]
                with torch.no_grad():
                    ego_tokens_list, _, _, _, _ = self.aggregator(ego_images)
                    seq_ego_tokens = ego_tokens_list[-1] # [B, 2, P, C]
                    f_ego_tokens = seq_ego_tokens[:, :, patch_start_idx:] # [B, 2, P_spatial, C]

                # Reshape to spatial: [B, 2, C, H_patch, W_patch]
                f_ego_spatial_2 = f_ego_tokens.transpose(2, 3).contiguous().view(B, 2, C, H_patch, W_patch)

                # Export f_ego_spatial_2 for Denoising Loss in train.py
                self.f_ego_reference = f_ego_spatial_2.detach()

                # Expand f_ego from S=2 to match S=4: [Ego_t0, Ego_t0, Ego_t1, Ego_t1]
                f_ego_expanded = torch.zeros(B, S, C, H_patch, W_patch, device=f_ego_spatial_2.device, dtype=f_ego_spatial_2.dtype)
                f_ego_expanded[:, 0] = f_ego_spatial_2[:, 0]
                f_ego_expanded[:, 1] = f_ego_spatial_2[:, 0] # Collab t0 uses Ego t0 as geometric reference
                f_ego_expanded[:, 2] = f_ego_spatial_2[:, 1]
                f_ego_expanded[:, 3] = f_ego_spatial_2[:, 1] # Collab t1 uses Ego t1 as geometric reference

                f_ego_ref_spatial = f_ego_expanded.view(B * S, C, H_patch, W_patch)

                # Pass the entire sequence to prior_denoising
                f_denoised_spatial, m_occ_all = self.prior_denoising(
                    f_mixed=f_mixed_spatial,
                    p_dyn_list=p_dyn_list,
                    f_ego_ref=f_ego_ref_spatial
                )

                # Reconstruct fused tokens
                fused = seq_tokens.clone()
                fused[:, :, patch_start_idx:] = f_denoised_spatial

                # Update the sequence tokens
                seq_tokens_updated = fused
                aggregated_tokens_list[-1] = seq_tokens_updated

                # Reconstruct image_tokens_list for gs_head
                image_tokens_list[-1] = torch.cat([dino_token_list[-1], seq_tokens_updated], dim=-1)

                # Save m_occ (only for t1: Ego_t1=2, Collab_t1=3 to match expected shape [B, 2, 1, H, W])
                predictions["m_occ"] = m_occ_all[:, 2:] # [B, 2, 1, H, W]

                # Save u_ego (padded m_occ for backwards compatibility)
                m_occ_t1 = m_occ_all[:, 2:] # [B, 2, 1, H, W]
                m_occ_t1_flat = m_occ_t1.view(B, 2, -1, 1) # [B, 2, P_spatial, 1]
                pad_len = patch_start_idx
                # pad the 3rd dim (P_spatial) at the beginning
                m_occ_padded = F.pad(m_occ_t1_flat, (0, 0, pad_len, 0)) # [B, 2, P, 1]
                predictions["u_ego"] = m_occ_padded

                # Assemble output for Stage 2
                predictions.update({
                    "m_occ": m_occ_all.view(B, S, 1, H_patch, W_patch) if 'm_occ_all' in locals() else None,
                    "p_dyn": p_mov, # Dynamic probability [B, S, 1, H, W].
                    "f_denoised": f_denoised_spatial.view(B, S, C, H_patch, W_patch) if 'f_denoised_spatial' in locals() else None,
                    "f_ego": self.f_ego_reference if hasattr(self, 'f_ego_reference') else None
                })



            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf


            if self.gs_head is not None:
                gs_map, gs_conf = self.gs_head(image_tokens_list, images, patch_start_idx)
                predictions["gs_map"] = gs_map
                predictions["gs_conf"] = gs_conf

            if self.semantic_head is not None:
                semantic_logits, _ = self.semantic_head(dino_token_list, images, patch_start_idx)
                predictions["semantic_logits"] = semantic_logits

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        predictions["images"] = images

        return predictions
