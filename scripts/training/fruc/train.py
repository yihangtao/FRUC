"""Train FRUC Gaussian reconstruction and collaborative adaptation on V2X-Real."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import lpips
import matplotlib.pyplot as plt
import numpy as np

from fruc.models.vggt import VGGT
from fruc.utils.checkpoint import extract_state_dict
from fruc.utils.pose_enc import pose_encoding_to_extri_intri
from fruc.utils.geometry import unproject_depth_map_to_point_map
from fruc.utils.gs import concat_list, get_split_gs
from gsplat.rendering import rasterization
from datasets.v2xreal_dataset import V2XRealDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def compute_lifespan_loss(gamma):
    return torch.mean(torch.abs(1 / (gamma + 1e-6)))

def alpha_t(t, t0, alpha, gamma0 = 1, gamma1 = 0.1):
    sigma = torch.log(torch.tensor(gamma1)).to(gamma0.device) /  ((gamma0)**2 + 1e-6)
    conf = torch.exp(sigma*(t0-t)**2)
    alpha_ = alpha * conf
    return alpha_.float()

def parse_args():
    parser = argparse.ArgumentParser(description="Train FRUC reconstruction on V2X-Real sequences.")
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='logs/fruc_v2xreal')
    parser.add_argument('--sequence_length', type=int, default=4, help='Single-agent sequence length; cooperative samples always use two timestamps')
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--save_image', type=int, default=100)
    parser.add_argument('--save_ckpt', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dataset_type', type=str, default='v2xreal_coop', choices=['v2xreal', 'v2xreal_coop'])
    parser.add_argument('--fruc_loss_weight', type=float, default=0.2, help='Weight for FRUC cooperative gain loss')
    parser.add_argument('--cam0_only', action='store_true', help='Only load ego cam 0 and collab cam 0 pairs')
    args = parser.parse_args()
    if args.batch_size != 1:
        parser.error('The FRUC trainer supports batch_size=1 per GPU.')
    if args.max_epoch < 1:
        parser.error('max_epoch must be positive.')
    return args

def main(args):
    dist.init_process_group(backend='nccl')
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    dtype = torch.bfloat16

    if args.dataset_type == 'v2xreal':
        dataset = V2XRealDataset(args.image_dir, scene_names=None, sequence_length=args.sequence_length, mode=1, views=1)
    elif args.dataset_type == 'v2xreal_coop':
        from datasets.v2xreal_coop_dataset import V2XRealCoopDataset
        # Each cooperative sample interleaves two agents at two timestamps.
        dataset = V2XRealCoopDataset(args.image_dir, split='train', cam0_only=args.cam0_only)
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    if len(dataset) == 0:
        raise ValueError('No training samples found. Check the dataset path and context.json view associations.')
    sampler = DistributedSampler(dataset,shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)

    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, "ckpt"), exist_ok=True)

    # Validation Set Preparation
    fixed_val_samples = []
    if (args.dataset_type == 'v2xreal_coop') and args.local_rank == 0:
        from datasets.v2xreal_coop_dataset import V2XRealCoopDataset
        # Try to infer validation directory from image_dir
        val_dir = args.image_dir.replace('train', 'val')
        if not os.path.exists(val_dir):
             # Fallback: assume image_dir is root, or structure is different
             val_dir = args.image_dir # Just use train set if val not found for debugging, or user manual input
             print(f"Validation dir {val_dir.replace('train', 'val')} not found, using {val_dir} for validation sampling.")

        val_dataset = V2XRealCoopDataset(val_dir, split='val', for_vis=True, cam0_only=args.cam0_only)
        # Select 1 sample per scene
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False) # Shuffle False to keep deterministic

        print(f"Collecting fixed validation samples from {val_dir} (1 per distinct scene)...")
        unique_scene_names = set()
        for i, batch in enumerate(val_dataloader):
            scene_ok = False
            scene_name = None
            if isinstance(batch, dict) and 'scene_name' in batch:
                try:
                    scene_name = batch['scene_name'][0]
                except Exception:
                    pass
            if scene_name is not None:
                if scene_name not in unique_scene_names:
                    unique_scene_names.add(scene_name)
                    fixed_val_samples.append(batch)
            else:
                if not fixed_val_samples: # Add at least one if no scene name
                    fixed_val_samples.append(batch)
        print(f"Collected {len(fixed_val_samples)} validation samples from {len(unique_scene_names) if unique_scene_names else 'unknown'} distinct scenes.")

    model = VGGT().to(device)
    if args.ckpt_path:
        print(f"Loading checkpoint from {args.ckpt_path}")
        checkpoint = torch.load(args.ckpt_path, map_location="cpu")
        state_dict = extract_state_dict(checkpoint)
        incompatible = model.load_state_dict(state_dict, strict=False)
        print(f'Checkpoint: {len(incompatible.missing_keys)} missing and {len(incompatible.unexpected_keys)} unexpected keys.')
    else:
        print("No checkpoint provided, initializing model from scratch.")

    model.train()
    # To fix "Expected to have finished reduction in the prior iteration before starting a new one"
    # The issue is that find_unused_parameters=True is needed because some outputs (like those from
    # instance_head or sky_model) are computed but not used in the loss if we don't have certain targets,
    # or because we freeze certain heads and their parameters might not receive gradients in a way DDP expects.
    model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)
    # When find_unused_parameters=True, we should not call _set_static_graph()
    # model._set_static_graph()

    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)


    binary_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
    semantic_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

    for param in model.module.parameters():
        param.requires_grad = False

    trainable_heads = ["gs_head", "depth_head", "prior_denoising", "sky_model"]
    for head_name in trainable_heads:
        if hasattr(model.module, head_name) and getattr(model.module, head_name) is not None:
            for param in getattr(model.module, head_name).parameters():
                param.requires_grad = True

    if args.local_rank == 0:
        print("="*40)
        print("Trainable Modules and Parameter Counts:")
        total_trainable_params = 0
        for head_name in trainable_heads:
            head_module = getattr(model.module, head_name, None)
            if head_module is not None:
                head_params = sum(p.numel() for p in head_module.parameters() if p.requires_grad)
                if head_params > 0:
                    print(f" - {head_name}: {head_params / 1e6:.2f} M")
                    total_trainable_params += head_params
        print(f"Total Trainable Parameters: {total_trainable_params / 1e6:.2f} M")
        print("="*40)

    optimizer_params = []
    if hasattr(model.module, 'gs_head') and model.module.gs_head is not None:
        optimizer_params.append({'params': model.module.gs_head.parameters(), 'lr': 4e-5})
    if hasattr(model.module, 'depth_head') and model.module.depth_head is not None:
        optimizer_params.append({'params': model.module.depth_head.parameters(), 'lr': 4e-5})
    if hasattr(model.module, 'prior_denoising') and model.module.prior_denoising is not None:
        optimizer_params.append({'params': model.module.prior_denoising.parameters(), 'lr': 1e-4})
    if hasattr(model.module, 'sky_model') and model.module.sky_model is not None:
        optimizer_params.append({'params': model.module.sky_model.parameters(), 'lr': 1e-4})

    optimizer = AdamW(optimizer_params, weight_decay=1e-4)

    warmup_iterations = 1000
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: min((step + 1) / warmup_iterations, 1.0) * 0.5 * (
            1 + torch.cos(torch.tensor(torch.pi * step / args.max_epoch)))
    )

    global_step = 0
    for epoch in tqdm(range(args.max_epoch)):
        step = epoch
        sampler.set_epoch(epoch)
        for batch in dataloader:
            global_step += 1
            images = batch['images'].to(device)
            sky_mask = batch['masks'].to(device).permute(0, 1, 3, 4, 2)
            bg_mask = (sky_mask == 0).any(dim=-1)
            timestamps = batch['timestamps'][0].to(device)

            if 'dynamic_mask' in batch:
                dynamic_masks = batch['dynamic_mask'].to(device)[:, :, 0, :, :]

            optimizer.zero_grad()

            predictions = model(images, apply_fruc=(args.dataset_type in ['v2xreal_coop']))
            images_for_render = images
            sky_mask_for_render = sky_mask
            bg_mask_for_render = bg_mask
            timestamps_for_render = timestamps
            dynamic_masks_for_render = dynamic_masks if 'dynamic_mask' in batch else None
            with torch.amp.autocast('cuda', dtype=dtype):
                H, W = images_for_render.shape[-2:]
                extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], (H, W))
                extrinsic = extrinsics[0]
                bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsic.device).view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)
                extrinsic = torch.cat([extrinsic, bottom], dim=1)
                intrinsic = intrinsics[0]
                depth_map = predictions["depth"][0]
                point_map = unproject_depth_map_to_point_map(depth_map.detach(), extrinsics[0], intrinsics[0])[None,...]
                point_map = torch.from_numpy(point_map).to(device).float()
                gs_map = predictions["gs_map"]
                gs_conf = predictions["gs_conf"]
                dy_map = predictions["dynamic_conf"].squeeze(-1)
                static_mask = torch.ones_like(bg_mask_for_render)
                static_points = point_map[static_mask].reshape(-1, 3)
                gs_dynamic_list = dy_map[static_mask].sigmoid()
                static_rgbs, static_opacity, static_scales, static_rotations = get_split_gs(gs_map, static_mask)
                static_opacity = static_opacity * (1 - gs_dynamic_list)
                static_gs_conf = gs_conf[static_mask]
                frame_idx = torch.nonzero(static_mask, as_tuple=False)[:,1]
                gs_timestamps = timestamps_for_render[frame_idx]
                dynamic_points, dynamic_rgbs, dynamic_opacitys, dynamic_scales, dynamic_rotations = [], [], [], [], []
                for i in range(dy_map.shape[1]):
                    point_map_i = point_map[:, i]
                    bg_mask_i = bg_mask_for_render[:, i]
                    dynamic_point = point_map_i[bg_mask_i].reshape(-1, 3)
                    dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation = get_split_gs(gs_map[:, i], bg_mask_i)
                    gs_dynamic_list_i = dy_map[:, i][bg_mask_i].sigmoid()
                    dynamic_opacity = dynamic_opacity * gs_dynamic_list_i
                    dynamic_points.append(dynamic_point)
                    dynamic_rgbs.append(dynamic_rgb)
                    dynamic_opacitys.append(dynamic_opacity)
                    dynamic_scales.append(dynamic_scale)
                    dynamic_rotations.append(dynamic_rotation)
                chunked_renders, chunked_alphas = [], []
                S = extrinsic.shape[0]
                for idx in range(S):
                    t0 = timestamps_for_render[idx]
                    static_opacity_ = alpha_t(gs_timestamps, t0, static_opacity, gamma0 = static_gs_conf)
                    static_gs_list = [static_points, static_rgbs, static_opacity_, static_scales, static_rotations]
                    if dynamic_points:
                        world_points, rgbs, opacity, scales, rotation = concat_list(
                            static_gs_list,
                            [dynamic_points[idx], dynamic_rgbs[idx], dynamic_opacitys[idx], dynamic_scales[idx], dynamic_rotations[idx]]
                        )
                    renders_chunk, alphas_chunk, _ = rasterization(
                        means=world_points,
                        quats=rotation,
                        scales=scales,
                        opacities=opacity,
                        colors=rgbs,
                        viewmats=extrinsic[idx][None],
                        Ks=intrinsic[idx][None],
                        width=W,
                        height=H,
                    )
                    chunked_renders.append(renders_chunk)
                    chunked_alphas.append(alphas_chunk)
                renders = torch.cat(chunked_renders, dim=0)
                alphas = torch.cat(chunked_alphas, dim=0)
                bg_render = predictions["bg_render"]
                renders = alphas * renders + (1 - alphas) * bg_render
                rendered_image = renders.permute(0, 3, 1, 2)
                target_image = images_for_render[0]
                chunked_renders_nody, chunked_alphas_nody = [], []
                for idx in range(S):
                    t0 = timestamps_for_render[idx]
                    static_opacity_ = alpha_t(gs_timestamps, t0, static_opacity, gamma0 = static_gs_conf)
                    world_points_s, rgbs_s, opacity_s, scales_s, rotation_s = static_points, static_rgbs, static_opacity_, static_scales, static_rotations
                    renders_chunk_s, alphas_chunk_s, _ = rasterization(
                        means=world_points_s,
                        quats=rotation_s,
                        scales=scales_s,
                        opacities=opacity_s,
                        colors=rgbs_s,
                        viewmats=extrinsic[idx][None],
                        Ks=intrinsic[idx][None],
                        width=W,
                        height=H,
                    )
                    chunked_renders_nody.append(renders_chunk_s)
                    chunked_alphas_nody.append(alphas_chunk_s)
                renders_nody = torch.cat(chunked_renders_nody, dim=0)
                alphas_nody = torch.cat(chunked_alphas_nody, dim=0)
                renders_nody = alphas_nody * renders_nody + (1 - alphas_nody) * bg_render
                rendered_image_nody = renders_nody.permute(0, 3, 1, 2)
                # Apply weight mask to mainly optimize Ego.
                # Sequence order: Ego_t0, Collab_t0, Ego_t1, Collab_t1
                # Ego is at index 0 and 2.
                weight_mask = torch.ones_like(target_image)
                if rendered_image.shape[0] == 4 and args.dataset_type in ['v2xreal_coop']:
                    weight_mask[0] = 1.0  # Ego, t0
                    weight_mask[1] = 0.0  # Collab, t0 (No constraint)
                    weight_mask[2] = 1.0  # Ego, t1
                    weight_mask[3] = 0.0  # Collab, t1 (No constraint)

                loss = F.l1_loss(rendered_image, target_image, reduction='none')
                # Use mean over the spatial/channel dimensions first, then apply the mask over the batch dimension
                loss = (loss.mean(dim=(1, 2, 3)) * weight_mask[:, 0, 0, 0]).sum() / (weight_mask[:, 0, 0, 0].sum() + 1e-6)

                sky_mask_loss = F.l1_loss(alphas, 1 - sky_mask_for_render[0, ..., 0][..., None], reduction='none')
                weight_mask_alpha = weight_mask.mean(dim=(1, 2, 3)).view(-1, 1, 1, 1) # [S, 1, 1, 1]
                # Same here: mean over spatial dims first, then apply the 1D mask over batch dimension
                sky_mask_loss = (sky_mask_loss.mean(dim=(1, 2, 3)) * weight_mask_alpha[:, 0, 0, 0]).sum() / (weight_mask_alpha[:, 0, 0, 0].sum() + 1e-6)

                loss +=  sky_mask_loss
                gs_conf_loss = compute_lifespan_loss(static_gs_conf)
                loss += 0.01 * gs_conf_loss
                if dynamic_masks_for_render is not None:
                    dyn_l = F.binary_cross_entropy_with_logits(dy_map[0].unsqueeze(1), dynamic_masks_for_render[0].float().unsqueeze(1), reduction='none')
                    dynamic_loss = (dyn_l.mean(dim=(1, 2, 3)) * weight_mask_alpha[:, 0, 0, 0]).sum() / (weight_mask_alpha[:, 0, 0, 0].sum() + 1e-6)
                else:
                    dyn_l = F.binary_cross_entropy_with_logits(dy_map[0].unsqueeze(1), torch.zeros_like(dy_map[0].unsqueeze(1)), reduction='none')
                    dynamic_loss = (dyn_l.mean(dim=(1, 2, 3)) * weight_mask_alpha[:, 0, 0, 0]).sum() / (weight_mask_alpha[:, 0, 0, 0].sum() + 1e-6)
                loss = loss + 0.05 * dynamic_loss
                fruc_loss = 0.0
                coop_gain_loss = 0.0
                count_frames = 0
                for idx in range(rendered_image_nody.shape[0]):
                    # Determine weight: Only apply to Ego (idx 0, 2)
                    weight = 1.0
                    if args.dataset_type in ['v2xreal_coop']:
                        if idx % 2 != 0:
                            continue # Skip Collab vehicle frames for coop_gain_loss
                        weight = 1.0 # Apply only to Ego

                    dyn_prob = torch.sigmoid(dy_map[0, idx])
                    dyn_bin = (dyn_prob > 0.5).float().unsqueeze(0)
                    dilated = F.max_pool2d(dyn_bin, kernel_size=7, stride=1, padding=3).squeeze(0)
                    ring = (dilated > 0.5) & (dyn_bin.squeeze(0) < 0.5)
                    non_sky = (sky_mask_for_render[0, idx, ..., 0] == 0)
                    valid_alpha = (alphas_nody[idx, ..., 0] > 0.05)
                    mask2d = (ring & non_sky & valid_alpha).float()
                    if mask2d.sum() > 0:
                        mask3 = mask2d.unsqueeze(0).expand_as(rendered_image_nody[idx])
                        l = F.l1_loss(rendered_image_nody[idx] * mask3, target_image[idx] * mask3, reduction='sum') / (mask2d.sum() * rendered_image_nody.shape[1] + 1e-6)
                        coop_gain_loss = coop_gain_loss + l * weight
                        count_frames += 1
                if count_frames > 0:
                    coop_gain_loss = coop_gain_loss / count_frames
                    fruc_loss += args.fruc_loss_weight * coop_gain_loss
                loss += fruc_loss

                # Latent FRUC module loss
                latent_fruc_loss = 0.0
                l_reg_val = l_var_val = l_denoise_val = 0.0
                if "m_occ" in predictions:
                    m_occ = predictions["m_occ"] # [B, S, 1, H, W]

                    # Regularize the causal occlusion field toward a dilated dynamic prior.
                    if "dynamic_conf" in predictions:
                        p_mov = predictions["dynamic_conf"]
                        # p_mov shape is [B, S, H_feat, W_feat, 1] before permute
                        p_dyn = torch.sigmoid(p_mov[..., 0:1]).permute(0, 1, 4, 2, 3) # [B, S, 1, H_feat, W_feat]
                        p_dyn_t1 = p_dyn[:, 2:] # [B, 2, 1, H, W], extract t1 for Ego and Collab

                        # Ensure spatial dimensions match M_occ
                        if p_dyn_t1.shape[-2:] != m_occ.shape[-2:]:
                            B_dyn, S_dyn, C_dyn, H_dyn, W_dyn = p_dyn_t1.shape
                            p_dyn_t1_flat = p_dyn_t1.reshape(-1, C_dyn, H_dyn, W_dyn)
                            p_dyn_t1_resized = F.interpolate(p_dyn_t1_flat, size=m_occ.shape[-2:], mode='bilinear', align_corners=False)
                            p_dyn_t1 = p_dyn_t1_resized.view(B_dyn, S_dyn, C_dyn, m_occ.shape[-2], m_occ.shape[-1])

                        # Dilate the dynamic prior to allow M_occ to expand safely around objects
                        # Max pooling acts as a morphological dilation
                        p_dyn_t1_dilated = F.max_pool2d(p_dyn_t1.squeeze(2), kernel_size=5, stride=1, padding=2).unsqueeze(2)

                        # Expand p_dyn_t1_dilated to match S=4 if M_occ is S=4
                        if p_dyn_t1_dilated.shape[1] != m_occ.shape[1]:
                            p_dyn_expanded = torch.zeros_like(m_occ)
                            # p_dyn_t1 has 2 elements (Ego t1, Collab t1). Map them appropriately.
                            # M_occ has 4 elements: Ego t0, Collab t0, Ego t1, Collab t1
                            # Assuming p_dyn_t1_dilated contains Ego t1 at [0] and Collab t1 at [1]
                            p_dyn_expanded[:, 0] = p_dyn_t1_dilated[:, 0] # Use Ego t1 for Ego t0 (approx)
                            p_dyn_expanded[:, 1] = p_dyn_t1_dilated[:, 1] # Use Collab t1 for Collab t0 (approx)
                            p_dyn_expanded[:, 2] = p_dyn_t1_dilated[:, 0] # Ego t1
                            p_dyn_expanded[:, 3] = p_dyn_t1_dilated[:, 1] # Collab t1
                            p_dyn_t1_dilated = p_dyn_expanded

                        # 1. Prior Regularization Loss (L_reg):
                        # We need to explicitly encourage the network to EXPLORE the dynamic regions (p_dyn).
                        # If we only penalize over-expansion, the zero-initialized network will be too conservative.
                        # Therefore, we use a standard L1 loss to actively pull M_occ towards p_dyn.
                        # This gives the network a strong initial push to "open up" the blind spots.
                        loss_reg = torch.mean(torch.abs(m_occ - p_dyn_t1_dilated.detach()))
                    else:
                        # Fallback to global L1 if p_dyn is missing
                        loss_reg = torch.mean(torch.abs(m_occ))

                    # Add Variance Loss to prevent M_occ from collapsing to a constant prior
                    loss_var = -0.01 * torch.var(m_occ)

                    # Extract individual latent loss components if available
                    l_reg_val = loss_reg.item() if 'loss_reg' in locals() and isinstance(loss_reg, torch.Tensor) else 0.0
                    l_var_val = loss_var.item() if 'loss_var' in locals() and isinstance(loss_var, torch.Tensor) else 0.0

                    latent_fruc_loss = (loss_reg + loss_var)

                    # 3. FRUC latent residual denoising loss (L_denoise).
                    # Regularize denoised features toward the detached ego-only reference.
                    if "f_denoised" in predictions and "f_ego" in predictions:
                        f_denoised = predictions["f_denoised"] # [B, S, C, H_feat, W_feat]
                        f_ego_ref = predictions["f_ego"] # [B, 2, C, H_feat, W_feat]

                        if f_denoised is not None and f_ego_ref is not None:
                            B_feat, S_feat, C_feat, H_feat, W_feat = f_denoised.shape

                            # Expand f_ego_ref from S=2 to S=4 to match f_denoised
                            f_ego_expanded = torch.zeros_like(f_denoised)
                            f_ego_expanded[:, 0] = f_ego_ref[:, 0]
                            f_ego_expanded[:, 1] = f_ego_ref[:, 0]
                            f_ego_expanded[:, 2] = f_ego_ref[:, 1]
                            f_ego_expanded[:, 3] = f_ego_ref[:, 1]

                            # Resize m_occ to match feature spatial dimensions
                            if m_occ.shape[-2:] != f_denoised.shape[-2:]:
                                m_occ_resized = F.interpolate(m_occ.reshape(-1, 1, m_occ.shape[-2], m_occ.shape[-1]),
                                                              size=(H_feat, W_feat), mode='bilinear', align_corners=False)
                                m_occ_resized = m_occ_resized.view(B_feat, S_feat, 1, H_feat, W_feat)
                            else:
                                m_occ_resized = m_occ

                            # The released recipe applies an unmasked mean-squared error.
                            # It balances ego-reference preservation with collaborative
                            # corrections encouraged by the photometric reconstruction loss.
                            loss_denoise = F.mse_loss(f_denoised, f_ego_expanded.detach(), reduction='mean')

                            l_denoise_val = loss_denoise.item() if 'loss_denoise' in locals() and isinstance(loss_denoise, torch.Tensor) else 0.0
                            latent_fruc_loss += loss_denoise * 10.0 # Scale up the latent loss

                    latent_fruc_loss = latent_fruc_loss * args.fruc_loss_weight
                    loss = loss + latent_fruc_loss
                else:
                    l_reg_val = 0.0
                    l_var_val = 0.0
                    l_denoise_val = 0.0

                # Perceptual reconstruction loss with an overall weight of 0.1.
                lpips_val = lpips_loss_fn(rendered_image, target_image)
                # Released per-view weights: 0.1 for ego, 1.0 for collaborator views.
                if rendered_image.shape[0] == 4 and (args.dataset_type == 'v2xreal_coop'):
                    lpips_weights = torch.ones(4, device=lpips_val.device)
                    lpips_weights[0] = 0.1
                    lpips_weights[1] = 1.0
                    lpips_weights[2] = 0.1
                    lpips_weights[3] = 1.0
                    lpips_val = lpips_val.view(4, -1).mean(dim=1) * lpips_weights
                    loss = loss + 0.1 * (lpips_val.sum() / (lpips_weights.sum() + 1e-6))
                else:
                    loss = loss + 0.1 * lpips_val.mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

            if args.local_rank == 0 and global_step % 10 == 0:
                fruc_loss_val = fruc_loss.item() if isinstance(fruc_loss, torch.Tensor) else 0.0

                # Use previously extracted latent loss values
                l_coop_gain_val = coop_gain_loss.item() if isinstance(coop_gain_loss, torch.Tensor) else coop_gain_loss

                print(f"[Epoch {epoch}/{args.max_epoch}][Step {global_step}] Loss: {loss.item():.4f} | L1: {F.l1_loss(rendered_image, target_image).item():.4f} | Sky: {sky_mask_loss.item():.4f} | Dyn: {dynamic_loss.item():.4f} | FRUC: {fruc_loss_val:.4f} (Reg: {l_reg_val:.4f}, CoopGain: {l_coop_gain_val:.4f}, Var: {l_var_val:.4f}, Denoise: {l_denoise_val:.4f}) | LPIPS: {lpips_val.mean().item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

            if args.local_rank == 0 and (global_step % args.save_ckpt == 0):
                ckpt_path = os.path.join(args.log_dir, "ckpt", "model_latest.pth")
                torch.save(model.module.state_dict(), ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

            if args.local_rank == 0 and (global_step == 1 or global_step % args.save_image == 0):
                # Save all frames in the batch to visualize both vehicles
                # Use fixed_val_samples if available

                samples_to_vis = []
                if len(fixed_val_samples) > 0:
                    samples_to_vis = fixed_val_samples
                    print(f"[Validation] Visualizing {len(samples_to_vis)} fixed samples...")
                else:
                    samples_to_vis = [batch] # Fallback to current batch

                model.eval() # Ensure eval mode

                for sample_idx, val_batch in enumerate(samples_to_vis):
                    with torch.no_grad():
                        # Move val_batch to device
                        v_images = val_batch['images'].to(device)
                        v_masks = val_batch['masks'].to(device).permute(0, 1, 3, 4, 2)
                        v_bg_mask = (v_masks == 0).any(dim=-1)
                        v_timestamps = val_batch['timestamps'][0].to(device)

                        v_dynamic_masks = None
                        if 'dynamic_mask' in val_batch:
                             v_dynamic_masks = val_batch['dynamic_mask'].to(device)[:, :, 0, :, :]

                        with torch.amp.autocast('cuda', dtype=dtype):
                            if args.dataset_type in ['v2xreal_coop']:
                                v_predictions = model(v_images, apply_fruc=True)
                                v_images_for_render = v_images
                                v_bg_mask_for_render = v_bg_mask
                                v_timestamps_for_render = v_timestamps
                            else:
                                v_predictions = model(v_images)
                                v_images_for_render = v_images
                                v_bg_mask_for_render = v_bg_mask
                                v_timestamps_for_render = v_timestamps

                            v_H, v_W = v_images_for_render.shape[-2:]
                            v_extrinsics, v_intrinsics = pose_encoding_to_extri_intri(v_predictions['pose_enc'], (v_H, v_W))
                            v_extrinsic = v_extrinsics[0] # Assuming batch size 1
                            v_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=v_extrinsic.device).view(1, 1, 4).expand(v_extrinsic.shape[0], 1, 4)
                            v_extrinsic = torch.cat([v_extrinsic, v_bottom], dim=1)
                            v_intrinsic = v_intrinsics[0]

                            if True: # use_depth (Match training)
                                v_depth_map = v_predictions["depth"][0]
                                v_point_map = unproject_depth_map_to_point_map(v_depth_map.detach(), v_extrinsics[0], v_intrinsics[0])[None,...]
                                v_point_map = torch.from_numpy(v_point_map).to(device).float()
                            else:
                                v_point_map = v_predictions["world_points"]

                            v_gs_map = v_predictions["gs_map"]
                            v_gs_conf = v_predictions["gs_conf"]
                            v_dy_map = v_predictions["dynamic_conf"].squeeze(-1)

                            v_static_mask = torch.ones_like(v_bg_mask_for_render)
                            v_static_points = v_point_map[v_static_mask].reshape(-1, 3)
                            v_gs_dynamic_list = v_dy_map[v_static_mask].sigmoid()
                            v_static_rgbs, v_static_opacity, v_static_scales, v_static_rotations = get_split_gs(v_gs_map, v_static_mask)
                            v_static_opacity = v_static_opacity * (1 - v_gs_dynamic_list)
                            v_static_gs_conf = v_gs_conf[v_static_mask]
                            v_frame_idx = torch.nonzero(v_static_mask, as_tuple=False)[:,1]
                            v_gs_timestamps = v_timestamps_for_render[v_frame_idx]

                            v_dynamic_points, v_dynamic_rgbs, v_dynamic_opacitys, v_dynamic_scales, v_dynamic_rotations = [], [], [], [], []
                            for i in range(v_dy_map.shape[1]):
                                v_point_map_i = v_point_map[:, i]
                                v_bg_mask_i = v_bg_mask_for_render[:, i]
                                v_dynamic_point = v_point_map_i[v_bg_mask_i].reshape(-1, 3)
                                v_dynamic_rgb, v_dynamic_opacity, v_dynamic_scale, v_dynamic_rotation = get_split_gs(v_gs_map[:, i], v_bg_mask_i)
                                v_gs_dynamic_list_i = v_dy_map[:, i][v_bg_mask_i].sigmoid()
                                v_dynamic_opacity = v_dynamic_opacity * v_gs_dynamic_list_i
                                v_dynamic_points.append(v_dynamic_point)
                                v_dynamic_rgbs.append(v_dynamic_rgb)
                                v_dynamic_opacitys.append(v_dynamic_opacity)
                                v_dynamic_scales.append(v_dynamic_scale)
                                v_dynamic_rotations.append(v_dynamic_rotation)

                            v_chunked_renders, v_chunked_alphas = [], []
                            v_S = v_extrinsic.shape[0]
                            for idx in range(v_S):
                                v_t0 = v_timestamps_for_render[idx]
                                v_static_opacity_ = alpha_t(v_gs_timestamps, v_t0, v_static_opacity, gamma0 = v_static_gs_conf)
                                v_static_gs_list = [v_static_points, v_static_rgbs, v_static_opacity_, v_static_scales, v_static_rotations]
                                if v_dynamic_points:
                                    v_world_points, v_rgbs, v_opacity, v_scales, v_rotation = concat_list(
                                        v_static_gs_list,
                                        [v_dynamic_points[idx], v_dynamic_rgbs[idx], v_dynamic_opacitys[idx], v_dynamic_scales[idx], v_dynamic_rotations[idx]]
                                    )
                                v_renders_chunk, v_alphas_chunk, _ = rasterization(
                                    means=v_world_points,
                                    quats=v_rotation,
                                    scales=v_scales,
                                    opacities=v_opacity,
                                    colors=v_rgbs,
                                    viewmats=v_extrinsic[idx][None],
                                    Ks=v_intrinsic[idx][None],
                                    width=v_W,
                                    height=v_H,
                                )
                                v_chunked_renders.append(v_renders_chunk)
                                v_chunked_alphas.append(v_alphas_chunk)

                            v_renders = torch.cat(v_chunked_renders, dim=0)
                            v_alphas = torch.cat(v_chunked_alphas, dim=0)
                            v_bg_render = v_predictions["bg_render"]
                            v_renders = v_alphas * v_renders + (1 - v_alphas) * v_bg_render

                            v_rendered_image = v_renders.permute(0, 3, 1, 2)
                            v_target_image = v_images_for_render[0]
                            # pred_images = v_rendered_image[[0, 2]] if v_rendered_image.shape[0] == 4 else v_rendered_image

                            # Also render Edited variant: remove all dynamic objects (static-only)
                            v_chunked_renders_nody, v_chunked_alphas_nody = [], []
                            v_chunked_renders_dy, v_chunked_alphas_dy = [], []
                            for idx in range(v_S):
                                v_t0 = v_timestamps_for_render[idx]
                                v_static_opacity_ = alpha_t(v_gs_timestamps, v_t0, v_static_opacity, gamma0 = v_static_gs_conf)
                                v_world_points_s, v_rgbs_s, v_opacity_s, v_scales_s, v_rotation_s = v_static_points, v_static_rgbs, v_static_opacity_, v_static_scales, v_static_rotations
                                v_renders_chunk_s, v_alphas_chunk_s, _ = rasterization(
                                    means=v_world_points_s,
                                    quats=v_rotation_s,
                                    scales=v_scales_s,
                                    opacities=v_opacity_s,
                                    colors=v_rgbs_s,
                                    viewmats=v_extrinsic[idx][None],
                                    Ks=v_intrinsic[idx][None],
                                    width=v_W,
                                    height=v_H,
                                )
                                v_chunked_renders_nody.append(v_renders_chunk_s)
                                v_chunked_alphas_nody.append(v_alphas_chunk_s)

                                has_dynamic = idx < len(v_dynamic_points) and v_dynamic_points[idx].numel() > 0
                                if has_dynamic:
                                    v_renders_chunk_d, v_alphas_chunk_d, _ = rasterization(
                                        means=v_dynamic_points[idx],
                                        quats=v_dynamic_rotations[idx],
                                        scales=v_dynamic_scales[idx],
                                        opacities=v_dynamic_opacitys[idx],
                                        colors=v_dynamic_rgbs[idx],
                                        viewmats=v_extrinsic[idx][None],
                                        Ks=v_intrinsic[idx][None],
                                        width=v_W,
                                        height=v_H,
                                    )
                                else:
                                    v_renders_chunk_d = torch.zeros((1, v_H, v_W, 3), device=device, dtype=v_point_map.dtype)
                                    v_alphas_chunk_d = torch.zeros((1, v_H, v_W, 1), device=device, dtype=v_point_map.dtype)
                                v_chunked_renders_dy.append(v_renders_chunk_d)
                                v_chunked_alphas_dy.append(v_alphas_chunk_d)
                            v_renders_nody = torch.cat(v_chunked_renders_nody, dim=0)
                            v_alphas_nody = torch.cat(v_chunked_alphas_nody, dim=0)
                            v_renders_nody = v_alphas_nody * v_renders_nody + (1 - v_alphas_nody) * v_bg_render
                            v_rendered_image_nody = v_renders_nody.permute(0, 3, 1, 2)
                            v_renders_dy = torch.cat(v_chunked_renders_dy, dim=0)
                            v_alphas_dy = torch.cat(v_chunked_alphas_dy, dim=0)
                            v_white_bg = torch.ones_like(v_renders_dy)
                            v_renders_dy = v_alphas_dy * v_renders_dy + (1 - v_alphas_dy) * v_white_bg
                            v_rendered_image_dy = v_renders_dy.permute(0, 3, 1, 2)

                        # Visualization Saving Logic
                        all_frames_combined = []

                        # Helper to apply colormap
                        def apply_colormap(tensor_2d, cmap_name='plasma'):
                            """Converts [H, W] tensor in [0, 1] to [3, H, W] RGB using matplotlib colormap."""
                            np_map = tensor_2d.detach().cpu().numpy()
                            np_map = np.clip(np_map, 0, 1)
                            cmap = plt.get_cmap(cmap_name)
                            rgb_map = cmap(np_map)[..., :3] # [H, W, 3]
                            rgb_map = torch.from_numpy(rgb_map).permute(2, 0, 1).float() # [3, H, W]
                            return rgb_map

                        for frame_idx in range(v_rendered_image.shape[0]):
                            rendered = v_rendered_image[frame_idx].detach().cpu().clamp(0, 1)
                            rendered_edit = v_rendered_image_nody[frame_idx].detach().cpu().clamp(0, 1)
                            rendered_dynamic = v_rendered_image_dy[frame_idx].detach().cpu().clamp(0, 1)
                            target = v_target_image[frame_idx].detach().cpu().clamp(0, 1)

                            dy_map_sigmoid = torch.sigmoid(v_dy_map[0, frame_idx]).detach().cpu()  # shape: (H, W)
                            dy_map_rgb = apply_colormap(dy_map_sigmoid, 'hot')  # [3, H, W]

                            sem_rgb = v_alphas[frame_idx, ..., 0].unsqueeze(0).repeat(3, 1, 1).cpu()  # [3, H, W]

                            # Depth visualization
                            if True: # use_depth
                                depth_frame = v_depth_map[frame_idx].detach().cpu()
                            else:
                                depth_frame = torch.zeros_like(dy_map_sigmoid)

                            if depth_frame.dim() == 3:
                                if depth_frame.shape[0] == 1:
                                    depth_frame = depth_frame.squeeze(0)
                                elif depth_frame.shape[-1] == 1:
                                    depth_frame = depth_frame.squeeze(-1)

                            depth_max = depth_frame.max()
                            if depth_max > 0:
                                depth_norm = depth_frame / depth_max
                            else:
                                depth_norm = depth_frame

                            depth_rgb = apply_colormap(depth_norm, 'turbo')

                            if depth_rgb.shape[-2:] != target.shape[-2:]:
                                depth_rgb = F.interpolate(depth_rgb.unsqueeze(0), size=target.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)

                            # FRUC temporal-spatial context visualization
                            fruc_vis = []
                            if (args.dataset_type == 'v2xreal_coop') and "u_ego" in v_predictions:
                                patch_start_idx = 5 # Matches patch_start_idx default

                                # Since v_predictions["u_ego"] is output per agent [B, K, P, 1] (where P includes special tokens)
                                # We need to extract only the spatial tokens
                                num_spatial = v_predictions["u_ego"].shape[2] - patch_start_idx

                                # Use the target image's aspect ratio to determine H_feat and W_feat
                                # rather than assuming it's a square image (which causes stripe artifacts if not square)
                                _, H_target, W_target = target.shape
                                aspect_ratio = W_target / H_target
                                H_feat = int((num_spatial / aspect_ratio) ** 0.5)
                                W_feat = int(H_feat * aspect_ratio)

                                # Make sure H_feat * W_feat matches num_spatial, if rounding caused a mismatch
                                # Try to adjust W_feat to make it match exactly
                                if H_feat * W_feat != num_spatial:
                                    W_feat = num_spatial // H_feat

                                # u_ego has shape [B, 2, P, 1] for the ego/collaborator at t1.
                                # Agent index 0 is ego; agent index 1 is the collaborator.
                                # Frame indices follow [Ego_t0, Collab_t0, Ego_t1, Collab_t1].
                                # Select the corresponding agent index for the uncertainty output.
                                agent_idx = 0 if frame_idx in [0, 2] else 1

                                # Skip the special tokens (e.g., first 5 tokens) to reshape the spatial map
                                v_u_ego_spatial = v_predictions["u_ego"][0, agent_idx, patch_start_idx:].view(H_feat, W_feat).detach().cpu()

                                # Apply colormaps (u_ego now represents occlusion prior m_occ)
                                u_ego_rgb = apply_colormap(v_u_ego_spatial, 'viridis')

                                if u_ego_rgb.shape[-2:] != target.shape[-2:]:
                                    u_ego_rgb = F.interpolate(u_ego_rgb.unsqueeze(0), size=target.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)

                                fruc_vis = [u_ego_rgb]

                            # Columns: GT | Rendered | Rendered(Static) | Rendered(Dynamic) | Dynamic Map | Semantics | Depth | Occlusion Prior
                            combined = torch.cat([target, rendered, rendered_edit, rendered_dynamic, dy_map_rgb, sem_rgb, depth_rgb] + fruc_vis, dim=-1)
                            all_frames_combined.append(combined)

                        final_combined = torch.cat(all_frames_combined, dim=1)
                        T.ToPILImage()(final_combined).save(os.path.join(args.log_dir, "images", f"step_{global_step:05d}_val_{sample_idx}_merged.jpg"))

                model.train() # Revert to train mode

if __name__ == "__main__":
    args = parse_args()
    main(args)
