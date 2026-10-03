#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.v2xreal_coop_dataset import V2XRealCoopDataset
from fruc.models.vggt import VGGT
from fruc.utils.checkpoint import extract_state_dict
from fruc.utils.geometry import unproject_depth_map_to_point_map
from fruc.utils.gs import get_split_gs
from fruc.utils.pose_enc import pose_encoding_to_extri_intri

try:
    from gsplat.rendering import rasterization
except ImportError:
    rasterization = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render FRUC cooperative novel views by moving the ego t0 camera."
    )
    parser.add_argument("--image_dir", type=str, required=True, help="Path to V2X-Real root or split directory")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to FRUC checkpoint")
    parser.add_argument("--output_dir", type=str, default="output/v2xreal_coop_novel_view")
    parser.add_argument("--scene_names", type=str, nargs="+", default=None, help="Optional scene filter")
    parser.add_argument("--scene_id", type=int, default=None, help="Optional scene index filter (0, 1, 2, ...)")
    parser.add_argument("--sample_id", type=int, default=None, help="Optional sample index after scene filtering")
    parser.add_argument("--start_frame", type=int, default=None, help="Optional frame index filter")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit_samples", type=int, default=-1)
    parser.add_argument("--cam0_only", action="store_true", help="Use only cam0 pair as in default FRUC eval")
    parser.add_argument("--disable_fruc", action="store_true", help="Disable cooperative FRUC fusion")
    parser.add_argument(
        "--edit_scope",
        type=str,
        default="ego_only",
        choices=["ego_only", "all_t0"],
        help="Which dynamic Gaussian subset to edit: only ego t0 or all t0 cooperative dynamics",
    )
    parser.add_argument("--remove_all_dynamic", action="store_true", help="Remove all selected dynamic Gaussians in the chosen edit scope")
    parser.add_argument(
        "--remove_dynamic_box",
        type=float,
        nargs=6,
        default=None,
        help="Remove dynamic Gaussians inside an axis-aligned box in ego t0 coordinates: x y z l w h",
    )
    parser.add_argument(
        "--add_source_box",
        type=float,
        nargs=6,
        default=None,
        help="Select dynamic Gaussians inside an axis-aligned source box to duplicate: x y z l w h",
    )
    parser.add_argument(
        "--add_translation",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Translation applied to duplicated dynamic Gaussians in ego t0 coordinates",
    )
    parser.add_argument(
        "--add_target_center",
        type=float,
        nargs=3,
        default=None,
        help="Optional target center for duplicated object centroid; overrides add_translation when set",
    )
    parser.add_argument("--add_repeat", type=int, default=1, help="How many translated copies to add")
    parser.add_argument(
        "--direction_mode",
        type=str,
        default="vertical",
        choices=["vertical", "trajectory", "lateral_left", "lateral_right", "custom"],
        help="How to choose the novel-view translation direction in ego t0 coordinates",
    )
    parser.add_argument(
        "--move_direction",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 1.0],
        help="Custom translation direction in ego t0 coordinates, used when direction_mode=custom",
    )
    parser.add_argument(
        "--offset_scales",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0],
        help="Translation magnitude multipliers relative to the ego t0->t1 trajectory length",
    )
    return parser.parse_args()


def infer_split_name(image_dir: str) -> str:
    base = os.path.basename(os.path.normpath(image_dir))
    if base in {"train", "val", "test"}:
        return base
    return "val"


def build_dataset_with_fallback(args):
    split = infer_split_name(args.image_dir)
    dataset = V2XRealCoopDataset(
        args.image_dir,
        split=split,
        scene_names=args.scene_names,
        start_frame=args.start_frame,
        for_vis=False,
        cam0_only=args.cam0_only,
    )
    used_start_frame = args.start_frame
    if len(dataset) == 0 and args.start_frame is not None:
        print(
            f"No cooperative samples found for start_frame={args.start_frame}. "
            "Retrying without start_frame filter...",
            flush=True,
        )
        dataset = V2XRealCoopDataset(
            args.image_dir,
            split=split,
            scene_names=args.scene_names,
            start_frame=None,
            for_vis=False,
            cam0_only=args.cam0_only,
        )
        used_start_frame = None
    return dataset, used_start_frame


def apply_scene_and_sample_filters(dataset, args):
    if args.scene_id is not None:
        scene_order = list(dict.fromkeys(sample["scene"] for sample in dataset.samples))
        if args.scene_id < 0 or args.scene_id >= len(scene_order):
            raise ValueError(f"scene_id={args.scene_id} is out of range. Available scene indices: 0 to {len(scene_order) - 1}")
        selected_scene = scene_order[args.scene_id]
        dataset.samples = [sample for sample in dataset.samples if sample["scene"] == selected_scene]
        print(f"Resolved scene_id={args.scene_id} to scene '{selected_scene}', keeping {len(dataset.samples)} samples.")

    if args.sample_id is not None:
        if args.sample_id < 0 or args.sample_id >= len(dataset.samples):
            raise ValueError(f"sample_id={args.sample_id} is out of range for {len(dataset.samples)} filtered samples.")
        selected_sample = dataset.samples[args.sample_id]
        dataset.samples = [selected_sample]
        print(
            f"Resolved sample_id={args.sample_id} to frame_idx={selected_sample['frame_idx']} "
            f"in scene '{selected_sample['scene']}'."
        )

    return dataset


def alpha_t(t, t0, alpha, gamma0=1, gamma1=0.1):
    sigma = torch.log(torch.tensor(gamma1, device=gamma0.device)) / ((gamma0) ** 2 + 1e-6)
    conf = torch.exp(sigma * (t0 - t) ** 2)
    return (alpha * conf).float()


def normalize_vec(vec: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(vec)
    if float(norm) < 1e-8:
        return fallback
    return vec / norm


def get_camera_center(extrinsic_4x4: torch.Tensor) -> torch.Tensor:
    return torch.inverse(extrinsic_4x4)[:3, 3]


def direction_from_mode(
    direction_mode: str,
    trajectory_vec: torch.Tensor,
    custom_direction: Sequence[float],
    device: torch.device,
) -> torch.Tensor:
    up = torch.tensor([0.0, 0.0, 1.0], device=device)
    fallback = torch.tensor([1.0, 0.0, 0.0], device=device)
    traj_unit = normalize_vec(trajectory_vec, fallback)

    if direction_mode == "vertical":
        return up
    if direction_mode == "trajectory":
        return traj_unit
    if direction_mode == "lateral_left":
        return normalize_vec(torch.cross(up, traj_unit), fallback)
    if direction_mode == "lateral_right":
        return normalize_vec(torch.cross(traj_unit, up), fallback)
    custom = torch.tensor(custom_direction, dtype=torch.float32, device=device)
    return normalize_vec(custom, up)


def to_uint8_image(rgb: torch.Tensor) -> np.ndarray:
    image = rgb.detach().cpu().clamp(0, 1)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = image.permute(1, 2, 0)
    image = image.numpy()
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    return (image * 255.0).astype(np.uint8)


def scale_tag(scale: float) -> str:
    return str(scale).replace("-", "neg").replace(".", "p")


def save_side_by_side_comparison(original_rgb: torch.Tensor, edited_rgb: torch.Tensor, output_path: str) -> None:
    original_img = to_uint8_image(original_rgb)
    edited_img = to_uint8_image(edited_rgb)
    if original_img.ndim == 2:
        original_img = np.repeat(original_img[..., None], 3, axis=-1)
    if edited_img.ndim == 2:
        edited_img = np.repeat(edited_img[..., None], 3, axis=-1)
    separator = np.full((original_img.shape[0], 12, 3), 255, dtype=np.uint8)
    comparison = np.concatenate([original_img, separator, edited_img], axis=1)
    Image.fromarray(comparison).save(output_path)


def merge_dynamic_gaussians(
    dynamic_points_list: Sequence[Optional[torch.Tensor]],
    dynamic_rgbs_list: Sequence[Optional[torch.Tensor]],
    dynamic_opacitys_list: Sequence[Optional[torch.Tensor]],
    dynamic_scales_list: Sequence[Optional[torch.Tensor]],
    dynamic_rotations_list: Sequence[Optional[torch.Tensor]],
    selected_slots: Sequence[int],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    merged_points = []
    merged_rgbs = []
    merged_opacitys = []
    merged_scales = []
    merged_rotations = []

    for slot in selected_slots:
        if dynamic_points_list[slot] is None:
            continue
        merged_points.append(dynamic_points_list[slot])
        merged_rgbs.append(dynamic_rgbs_list[slot])
        merged_opacitys.append(dynamic_opacitys_list[slot])
        merged_scales.append(dynamic_scales_list[slot])
        merged_rotations.append(dynamic_rotations_list[slot])

    if not merged_points:
        return None, None, None, None, None

    return (
        torch.cat(merged_points, dim=0),
        torch.cat(merged_rgbs, dim=0),
        torch.cat(merged_opacitys, dim=0),
        torch.cat(merged_scales, dim=0),
        torch.cat(merged_rotations, dim=0),
    )


def gaussian_set_is_empty(gaussians):
    return gaussians[0] is None or gaussians[0].numel() == 0


def concat_gaussian_sets(*gaussian_sets):
    valid_sets = [gs for gs in gaussian_sets if gs is not None and not gaussian_set_is_empty(gs)]
    if not valid_sets:
        return (None, None, None, None, None)
    return tuple(torch.cat([gs[i] for gs in valid_sets], dim=0) for i in range(5))


def split_gaussians_by_box(gaussians, box):
    if gaussians is None or gaussian_set_is_empty(gaussians) or box is None:
        empty = (None, None, None, None, None)
        return gaussians, empty

    points, rgbs, opacities, scales, rotations = gaussians
    x, y, z, l, w, h = box
    mask_x = (points[:, 0] >= x - l / 2) & (points[:, 0] <= x + l / 2)
    mask_y = (points[:, 1] >= y - w / 2) & (points[:, 1] <= y + w / 2)
    mask_z = (points[:, 2] >= z - h / 2) & (points[:, 2] <= z + h / 2)
    inside = mask_x & mask_y & mask_z
    outside = ~inside

    def gather(mask):
        if int(mask.sum()) == 0:
            return (None, None, None, None, None)
        return (
            points[mask],
            rgbs[mask],
            opacities[mask],
            scales[mask],
            rotations[mask],
        )

    return gather(outside), gather(inside)


def translate_gaussian_set(gaussians, translation):
    if gaussians is None or gaussian_set_is_empty(gaussians):
        return (None, None, None, None, None)
    translated_points = gaussians[0] + translation.view(1, 3)
    return (translated_points, gaussians[1], gaussians[2], gaussians[3], gaussians[4])


def duplicate_gaussian_set(gaussians, translation, repeat_count):
    if gaussians is None or gaussian_set_is_empty(gaussians) or repeat_count <= 0:
        return (None, None, None, None, None)

    copies = []
    for repeat_idx in range(repeat_count):
        delta = translation * float(repeat_idx + 1)
        copies.append(translate_gaussian_set(gaussians, delta))
    return concat_gaussian_sets(*copies)


def load_model(ckpt_path: str, device: torch.device) -> VGGT:
    model = VGGT().to(device)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    required = ("gs_head.", "instance_head.", "prior_denoising.", "sky_model.")
    missing = [key for key in incompatible.missing_keys if key.startswith(required)]
    if missing:
        raise ValueError(f"Expected trained FRUC weights; {len(missing)} reconstruction parameters are missing.")
    model.eval()
    return model


def build_bg_render(
    model: VGGT,
    images_input: torch.Tensor,
    source_extrinsics: torch.Tensor,
    source_intrinsics: torch.Tensor,
    target_extrinsic: torch.Tensor,
    target_intrinsic: torch.Tensor,
) -> torch.Tensor:
    if not hasattr(model, "sky_model") or model.sky_model is None:
        _, _, h, w = images_input.shape[1:]
        return torch.zeros((1, h, w, 3), device=images_input.device)

    bg_render = model.sky_model.forward_with_new_pose(
        images=images_input,
        extrinsics=source_extrinsics,
        intrinsics=source_intrinsics,
        extrinsics_=target_extrinsic[None],
        intrinsics_=target_intrinsic[None],
    )
    if bg_render.shape[-1] != 3:
        bg_render = bg_render.permute(0, 2, 3, 1)
    return bg_render


def render_novel_view(
    model: VGGT,
    world_points: torch.Tensor,
    rgbs: torch.Tensor,
    opacity: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    target_extrinsic: torch.Tensor,
    target_intrinsic: torch.Tensor,
    images_input: torch.Tensor,
    source_extrinsics: torch.Tensor,
    source_intrinsics: torch.Tensor,
    image_hw: Tuple[int, int],
    bg_color: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    h, w = image_hw
    if rasterization is None:
        raise RuntimeError("gsplat is required for rendering novel views.")

    render_rgb, alphas, _ = rasterization(
        means=world_points,
        quats=rotations,
        scales=scales,
        opacities=opacity,
        colors=rgbs,
        viewmats=target_extrinsic[None],
        Ks=target_intrinsic[None],
        width=w,
        height=h,
    )
    if bg_color is not None:
        bg_render = bg_color.view(1, 1, 1, 3).expand(1, h, w, 3).to(render_rgb.device)
    else:
        bg_render = build_bg_render(
            model=model,
            images_input=images_input,
            source_extrinsics=source_extrinsics,
            source_intrinsics=source_intrinsics,
            target_extrinsic=target_extrinsic,
            target_intrinsic=target_intrinsic,
        )
    return alphas[0] * render_rgb[0] + (1.0 - alphas[0]) * bg_render[0]


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    if rasterization is None:
        raise RuntimeError("gsplat is not available in the current environment.")

    os.makedirs(args.output_dir, exist_ok=True)

    dataset, used_start_frame = build_dataset_with_fallback(args)
    dataset = apply_scene_and_sample_filters(dataset, args)
    if len(dataset) == 0:
        raise RuntimeError(
            "No cooperative samples were found for the requested scene/filter settings."
        )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = load_model(args.ckpt_path, device)

    print(f"Loaded {len(dataset)} cooperative samples from {args.image_dir}")
    print(f"Effective start_frame: {used_start_frame if used_start_frame is not None else '<auto>'}")
    print(f"Edit scope: {args.edit_scope}")
    print(f"Novel-view direction mode: {args.direction_mode}")
    print(f"Offset scales: {args.offset_scales}")

    with torch.no_grad():
        for sample_idx, batch in enumerate(tqdm(dataloader)):
            if args.limit_samples > 0 and sample_idx >= args.limit_samples:
                break

            sample_meta = dataset.samples[sample_idx]
            scene_name = batch["scene_name"][0]
            frame_idx = sample_meta["frame_idx"]
            scene_dir = os.path.join(args.output_dir, scene_name, f"sample_{sample_idx:04d}_frame_{frame_idx:03d}")
            os.makedirs(scene_dir, exist_ok=True)

            images = batch["images"].to(device)
            masks = batch["masks"].to(device)
            timestamps = batch["timestamps"][0].to(device)

            apply_fruc = not args.disable_fruc
            predictions = model(images, apply_fruc=apply_fruc)

            _, _, _, h, w = images.shape
            extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], (h, w))
            extrinsic = extrinsics[0]
            intrinsic = intrinsics[0]

            bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)
            extrinsic_4x4 = torch.cat([extrinsic, bottom], dim=1)

            depth_map = predictions["depth"][0]
            point_map_np = unproject_depth_map_to_point_map(depth_map, extrinsics[0], intrinsics[0])
            point_map = torch.from_numpy(point_map_np).to(device).float()

            gs_map = predictions["gs_map"][0]
            gs_conf = predictions["gs_conf"][0]
            dy_map = predictions["dynamic_conf"].squeeze(-1)[0]

            sky_mask = masks[0].permute(0, 2, 3, 1)
            bg_mask = (sky_mask[..., 0] == 0)
            static_mask = torch.ones_like(bg_mask)

            static_points = point_map[static_mask].reshape(-1, 3)
            gs_dynamic_list = dy_map[static_mask].sigmoid()
            static_rgbs, static_opacity, static_scales, static_rotations = get_split_gs(gs_map, static_mask)
            static_opacity = static_opacity * (1 - gs_dynamic_list)
            static_gs_conf = gs_conf[static_mask]
            frame_indices = torch.nonzero(static_mask, as_tuple=False)[:, 0]
            gs_timestamps_static = timestamps[frame_indices]

            dynamic_points_list = []
            dynamic_rgbs_list = []
            dynamic_opacitys_list = []
            dynamic_scales_list = []
            dynamic_rotations_list = []
            for frame_slot in range(extrinsic_4x4.shape[0]):
                bg_mask_slot = bg_mask[frame_slot]
                if int(bg_mask_slot.sum()) == 0:
                    dynamic_points_list.append(None)
                    dynamic_rgbs_list.append(None)
                    dynamic_opacitys_list.append(None)
                    dynamic_scales_list.append(None)
                    dynamic_rotations_list.append(None)
                    continue
                dyn_points = point_map[frame_slot][bg_mask_slot].reshape(-1, 3)
                dyn_rgb, dyn_opacity, dyn_scale, dyn_rotation = get_split_gs(gs_map[frame_slot], bg_mask_slot)
                dyn_conf = dy_map[frame_slot][bg_mask_slot].sigmoid()
                dyn_opacity = dyn_opacity * dyn_conf
                dynamic_points_list.append(dyn_points)
                dynamic_rgbs_list.append(dyn_rgb)
                dynamic_opacitys_list.append(dyn_opacity)
                dynamic_scales_list.append(dyn_scale)
                dynamic_rotations_list.append(dyn_rotation)

            ref_idx = 0
            ego_t1_idx = extrinsic_4x4.shape[0] // 2 if extrinsic_4x4.shape[0] >= 2 else 0
            t0_timestamp = timestamps[ref_idx]
            t0_slots = [idx for idx, ts in enumerate(timestamps.tolist()) if abs(ts - float(t0_timestamp.item())) < 1e-6]
            ego_dyn = merge_dynamic_gaussians(
                dynamic_points_list,
                dynamic_rgbs_list,
                dynamic_opacitys_list,
                dynamic_scales_list,
                dynamic_rotations_list,
                [ref_idx],
            )
            merged_dyn = merge_dynamic_gaussians(
                dynamic_points_list,
                dynamic_rgbs_list,
                dynamic_opacitys_list,
                dynamic_scales_list,
                dynamic_rotations_list,
                t0_slots,
            )
            collab_t0_slots = [idx for idx in t0_slots if idx != ref_idx]
            collab_dyn = merge_dynamic_gaussians(
                dynamic_points_list,
                dynamic_rgbs_list,
                dynamic_opacitys_list,
                dynamic_scales_list,
                dynamic_rotations_list,
                collab_t0_slots,
            )

            ref_extrinsic = extrinsic_4x4[ref_idx]
            ref_intrinsic = intrinsic[ref_idx]
            ref_c2w = torch.inverse(ref_extrinsic)

            center_t0 = get_camera_center(extrinsic_4x4[ref_idx])
            center_t1 = get_camera_center(extrinsic_4x4[ego_t1_idx])
            trajectory_vec = center_t1 - center_t0
            trajectory_scale = float(torch.linalg.norm(trajectory_vec).item())
            if trajectory_scale < 1e-6:
                trajectory_scale = 1.0

            move_dir = direction_from_mode(
                args.direction_mode,
                trajectory_vec,
                args.move_direction,
                device,
            )

            if args.edit_scope == "ego_only":
                editable_dyn = ego_dyn
                non_editable_dyn = collab_dyn
            else:
                editable_dyn = merged_dyn
                non_editable_dyn = (None, None, None, None, None)

            edited_editable_dyn = editable_dyn
            removed_dyn = (None, None, None, None, None)
            if args.remove_all_dynamic:
                removed_dyn = editable_dyn
                edited_editable_dyn = (None, None, None, None, None)
            elif args.remove_dynamic_box is not None:
                edited_editable_dyn, removed_dyn = split_gaussians_by_box(editable_dyn, args.remove_dynamic_box)

            add_source_dyn = editable_dyn
            selected_for_copy = (None, None, None, None, None)
            added_dyn = (None, None, None, None, None)
            if args.add_source_box is not None:
                _, selected_for_copy = split_gaussians_by_box(add_source_dyn, args.add_source_box)
                if not gaussian_set_is_empty(selected_for_copy):
                    if args.add_target_center is not None:
                        source_center = selected_for_copy[0].mean(dim=0)
                        target_center = torch.tensor(args.add_target_center, device=device, dtype=source_center.dtype)
                        add_translation = target_center - source_center
                    else:
                        add_translation = torch.tensor(args.add_translation, device=device, dtype=selected_for_copy[0].dtype)
                    added_dyn = duplicate_gaussian_set(selected_for_copy, add_translation, args.add_repeat)

            final_dyn = concat_gaussian_sets(non_editable_dyn, edited_editable_dyn, added_dyn)

            static_opacity_t0 = alpha_t(gs_timestamps_static, t0_timestamp, static_opacity, gamma0=static_gs_conf)
            if final_dyn[0] is not None:
                world_points = torch.cat([static_points, final_dyn[0]], dim=0)
                rgbs = torch.cat([static_rgbs, final_dyn[1]], dim=0)
                opacity = torch.cat([static_opacity_t0, final_dyn[2]], dim=0)
                scales = torch.cat([static_scales, final_dyn[3]], dim=0)
                rotations = torch.cat([static_rotations, final_dyn[4]], dim=0)
            else:
                world_points = static_points
                rgbs = static_rgbs
                opacity = static_opacity_t0
                scales = static_scales
                rotations = static_rotations

            if merged_dyn[0] is not None:
                original_world_points = torch.cat([static_points, merged_dyn[0]], dim=0)
                original_rgbs = torch.cat([static_rgbs, merged_dyn[1]], dim=0)
                original_opacity = torch.cat([static_opacity_t0, merged_dyn[2]], dim=0)
                original_scales = torch.cat([static_scales, merged_dyn[3]], dim=0)
                original_rotations = torch.cat([static_rotations, merged_dyn[4]], dim=0)
            else:
                original_world_points = static_points
                original_rgbs = static_rgbs
                original_opacity = static_opacity_t0
                original_scales = static_scales
                original_rotations = static_rotations

            intrinsic_source = intrinsic
            images_input = images

            Image.fromarray(to_uint8_image(images[0, ref_idx])).save(
                os.path.join(scene_dir, "ego_t0_input.png")
            )

            metadata = {
                "scene_name": scene_name,
                "sample_index": sample_idx,
                "frame_idx": int(frame_idx),
                "trajectory_scale": trajectory_scale,
                "direction_mode": args.direction_mode,
                "move_direction_unit": move_dir.detach().cpu().tolist(),
                "offset_scales": list(args.offset_scales),
                "t0_slots": t0_slots,
                "apply_fruc": bool(apply_fruc),
                "edit_scope": args.edit_scope,
                "remove_all_dynamic": bool(args.remove_all_dynamic),
                "remove_dynamic_box": args.remove_dynamic_box,
                "add_source_box": args.add_source_box,
                "add_translation": list(args.add_translation),
                "add_target_center": args.add_target_center,
                "add_repeat": int(args.add_repeat),
                "original_dynamic_count": int(0 if gaussian_set_is_empty(merged_dyn) else merged_dyn[0].shape[0]),
                "removed_dynamic_count": int(0 if gaussian_set_is_empty(removed_dyn) else removed_dyn[0].shape[0]),
                "added_dynamic_count": int(0 if gaussian_set_is_empty(added_dyn) else added_dyn[0].shape[0]),
                "final_dynamic_count": int(0 if gaussian_set_is_empty(final_dyn) else final_dyn[0].shape[0]),
            }

            for offset_scale in args.offset_scales:
                delta = move_dir * (trajectory_scale * float(offset_scale))
                target_c2w = ref_c2w.clone()
                target_c2w[:3, 3] = ref_c2w[:3, 3] + delta
                target_extrinsic = torch.inverse(target_c2w)

                rendered_original = render_novel_view(
                    model=model,
                    world_points=original_world_points,
                    rgbs=original_rgbs,
                    opacity=original_opacity,
                    scales=original_scales,
                    rotations=original_rotations,
                    target_extrinsic=target_extrinsic,
                    target_intrinsic=ref_intrinsic,
                    images_input=images_input,
                    source_extrinsics=extrinsic_4x4,
                    source_intrinsics=intrinsic_source,
                    image_hw=(h, w),
                )
                rendered = render_novel_view(
                    model=model,
                    world_points=world_points,
                    rgbs=rgbs,
                    opacity=opacity,
                    scales=scales,
                    rotations=rotations,
                    target_extrinsic=target_extrinsic,
                    target_intrinsic=ref_intrinsic,
                    images_input=images_input,
                    source_extrinsics=extrinsic_4x4,
                    source_intrinsics=intrinsic_source,
                    image_hw=(h, w),
                )
                out_name = f"edited_offset_{scale_tag(float(offset_scale))}.png"
                out_orig_name = f"original_offset_{scale_tag(float(offset_scale))}.png"
                out_compare_name = f"compare_offset_{scale_tag(float(offset_scale))}.png"
                Image.fromarray(to_uint8_image(rendered_original)).save(os.path.join(scene_dir, out_orig_name))
                Image.fromarray(to_uint8_image(rendered)).save(os.path.join(scene_dir, out_name))
                save_side_by_side_comparison(
                    rendered_original,
                    rendered,
                    os.path.join(scene_dir, out_compare_name),
                )
                metadata[out_name] = {
                    "offset_scale": float(offset_scale),
                    "offset_world": delta.detach().cpu().tolist(),
                    "target_camera_center": target_c2w[:3, 3].detach().cpu().tolist(),
                    "comparison_image": out_compare_name,
                }

            with open(os.path.join(scene_dir, "novel_view_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

    print(f"Done. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main(parse_args())
