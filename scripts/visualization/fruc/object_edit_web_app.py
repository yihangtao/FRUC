#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
from uuid import uuid4

import torch
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.visualization.fruc.demo_v2xreal_coop_novel_view import (
    alpha_t,
    concat_gaussian_sets,
    direction_from_mode,
    duplicate_gaussian_set,
    gaussian_set_is_empty,
    get_camera_center,
    infer_split_name,
    load_model,
    merge_dynamic_gaussians,
    render_novel_view,
    save_side_by_side_comparison,
    scale_tag,
    to_uint8_image,
)
from datasets.v2xreal_coop_dataset import V2XRealCoopDataset
from fruc.utils.geometry import unproject_depth_map_to_point_map
from fruc.utils.gs import get_split_gs
from fruc.utils.pose_enc import pose_encoding_to_extri_intri


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive FRUC object editing web app.")
    parser.add_argument("--image_dir", type=str, default="data/v2xreal/val")
    parser.add_argument("--ckpt_path", type=str, default="pretrained/fruc/model_fruc_v2xreal.pth")
    parser.add_argument("--output_dir", type=str, default="output/fruc_object_edit_web")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5010)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cam0_only", action="store_true", help="Use only ego cam0 with the default collab mapping.")
    parser.add_argument("--disable_fruc", action="store_true")
    parser.add_argument("--dynamic_thresh", type=float, default=0.25)
    return parser.parse_args()


ARGS = parse_args()
DEVICE = torch.device(f"cuda:{ARGS.gpu}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(DEVICE)

app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "templates"),
)

_MODEL_CACHE = {}
_DATASET_CACHE = {}
_SESSION_CACHE = {}


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def get_dataset(image_dir: str, cam0_only: bool):
    image_dir = resolve_path(image_dir)
    cache_key = (image_dir, cam0_only)
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]
    split = infer_split_name(image_dir)
    dataset = V2XRealCoopDataset(
        image_dir,
        split=split,
        for_vis=False,
        cam0_only=cam0_only,
    )
    _DATASET_CACHE[cache_key] = dataset
    return dataset


def get_scene_samples(image_dir: str, cam0_only: bool):
    dataset = get_dataset(image_dir, cam0_only)
    scene_order = list(dict.fromkeys(sample["scene"] for sample in dataset.samples))
    scene_to_samples = {scene: [] for scene in scene_order}
    for sample_idx, sample in enumerate(dataset.samples):
        scene_to_samples[sample["scene"]].append(
            {
                "sample_id": len(scene_to_samples[sample["scene"]]),
                "global_sample_index": sample_idx,
                "frame_idx": int(sample["frame_idx"]),
                "scene_name": sample["scene"],
            }
        )
    return scene_order, scene_to_samples


def get_model(ckpt_path: str):
    ckpt_path = resolve_path(ckpt_path)
    if ckpt_path not in _MODEL_CACHE:
        _MODEL_CACHE[ckpt_path] = load_model(ckpt_path, DEVICE)
    return _MODEL_CACHE[ckpt_path]


def build_boxes_mask(image_hw, boxes):
    h, w = image_hw
    mask = torch.zeros((h, w), dtype=torch.bool, device=DEVICE)
    for box in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return mask


def gaussian_subset_by_image_boxes(gaussians, bg_mask_slot, dyn_conf_slot, image_boxes):
    if gaussians is None or gaussian_set_is_empty(gaussians):
        empty = (None, None, None, None, None)
        return empty, empty, 0

    image_mask = build_boxes_mask(tuple(bg_mask_slot.shape), image_boxes)
    selected = image_mask[bg_mask_slot]
    if dyn_conf_slot is not None:
        selected = selected & (dyn_conf_slot > ARGS.dynamic_thresh)
    outside = ~selected

    def gather(mask):
        if int(mask.sum()) == 0:
            return (None, None, None, None, None)
        return tuple(t[mask] for t in gaussians)

    return gather(outside), gather(selected), int(selected.sum().item())


def estimate_ground_anchor(points: torch.Tensor) -> torch.Tensor:
    if points is None or points.numel() == 0:
        raise ValueError("Cannot estimate ground anchor from empty points.")
    z_min = points[:, 2].min()
    z_max = points[:, 2].max()
    z_band = torch.clamp((z_max - z_min) * 0.12, min=0.08, max=0.35)
    bottom_mask = points[:, 2] <= (z_min + z_band)
    if int(bottom_mask.sum()) == 0:
        bottom_mask = points[:, 2] == z_min
    anchor = points[bottom_mask].mean(dim=0)
    return anchor


def save_selection_overlay(rgb_tensor: torch.Tensor, boxes, output_path: str):
    image = Image.fromarray(to_uint8_image(rgb_tensor))
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        draw.rectangle([x1, y1, x2, y2], outline=(255, 64, 64), width=3)
    image.save(output_path)


def prepare_sample_state(image_dir: str, ckpt_path: str, scene_id: int, sample_id: int, cam0_only: bool, disable_fruc: bool):
    dataset = get_dataset(image_dir, cam0_only)
    scene_order = list(dict.fromkeys(sample["scene"] for sample in dataset.samples))
    if scene_id < 0 or scene_id >= len(scene_order):
        raise ValueError(f"scene_id={scene_id} out of range [0, {len(scene_order) - 1}]")
    selected_scene = scene_order[scene_id]
    scene_samples = [sample for sample in dataset.samples if sample["scene"] == selected_scene]
    if sample_id < 0 or sample_id >= len(scene_samples):
        raise ValueError(f"sample_id={sample_id} out of range for scene '{selected_scene}'")

    selected_sample = scene_samples[sample_id]
    temp_dataset = V2XRealCoopDataset(
        resolve_path(image_dir),
        split=infer_split_name(resolve_path(image_dir)),
        scene_names=[selected_scene],
        start_frame=selected_sample["frame_idx"],
        for_vis=False,
        cam0_only=cam0_only,
    )
    temp_dataset.samples = [sample for sample in temp_dataset.samples if sample["frame_idx"] == selected_sample["frame_idx"]][:1]
    dataloader = DataLoader(temp_dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))

    model = get_model(ckpt_path)
    with torch.no_grad():
        images = batch["images"].to(DEVICE)
        masks = batch["masks"].to(DEVICE)
        timestamps = batch["timestamps"][0].to(DEVICE)
        predictions = model(images, apply_fruc=not disable_fruc)

        _, _, _, h, w = images.shape
        extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], (h, w))
        extrinsic = extrinsics[0]
        intrinsic = intrinsics[0]
        bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=DEVICE).view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)
        extrinsic_4x4 = torch.cat([extrinsic, bottom], dim=1)

        depth_map = predictions["depth"][0]
        point_map_np = unproject_depth_map_to_point_map(depth_map, extrinsics[0], intrinsics[0])
        point_map = torch.from_numpy(point_map_np).to(DEVICE).float()

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
        dynamic_conf_list = []
        dynamic_sets_by_slot = []
        for frame_slot in range(extrinsic_4x4.shape[0]):
            bg_mask_slot = bg_mask[frame_slot]
            if int(bg_mask_slot.sum()) == 0:
                dynamic_points_list.append(None)
                dynamic_rgbs_list.append(None)
                dynamic_opacitys_list.append(None)
                dynamic_scales_list.append(None)
                dynamic_rotations_list.append(None)
                dynamic_conf_list.append(None)
                dynamic_sets_by_slot.append((None, None, None, None, None))
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
            dynamic_conf_list.append(dyn_conf)
            dynamic_sets_by_slot.append((dyn_points, dyn_rgb, dyn_opacity, dyn_scale, dyn_rotation))

        ref_idx = 0
        ego_t1_idx = extrinsic_4x4.shape[0] // 2 if extrinsic_4x4.shape[0] >= 2 else 0
        t0_timestamp = timestamps[ref_idx]
        t0_slots = [idx for idx, ts in enumerate(timestamps.tolist()) if abs(ts - float(t0_timestamp.item())) < 1e-6]
        collab_t0_slots = [idx for idx in t0_slots if idx != ref_idx]
        merged_dyn = merge_dynamic_gaussians(
            dynamic_points_list,
            dynamic_rgbs_list,
            dynamic_opacitys_list,
            dynamic_scales_list,
            dynamic_rotations_list,
            t0_slots,
        )
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

        static_opacity_t0 = alpha_t(gs_timestamps_static, t0_timestamp, static_opacity, gamma0=static_gs_conf)
        original_world_points = static_points if gaussian_set_is_empty(merged_dyn) else torch.cat([static_points, merged_dyn[0]], dim=0)
        original_rgbs = static_rgbs if gaussian_set_is_empty(merged_dyn) else torch.cat([static_rgbs, merged_dyn[1]], dim=0)
        original_opacity = static_opacity_t0 if gaussian_set_is_empty(merged_dyn) else torch.cat([static_opacity_t0, merged_dyn[2]], dim=0)
        original_scales = static_scales if gaussian_set_is_empty(merged_dyn) else torch.cat([static_scales, merged_dyn[3]], dim=0)
        original_rotations = static_rotations if gaussian_set_is_empty(merged_dyn) else torch.cat([static_rotations, merged_dyn[4]], dim=0)

        black_bg = torch.tensor([0.0, 0.0, 0.0], device=DEVICE)
        rendered_dynamic_t0 = render_novel_view(
            model=model,
            world_points=merged_dyn[0] if not gaussian_set_is_empty(merged_dyn) else torch.zeros((1, 3), device=DEVICE),
            rgbs=merged_dyn[1] if not gaussian_set_is_empty(merged_dyn) else torch.zeros((1, 3), device=DEVICE),
            opacity=merged_dyn[2] if not gaussian_set_is_empty(merged_dyn) else torch.zeros((1, 1), device=DEVICE),
            scales=merged_dyn[3] if not gaussian_set_is_empty(merged_dyn) else torch.zeros((1, 3), device=DEVICE),
            rotations=merged_dyn[4] if not gaussian_set_is_empty(merged_dyn) else torch.zeros((1, 4), device=DEVICE),
            target_extrinsic=ref_c2w.inverse(),
            target_intrinsic=ref_intrinsic,
            images_input=images,
            source_extrinsics=extrinsic_4x4,
            source_intrinsics=intrinsic,
            image_hw=(h, w),
            bg_color=black_bg,
        )

    return {
        "image_dir": resolve_path(image_dir),
        "ckpt_path": resolve_path(ckpt_path),
        "scene_name": selected_scene,
        "frame_idx": int(selected_sample["frame_idx"]),
        "images": images,
        "point_map_ref": point_map[ref_idx],
        "intrinsic": intrinsic,
        "intrinsic_ref": ref_intrinsic,
        "extrinsic_4x4": extrinsic_4x4,
        "ref_c2w": ref_c2w,
        "trajectory_vec": trajectory_vec,
        "trajectory_scale": trajectory_scale,
        "ref_idx": ref_idx,
        "t0_slots": t0_slots,
        "collab_t0_slots": collab_t0_slots,
        "collab_dyn": collab_dyn,
        "ego_dyn": dynamic_sets_by_slot[ref_idx],
        "ego_bg_mask": bg_mask[ref_idx],
        "ego_dyn_conf": dynamic_conf_list[ref_idx],
        "static_points": static_points,
        "static_rgbs": static_rgbs,
        "static_opacity_t0": static_opacity_t0,
        "static_scales": static_scales,
        "static_rotations": static_rotations,
        "original_world_points": original_world_points,
        "original_rgbs": original_rgbs,
        "original_opacity": original_opacity,
        "original_scales": original_scales,
        "original_rotations": original_rotations,
        "rendered_dynamic_t0": rendered_dynamic_t0,
        "model": model,
        "image_hw": (h, w),
    }


def save_rgb(rgb_tensor: torch.Tensor, output_path: str):
    Image.fromarray(to_uint8_image(rgb_tensor)).save(output_path)


def render_edit_run(session_id: str, request_data: dict):
    state = _SESSION_CACHE[session_id]
    output_root = resolve_path(ARGS.output_dir)
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    run_dir = os.path.join(output_root, state["scene_name"], f"sample_{state['frame_idx']:03d}", run_id)
    os.makedirs(run_dir, exist_ok=True)

    selection_boxes = request_data.get("selection_boxes")
    if not selection_boxes or not isinstance(selection_boxes, list):
        raise ValueError("selection_boxes must be a list of boxes.")
    selection_boxes = [[float(v) for v in box] for box in selection_boxes]

    remove_original = bool(request_data.get("remove_original", True))
    duplicate_selected = bool(request_data.get("duplicate_selected", False))
    add_repeat = int(request_data.get("add_repeat", 1))
    gaussian_dtype = state["ego_dyn"][0].dtype if state["ego_dyn"][0] is not None else torch.float32
    add_translation = torch.tensor(
        request_data.get("add_translation", [0.0, 0.0, 0.0]),
        device=DEVICE,
        dtype=gaussian_dtype,
    )
    target_click = request_data.get("target_click")
    offset_scales = [float(v) for v in request_data.get("offset_scales", [0.0, 1.0])]
    direction_mode = request_data.get("direction_mode", "vertical")
    move_direction = [float(v) for v in request_data.get("move_direction", [0.0, 0.0, 1.0])]

    kept_ego_dyn, selected_dyn, selected_count = gaussian_subset_by_image_boxes(
        state["ego_dyn"],
        state["ego_bg_mask"],
        state["ego_dyn_conf"],
        selection_boxes,
    )
    if selected_count == 0:
        raise RuntimeError("No dynamic Gaussian was selected inside the image box. Try a tighter box around the target vehicle.")

    target_center_world = None
    if target_click is not None:
        if len(target_click) != 2:
            raise ValueError("target_click must contain [x, y].")
        px = int(round(float(target_click[0])))
        py = int(round(float(target_click[1])))
        h, w = state["image_hw"]
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))
        target_center_world = state["point_map_ref"][py, px].to(dtype=gaussian_dtype)

    edited_ego_dyn = kept_ego_dyn if remove_original else state["ego_dyn"]
    if duplicate_selected:
        if target_center_world is not None:
            source_anchor = estimate_ground_anchor(selected_dyn[0])
            add_translation = target_center_world - source_anchor
        added_dyn = duplicate_gaussian_set(selected_dyn, add_translation, add_repeat)
    else:
        added_dyn = (None, None, None, None, None)
    final_dyn = concat_gaussian_sets(state["collab_dyn"], edited_ego_dyn, added_dyn)

    if final_dyn[0] is not None:
        world_points = torch.cat([state["static_points"], final_dyn[0]], dim=0)
        rgbs = torch.cat([state["static_rgbs"], final_dyn[1]], dim=0)
        opacity = torch.cat([state["static_opacity_t0"], final_dyn[2]], dim=0)
        scales = torch.cat([state["static_scales"], final_dyn[3]], dim=0)
        rotations = torch.cat([state["static_rotations"], final_dyn[4]], dim=0)
    else:
        world_points = state["static_points"]
        rgbs = state["static_rgbs"]
        opacity = state["static_opacity_t0"]
        scales = state["static_scales"]
        rotations = state["static_rotations"]

    move_dir = direction_from_mode(
        direction_mode,
        state["trajectory_vec"],
        move_direction,
        DEVICE,
    )

    input_path = os.path.join(run_dir, "ego_t0_input.png")
    dynamic_path = os.path.join(run_dir, "ego_t0_dynamic.png")
    selection_path = os.path.join(run_dir, "ego_t0_selection.png")
    save_rgb(state["images"][0, state["ref_idx"]], input_path)
    save_rgb(state["rendered_dynamic_t0"], dynamic_path)
    save_selection_overlay(state["rendered_dynamic_t0"], selection_boxes, selection_path)
    target_path = None
    if target_click is not None:
        target_path = os.path.join(run_dir, "ego_t0_target.png")
        image = Image.fromarray(to_uint8_image(state["rendered_dynamic_t0"]))
        draw = ImageDraw.Draw(image)
        px = int(round(float(target_click[0])))
        py = int(round(float(target_click[1])))
        radius = 8
        draw.ellipse([px - radius, py - radius, px + radius, py + radius], outline=(64, 255, 64), width=3)
        draw.line([px - 12, py, px + 12, py], fill=(64, 255, 64), width=2)
        draw.line([px, py - 12, px, py + 12], fill=(64, 255, 64), width=2)
        image.save(target_path)

    outputs = []
    for offset_scale in offset_scales:
        delta = move_dir * (state["trajectory_scale"] * float(offset_scale))
        target_c2w = state["ref_c2w"].clone()
        target_c2w[:3, 3] = state["ref_c2w"][:3, 3] + delta
        target_extrinsic = torch.inverse(target_c2w)
        rendered_original = render_novel_view(
            model=state["model"],
            world_points=state["original_world_points"],
            rgbs=state["original_rgbs"],
            opacity=state["original_opacity"],
            scales=state["original_scales"],
            rotations=state["original_rotations"],
            target_extrinsic=target_extrinsic,
            target_intrinsic=state["intrinsic_ref"],
            images_input=state["images"],
            source_extrinsics=state["extrinsic_4x4"],
            source_intrinsics=state["intrinsic"],
            image_hw=state["image_hw"],
        )
        rendered_static = render_novel_view(
            model=state["model"],
            world_points=state["static_points"],
            rgbs=state["static_rgbs"],
            opacity=state["static_opacity_t0"],
            scales=state["static_scales"],
            rotations=state["static_rotations"],
            target_extrinsic=target_extrinsic,
            target_intrinsic=state["intrinsic_ref"],
            images_input=state["images"],
            source_extrinsics=state["extrinsic_4x4"],
            source_intrinsics=state["intrinsic"],
            image_hw=state["image_hw"],
        )
        rendered = render_novel_view(
            model=state["model"],
            world_points=world_points,
            rgbs=rgbs,
            opacity=opacity,
            scales=scales,
            rotations=rotations,
            target_extrinsic=target_extrinsic,
            target_intrinsic=state["intrinsic_ref"],
            images_input=state["images"],
            source_extrinsics=state["extrinsic_4x4"],
            source_intrinsics=state["intrinsic"],
            image_hw=state["image_hw"],
        )
        orig_name = f"original_offset_{scale_tag(float(offset_scale))}.png"
        static_name = f"static_offset_{scale_tag(float(offset_scale))}.png"
        edit_name = f"edited_offset_{scale_tag(float(offset_scale))}.png"
        compare_name = f"compare_offset_{scale_tag(float(offset_scale))}.png"
        orig_path = os.path.join(run_dir, orig_name)
        static_path = os.path.join(run_dir, static_name)
        edit_path = os.path.join(run_dir, edit_name)
        compare_path = os.path.join(run_dir, compare_name)
        save_rgb(rendered_original, orig_path)
        save_rgb(rendered_static, static_path)
        save_rgb(rendered, edit_path)
        save_side_by_side_comparison(rendered_original, rendered, compare_path)
        outputs.append(
            {
                "offset_scale": float(offset_scale),
                "original": orig_path,
                "static": static_path,
                "edited": edit_path,
                "compare": compare_path,
            }
        )

    metadata = {
        "scene_name": state["scene_name"],
        "frame_idx": state["frame_idx"],
        "selection_boxes": selection_boxes,
        "selected_dynamic_count": selected_count,
        "remove_original": remove_original,
        "duplicate_selected": duplicate_selected,
        "add_translation": add_translation.detach().cpu().tolist(),
        "target_click": target_click,
        "target_center_world": None if target_center_world is None else target_center_world.detach().cpu().tolist(),
        "source_ground_anchor": None if target_center_world is None else estimate_ground_anchor(selected_dyn[0]).detach().cpu().tolist(),
        "add_repeat": add_repeat,
        "direction_mode": direction_mode,
        "move_direction": move_direction,
        "offset_scales": offset_scales,
        "run_dir": run_dir,
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    return input_path, dynamic_path, selection_path, target_path, outputs, metadata


@app.route("/")
def index():
    return render_template(
        "fruc_object_editor.html",
        default_image_dir=ARGS.image_dir,
        default_ckpt_path=ARGS.ckpt_path,
        default_dynamic_thresh=ARGS.dynamic_thresh,
    )


@app.route("/file")
def file_route():
    path = request.args.get("path", "")
    abs_path = os.path.abspath(path)
    return send_file(abs_path)


@app.route("/api/scenes")
def api_scenes():
    image_dir = request.args.get("image_dir", ARGS.image_dir)
    cam0_only = request.args.get("cam0_only", "true").lower() == "true"
    scene_order, scene_to_samples = get_scene_samples(image_dir, cam0_only)
    payload = []
    for idx, scene_name in enumerate(scene_order):
        payload.append(
            {
                "scene_id": idx,
                "scene_name": scene_name,
                "sample_count": len(scene_to_samples[scene_name]),
            }
        )
    return jsonify(payload)


@app.route("/api/samples")
def api_samples():
    image_dir = request.args.get("image_dir", ARGS.image_dir)
    cam0_only = request.args.get("cam0_only", "true").lower() == "true"
    scene_id = int(request.args["scene_id"])
    scene_order, scene_to_samples = get_scene_samples(image_dir, cam0_only)
    scene_name = scene_order[scene_id]
    return jsonify(
        {
            "scene_name": scene_name,
            "samples": scene_to_samples[scene_name],
        }
    )


@app.route("/api/load_sample", methods=["POST"])
def api_load_sample():
    try:
        data = request.get_json()
        scene_id = int(data["scene_id"])
        sample_id = int(data["sample_id"])
        image_dir = data.get("image_dir", ARGS.image_dir)
        ckpt_path = data.get("ckpt_path", ARGS.ckpt_path)
        cam0_only = bool(data.get("cam0_only", True))
        disable_fruc = bool(data.get("disable_fruc", False))

        state = prepare_sample_state(image_dir, ckpt_path, scene_id, sample_id, cam0_only, disable_fruc)
        session_id = uuid4().hex
        _SESSION_CACHE[session_id] = state

        preview_dir = os.path.join(resolve_path(ARGS.output_dir), "preview", session_id)
        os.makedirs(preview_dir, exist_ok=True)
        input_path = os.path.join(preview_dir, "ego_t0_input.png")
        save_rgb(state["images"][0, state["ref_idx"]], input_path)
        collab_inputs = []
        for collab_idx, slot in enumerate(state["collab_t0_slots"]):
            collab_path = os.path.join(preview_dir, f"collab_t0_input_{collab_idx:02d}_slot_{slot}.png")
            save_rgb(state["images"][0, slot], collab_path)
            collab_inputs.append(
                {
                    "slot": int(slot),
                    "label": f"collab_t0_slot_{slot}",
                    "path": collab_path,
                }
            )

        return jsonify(
            {
                "session_id": session_id,
                "scene_name": state["scene_name"],
                "frame_idx": state["frame_idx"],
                "trajectory_scale": state["trajectory_scale"],
                "image_width": state["image_hw"][1],
                "image_height": state["image_hw"][0],
                "input_image": input_path,
                "collab_inputs": collab_inputs,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/render_edit", methods=["POST"])
def api_render_edit():
    try:
        data = request.get_json()
        session_id = data["session_id"]
        input_path, dynamic_path, selection_path, target_path, outputs, metadata = render_edit_run(session_id, data)
        return jsonify(
            {
                "input_image": input_path,
                "dynamic_image": dynamic_path,
                "selection_image": selection_path,
                "target_image": target_path,
                "outputs": outputs,
                "metadata": metadata,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    os.makedirs(resolve_path(ARGS.output_dir), exist_ok=True)
    app.run(host=ARGS.host, port=ARGS.port, debug=False)
