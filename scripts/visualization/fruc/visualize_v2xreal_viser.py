"""Inspect FRUC cooperative V2X-Real reconstructions in Viser."""

import os
import argparse
import torch
import numpy as np
import viser
import time
import sys

# Add the repository root to the Python import path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from fruc.models.vggt import VGGT
from fruc.utils.checkpoint import extract_state_dict
from fruc.utils.pose_enc import pose_encoding_to_extri_intri
from fruc.utils.geometry import unproject_depth_map_to_point_map
from fruc.utils.gs import get_split_gs
from datasets.v2xreal_coop_dataset import V2XRealCoopDataset

def compute_cov3d(scales, quats):
    # quats: (N, 4) w, x, y, z
    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / norm

    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    R = np.zeros((quats.shape[0], 3, 3))

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)

    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)

    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    M = R * scales[:, np.newaxis, :]
    Cov = np.matmul(M, np.transpose(M, (0, 2, 1)))
    return Cov.astype(np.float32)

def alpha_t(t, t0, alpha, gamma0=1, gamma1=0.1):
    sigma = torch.log(torch.tensor(gamma1)).to(gamma0.device) / ((gamma0)**2 + 1e-6)
    conf = torch.exp(sigma*(t0-t)**2)
    alpha_ = alpha * conf
    return alpha_.float()

def main():
    parser = argparse.ArgumentParser(description="Inspect FRUC cooperative V2X-Real reconstructions in Viser.")
    parser.add_argument("--image_dir", type=str, default="data/v2xreal/val", help="Path to V2X-Real root directory")
    parser.add_argument("--ckpt_path", type=str, default="pretrained/fruc/model_fruc_v2xreal.pth", help="Path to the FRUC checkpoint")
    parser.add_argument("--scene_idx", type=int, default=0, help="Index of the scene to visualize")
    parser.add_argument("--sample_idx", type=int, default=0, help="Index of the sample within the scene")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # 1. Load Dataset
    print(f"Loading V2X-Real Cooperative dataset from {args.image_dir}...")
    dataset = V2XRealCoopDataset(root_dir=args.image_dir, split='train' if 'train' in args.image_dir else 'val', cam0_only=True)
    if len(dataset) == 0:
        print("No samples found in the dataset.")
        return

    scene_order = list(dict.fromkeys(sample["scene"] for sample in dataset.samples))
    if args.scene_idx < 0 or args.scene_idx >= len(scene_order):
        print(f"Error: scene_idx {args.scene_idx} out of bounds [0, {len(scene_order)-1}].")
        return
    selected_scene = scene_order[args.scene_idx]

    scene_samples = [idx for idx, sample in enumerate(dataset.samples) if sample["scene"] == selected_scene]
    if len(scene_samples) == 0:
        print(f"Error: No samples found for scene {selected_scene}.")
        return

    if args.sample_idx < 0 or args.sample_idx >= len(scene_samples):
        print(f"Warning: sample_idx {args.sample_idx} out of bounds [0, {len(scene_samples)-1}]. Clamping.")

    clamped_sample_idx = max(0, min(args.sample_idx, len(scene_samples) - 1))
    global_sample_idx = scene_samples[clamped_sample_idx]

    sample = dataset[global_sample_idx]

    # 2. Load Model
    print(f"Loading model from {args.ckpt_path}...")
    model = VGGT().to(device)

    checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(checkpoint)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 3. Process Sample
    print(f"Processing scene: {sample['scene_name']}")
    with torch.no_grad():
        # Add batch dimension
        images = sample['images'].unsqueeze(0).to(device)
        sky_mask = sample['masks'].unsqueeze(0).to(device).permute(0, 1, 3, 4, 2)
        timestamps = sample['timestamps'].to(device)

        images_s = images[0]
        sky_mask_s = sky_mask[0]

        # Pass all 4 frames to the model to trigger FRUC cooperative fusion
        # (Ego_t0, Collab_t0, Ego_t1, Collab_t1)
        images_input = images
        sky_mask_input_s = sky_mask_s
        timestamps_input = timestamps
        bg_mask = (sky_mask_input_s[..., 0] == 0)

        print("Running forward pass with FRUC cooperative fusion...")
        predictions = model(images_input, apply_fruc=True)

        H, W = images_s.shape[-2:]
        extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], (H, W))
        extrinsic = extrinsics[0]
        intrinsic = intrinsics[0]

        # Depth to Point map
        depth_map = predictions["depth"][0]
        point_map = unproject_depth_map_to_point_map(depth_map.cpu().numpy(), extrinsics[0].cpu().numpy(), intrinsics[0].cpu().numpy())[None,...]
        point_map = torch.from_numpy(point_map).to(device).float()

        gs_map = predictions["gs_map"][0]
        gs_conf = predictions["gs_conf"][0]
        dy_map = predictions["dynamic_conf"].squeeze(-1)[0]
        point_map = point_map[0]

        static_mask = torch.ones_like(bg_mask)

        # Extract Static GS
        static_points = point_map[static_mask].reshape(-1, 3)
        gs_dynamic_list = dy_map[static_mask].sigmoid()
        static_rgbs, static_opacity, static_scales, static_rotations = get_split_gs(gs_map, static_mask)
        static_opacity = static_opacity * (1 - gs_dynamic_list)
        static_gs_conf = gs_conf[static_mask]

        frame_indices = torch.nonzero(static_mask, as_tuple=False)[:, 0]
        gs_timestamps_static = timestamps[frame_indices]

        # We will visualize the first frame (t0)
        target_frame_idx = 0
        t0 = timestamps_input[target_frame_idx]

        # Apply temporal decay to static opacity
        static_opacity_t0 = alpha_t(gs_timestamps_static, t0, static_opacity, gamma0=static_gs_conf)

        # Extract Dynamic GS for the target frame (Ego_t0 and Collab_t0)
        # Sequence order is Ego_t0, Collab_t0, Ego_t1, Collab_t1
        # So idx=0 and idx=1 are the t0 frames for both agents.
        dyn_points_list, dyn_rgbs_list, dyn_op_list, dyn_scales_list, dyn_rot_list = [], [], [], [], []

        for idx in [0, 1]:  # Ego_t0 and Collab_t0
            bg_mask_idx = bg_mask[idx]
            if bg_mask_idx.sum() > 0:
                p = point_map[idx][bg_mask_idx].reshape(-1, 3)
                rgb, op, s, r = get_split_gs(gs_map[idx], bg_mask_idx)
                dyn_conf = dy_map[idx][bg_mask_idx].sigmoid()
                op = op * dyn_conf

                dyn_points_list.append(p)
                dyn_rgbs_list.append(rgb)
                dyn_op_list.append(op)
                dyn_scales_list.append(s)
                dyn_rot_list.append(r)

        if len(dyn_points_list) > 0:
            dyn_p = torch.cat(dyn_points_list, dim=0)
            dyn_rgb = torch.cat(dyn_rgbs_list, dim=0)
            dyn_op = torch.cat(dyn_op_list, dim=0)
            dyn_s = torch.cat(dyn_scales_list, dim=0)
            dyn_r = torch.cat(dyn_rot_list, dim=0)

            # Combine static and dynamic
            world_points = torch.cat([static_points, dyn_p], dim=0)
            rgbs = torch.cat([static_rgbs, dyn_rgb], dim=0)
            opacity = torch.cat([static_opacity_t0, dyn_op], dim=0)
            scales = torch.cat([static_scales, dyn_s], dim=0)
            rotation = torch.cat([static_rotations, dyn_r], dim=0)
        else:
            world_points = static_points
            rgbs = static_rgbs
            opacity = static_opacity_t0
            scales = static_scales
            rotation = static_rotations

        # Convert to numpy
        means_np = world_points.cpu().numpy()
        rgbs_np = rgbs.cpu().numpy()
        opacities_np = opacity.cpu().numpy()
        if opacities_np.ndim == 1:
            opacities_np = opacities_np[:, np.newaxis]
        scales_np = scales.cpu().numpy()
        quats_np = rotation.cpu().numpy()

        # Convert scales from log space if needed (get_split_gs might already do this or not)
        # Checking get_split_gs in typical 3DGS: scales are usually exponentiated before rendering
        # Wait, get_split_gs typically returns raw scales or exp scales?
        # Let's just pass them as is, and check if we need to exp.
        # Usually gsplat rasterization takes raw scales if packed=False but wait, gsplat takes exp scales?
        # Let's look at `alpha_t` and `gs_dynamic_list`...

        # Wait, if get_split_gs doesn't exp scales, we should. Let's assume it doesn't exp.
        # Actually in fruc `get_split_gs` often returns torch.exp(scales). Let's see if we get very large values.

        # Compute covariances for Viser
        # Viser add_gaussian_splats can take scales and quats directly in recent versions, but covariance is safer.
        try:
            covariances_np = compute_cov3d(scales_np, quats_np)
        except Exception as e:
            print(f"Failed to compute covariances: {e}")
            covariances_np = None

    # 4. Start Viser Server
    print(f"Starting Viser server on port {args.port}...")
    server = viser.ViserServer(port=args.port)

    try:
        if covariances_np is not None:
            server.scene.add_gaussian_splats(
                "/gaussians",
                centers=means_np,
                covariances=covariances_np,
                opacities=opacities_np,
                rgbs=rgbs_np
            )
        else:
            # Fallback if compute_cov3d fails
            server.scene.add_gaussian_splats(
                "/gaussians",
                centers=means_np,
                scales=scales_np,
                quats=quats_np,
                opacities=opacities_np,
                rgbs=rgbs_np
            )
        print("3DGS scene added successfully!")
    except Exception as e:
        print(f"Error adding 3DGS: {e}")

    # Set initial camera to match Ego (extrinsic)
        try:
            # extrinsics[0] is the 3x4 matrix for Ego t0
            extrinsic_t0 = extrinsic.cpu().numpy() # [3, 4]
            # Viser camera is typically defined by w2c or c2w
            # PyTorch3D / FRUC extrinsic is usually World-to-Camera
            # R, T = w2c[:3,:3], w2c[:3,3]
            # c2w = inv(w2c)
            R_w2c = extrinsic_t0[:3, :3]
            T_w2c = extrinsic_t0[:3, 3]
            R_c2w = R_w2c.T
            T_c2w = -R_w2c.T @ T_w2c

            # Viser uses quaternion for rotation
            # We can set the initial camera pose
            # Convert R_c2w to quaternion (w, x, y, z)
            import transforms3d.quaternions as quat
            q_c2w = quat.mat2quat(R_c2w)

            # Create a camera handle to set position
            client = server.get_clients()

            # Since clients might connect later, we set the global up direction and default camera
            # Set up direction (often Y down in CV, Z up in others)
            # We will just set the initial camera position
            @server.on_client_connect
            def on_client_connect(client: viser.ClientHandle) -> None:
                client.camera.position = T_c2w
                client.camera.wxyz = q_c2w
                # Allow infinite zoom by adjusting near plane and controls
                client.camera.near = 0.001
        except Exception as e:
            print(f"Could not set initial camera: {e}")

    print(f"Viewer running at http://localhost:{args.port}")
    print("Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    main()
