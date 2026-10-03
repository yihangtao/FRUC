import argparse
import os
import shutil
import numpy as np
import json
import math
from tqdm import tqdm
import yaml

def pose_vec2mat(vec):
    """
    Converts a pose vector to a transformation matrix using V2X-Real logic.
    Args:
        vec: [x, y, z, roll, yaw, pitch] in degrees
    Returns:
        4x4 matrix
    """
    x, y, z, roll, yaw, pitch = vec
    # Convert to radians with correction (flip roll and pitch)
    # V2X-Real specific: roll = -roll, pitch = -pitch
    roll_rad = math.radians(-roll)
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(-pitch)

    # Rotation matrices
    R_roll = np.array([[1, 0, 0], [0, math.cos(roll_rad), -math.sin(roll_rad)], [0, math.sin(roll_rad), math.cos(roll_rad)]])
    R_yaw = np.array([[math.cos(yaw_rad), -math.sin(yaw_rad), 0], [math.sin(yaw_rad), math.cos(yaw_rad), 0], [0, 0, 1]])
    R_pitch = np.array([[math.cos(pitch_rad), 0, math.sin(pitch_rad)], [0, 1, 0], [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]])

    # R = R_yaw @ R_pitch @ R_roll
    R = R_yaw @ R_pitch @ R_roll

    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = [x, y, z]
    return mat

def process_scene_vehicle(scene_path, vehicle_id, target_dir, scene_name, camera_index_base=0):
    """
    Process a specific vehicle's data from a scene.
    scene_path: path to the raw scene directory (e.g. .../2023-03-17-15-53-02)
    vehicle_id: '1' or '2'
    target_dir: output root directory
    scene_name: original scene name
    """
    vehicle_path = os.path.join(scene_path, str(vehicle_id))

    seq_name = f"{scene_name}_{vehicle_id}"
    seq_dir = os.path.join(target_dir, seq_name)

    # Breakpoint Resume Check
    # Check if images exist and are populated
    img_dir = os.path.join(seq_dir, 'images')
    if os.path.exists(seq_dir):
        if os.path.exists(img_dir) and len(os.listdir(img_dir)) > 0:
            tqdm.write(f"Skipping {seq_name}, already processed (found {len(os.listdir(img_dir))} images).")
            return
        else:
            tqdm.write(f"Resuming {seq_name}, directory exists but empty or incomplete (found {len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0} images).")

    os.makedirs(seq_dir, exist_ok=True)
    os.makedirs(os.path.join(seq_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(seq_dir, "intrinsics"), exist_ok=True)
    os.makedirs(os.path.join(seq_dir, "extrinsics"), exist_ok=True)
    os.makedirs(os.path.join(seq_dir, "ego_pose"), exist_ok=True)
    os.makedirs(os.path.join(seq_dir, "lidar"), exist_ok=True)

    # Get all frames (yaml files)
    files = sorted([f for f in os.listdir(vehicle_path) if f.endswith('.yaml')])
    frames = []
    for f in files:
        try:
            frame_idx = int(f.split('.')[0])
            frames.append(frame_idx)
        except ValueError:
            continue

    frames = sorted(list(set(frames)))

    # Save frame info
    frame_info = {
        "scene_name": seq_name,
        "num_frames": len(frames)
    }
    with open(os.path.join(seq_dir, "frame_info.json"), 'w') as f:
        json.dump(frame_info, f, indent=4)

    for i, frame_idx in enumerate(frames):
        yaml_file = os.path.join(vehicle_path, f"{frame_idx:06d}.yaml")
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)

        # Ego pose
        # Handle inconsistent keys in V2X-Real yaml files
        if 'true_ego_pos' in data:
            ego_pose_vec = data['true_ego_pos']
        elif 'true_ego_pose' in data:
            ego_pose_vec = data['true_ego_pose']
        elif 'lidar_pose' in data:
             # Fallback to lidar pose if ego pose is missing (often they are the same frame)
            ego_pose_vec = data['lidar_pose']
        else:
            print(f"Warning: Could not find ego pose in {yaml_file}. Skipping frame.")
            continue

        ego_pose_mat = pose_vec2mat(ego_pose_vec)

        # Save ego pose
        np.savetxt(os.path.join(seq_dir, "ego_pose", f"{frame_idx:06d}.txt"), ego_pose_mat)

        # Lidar
        lidar_src = os.path.join(vehicle_path, f"{frame_idx:06d}.bin")
        if os.path.exists(lidar_src):
            # Preserve LiDAR for offline dataset inspection.
            lidar_dst = os.path.join(seq_dir, "lidar", f"{frame_idx:03d}.bin")
            shutil.copy2(lidar_src, lidar_dst)
        else:
            # Optional: Warning or skip
            pass

        # Process cameras
        # V2X-Real has 4 cameras
        for source_cam_id in range(camera_index_base, camera_index_base + 4):
            cam_id = source_cam_id - camera_index_base
            # Try different naming patterns
            # Accept both numbered-camera and cam-prefixed image names.

            found_src = False
            src_img_path = ""

            for pattern in [f"{frame_idx:06d}_{source_cam_id}", f"{frame_idx:06d}_cam{source_cam_id}"]:
                for ext in ['.jpg', '.jpeg', '.png']:
                    temp_name = pattern + ext
                    temp_path = os.path.join(vehicle_path, temp_name)
                    if os.path.exists(temp_path):
                        src_img_path = temp_path
                        found_src = True
                        break
                if found_src:
                    break

            if not found_src:
                 # print(f"Warning: Image for frame {frame_idx} cam {cam_id} not found.")
                 continue

            dst_img_name = f"{frame_idx:06d}_{cam_id}.jpg"
            dst_img_path = os.path.join(seq_dir, "images", dst_img_name)
            shutil.copy2(src_img_path, dst_img_path)

            # Extrinsics & Intrinsics

            # Check for nested cam dict structure (common in this dataset)
            cam_key = f"cam{source_cam_id}"
            if cam_key in data:
                cam_data = data[cam_key]

                # Extrinsics
                # YAML 'extrinsic' is typically lidar_to_cam (point transformation)
                # We need to save cam_to_lidar (camera pose)
                if 'extrinsic' in cam_data:
                    lidar_to_cam = np.array(cam_data['extrinsic'])
                    cam_to_lidar = np.linalg.inv(lidar_to_cam)
                    np.savetxt(os.path.join(seq_dir, "extrinsics", f"{cam_id}.txt"), cam_to_lidar)

                # Intrinsics
                if 'intrinsic' in cam_data:
                    K = np.array(cam_data['intrinsic'])
                    np.savetxt(os.path.join(seq_dir, "intrinsics", f"{cam_id}.txt"), K)

            # Fallback/Legacy: Check for flat keys
            elif f'lidar_to_cam{source_cam_id}' in data:
                lidar_to_cam_vec = data[f'lidar_to_cam{source_cam_id}']
                lidar_to_cam = pose_vec2mat(lidar_to_cam_vec)
                # We save cam_to_lidar (inverse) as extrinsic
                cam_to_lidar = np.linalg.inv(lidar_to_cam)
                np.savetxt(os.path.join(seq_dir, "extrinsics", f"{cam_id}.txt"), cam_to_lidar)

                if f'intrinsic{source_cam_id}' in data:
                    K = np.array(data[f'intrinsic{source_cam_id}']).reshape(3, 3)
                    np.savetxt(os.path.join(seq_dir, "intrinsics", f"{cam_id}.txt"), K)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--target_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--camera_index_base', type=int, choices=[0, 1], default=0, help='Raw camera ids start at 0 (official format) or 1 (legacy export); output ids always start at 0.')
    args = parser.parse_args()

    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    scenes = sorted([d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))])

    print(f"Found {len(scenes)} scenes in {args.data_root}")

    for scene in tqdm(scenes, desc="Processing Scenes"):
        scene_path = os.path.join(args.data_root, scene)

        # Iterate over potential vehicle directories
        # V2X-Real contains '1', '2' (vehicles) and '-1', '-2' (RSUs).
        # We only process '1' and '2'.

        found_vehicle = False
        for vid in [1, 2]:
            if os.path.exists(os.path.join(scene_path, str(vid))):
                found_vehicle = True
                process_scene_vehicle(scene_path, vid, args.target_dir, scene, args.camera_index_base)

        if not found_vehicle:
            tqdm.write(f"Warning: No vehicle data found for scene {scene}")

if __name__ == '__main__':
    main()
