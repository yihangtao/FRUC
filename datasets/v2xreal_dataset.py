
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import yaml
import math
from torchvision import transforms

def pose_vec2mat_corrected(vec):
    """
    Convert 6DoF pose vector to 4x4 transformation matrix with V2X-Real correction.
    Args:
        vec: [6] (x, y, z, roll, yaw, pitch) in degrees
    Returns:
        mat: [4, 4] tensor
    """
    x, y, z, roll, yaw, pitch = vec

    # Flip roll and pitch, convert to radians
    # V2X-Real specific: roll = -roll, pitch = -pitch
    roll_rad = math.radians(-roll)
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(-pitch)

    cos_r, sin_r = math.cos(roll_rad), math.sin(roll_rad)
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)

    # Rotation matrices (yaw->pitch->roll order)
    R_roll = np.array([[1, 0, 0], [0, cos_r, -sin_r], [0, sin_r, cos_r]])
    R_yaw = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])
    R_pitch = np.array([[cos_p, 0, sin_p], [0, 1, 0], [-sin_p, 0, cos_p]])

    R = R_yaw @ R_pitch @ R_roll

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return torch.from_numpy(T).float()

def read_matrix(path, shape=(4, 4)):
    """Read a matrix from a text file."""
    if not os.path.exists(path):
        return torch.eye(shape[0])
    with open(path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        parts = line.strip().split()
        if not parts: continue
        data.extend([float(x) for x in parts])

    if len(data) != shape[0] * shape[1]:
        # Handle cases where the file might be malformed or different format
        return torch.eye(shape[0])

    return torch.tensor(data).reshape(shape).float()

class V2XRealDataset(Dataset):
    def __init__(self, image_dir, scene_names=None, sequence_length=4, start_idx=-1, mode=2, views=1, intervals=1, camera=None, start_frame=None):
        self.image_dir = image_dir
        self.sequence_length = sequence_length
        self.interval = intervals
        self.mode = mode
        self.views = views
        self.start_idx = start_idx
        self.camera = str(camera) if camera is not None else None
        self.start_frame = start_frame

        self.sequences = []
        self.structure_type = 'original' # Default

        # If scene_names are provided, filter by them
        if scene_names is None:
            # Find all available scenes
            if os.path.exists(image_dir):
                possible_scenes = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])
                scene_names = possible_scenes
            else:
                scene_names = []
        else:
            # Filter valid scenes
            scene_names = [s for s in scene_names if os.path.isdir(os.path.join(image_dir, s))]

        print(f"Scanning {len(scene_names)} scenes in {image_dir}...")

        if len(scene_names) > 0:
            # Detect structure type based on the first scene
            first_scene_path = os.path.join(image_dir, scene_names[0])
            if os.path.exists(os.path.join(first_scene_path, 'images')):
                self.structure_type = 'plus'
                print("Detected V2X-Real+ structure (flat scenes with 'images' folder).")
                self._scan_plus(scene_names)
            else:
                self.structure_type = 'original'
                print("Detected Original V2X-Real structure (nested agent folders).")
                self._scan_original(scene_names)
        else:
             print("No scenes found.")

        # Filter by camera if specified
        if self.camera is not None:
            self.sequences = [seq for seq in self.sequences if seq.get('camera') == self.camera]
            print(f"Filtered sequences by camera '{self.camera}'. Remaining: {len(self.sequences)}")

        print(f"Found {len(self.sequences)} valid sequences.")

    def _scan_original(self, scene_names):
        for scene_name in scene_names:
            scene_path = os.path.join(self.image_dir, scene_name)

            # Find agent directories (1, 2, etc. skipping -1, -2)
            agent_dirs = sorted([
                d for d in os.listdir(scene_path)
                if os.path.isdir(os.path.join(scene_path, d))
            ])

            for agent_name in agent_dirs:
                if agent_name in ['-1', '-2']:
                    continue
                if agent_name not in ['1', '2']: # Limit to vehicles
                    continue

                agent_path = os.path.join(scene_path, agent_name)

                # Check files
                files = sorted(os.listdir(agent_path))
                jpg_files = [f for f in files if f.endswith('.jpg') or f.endswith('.jpeg')]

                if not jpg_files:
                    continue

                # Extract frame indices
                frame_indices = set()
                for f in jpg_files:
                    parts = f.split('_')
                    if len(parts) >= 2 and parts[0].isdigit():
                        frame_indices.add(int(parts[0]))

                if not frame_indices:
                    continue

                sorted_indices = sorted(list(frame_indices))

                # Determine camera
                frame0 = sorted_indices[0]
                ext = '.jpeg' if any(f.endswith('.jpeg') for f in jpg_files) else '.jpg'

                # For simplicity, use cam1 for single view
                target_cam = 'cam1'
                fname = f"{frame0:06d}_{target_cam}{ext}"
                if not os.path.exists(os.path.join(agent_path, fname)):
                    # Try to find available camera
                    for c in range(1, 5):
                        if os.path.exists(os.path.join(agent_path, f"{frame0:06d}_cam{c}{ext}")):
                            target_cam = f"cam{c}"
                            break

                seq_data = {
                    'root': agent_path,
                    'scene_name': scene_name,
                    'agent_name': agent_name,
                    'frames': [],
                    'ext': ext,
                    'camera': target_cam
                }

                for idx in sorted_indices:
                    frame_info = {
                        'idx': idx,
                        'file_path': os.path.join(agent_path, f"{idx:06d}_{target_cam}{ext}"),
                        'yaml_path': os.path.join(agent_path, f"{idx:06d}.yaml")
                    }
                    if os.path.exists(frame_info['file_path']):
                        seq_data['frames'].append(frame_info)

                if len(seq_data['frames']) >= self.sequence_length:
                     self.sequences.append(seq_data)

    def _scan_plus(self, scene_names):
        for scene_name in scene_names:
            scene_path = os.path.join(self.image_dir, scene_name)
            images_dir = os.path.join(scene_path, 'images')

            if not os.path.exists(images_dir):
                continue

            files = sorted(os.listdir(images_dir))
            # Support multiple extensions
            valid_exts = ('.jpg', '.jpeg', '.png')
            image_files = [f for f in files if f.lower().endswith(valid_exts)]

            if not image_files:
                continue

            # Parse files to find cameras and frames
            # Format: FRAME_CAM.ext (e.g., 000000_1.jpg or 000_0.jpeg)
            frames_per_cam = {} # cam_id -> list of indices
            file_map = {} # (cam_id, frame_idx) -> filename
            frame_str_map = {} # frame_idx -> original frame string (for padding)

            for f in image_files:
                name_part, ext = os.path.splitext(f)
                parts = name_part.split('_')
                if len(parts) >= 2 and parts[0].isdigit():
                    frame_str = parts[0]
                    frame_idx = int(frame_str)
                    # cam_id is usually the last part before extension if formatted as FRAME_CAM
                    # But let's stick to the observed format FRAME_CAM
                    cam_id = parts[1] # e.g., '1', '2' or '0'
                    if cam_id.isdigit():
                        cam_id = f"cam{cam_id}"

                    if cam_id not in frames_per_cam:
                        frames_per_cam[cam_id] = []
                    frames_per_cam[cam_id].append(frame_idx)

                    file_map[(cam_id, frame_idx)] = f
                    frame_str_map[frame_idx] = frame_str

            # Create sequences for each camera
            for cam_id in sorted(frames_per_cam.keys()):
                indices = sorted(list(set(frames_per_cam[cam_id])))
                if not indices:
                    continue

                seq_data = {
                    'root': scene_path,
                    'scene_name': scene_name,
                    'agent_name': 'ego', # V2X-Real+ seems to be ego-centric
                    'frames': [],
                    'camera': cam_id,
                    'type': 'plus'
                }

                for idx in indices:
                    # Retrieve original filename and frame string to handle padding correctly
                    filename = file_map.get((cam_id, idx))
                    frame_str = frame_str_map.get(idx)

                    if not filename or not frame_str:
                        continue

                    # Construct ego pose path using the same frame string padding
                    ego_pose_name = f"{frame_str}.txt"

                    frame_info = {
                        'idx': idx,
                        'file_path': os.path.join(images_dir, filename),
                        'ego_pose_path': os.path.join(scene_path, 'ego_pose', ego_pose_name),
                        'extrinsic_path': os.path.join(scene_path, 'extrinsics', f"{cam_id}.txt"),
                        'intrinsic_path': os.path.join(scene_path, 'intrinsics', f"{cam_id}.txt")
                    }
                    seq_data['frames'].append(frame_info)

                if len(seq_data['frames']) >= self.sequence_length:
                    self.sequences.append(seq_data)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.start_frame is not None:
             # Find correct starting point within sequence
             # This logic depends on structure
             if self.structure_type == 'plus':
                 return self._getitem_plus(idx, force_start_idx=self.start_frame)
             else:
                 # The nested layout uses its standard sequence selection below.
                 pass

        if self.structure_type == 'plus':
            return self._getitem_plus(idx)
        else:
            return self._getitem_original(idx)

    def _getitem_plus(self, idx, force_start_idx=None):
        seq_info = self.sequences[idx]
        frames = seq_info['frames']

        # Determine frames to load
        # Randomly select a valid start index
        max_start = max(0, len(frames) - self.sequence_length * self.interval)

        if force_start_idx is not None:
            # If force_start_idx is provided (e.g. via start_frame), try to find the frame with that index
            # Frames list contains objects with 'idx' field
            start_pos = 0
            found = False
            for i, f in enumerate(frames):
                if f['idx'] == force_start_idx:
                    start_pos = i
                    found = True
                    break

            if found:
                start_idx = start_pos
            else:
                # Fallback or clamp
                start_idx = min(force_start_idx, max_start)
        elif self.start_idx >= 0:
             start_idx = min(self.start_idx, max_start)
        else:
            # Random for training, 0 for testing usually
            if self.mode == 2: # validation/test/reconstruction
                start_idx = 0
            else:
                start_idx = np.random.randint(0, max_start + 1)

        selected_frames = []
        for i in range(self.sequence_length):
            idx_in_list = start_idx + i * self.interval
            if idx_in_list < len(frames):
                selected_frames.append(frames[idx_in_list])
            else:
                selected_frames.append(frames[-1])

        target_size = (378, 672)
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])

        images = []
        intrinsics = []
        extrinsics = []
        timestamps = []
        masks = []
        dynamic_masks = []

        for i, frame_info in enumerate(selected_frames):
            # Load Image
            img_path = frame_info['file_path']
            img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img.size
            images.append(transform(img))

            # Timestamp
            timestamps.append(float(i))

            # Load Intrinsic
            K = read_matrix(frame_info['intrinsic_path'], shape=(3, 3))
            scale_x = target_size[1] / orig_w
            scale_y = target_size[0] / orig_h
            K[0, 0] *= scale_x
            K[1, 1] *= scale_y
            K[0, 2] *= scale_x
            K[1, 2] *= scale_y
            intrinsics.append(K)

            # Load Pose (Ego + Extrinsic)
            # Ego Pose: Vehicle to World
            T_ego = read_matrix(frame_info['ego_pose_path'], shape=(4, 4))

            # Extrinsic: Camera to Vehicle (Static)
            T_cam_vehicle = read_matrix(frame_info['extrinsic_path'], shape=(4, 4))

            # Final Pose: Camera to World = T_ego @ T_cam_vehicle
            T_cam_world = T_ego @ T_cam_vehicle
            extrinsics.append(T_cam_world)

            # Masks
            # Derive scene directory from image path (.../scene/images/img.jpg -> .../scene)
            img_dir = os.path.dirname(img_path)
            scene_dir = os.path.dirname(img_dir)
            filename = os.path.basename(img_path)
            name_no_ext = os.path.splitext(filename)[0]

            # Sky Mask
            sky_mask_path = os.path.join(scene_dir, 'sky_masks', name_no_ext + '.png')
            if os.path.exists(sky_mask_path):
                m = Image.open(sky_mask_path).convert('L')
                m = m.resize((target_size[1], target_size[0]), Image.NEAREST)
                masks.append((transforms.ToTensor()(m) > 0.5).float())
            else:
                masks.append(torch.ones(1, target_size[0], target_size[1]))

            # Dynamic Mask
            # Try fine_dynamic_masks/all first, then dynamic_masks
            dyn_mask_path = os.path.join(scene_dir, 'fine_dynamic_masks', 'all', name_no_ext + '.png')
            if not os.path.exists(dyn_mask_path):
                 dyn_mask_path = os.path.join(scene_dir, 'dynamic_masks', name_no_ext + '.png')

            if os.path.exists(dyn_mask_path):
                dm = Image.open(dyn_mask_path).convert('L')
                dm = dm.resize((target_size[1], target_size[0]), Image.NEAREST)
                dynamic_masks.append((transforms.ToTensor()(dm) > 0.5).float())
            else:
                dynamic_masks.append(torch.zeros(1, target_size[0], target_size[1]))

        images = torch.stack(images)
        intrinsics = torch.stack(intrinsics)
        extrinsics = torch.stack(extrinsics)
        timestamps = torch.tensor(timestamps)
        masks = torch.stack(masks)
        dynamic_masks = torch.stack(dynamic_masks)

        gt_depth = torch.zeros((len(images), target_size[0], target_size[1], 1))

        return {
            'images': images,
            'masks': masks,
            'dynamic_mask': dynamic_masks,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'timestamps': timestamps,
            'gt_depth': gt_depth,
            'targets': images,
            'target_masks': masks,
            'scene_name': seq_info['scene_name']
        }

    def _getitem_original(self, idx):
        seq_info = self.sequences[idx]
        frames = seq_info['frames']

        start_idx = self.start_idx
        if start_idx + self.sequence_length * self.interval > len(frames):
            start_idx = 0

        selected_frames = []
        for i in range(self.sequence_length):
            idx_in_list = start_idx + i * self.interval
            if idx_in_list < len(frames):
                selected_frames.append(frames[idx_in_list])
            else:
                selected_frames.append(frames[-1])

        image_paths = [f['file_path'] for f in selected_frames]
        yaml_paths = [f['yaml_path'] for f in selected_frames]
        camera_name = seq_info['camera']

        target_size = (378, 672)
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])

        images = []
        intrinsics = []
        extrinsics = []
        timestamps = []
        masks = []
        dynamic_masks = []

        for i, (img_path, yml_path) in enumerate(zip(image_paths, yaml_paths)):
            img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img.size
            images.append(transform(img))

            timestamps.append(float(i))

            if os.path.exists(yml_path):
                with open(yml_path, 'r') as f:
                    data = yaml.safe_load(f)

                if camera_name in data:
                    cam_data = data[camera_name]
                    K = np.array(cam_data['intrinsic'])
                    scale_x = target_size[1] / orig_w
                    scale_y = target_size[0] / orig_h
                    K[0, 0] *= scale_x
                    K[1, 1] *= scale_y
                    K[0, 2] *= scale_x
                    K[1, 2] *= scale_y
                    intrinsics.append(torch.from_numpy(K).float())

                    if 'cords' in cam_data:
                        pose = pose_vec2mat_corrected(cam_data['cords'])
                        extrinsics.append(pose)
                    else:
                        E = np.array(cam_data['extrinsic'])
                        extrinsics.append(torch.from_numpy(E).float())
                else:
                    intrinsics.append(torch.eye(3))
                    extrinsics.append(torch.eye(4))
            else:
                intrinsics.append(torch.eye(3))
                extrinsics.append(torch.eye(4))

            # Masks
            dirname = os.path.dirname(img_path)
            basename = os.path.basename(img_path)
            name_no_ext = os.path.splitext(basename)[0]

            sky_mask_path = os.path.join(dirname, 'sky_masks', name_no_ext + '.png')
            if os.path.exists(sky_mask_path):
                m = Image.open(sky_mask_path).convert('L')
                m = m.resize((target_size[1], target_size[0]), Image.NEAREST)
                masks.append((transforms.ToTensor()(m) > 0.5).float())
            else:
                masks.append(torch.ones(1, target_size[0], target_size[1]))

            dyn_mask_path = os.path.join(dirname, 'dynamic_masks', name_no_ext + '.png')
            if os.path.exists(dyn_mask_path):
                dm = Image.open(dyn_mask_path).convert('L')
                dm = dm.resize((target_size[1], target_size[0]), Image.NEAREST)
                dynamic_masks.append((transforms.ToTensor()(dm) > 0.5).float())
            else:
                dynamic_masks.append(torch.zeros(1, target_size[0], target_size[1]))

        images = torch.stack(images)
        intrinsics = torch.stack(intrinsics)
        extrinsics = torch.stack(extrinsics)
        timestamps = torch.tensor(timestamps)
        masks = torch.stack(masks)
        dynamic_masks = torch.stack(dynamic_masks)

        gt_depth = torch.zeros((len(images), target_size[0], target_size[1], 1))

        return {
            'images': images,
            'masks': masks,
            'dynamic_mask': dynamic_masks,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'timestamps': timestamps,
            'gt_depth': gt_depth,
            'targets': images,
            'target_masks': masks,
            'scene_name': f"{seq_info['scene_name']}_{seq_info['agent_name']}"
        }
