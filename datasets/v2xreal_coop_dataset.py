import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

def read_matrix(path, shape=(4, 4)):
    with open(path, 'r') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            data.extend([float(x) for x in line.split()])
        return np.array(data).reshape(shape)

class V2XRealCoopDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split='train',
        scene_names=None,
        start_frame=None,
        cam_id=None,
        for_vis=False,
        cam0_only=True,
        collab_view_count=1,
    ):
        self.root_dir = root_dir
        self.split = split
        self.scene_names = scene_names
        self.start_frame = start_frame
        self.cam_id = cam_id
        self.for_vis = for_vis
        self.cam0_only = cam0_only
        self.collab_view_count = max(0, int(collab_view_count))
        # If root_dir already points to a concrete split directory, don't append split again.
        if os.path.basename(os.path.normpath(root_dir)) in {'train', 'val', 'test'}:
            self.split_dir = root_dir
        else:
            self.split_dir = os.path.join(root_dir, split)
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        if not os.path.exists(self.split_dir):
            print(f"Warning: Split directory {self.split_dir} does not exist.")
            return

        # 1. Group directories by scene prefix
        # Structure: TIMESTAMP_SCENE_ID_SOMETHING_AGENTID
        # e.g. 2023-03-17-16-10-12_1_0_1
        # We want to group by everything except the last digit

        all_dirs = sorted([d for d in os.listdir(self.split_dir) if os.path.isdir(os.path.join(self.split_dir, d))])
        scenes = {}

        for d in all_dirs:
            # Parse directory name
            # Heuristic: split by '_'
            parts = d.split('_')
            if len(parts) < 2:
                continue

            # Assume last part is agent_id
            agent_id = parts[-1]
            scene_prefix = "_".join(parts[:-1])

            if scene_prefix not in scenes:
                scenes[scene_prefix] = {}
            scenes[scene_prefix][agent_id] = os.path.join(self.split_dir, d)

        # 2. Find valid pairs (Agent 1 and Agent 2)
        valid_scenes = []
        for scene_prefix, agents in scenes.items():
            if '1' in agents and '2' in agents:
                # If scene_names is provided, check if this scene matches
                if self.scene_names is not None:
                    # Allow matching by prefix or full agent directory name
                    match = False
                    for name in self.scene_names:
                        if name == scene_prefix:
                            match = True
                            break
                        # Check if name is an agent directory (e.g., ..._1_0_2)
                        # Extract prefix from name
                        parts = name.split('_')
                        if len(parts) >= 2:
                            name_prefix = "_".join(parts[:-1])
                            if name_prefix == scene_prefix:
                                match = True
                                break
                    if not match:
                        continue

                valid_scenes.append({
                    'prefix': scene_prefix,
                    'agent1_path': agents['1'],
                    'agent2_path': agents['2']
                })

        print(f"Found {len(valid_scenes)} scenes with both Agent 1 and Agent 2.")

        # 3. Find continuous frames for each scene based on context.json
        for scene in valid_scenes:
            context_path = os.path.join(scene['agent1_path'], 'context.json')
            if not os.path.exists(context_path):
                # Try agent2's context.json just in case
                context_path = os.path.join(scene['agent2_path'], 'context.json')
                if not os.path.exists(context_path):
                    continue

            with open(context_path, 'r') as f:
                try:
                    context_data = json.load(f)
                except json.JSONDecodeError:
                    continue

            frames = sorted([int(k) for k in context_data.keys() if k.isdigit()])

            scene_samples = []

            for i in range(len(frames) - 1):
                f1 = frames[i]
                f2 = frames[i+1]

                # Check continuity
                if f2 == f1 + 1:
                    if self.start_frame is not None and f1 != self.start_frame:
                        continue

                    anno1 = context_data.get(f'{f1:06d}') or context_data.get(f'{f1:03d}') or context_data.get(str(f1))
                    anno2 = context_data.get(f'{f2:06d}') or context_data.get(f'{f2:03d}') or context_data.get(str(f2))

                    if not anno1 or not anno2:
                        continue

                    if 'ego' not in anno1 or 'ego' not in anno2:
                        continue
                    if anno1['ego'] != anno2['ego'] or not anno1['ego']:
                        continue

                    ego_agent_name = anno1['ego'] # 'agent1' or 'agent2'
                    ego_path = scene['agent1_path'] if ego_agent_name == 'agent1' else scene['agent2_path']
                    collab_path = scene['agent2_path'] if ego_agent_name == 'agent1' else scene['agent1_path']

                    mapping1 = anno1.get('mapping', {})
                    mapping2 = anno2.get('mapping', {})

                    if self.collab_view_count == 0:
                        ego_cams = [0] if self.cam0_only else sorted(
                            {
                                int(cam)
                                for cam in list(mapping1.keys()) + list(mapping2.keys())
                                if str(cam).isdigit()
                            }
                        )
                        for ego_cam in ego_cams:
                            scene_samples.append({
                                'scene': scene['prefix'],
                                'frame_idx': f1,
                                'frame_next_idx': f2,
                                'ego_path': ego_path,
                                'collab_path': collab_path,
                                'ego_cam': int(ego_cam),
                                'collab_cams': [],
                            })
                        continue

                    # Find consistent multi-view collab sets between f1 and f2.
                    for ego_cam_str, collab_cams1 in mapping1.items():
                        try:
                            ego_cam = int(ego_cam_str)
                        except Exception:
                            continue

                        if self.cam0_only and ego_cam != 0:
                            continue

                        if ego_cam_str not in mapping2:
                            continue

                        collab_cams2 = mapping2[ego_cam_str]
                        common_collab_cams = sorted(
                            {int(cam) for cam in collab_cams1 if str(cam).isdigit()}.intersection(
                                {int(cam) for cam in collab_cams2 if str(cam).isdigit()}
                            )
                        )

                        if self.cam0_only and self.collab_view_count == 1:
                            common_collab_cams = [cam for cam in common_collab_cams if cam == 0]

                        if len(common_collab_cams) < self.collab_view_count:
                            continue

                        selected_collab_cams = common_collab_cams[:self.collab_view_count]
                        scene_samples.append({
                            'scene': scene['prefix'],
                            'frame_idx': f1,
                            'frame_next_idx': f2,
                            'ego_path': ego_path,
                            'collab_path': collab_path,
                            'ego_cam': ego_cam,
                            'collab_cams': selected_collab_cams,
                        })

            if self.for_vis:
                # Visualize the first valid ego/collaborator cam0 sample in each scene.
                cam0_samples = [
                    s for s in scene_samples
                    if s['ego_cam'] == 0 and len(s.get('collab_cams', [])) == self.collab_view_count
                ]
                if cam0_samples:
                    cam0_samples.sort(key=lambda x: x['frame_idx'])
                    self.samples.append(cam0_samples[0])
            else:
                self.samples.extend(scene_samples)

        print(f"Loaded {len(self.samples)} continuous samples from {self.split_dir}")

    def _get_frames(self, agent_path):
        pose_dir = os.path.join(agent_path, 'ego_pose')
        if not os.path.exists(pose_dir):
            return set()
        # Parse 000.txt -> 0
        frames = set()
        for f in os.listdir(pose_dir):
            if f.endswith('.txt'):
                try:
                    frames.add(int(f.split('.')[0]))
                except ValueError:
                    pass
        return frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        collab_cams = sample.get('collab_cams')
        if collab_cams is None:
            collab_cam = sample.get('collab_cam')
            collab_cams = [] if collab_cam is None else [collab_cam]

        # Load data for Ego and Collab
        ego_data = self._load_agent_data(sample['ego_path'], sample['frame_idx'], sample['frame_next_idx'], sample['ego_cam'])
        collab_data_list = [
            self._load_agent_data(sample['collab_path'], sample['frame_idx'], sample['frame_next_idx'], collab_cam)
            for collab_cam in collab_cams
        ]

        # Base pose is Ego's current pose
        base_pose = ego_data['pose']
        try:
            base_pose_inv = np.linalg.inv(base_pose)
        except np.linalg.LinAlgError:
            base_pose_inv = np.eye(4)

        target_size = (378, 672)
        from torchvision import transforms
        try:
            from torchvision.transforms import InterpolationMode
            interpolation = InterpolationMode.NEAREST
        except ImportError:
            interpolation = Image.NEAREST

        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])

        transform_mask = transforms.Compose([
            transforms.Resize(target_size, interpolation=interpolation),
            transforms.ToTensor(),
        ])

        images = []
        intrinsics = []
        extrinsics = []
        timestamps = []
        masks = []
        dynamic_masks = []

        # Order: Ego t=0, Collab_0..N-1 t=0, Ego t=1, Collab_0..N-1 t=1
        data_sequence = [(ego_data, 0.0, False)]
        data_sequence.extend((collab_data, 0.0, False) for collab_data in collab_data_list)
        data_sequence.append((ego_data, 1.0, True))
        data_sequence.extend((collab_data, 1.0, True) for collab_data in collab_data_list)

        for agent_data, timestamp, is_next in data_sequence:
            suffix = '_next' if is_next else ''

            img = agent_data[f'image{suffix}']
            if img is None:
                img = Image.new('RGB', (1920, 1080), (0, 0, 0))

            orig_w, orig_h = img.size
            images.append(transform(img))
            timestamps.append(timestamp)

            # Intrinsics
            K = agent_data['intrinsics'].copy()
            scale_x = target_size[1] / orig_w
            scale_y = target_size[0] / orig_h
            K[0, 0] *= scale_x
            K[1, 1] *= scale_y
            K[0, 2] *= scale_x
            K[1, 2] *= scale_y
            intrinsics.append(torch.from_numpy(K).float())

            # Extrinsics: T_cam_world = base_pose_inv @ T_ego @ T_cam_vehicle
            T_ego = agent_data[f'pose{suffix}']
            T_cam_vehicle = agent_data['extrinsics']

            T_ego_rel = base_pose_inv @ T_ego
            T_cam_world = T_ego_rel @ T_cam_vehicle
            extrinsics.append(torch.from_numpy(T_cam_world).float())

            # Masks
            mask = agent_data[f'mask{suffix}']
            if mask is not None:
                masks.append(transform_mask(mask))
            else:
                masks.append(torch.zeros(1, target_size[0], target_size[1]))

            dy_mask = agent_data[f'dy_mask{suffix}']
            if dy_mask is not None:
                dynamic_masks.append(transform_mask(dy_mask))
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
            'scene_name': sample['scene'],
            'ego_cam': sample['ego_cam'],
            'collab_cams': torch.tensor(collab_cams, dtype=torch.long),
        }

    def _read_overlap(self, agent_path, frame_idx):
        # Overlap files are usually named with 6 digits
        # e.g. 000000_overlap.json
        names = [
            f"{frame_idx:06d}_overlap.json",
            f"{frame_idx:03d}_overlap.json"
        ]

        overlap_dir = os.path.join(agent_path, 'overlap')
        if not os.path.exists(overlap_dir):
            return None

        for name in names:
            path = os.path.join(overlap_dir, name)
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        return json.load(f)
                except:
                    print(f"Failed to load overlap json: {path}")
                    return None
        return None

    def _load_mask(self, agent_path, frame_idx, cam_id, mask_type='sky_masks'):
        # mask_type: 'sky_masks' or 'fine_dynamic_masks/all'

        names = [
            f"{frame_idx:03d}_{cam_id}.png",
            f"{frame_idx:06d}_{cam_id}.png"
        ]

        # Check direct path
        mask_dirs = [
            os.path.join(agent_path, mask_type),
            os.path.join(agent_path, mask_type, 'all') # Check 'all' subdirectory too
        ]

        for mask_dir in mask_dirs:
            if not os.path.exists(mask_dir):
                continue

            for name in names:
                path = os.path.join(mask_dir, name)
                if os.path.exists(path):
                    try:
                        return Image.open(path).convert('L') # Load as grayscale
                    except:
                        print(f"Failed to open mask: {path}")
                        continue
        return None

    def _load_agent_data(self, agent_path, frame_idx, next_frame_idx, cam_id):
        # Load Poses
        pose_curr = self._read_pose(agent_path, frame_idx)
        pose_next = self._read_pose(agent_path, next_frame_idx)

        # Load Extrinsics/Intrinsics (static per camera)
        ext = read_matrix(os.path.join(agent_path, 'extrinsics', f'{cam_id}.txt'))
        intr = read_matrix(os.path.join(agent_path, 'intrinsics', f'{cam_id}.txt'), shape=(3, 3))

        # Load Images
        img_curr = self._load_image(agent_path, frame_idx, cam_id)
        img_next = self._load_image(agent_path, next_frame_idx, cam_id)

        # Load Masks
        mask_curr = self._load_mask(agent_path, frame_idx, cam_id, 'sky_masks')
        mask_next = self._load_mask(agent_path, next_frame_idx, cam_id, 'sky_masks')
        dy_mask_curr = self._load_mask(agent_path, frame_idx, cam_id, 'fine_dynamic_masks')
        dy_mask_next = self._load_mask(agent_path, next_frame_idx, cam_id, 'fine_dynamic_masks')

        return {
            'pose': pose_curr,
            'pose_next': pose_next,
            'extrinsics': ext,
            'intrinsics': intr,
            'image': img_curr,
            'image_next': img_next,
            'mask': mask_curr,
            'mask_next': mask_next,
            'dy_mask': dy_mask_curr,
            'dy_mask_next': dy_mask_next
        }

    def _read_pose(self, agent_path, frame_idx):
        for width in (6, 3):
            path = os.path.join(agent_path, 'ego_pose', f'{frame_idx:0{width}d}.txt')
            if os.path.exists(path):
                return read_matrix(path)
        return np.eye(4)

    def _load_image(self, agent_path, frame_idx, cam_id):
        # Image naming: FRAMEID_CAMID.jpeg
        # e.g. 006_1.jpeg

        # Try finding the file
        names = [
            f"{frame_idx:03d}_{cam_id}.jpeg",
            f"{frame_idx:03d}_{cam_id}.jpg",
            f"{frame_idx:06d}_{cam_id}.jpeg",
            f"{frame_idx:06d}_{cam_id}.jpg"
        ]

        img_dir = os.path.join(agent_path, 'images')
        for name in names:
            path = os.path.join(img_dir, name)
            if os.path.exists(path):
                try:
                    return Image.open(path).convert('RGB')
                except:
                    print(f"Failed to open image: {path}")
                    return None

        # print(f"Image not found: {names} in {img_dir}")
        return None
