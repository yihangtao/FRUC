
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import segmentation_models_pytorch as smp
import argparse

# Cityscapes classes
# 0: road, 1: sidewalk, 2: building, 3: wall, 4: fence, 5: pole,
# 6: traffic light, 7: traffic sign, 8: vegetation, 9: terrain, 10: sky,
# 11: person, 12: rider, 13: car, 14: truck, 15: bus, 16: train, 17: motorcycle, 18: bicycle

DYNAMIC_CLASSES = [11, 12, 13, 14, 15, 16, 17, 18]
SKY_CLASS = [10]

def main():
    parser = argparse.ArgumentParser(description="Generate FRUC sky and dynamic masks for processed V2X-Real scenes.")
    parser.add_argument('--data_root', type=str, required=True, help="Processed V2X-Real split directory, e.g. data/v2xreal/train")
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load Model
    print("Loading SegFormer model...")
    # This will download weights automatically if not present
    checkpoint = "smp-hub/segformer-b5-1024x1024-city-160k"
    try:
        model = smp.from_pretrained(checkpoint).eval().to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have internet access or the weights are cached.")
        raise RuntimeError('Unable to load the segmentation checkpoint.') from e

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    data_root = args.data_root
    if not os.path.exists(data_root):
        raise FileNotFoundError(f'Data root {data_root} does not exist.')

    # Find scenes
    # The dataset structure after preprocess_v2xreal.py is:
    # data_root/
    #   ├── scene1_1/
    #   │   ├── images/
    #   │       ├── image_name.jpg
    #   │   ├── sky_masks/
    #   │   ├── fine_dynamic_masks/all/

    scene_dirs = []
    if os.path.exists(data_root):
        potential_dirs = sorted(os.listdir(data_root))
        for d in potential_dirs:
            full_path = os.path.join(data_root, d)
            if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, "images")):
                 scene_dirs.append(full_path)

    print(f"Found {len(scene_dirs)} scenes.")

    for scene_dir in tqdm(scene_dirs, desc="Processing Scenes"):
        scene_name = os.path.basename(scene_dir)
        image_dir = os.path.join(scene_dir, "images")

        # Paths for mask directories
        # Consistent with train split:
        # scene_dir/sky_masks/
        # scene_dir/fine_dynamic_masks/all/

        sky_mask_dir = os.path.join(scene_dir, "sky_masks")
        dynamic_mask_dir = os.path.join(scene_dir, "fine_dynamic_masks", "all")

        # Ensure directories exist
        os.makedirs(sky_mask_dir, exist_ok=True)
        os.makedirs(dynamic_mask_dir, exist_ok=True)

        # Check if masks already exist for all images
        images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if not images:
            continue

        # Check filenames, since counts can include masks from an older export.
        if all(
            os.path.isfile(os.path.join(sky_mask_dir, os.path.splitext(name)[0] + '.png'))
            and os.path.isfile(os.path.join(dynamic_mask_dir, os.path.splitext(name)[0] + '.png'))
            for name in images
        ):
            tqdm.write(f"Skipping {scene_name}, masks already exist for all {len(images)} images.")
            continue

        for img_name in tqdm(images, leave=False, desc=f"Scene {scene_name}"):
            img_path = os.path.join(image_dir, img_name)
            name_no_ext = os.path.splitext(img_name)[0]

            # Output paths
            sky_mask_path = os.path.join(sky_mask_dir, f"{name_no_ext}.png")
            dynamic_mask_path = os.path.join(dynamic_mask_dir, f"{name_no_ext}.png")

            # Skip if specific mask exists
            if os.path.exists(sky_mask_path) and os.path.exists(dynamic_mask_path):
                continue

            img = Image.open(img_path).convert('RGB')
            w, h = img.size

            # Resize logic:
            # SegFormer requires input divisible by 32.
            # 1920x1080 -> 1080 is not divisible by 32 (1080/32 = 33.75)
            # Closest divisible is 1088 (34*32).
            # We can pad the image to 1920x1088.

            target_h = ((h - 1) // 32 + 1) * 32
            target_w = ((w - 1) // 32 + 1) * 32

            pad_h = target_h - h
            pad_w = target_w - w

            # Prepare tensor
            img_np = np.array(img)
            img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0).to(device)

            # Normalize
            img_tensor = img_tensor / 255.0
            img_tensor = (img_tensor - mean) / std

            # Pad if needed
            if pad_h > 0 or pad_w > 0:
                img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

            # Inference
            with torch.no_grad():
                logits = model(img_tensor)
                # Logits shape: [1, num_classes, target_h, target_w]

                # Crop back to original size
                if pad_h > 0 or pad_w > 0:
                    logits = logits[:, :, :h, :w]

                pr_masks = logits.softmax(dim=1)
                pred = pr_masks.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # Generate Sky Mask
            sky_mask = np.isin(pred, SKY_CLASS).astype(np.uint8) * 255

            # Generate Dynamic Mask
            dynamic_mask = np.isin(pred, DYNAMIC_CLASSES).astype(np.uint8) * 255

            # Save
            Image.fromarray(sky_mask).save(sky_mask_path)
            Image.fromarray(dynamic_mask).save(dynamic_mask_path)

if __name__ == '__main__':
    main()
