# Preparing V2X-Real for FRUC

Obtain V2X-Real from the [official project](https://github.com/ucla-mobility/V2X-Real)
and follow its access terms. Keep the original train/validation split. Install
`requirements.txt` and `requirements_data.txt` in the FRUC environment.

## 1. Raw layout

```text
data/v2xreal_raw/
  train/
    <scene>/
      1/                         # Vehicle 1
        000000.yaml
        000000_cam0.jpeg
        000000_cam1.jpeg
        000000_cam2.jpeg
        000000_cam3.jpeg
      2/                         # Vehicle 2
  val/...
```

The converter keeps vehicle agents `1` and `2`; infrastructure agents are outside
this release. Images can be `.jpg`, `.jpeg` or `.png`, named `<frame>_cam<id>` or
`<frame>_<id>`. YAML files supply original camera matrices and ego poses. Use
`--camera_index_base` for official zero-indexed cameras or legacy one-indexed
exports. Processed cameras always use `0..3`.

## 2. Convert a split and generate masks

```bash
python datasets/preprocess_v2xreal.py \
    --data_root data/v2xreal_raw/train \
    --target_dir data/v2xreal/train --split train

python scripts/data_process/generate_masks_v2xreal_smp.py \
    --data_root data/v2xreal/train --device cuda:0
```

Repeat for `val`. For raw cameras numbered `1..4`, add `--camera_index_base 1`.
The official download names the validation split `validate`; use that raw path
and save it to the processed `val` directory. The all-split shell script handles
this alias automatically.
Use a fresh target directory when changing the indexing convention: conversion
resumes populated scenes. Masks use
`smp-hub/segformer-b5-1024x1024-city-160k`, downloaded on first use, with ImageNet
normalization and padding to multiples of 32. Output is restored to the original
image size. Cityscapes class 10 is sky; classes 11–18 are movable foreground.
Mask values are 0/255.

For all available splits:

```bash
bash scripts/data_process/prepare_v2xreal_dataset.sh \
    data/v2xreal_raw data/v2xreal cuda:0 0
```

Output:

```text
data/v2xreal/train/<scene>_1/
  images/000000_0.jpg
  intrinsics/0.txt                # Original 3×3 matrix
  extrinsics/0.txt                # Camera-to-vehicle 4×4 transform
  ego_pose/000000.txt             # Preserved per-frame pose
  lidar/                         # Optional original LiDAR
  frame_info.json
  sky_masks/000000_0.png
  fine_dynamic_masks/all/000000_0.png
  context.json                   # Add in the next step
```

The same layout is created for `<scene>_2`. Camera matrices and LiDAR are metadata
for offline inspection, not FRUC network inputs. Rendering uses predicted camera
parameters. Dynamic masks supervise training; FRUC predicts its own dynamic map
at inference.

## 3. Verify cross-agent view associations

The cooperative loader needs `context.json` to designate the ego agent and select
overlapping collaborator cameras. It uses consecutive frame ids with consistent
ego designation and associations. Missing/empty contexts yield no samples. Check
image overlap before assigning pairs; matching camera ids alone do not establish
overlap.

For verified pairs that remain valid throughout a scene:

```bash
python scripts/data_process/create_v2xreal_context.py \
    --data_root data/v2xreal/train --scene YOUR_SCENE_PREFIX \
    --ego_agent agent1 --pairs 0:0 1:2
```

This example maps ego camera 0 to collaborator camera 0, and ego camera 1 to
collaborator camera 2. Replace them with verified pairs. The script includes only
shared image frames and preserves existing contexts unless `--overwrite` is
supplied. For time-varying associations, run the annotation interface:

```bash
V2XREAL_ROOT=data/v2xreal python scripts/data_process/app.py
```

Open `http://127.0.0.1:5000`, choose split/scene/frame, select the ego vehicle, then
select ego and overlapping collaborator cameras and save. Annotate both timestamps
of each desired sample. Existing verified contexts can also be copied into the
corresponding processed agent folders.

Example:

```json
{
  "000000": {"ego": "agent1", "mapping": {"0": [0], "1": [2]}},
  "000001": {"ego": "agent1", "mapping": {"0": [0], "1": [2]}}
}
```

Context selects images offline. It is not fed to the Transformer and does not
supply calibration. Use the paper's verified associations for benchmark
reproduction; the benchmark metadata is not bundled here.

## 4. Check before training

Check both agent folders, frame ids, camera ids `0..3`, and a sky/dynamic mask for
every selected image. Samples follow `[Ego_t0, Collab_t0, Ego_t1, Collab_t1]` for
two adjacent frame ids. `--cam0_only` selects camera 0 for both agents and requires
verified overlap. Benchmark evaluation scripts are excluded from this release.
