"""Write explicitly verified ego/collaborator camera associations for V2X-Real.

These files select images; they are not camera calibration or network inputs.
"""
import argparse
import json
from pathlib import Path


def image_frames(agent_dir, camera_id):
    frames = set()
    for path in (agent_dir / "images").iterdir():
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        parts = path.stem.rsplit("_", 1)
        if len(parts) == 2 and parts[0].isdigit() and parts[1] == str(camera_id):
            frames.add(int(parts[0]))
    return frames


def parse_pairs(values):
    mapping = {}
    for value in values:
        try:
            ego, collab = value.split(":")
            cameras = [int(camera) for camera in collab.split(",")]
            ego = int(ego)
        except ValueError as error:
            raise ValueError(f"Invalid pair {value!r}; use EGO_CAM:COLLAB_CAM[,COLLAB_CAM].") from error
        if ego not in range(4) or any(camera not in range(4) for camera in cameras):
            raise ValueError("Camera ids must be in 0..3.")
        if str(ego) in mapping:
            raise ValueError(f"Duplicate ego camera {ego}.")
        mapping[str(ego)] = sorted(set(cameras))
    return mapping


def create_context(data_root, scene, ego_agent, mapping, overwrite=False):
    agents = [data_root / f"{scene}_1", data_root / f"{scene}_2"]
    for agent in agents:
        if not (agent / "images").is_dir():
            raise FileNotFoundError(f"Missing images directory: {agent / 'images'}")
        if (agent / "context.json").exists() and not overwrite:
            raise FileExistsError(f"Preserving {agent / 'context.json'}; use --overwrite to replace it.")
    ego_index = 0 if ego_agent == "agent1" else 1
    valid_by_pair = {}
    for ego_camera, collab_cameras in mapping.items():
        frames = image_frames(agents[ego_index], int(ego_camera))
        for collab_camera in collab_cameras:
            frames &= image_frames(agents[1 - ego_index], collab_camera)
        valid_by_pair[ego_camera] = frames
    all_frames = sorted(set().union(*valid_by_pair.values()))
    context = {
        f"{frame:06d}": {
            "ego": ego_agent,
            "mapping": {camera: mapping[camera] for camera, frames in valid_by_pair.items() if frame in frames},
        }
        for frame in all_frames
    }
    if not context:
        raise ValueError(f"No shared image frames for scene {scene} and the selected camera pairs.")
    for agent in agents:
        (agent / "context.json").write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")
    return len(context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True, help="Processed split directory, e.g. data/v2xreal/train")
    parser.add_argument("--scene", required=True, help="Shared scene prefix without the final agent suffix")
    parser.add_argument("--ego_agent", choices=["agent1", "agent2"], required=True)
    parser.add_argument("--pairs", nargs="+", required=True, help="Verified view pairs, e.g. 0:0 1:2,3")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing view associations")
    args = parser.parse_args()
    count = create_context(args.data_root, args.scene, args.ego_agent, parse_pairs(args.pairs), args.overwrite)
    print(f"Saved associations for {count} shared frames in {args.scene}.")


if __name__ == "__main__":
    main()
