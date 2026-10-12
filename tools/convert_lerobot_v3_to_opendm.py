import json
from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

SRC = Path("/home/je/code/lerobot/dataset/cr5_o6_20250917")
DST = Path("/home/je/code/opendm/data/cr5_o6_20250917")

JSONL_DIR = DST / "jsonl"

INFO_PATH = SRC / "meta/info.json"
EPISODES_PATH = SRC / "meta/episodes/chunk-000/file-000.parquet"
DATA_PATH = SRC / "data/chunk-000/file-000.parquet"


# ============================================================
# Config
# ============================================================

FPS = 20

CAMERAS = {
    "images_1": "observation.images.base_0_rgb",
    "images_2": "observation.images.left_wrist_0_rgb",
    "images_3": "observation.images.right_wrist_0_rgb",
}


# ============================================================
# Load metadata
# ============================================================

with open(INFO_PATH, "r") as f:
    info = json.load(f)

assert info["fps"] == FPS

episodes = pd.read_parquet(EPISODES_PATH)
data = pd.read_parquet(DATA_PATH)

print("Source dataset:")
print(f"  episodes : {len(episodes)}")
print(f"  frames   : {len(data)}")
print(f"  fps      : {FPS}")


# ============================================================
# Prepare output
# ============================================================

JSONL_DIR.mkdir(parents=True, exist_ok=True)

video_link = DST / "videos"

if not video_link.exists():
    video_link.symlink_to(SRC / "videos", target_is_directory=True)


# ============================================================
# Convert
# ============================================================

total_frames = 0

for _, ep in episodes.iterrows():

    episode_index = int(ep["episode_index"])

    start = int(ep["dataset_from_index"])
    end = int(ep["dataset_to_index"])

    ep_data = data.iloc[start:end]

    expected_length = int(ep["length"])

    assert len(ep_data) == expected_length, (
        f"Episode {episode_index}: "
        f"expected {expected_length} frames, "
        f"got {len(ep_data)}"
    )

    # --------------------------------------------------------
    # Task / prompt
    # --------------------------------------------------------

    tasks = ep["tasks"]

    if isinstance(tasks, str):
        try:
            tasks = json.loads(tasks)
        except Exception:
            pass

    if isinstance(tasks, (list, tuple)):
        prompt = str(tasks[0])
    else:
        prompt = str(tasks)

    # --------------------------------------------------------
    # OpenDM JSONL
    # --------------------------------------------------------

    output_path = (
        JSONL_DIR /
        f"episode_{episode_index:06d}.jsonl"
    )

    # --------------------------------------------------------
    # Video metadata for this episode
    # --------------------------------------------------------

    video_meta = {}

    for image_key, lerobot_key in CAMERAS.items():

        file_index = int(
            ep[f"videos/{lerobot_key}/file_index"]
        )

        from_timestamp = float(
            ep[f"videos/{lerobot_key}/from_timestamp"]
        )

        video_path = (
            f"{lerobot_key}/"
            f"chunk-000/"
            f"file-{file_index:03d}.mp4"
        )

        video_meta[image_key] = {
            "url": video_path,
            "from_timestamp": from_timestamp,
        }

    # --------------------------------------------------------
    # Write episode
    # --------------------------------------------------------

    with open(output_path, "w") as f:

        for _, row in ep_data.iterrows():

            timestamp = float(row["timestamp"])

            frame = {}

            # ------------------------------------------------
            # Images
            # ------------------------------------------------

            for image_key, meta in video_meta.items():

                frame_idx = int(
                    round(
                        (
                            meta["from_timestamp"]
                            + timestamp
                        )
                        * FPS
                    )
                )

                frame[image_key] = {
                    "type": "video",
                    "url": meta["url"],
                    "frame_idx": frame_idx,
                }

            # ------------------------------------------------
            # State
            # ------------------------------------------------

            state = row["observation.state"]

            if hasattr(state, "tolist"):
                state = state.tolist()

            state = [float(x) for x in state]

            assert len(state) == 12

            frame["state"] = state

            # ------------------------------------------------
            # Action
            # ------------------------------------------------

            action = row["action"]

            if hasattr(action, "tolist"):
                action = action.tolist()

            action = [float(x) for x in action]

            assert len(action) == 12

            frame["action"] = action

            # ------------------------------------------------
            # Prompt
            # ------------------------------------------------

            frame["prompt"] = prompt

            # ------------------------------------------------
            # Robot flag
            # ------------------------------------------------

            frame["is_robot"] = True

            # ------------------------------------------------
            # Write JSONL
            # ------------------------------------------------

            f.write(
                json.dumps(
                    frame,
                    ensure_ascii=False,
                )
                + "\n"
            )

    total_frames += len(ep_data)

    print(
        f"episode {episode_index:03d}: "
        f"{len(ep_data):4d} frames -> "
        f"{output_path.name}"
    )


# ============================================================
# Final validation
# ============================================================

assert total_frames == len(data)

print()
print("=" * 60)
print("Conversion finished")
print("=" * 60)
print(f"episodes : {len(episodes)}")
print(f"frames   : {total_frames}")
print(f"output   : {DST}")
