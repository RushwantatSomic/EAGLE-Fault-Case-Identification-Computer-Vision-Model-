import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import random
import shutil

SEGMENTS = Path("outputs/dataset/training_segments.csv")
VIDEO_DIR = Path("data/annotated_videos")
OUTPUT_DIR = Path("outputs/dataset/clips")

CLIP_LENGTH = 16
IMAGE_SIZE = 224

# Number of clips sampled from long segments.
# Keeping this modest prevents the same video from dominating training.
MAX_CLIPS_PER_SEGMENT = 30

SEED = 42
random.seed(SEED)


def get_video_info(path):
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return fps, frame_count


def make_clip(cap, frame_indices):
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()

        if not ok:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(
            frame,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        frames.append(frame)

    if len(frames) != CLIP_LENGTH:
        return None

    return np.stack(frames)


def save_clip(frames, output_path):
    # Save as .npy for fast loading during training.
    np.save(output_path, frames.astype(np.uint8))


def main():

    print("=" * 100)
    print("EAGLE-MARS TRAINING CLIP GENERATOR")
    print("=" * 100)

    if not SEGMENTS.exists():
        raise FileNotFoundError(SEGMENTS)

    df = pd.read_csv(SEGMENTS)

    # Clean output directory.
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val", "test"]:
        for label in ["NORMAL", "FAULT"]:
            (OUTPUT_DIR / split / label).mkdir(
                parents=True,
                exist_ok=True
            )

    # ---------------------------------------------------------
    # Split by VIDEO, never by clip.
    # This prevents frames from the same video appearing in
    # both training and validation/test.
    # ---------------------------------------------------------

    videos = sorted(df["video"].unique())

    random.shuffle(videos)

    n = len(videos)

    n_train = max(1, round(n * 0.70))
    n_val = max(1, round(n * 0.15))

    train_videos = videos[:n_train]
    val_videos = videos[n_train:n_train + n_val]
    test_videos = videos[n_train + n_val:]

    # Make sure test is not empty.
    if not test_videos:
        test_videos = [val_videos.pop()]

    split_map = {}

    for v in train_videos:
        split_map[v] = "train"

    for v in val_videos:
        split_map[v] = "val"

    for v in test_videos:
        split_map[v] = "test"

    print()
    print("VIDEO SPLIT")
    print("-" * 100)

    print("TRAIN:")
    for v in train_videos:
        print(" ", v)

    print("\nVAL:")
    for v in val_videos:
        print(" ", v)

    print("\nTEST:")
    for v in test_videos:
        print(" ", v)

    print()

    manifest_rows = []

    clip_id = 0

    # ---------------------------------------------------------
    # Process each video.
    # ---------------------------------------------------------

    for video, group in df.groupby("video"):

        video_path = VIDEO_DIR / video

        if not video_path.exists():
            print(f"WARNING: missing {video_path}")
            continue

        split = split_map[video]

        fps, recovered_count = get_video_info(video_path)

        # Original local frame count comes from the annotation timeline.
        original_max = int(
            group["end_frame_local"].max()
        )

        original_count = original_max + 1

        print(
            f"{video}: "
            f"original={original_count} frames | "
            f"recovered={recovered_count} frames | "
            f"split={split}"
        )

        cap = cv2.VideoCapture(str(video_path))

        for _, row in group.iterrows():

            label = row["label"]

            start_original = int(row["start_frame_local"])
            end_original = int(row["end_frame_local"])

            duration = end_original - start_original + 1

            if duration < CLIP_LENGTH:
                continue

            # -------------------------------------------------
            # Map original CVAT local frame coordinates onto
            # recovered screen-recording frame coordinates.
            # -------------------------------------------------

            start_recovered = round(
                start_original *
                (recovered_count - 1) /
                max(original_count - 1, 1)
            )

            end_recovered = round(
                end_original *
                (recovered_count - 1) /
                max(original_count - 1, 1)
            )

            recovered_duration = (
                end_recovered - start_recovered + 1
            )

            if recovered_duration < CLIP_LENGTH:
                continue

            # Generate evenly spaced starting positions.
            possible_starts = list(
                range(
                    start_recovered,
                    end_recovered - CLIP_LENGTH + 2,
                    CLIP_LENGTH
                )
            )

            if not possible_starts:
                continue

            # Limit samples from each segment.
            if len(possible_starts) > MAX_CLIPS_PER_SEGMENT:
                possible_starts = random.sample(
                    possible_starts,
                    MAX_CLIPS_PER_SEGMENT
                )

            for clip_start in sorted(possible_starts):

                frame_indices = list(
                    range(
                        clip_start,
                        clip_start + CLIP_LENGTH
                    )
                )

                frames = make_clip(cap, frame_indices)

                if frames is None:
                    continue

                filename = (
                    f"{clip_id:06d}_"
                    f"{label}_"
                    f"{video_path.stem}.npy"
                )

                output_path = (
                    OUTPUT_DIR /
                    split /
                    label /
                    filename
                )

                save_clip(frames, output_path)

                manifest_rows.append({
                    "clip_id": clip_id,
                    "video": video,
                    "split": split,
                    "label": label,
                    "start_frame_recovered": clip_start,
                    "end_frame_recovered": (
                        clip_start + CLIP_LENGTH - 1
                    ),
                    "fps_recovered": fps,
                    "frames": CLIP_LENGTH,
                    "file": str(output_path),
                })

                clip_id += 1

        cap.release()

    manifest = pd.DataFrame(manifest_rows)

    manifest_path = OUTPUT_DIR / "clips_manifest.csv"

    manifest.to_csv(
        manifest_path,
        index=False
    )

    print()
    print("=" * 100)
    print("CLIP GENERATION COMPLETE")
    print("=" * 100)

    print(f"Total clips: {len(manifest)}")
    print()

    if not manifest.empty:
        print("BY SPLIT:")
        print(manifest.groupby("split").size())
        print()

        print("BY LABEL:")
        print(manifest.groupby("label").size())
        print()

        print("SPLIT × LABEL:")
        print(
            manifest.groupby(
                ["split", "label"]
            ).size()
        )

    print()
    print(f"Output: {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()