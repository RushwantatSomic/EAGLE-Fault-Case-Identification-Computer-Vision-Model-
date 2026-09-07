import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
)


# ============================================================
# EAGLE-MARS — TEST TRAINED VIDEOMAE ON A REAL VIDEO
# ============================================================

MODEL_DIR = Path(
    "outputs/training/best_model"
)

VIDEO_PATH = Path(
    r"data\annotated_videos\Cartoning 2.mp4"
)

OUTPUT_DIR = Path(
    "outputs\inference"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CSV = (
    OUTPUT_DIR /
    "Cartoning_2_predictions.csv"
)

NUM_FRAMES = 16

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

ID2LABEL = {
    0: "NORMAL",
    1: "FAULT",
}


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 100)
print("EAGLE-MARS VIDEOMAE VIDEO TEST")
print("=" * 100)

print()
print("MODEL")
print("-" * 100)
print(f"Path   : {MODEL_DIR}")
print(f"Device : {DEVICE}")

if not MODEL_DIR.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_DIR}"
    )

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )

print()
print("Loading processor...")

processor = (
    VideoMAEImageProcessor
    .from_pretrained(
        MODEL_DIR
    )
)

print("Loading model...")

model = (
    VideoMAEForVideoClassification
    .from_pretrained(
        MODEL_DIR
    )
)

model.to(DEVICE)
model.eval()


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )

fps = cap.get(
    cv2.CAP_PROP_FPS
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

duration = (
    total_frames / fps
    if fps > 0
    else 0
)

print()
print("VIDEO")
print("-" * 100)

print(
    f"File     : {VIDEO_PATH.name}"
)

print(
    f"FPS      : {fps:.2f}"
)

print(
    f"Frames   : {total_frames}"
)

print(
    f"Duration : {duration:.2f} seconds"
)

print()


# ============================================================
# SLIDING WINDOW INFERENCE
# ============================================================

results = []

window_start = 0

with torch.no_grad():

    while (
        window_start + NUM_FRAMES
        <= total_frames
    ):

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            window_start
        )

        frames = []

        for _ in range(
            NUM_FRAMES
        ):

            ret, frame = (
                cap.read()
            )

            if not ret:
                break

            # OpenCV BGR -> RGB
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frames.append(
                frame
            )

        if len(frames) != NUM_FRAMES:
            break

        inputs = processor(
            frames,
            return_tensors="pt"
        )

        pixel_values = (
            inputs["pixel_values"]
            .to(DEVICE)
        )

        outputs = model(
            pixel_values=pixel_values
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

        predicted_id = int(
            torch.argmax(
                probabilities
            )
        )

        predicted_label = (
            ID2LABEL[
                predicted_id
            ]
        )

        confidence = float(
            probabilities[
                predicted_id
            ]
        )

        normal_probability = float(
            probabilities[0]
        )

        fault_probability = float(
            probabilities[1]
        )

        end_frame = (
            window_start
            + NUM_FRAMES
            - 1
        )

        timestamp = (
            window_start / fps
            if fps > 0
            else 0
        )

        results.append(
            {
                "start_frame": window_start,
                "end_frame": end_frame,
                "timestamp_seconds": timestamp,
                "prediction": predicted_label,
                "confidence": confidence,
                "normal_probability":
                    normal_probability,
                "fault_probability":
                    fault_probability,
            }
        )

        print(
            f"{timestamp:6.2f}s | "
            f"frames "
            f"{window_start:4d}-"
            f"{end_frame:4d} | "
            f"{predicted_label:6s} | "
            f"confidence "
            f"{confidence:.3f} | "
            f"FAULT "
            f"{fault_probability:.3f}"
        )

        # Move by 8 frames = approximately 0.27 s.
        window_start += 8


cap.release()


# ============================================================
# SAVE CSV
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "start_frame",
            "end_frame",
            "timestamp_seconds",
            "prediction",
            "confidence",
            "normal_probability",
            "fault_probability",
        ],
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# SUMMARY
# ============================================================

total_windows = len(
    results
)

fault_windows = sum(
    r["prediction"] == "FAULT"
    for r in results
)

normal_windows = (
    total_windows
    - fault_windows
)

fault_percentage = (
    100.0 * fault_windows
    / total_windows
    if total_windows
    else 0
)

print()
print("=" * 100)
print("INFERENCE SUMMARY")
print("=" * 100)

print(
    f"Windows tested   : "
    f"{total_windows}"
)

print(
    f"NORMAL windows   : "
    f"{normal_windows}"
)

print(
    f"FAULT windows    : "
    f"{fault_windows}"
)

print(
    f"FAULT percentage : "
    f"{fault_percentage:.2f}%"
)

if results:

    avg_fault_probability = np.mean(
        [
            r["fault_probability"]
            for r in results
        ]
    )

    print(
        f"Average FAULT probability : "
        f"{avg_fault_probability:.4f}"
    )

print()
print(
    f"Predictions saved:"
)
print(
    OUTPUT_CSV
)

print()
print("=" * 100)
print("VIDEO TEST COMPLETE")
print("=" * 100)