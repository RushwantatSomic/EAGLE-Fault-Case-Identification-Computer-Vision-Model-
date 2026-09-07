import os
import json
import cv2
import numpy as np
import pandas as pd
import torch

from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH = r"outputs\training\best_model"

VIDEO_DIR = r"data\annotated_videos"

OUTPUT_DIR = r"outputs\inference"

NUM_FRAMES = 16
IMAGE_SIZE = 224

# Sliding-window configuration
WINDOW_STRIDE = 8

# Test videos and their LOCAL CVAT frame ranges.
#
# These are the frame ranges established from:
# outputs/dataset/training_segments.csv
#
# NORMAL = 0
# FAULT  = 1
#
TEST_RANGES = {
    "Cartoning 2.mp4": [
        (0, 349, 0),
        (350, 556, 1),
    ],

    "Cartoning.mp4": [
        (0, 424, 0),
        (425, 425, 1),
    ],

    "covering.mp4": [
        (0, 41, 1),
        (42, 71, 0),
        (72, 204, 1),
        (205, 213, 0),
        (214, 514, 1),
    ],
}


# =============================================================================
# HELPERS
# =============================================================================

def frame_ground_truth(frame_index, ranges):
    """
    Return the ground-truth label for a local recovered-video frame.

    0 = NORMAL
    1 = FAULT
    """
    for start, end, label in ranges:
        if start <= frame_index <= end:
            return label

    return None


def load_video_frames(video_path):
    """
    Load the complete video into memory.

    The recovered videos are only ~20 seconds long, so this is acceptable
    for the current evaluation set.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # OpenCV BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame)

    cap.release()

    return frames, fps


def prepare_clip(frames, processor):
    """
    Convert 16 RGB frames into VideoMAE input tensor.
    """
    clip = frames[:NUM_FRAMES]

    if len(clip) != NUM_FRAMES:
        raise ValueError(
            f"Expected {NUM_FRAMES} frames, got {len(clip)}"
        )

    inputs = processor(
        list(clip),
        return_tensors="pt"
    )

    return inputs["pixel_values"]


def predict_clip(model, processor, frames, device):
    """
    Run VideoMAE inference on one 16-frame clip.
    """
    pixel_values = prepare_clip(frames, processor)

    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    # VideoMAE label mapping from the trained 2-class model:
    # id 0 = NORMAL
    # id 1 = FAULT
    #
    # We explicitly use probabilities rather than relying on the model's
    # id2label configuration.

    normal_probability = float(probabilities[0].item())
    fault_probability = float(probabilities[1].item())

    prediction = 1 if fault_probability >= normal_probability else 0

    confidence = max(
        normal_probability,
        fault_probability
    )

    return (
        prediction,
        confidence,
        normal_probability,
        fault_probability,
    )


def evaluate_video(
    video_name,
    model,
    processor,
    device,
):
    """
    Evaluate one complete test video.
    """

    video_path = os.path.join(
        VIDEO_DIR,
        video_name
    )

    print()
    print("=" * 100)
    print(f"VIDEO: {video_name}")
    print("=" * 100)

    if not os.path.exists(video_path):
        print(f"ERROR: Video not found:")
        print(video_path)
        return []

    ranges = TEST_RANGES[video_name]

    frames, fps = load_video_frames(video_path)

    total_frames = len(frames)

    print(f"Video path : {video_path}")
    print(f"FPS        : {fps:.2f}")
    print(f"Frames     : {total_frames}")
    print(f"Duration   : {total_frames / fps:.2f}s")

    print()
    print("GROUND TRUTH RANGES")
    print("-" * 100)

    for start, end, label in ranges:
        label_name = "FAULT" if label == 1 else "NORMAL"
        print(
            f"{start:4d} - {end:4d} : {label_name}"
        )

    results = []

    window_index = 0

    for start in range(
        0,
        total_frames - NUM_FRAMES + 1,
        WINDOW_STRIDE
    ):

        end = start + NUM_FRAMES - 1

        clip_frames = frames[start:end + 1]

        (
            prediction,
            confidence,
            normal_probability,
            fault_probability,
        ) = predict_clip(
            model,
            processor,
            clip_frames,
            device
        )

        # Determine ground truth using the CENTER frame of the window.
        center_frame = start + (NUM_FRAMES // 2)

        ground_truth = frame_ground_truth(
            center_frame,
            ranges
        )

        if ground_truth is None:
            # Window falls outside known annotation range.
            continue

        prediction_name = (
            "FAULT"
            if prediction == 1
            else "NORMAL"
        )

        ground_truth_name = (
            "FAULT"
            if ground_truth == 1
            else "NORMAL"
        )

        correct = prediction == ground_truth

        print(
            f"{start / fps:6.2f}s | "
            f"frames {start:4d}-{end:4d} | "
            f"GT={ground_truth_name:<6} | "
            f"PRED={prediction_name:<6} | "
            f"FAULT={fault_probability:.3f} | "
            f"{'OK' if correct else 'WRONG'}"
        )

        results.append({
            "video": video_name,
            "window": window_index,
            "start_frame": start,
            "end_frame": end,
            "center_frame": center_frame,
            "time_seconds": start / fps,
            "ground_truth": ground_truth,
            "ground_truth_label": ground_truth_name,
            "prediction": prediction,
            "prediction_label": prediction_name,
            "confidence": confidence,
            "normal_probability": normal_probability,
            "fault_probability": fault_probability,
            "correct": correct,
        })

        window_index += 1

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print("EAGLE-MARS TEST VIDEO EVALUATION")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("DEVICE")
    print("-" * 100)
    print(f"Device : {device}")

    if torch.cuda.is_available():
        print(
            f"GPU    : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"VRAM   : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    # -------------------------------------------------------------------------
    # Check model
    # -------------------------------------------------------------------------

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    print()
    print("MODEL")
    print("-" * 100)
    print(f"Path : {MODEL_PATH}")

    # -------------------------------------------------------------------------
    # Load processor
    # -------------------------------------------------------------------------

    print()
    print("Loading VideoMAE processor...")

    processor = VideoMAEImageProcessor.from_pretrained(
        MODEL_PATH
    )

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------

    print("Loading VideoMAE model...")

    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_PATH
    )

    model.to(device)
    model.eval()

    # -------------------------------------------------------------------------
    # Evaluate videos
    # -------------------------------------------------------------------------

    all_results = []

    for video_name in TEST_RANGES.keys():

        results = evaluate_video(
            video_name,
            model,
            processor,
            device
        )

        all_results.extend(results)

    if not all_results:
        print()
        print("ERROR: No evaluation results were generated.")
        return

    # -------------------------------------------------------------------------
    # DataFrame
    # -------------------------------------------------------------------------

    df = pd.DataFrame(all_results)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    csv_path = os.path.join(
        OUTPUT_DIR,
        "test_video_evaluation.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    # -------------------------------------------------------------------------
    # Overall metrics
    # -------------------------------------------------------------------------

    y_true = df["ground_truth"].astype(int)
    y_pred = df["prediction"].astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    # -------------------------------------------------------------------------
    # Confusion matrix values
    # -------------------------------------------------------------------------

    tn, fp, fn, tp = cm.ravel()

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )

    # -------------------------------------------------------------------------
    # Print final results
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("OVERALL TEST RESULTS")
    print("=" * 100)

    print()
    print(f"Windows evaluated : {len(df)}")
    print(f"Correct windows   : {int(df.correct.sum())}")
    print(f"Wrong windows     : {int((~df.correct).sum())}")

    print()
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1                : {f1:.4f}")

    print()
    print(f"False Positive Rate : {false_positive_rate:.4f}")
    print(f"False Negative Rate : {false_negative_rate:.4f}")

    print()
    print("CONFUSION MATRIX")
    print("-" * 100)
    print()
    print("                 Predicted")
    print("              NORMAL  FAULT")
    print(
        f"Actual NORMAL  {tn:6d} {fp:6d}"
    )
    print(
        f"Actual FAULT   {fn:6d} {tp:6d}"
    )

    # -------------------------------------------------------------------------
    # Classification report
    # -------------------------------------------------------------------------

    print()
    print("CLASSIFICATION REPORT")
    print("-" * 100)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "NORMAL",
                "FAULT"
            ],
            zero_division=0
        )
    )

    # -------------------------------------------------------------------------
    # Per-video results
    # -------------------------------------------------------------------------

    print()
    print("PER-VIDEO RESULTS")
    print("-" * 100)

    for video_name in TEST_RANGES.keys():

        video_df = df[
            df["video"] == video_name
        ]

        if len(video_df) == 0:
            continue

        video_true = video_df[
            "ground_truth"
        ].astype(int)

        video_pred = video_df[
            "prediction"
        ].astype(int)

        video_accuracy = accuracy_score(
            video_true,
            video_pred
        )

        video_precision = precision_score(
            video_true,
            video_pred,
            zero_division=0
        )

        video_recall = recall_score(
            video_true,
            video_pred,
            zero_division=0
        )

        video_f1 = f1_score(
            video_true,
            video_pred,
            zero_division=0
        )

        print()
        print(video_name)
        print(
            f"  Windows   : {len(video_df)}"
        )
        print(
            f"  Accuracy  : {video_accuracy:.4f}"
        )
        print(
            f"  Precision : {video_precision:.4f}"
        )
        print(
            f"  Recall    : {video_recall:.4f}"
        )
        print(
            f"  F1        : {video_f1:.4f}"
        )

        print(
            f"  GT FAULT windows   : "
            f"{int(video_true.sum())}"
        )

        print(
            f"  Pred FAULT windows : "
            f"{int(video_pred.sum())}"
        )

    # -------------------------------------------------------------------------
    # Average probabilities
    # -------------------------------------------------------------------------

    print()
    print("AVERAGE MODEL PROBABILITIES")
    print("-" * 100)

    print(
        f"Average NORMAL probability : "
        f"{df.normal_probability.mean():.4f}"
    )

    print(
        f"Average FAULT probability  : "
        f"{df.fault_probability.mean():.4f}"
    )

    # -------------------------------------------------------------------------
    # Save metrics
    # -------------------------------------------------------------------------

    metrics = {
        "model": MODEL_PATH,

        "num_windows": int(len(df)),

        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),

        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),

        "false_positive_rate": float(
            false_positive_rate
        ),

        "false_negative_rate": float(
            false_negative_rate
        ),

        "average_normal_probability": float(
            df.normal_probability.mean()
        ),

        "average_fault_probability": float(
            df.fault_probability.mean()
        ),
    }

    metrics_path = os.path.join(
        OUTPUT_DIR,
        "test_video_evaluation_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    # -------------------------------------------------------------------------
    # Complete
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("TEST VIDEO EVALUATION COMPLETE")
    print("=" * 100)

    print()
    print(f"Predictions : {csv_path}")
    print(f"Metrics     : {metrics_path}")

    print()


if __name__ == "__main__":
    main()