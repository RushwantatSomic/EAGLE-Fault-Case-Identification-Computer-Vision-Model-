# scripts/train_continual_final.py

import os
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONTINUAL_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "continual_learning"
    / "continual_annotations.json"
)

EXP4_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_exp4"
    / "clips_manifest.csv"
)

START_MODEL = (
    PROJECT_ROOT
    / "outputs"
    / "continual_learning"
    / "current_model"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "continual_learning"
    / "final_model"
)

MODEL_NAME = "MCG-NJU/videomae-small-finetuned-kinetics"

NUM_FRAMES = 16
IMAGE_SIZE = 224

BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION

LEARNING_RATE = 5e-6
EPOCHS = 5

EARLY_STOPPING = False

FAULT_THRESHOLD = 0.72

NUM_WORKERS = 0

SEED = 42


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =============================================================================
# DEVICE
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# HELPERS
# =============================================================================

def print_header(title):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def verify_path(path, description):
    if not path.exists():
        raise FileNotFoundError(
            f"\n{description} not found:\n{path}"
        )


# =============================================================================
# LOAD EXP4 DATASET
# =============================================================================

def load_exp4_manifest():

    verify_path(
        EXP4_MANIFEST,
        "Experiment 4 manifest"
    )

    df = pd.read_csv(EXP4_MANIFEST)

    if "file" not in df.columns:
        raise ValueError("Manifest does not contain 'file' column.")

    if "label" not in df.columns:
        raise ValueError("Manifest does not contain 'label' column.")

    print_header("EXPERIMENT 4 DATASET")

    print(df.groupby(["split", "label"]).size())
    print()
    print("Total clips:", len(df))

    return df


# =============================================================================
# LOAD CONTINUAL ANNOTATIONS
# =============================================================================

def load_continual_annotations():

    verify_path(
        CONTINUAL_JSON,
        "Continual annotation JSON"
    )

    with open(CONTINUAL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "continual_annotations.json must contain a list."
        )

    print_header("CONTINUAL DATASET")

    print("Videos:", len(data))

    total_windows = 0
    total_fault = 0
    total_normal = 0

    for item in data:

        windows = item.get("windows", [])

        total_windows += len(windows)

        for w in windows:

            if int(w["label"]) == 1:
                total_fault += 1
            else:
                total_normal += 1

    print("Windows:", total_windows)
    print("NORMAL :", total_normal)
    print("FAULT  :", total_fault)

    return data


# =============================================================================
# VIDEO WINDOW DATASET
# =============================================================================

class ContinualVideoDataset(Dataset):

    def __init__(
        self,
        annotation_data,
        processor,
        num_frames=16,
    ):

        self.processor = processor
        self.num_frames = num_frames

        self.samples = []

        for video_entry in annotation_data:

            video_path = Path(video_entry["video"])

            if not video_path.exists():

                print(
                    f"WARNING: video not found:\n"
                    f"{video_path}"
                )

                continue

            for window in video_entry.get("windows", []):

                start_frame = int(window["start_frame"])
                end_frame = int(window["end_frame"])
                label = int(window["label"])

                self.samples.append(
                    {
                        "video": str(video_path),
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "label": label,
                    }
                )

        print(
            f"Usable continual-learning windows: "
            f"{len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def read_frames(self, video_path, start_frame, end_frame):

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video:\n{video_path}"
            )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            start_frame
        )

        frames = []

        for _ in range(self.num_frames):

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frame = cv2.resize(
                frame,
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            frames.append(frame)

        cap.release()

        if len(frames) == 0:

            raise RuntimeError(
                f"No frames read from:\n"
                f"{video_path}\n"
                f"{start_frame}-{end_frame}"
            )

        # Repeat last frame if necessary.
        while len(frames) < self.num_frames:
            frames.append(frames[-1].copy())

        return np.stack(frames)

    def __getitem__(self, index):

        sample = self.samples[index]

        frames = self.read_frames(
            sample["video"],
            sample["start_frame"],
            sample["end_frame"],
        )

        inputs = self.processor(
            list(frames),
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(
                sample["label"],
                dtype=torch.long
            ),
        }


# =============================================================================
# EXP4 CLIP DATASET
# =============================================================================

class Exp4ClipDataset(Dataset):

    def __init__(
        self,
        dataframe,
        processor,
    ):

        self.df = dataframe.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        path = Path(row["file"])

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        frames = np.load(path)

        if frames.shape[0] != NUM_FRAMES:

            raise RuntimeError(
                f"Invalid clip shape: "
                f"{frames.shape}"
            )

        inputs = self.processor(
            list(frames),
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"].squeeze(0)

        label = 1 if row["label"] == "FAULT" else 0

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(
                label,
                dtype=torch.long
            ),
        }


# =============================================================================
# COMBINED TRAINING DATASET
# =============================================================================

class CombinedDataset(Dataset):

    def __init__(
        self,
        exp4_dataset,
        continual_dataset,
    ):

        self.exp4_dataset = exp4_dataset
        self.continual_dataset = continual_dataset

    def __len__(self):

        return (
            len(self.exp4_dataset)
            + len(self.continual_dataset)
        )

    def __getitem__(self, index):

        if index < len(self.exp4_dataset):

            return self.exp4_dataset[index]

        return self.continual_dataset[
            index - len(self.exp4_dataset)
        ]


# =============================================================================
# VALIDATION DATASET
# =============================================================================

class ValidationDataset(Dataset):

    def __init__(
        self,
        dataframe,
        processor,
    ):

        self.dataset = Exp4ClipDataset(
            dataframe,
            processor,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(
    model,
    loader,
):

    model.eval()

    losses = []

    predictions = []
    labels = []

    with torch.no_grad():

        for batch in loader:

            pixel_values = batch["pixel_values"].to(
                DEVICE,
                non_blocking=True
            )

            target = batch["labels"].to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(
                pixel_values=pixel_values,
                labels=target,
            )

            losses.append(
                outputs.loss.item()
            )

            probs = torch.softmax(
                outputs.logits,
                dim=1
            )

            pred = torch.argmax(
                probs,
                dim=1
            )

            predictions.extend(
                pred.cpu().numpy().tolist()
            )

            labels.extend(
                target.cpu().numpy().tolist()
            )

    loss = float(np.mean(losses))

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        pos_label=1,
        zero_division=0
    )

    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "labels": labels,
        "predictions": predictions,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print_header(
        "EAGLE-MARS CONTINUAL LEARNING FINAL CONSOLIDATION"
    )

    print()
    print("DEVICE")
    print("-" * 90)
    print("Device :", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU    :",
            torch.cuda.get_device_name(0)
        )

        print(
            "VRAM   :",
            round(
                torch.cuda.get_device_properties(0)
                .total_memory / 1024**3,
                2
            ),
            "GB"
        )

    print()
    print("CONFIGURATION")
    print("-" * 90)
    print("Starting model       :", START_MODEL)
    print("Exp4 manifest        :", EXP4_MANIFEST)
    print("Continual annotations:", CONTINUAL_JSON)
    print("Output               :", OUTPUT_DIR)
    print("Frames               :", NUM_FRAMES)
    print("Resolution           :", f"{IMAGE_SIZE}x{IMAGE_SIZE}")
    print("Batch size           :", BATCH_SIZE)
    print("Gradient accumulation:", GRADIENT_ACCUMULATION)
    print("Effective batch      :", EFFECTIVE_BATCH_SIZE)
    print("Learning rate        :", LEARNING_RATE)
    print("Epochs               :", EPOCHS)
    print("Fault threshold      :", FAULT_THRESHOLD)

    # -------------------------------------------------------------------------
    # Verify model
    # -------------------------------------------------------------------------

    verify_path(
        START_MODEL,
        "Starting continual-learning model"
    )

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    exp4_df = load_exp4_manifest()
    continual_data = load_continual_annotations()

    # -------------------------------------------------------------------------
    # Split Exp4
    # -------------------------------------------------------------------------

    train_df = exp4_df[
        exp4_df["split"] == "train"
    ].copy()

    val_df = exp4_df[
        exp4_df["split"] == "val"
    ].copy()

    test_df = exp4_df[
        exp4_df["split"] == "test"
    ].copy()

    # -------------------------------------------------------------------------
    # Processor
    # -------------------------------------------------------------------------

    print_header("LOADING PROCESSOR")

    processor = VideoMAEImageProcessor.from_pretrained(
        MODEL_NAME
    )

    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------

    print_header("BUILDING DATASETS")

    exp4_train_dataset = Exp4ClipDataset(
        train_df,
        processor
    )

    continual_dataset = ContinualVideoDataset(
        continual_data,
        processor,
        NUM_FRAMES,
    )

    val_dataset = ValidationDataset(
        val_df,
        processor
    )

    test_dataset = ValidationDataset(
        test_df,
        processor
    )

    combined_dataset = CombinedDataset(
        exp4_train_dataset,
        continual_dataset
    )

    print()
    print("EXP4 training clips :", len(exp4_train_dataset))
    print("Continual windows   :", len(continual_dataset))
    print("Combined training   :", len(combined_dataset))
    print("Validation clips     :", len(val_dataset))
    print("Test clips           :", len(test_dataset))

    # -------------------------------------------------------------------------
    # Combined labels for sampler
    # -------------------------------------------------------------------------

    combined_labels = []

    for _, row in train_df.iterrows():

        combined_labels.append(
            1 if row["label"] == "FAULT" else 0
        )

    for sample in continual_dataset.samples:

        combined_labels.append(
            int(sample["label"])
        )

    combined_labels = np.array(
        combined_labels
    )

    normal_count = int(
        np.sum(combined_labels == 0)
    )

    fault_count = int(
        np.sum(combined_labels == 1)
    )

    print()
    print("COMBINED CLASS COUNTS")
    print("-" * 90)
    print("NORMAL:", normal_count)
    print("FAULT :", fault_count)

    # Balanced sampler
    class_counts = np.bincount(
        combined_labels,
        minlength=2
    )

    class_weights = 1.0 / np.maximum(
        class_counts,
        1
    )

    sample_weights = np.array(
        [
            class_weights[label]
            for label in combined_labels
        ],
        dtype=np.float64
    )

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            sample_weights,
            dtype=torch.double
        ),
        num_samples=len(sample_weights),
        replacement=True,
    )

    # -------------------------------------------------------------------------
    # DataLoaders
    # -------------------------------------------------------------------------

    train_loader = DataLoader(
        combined_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    print_header("LOADING STARTING MODEL")

    model = VideoMAEForVideoClassification.from_pretrained(
        START_MODEL
    )

    model.to(DEVICE)

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    # -------------------------------------------------------------------------
    # Output directory
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    best_f1 = -1.0
    best_epoch = 0

    history = []

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    print_header(
        "STARTING 5-EPOCH CONSOLIDATION TRAINING"
    )

    for epoch in range(1, EPOCHS + 1):

        model.train()

        running_loss = 0.0

        optimizer.zero_grad(
            set_to_none=True
        )

        for step, batch in enumerate(
            train_loader,
            start=1
        ):

            pixel_values = batch[
                "pixel_values"
            ].to(
                DEVICE,
                non_blocking=True
            )

            labels = batch[
                "labels"
            ].to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(
                pixel_values=pixel_values,
                labels=labels,
            )

            loss = (
                outputs.loss
                / GRADIENT_ACCUMULATION
            )

            loss.backward()

            running_loss += (
                outputs.loss.item()
            )

            if (
                step % GRADIENT_ACCUMULATION == 0
                or step == len(train_loader)
            ):

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

        train_loss = (
            running_loss
            / len(train_loader)
        )

        val_metrics = evaluate(
            model,
            val_loader
        )

        print()
        print(f"Epoch {epoch}/{EPOCHS}")
        print(
            f"Train Loss : {train_loss:.4f}"
        )
        print(
            f"Val Loss   : {val_metrics['loss']:.4f}"
        )
        print(
            f"Val Acc    : {val_metrics['accuracy']:.4f}"
        )
        print(
            f"Val Prec   : {val_metrics['precision']:.4f}"
        )
        print(
            f"Val Recall : {val_metrics['recall']:.4f}"
        )
        print(
            f"Val F1     : {val_metrics['f1']:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
            }
        )

        # Save best model
        if val_metrics["f1"] > best_f1:

            best_f1 = val_metrics["f1"]
            best_epoch = epoch

            best_dir = (
                OUTPUT_DIR / "best_model"
            )

            if best_dir.exists():
                shutil.rmtree(best_dir)

            model.save_pretrained(
                best_dir
            )

            processor.save_pretrained(
                best_dir
            )

            print()
            print("BEST MODEL SAVED")
            print(
                f"Validation F1: {best_f1:.4f}"
            )

    # -------------------------------------------------------------------------
    # Save training history
    # -------------------------------------------------------------------------

    with open(
        OUTPUT_DIR / "training_history.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )

    # -------------------------------------------------------------------------
    # Load best model
    # -------------------------------------------------------------------------

    print_header(
        "LOADING BEST CONSOLIDATED MODEL"
    )

    best_model = (
        VideoMAEForVideoClassification
        .from_pretrained(
            OUTPUT_DIR / "best_model"
        )
    )

    best_model.to(DEVICE)

    # -------------------------------------------------------------------------
    # Final test
    # -------------------------------------------------------------------------

    print_header(
        "FINAL TEST"
    )

    test_metrics = evaluate(
        best_model,
        test_loader
    )

    print(
        f"Test Loss     : "
        f"{test_metrics['loss']:.4f}"
    )

    print(
        f"Test Accuracy : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Test Recall   : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"Test F1       : "
        f"{test_metrics['f1']:.4f}"
    )

    print()
    print("CLASSIFICATION REPORT")
    print("-" * 90)

    report = classification_report(
        test_metrics["labels"],
        test_metrics["predictions"],
        target_names=[
            "NORMAL",
            "FAULT"
        ],
        zero_division=0,
    )

    print(report)

    print("CONFUSION MATRIX")
    print("-" * 90)

    cm = confusion_matrix(
        test_metrics["labels"],
        test_metrics["predictions"],
    )

    print(
        "                 Predicted"
    )

    print(
        "              NORMAL  FAULT"
    )

    print(
        f"Actual NORMAL   "
        f"{cm[0][0]:6d}  "
        f"{cm[0][1]:5d}"
    )

    print(
        f"       FAULT    "
        f"{cm[1][0]:6d}  "
        f"{cm[1][1]:5d}"
    )

    # -------------------------------------------------------------------------
    # Save metrics
    # -------------------------------------------------------------------------

    metrics_to_save = {
        "best_epoch": best_epoch,
        "best_validation_f1": best_f1,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "confusion_matrix": cm.tolist(),
        "configuration": {
            "epochs": EPOCHS,
            "frames": NUM_FRAMES,
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "fault_threshold": FAULT_THRESHOLD,
        },
        "datasets": {
            "exp4_training_clips": len(
                exp4_train_dataset
            ),
            "continual_windows": len(
                continual_dataset
            ),
            "combined_training_samples": len(
                combined_dataset
            ),
            "validation_clips": len(
                val_dataset
            ),
            "test_clips": len(
                test_dataset
            ),
        },
    }

    with open(
        OUTPUT_DIR / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics_to_save,
            f,
            indent=2
        )

    # -------------------------------------------------------------------------
    # Save predictions
    # -------------------------------------------------------------------------

    predictions_df = pd.DataFrame(
        {
            "actual": test_metrics["labels"],
            "prediction": test_metrics["predictions"],
        }
    )

    predictions_df["actual_label"] = (
        predictions_df["actual"]
        .map(
            {
                0: "NORMAL",
                1: "FAULT"
            }
        )
    )

    predictions_df["prediction_label"] = (
        predictions_df["prediction"]
        .map(
            {
                0: "NORMAL",
                1: "FAULT"
            }
        )
    )

    predictions_df.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False
    )

    # -------------------------------------------------------------------------
    # Complete
    # -------------------------------------------------------------------------

    print_header(
        "CONTINUAL LEARNING CONSOLIDATION COMPLETE"
    )

    print()
    print(
        "Best epoch       :",
        best_epoch
    )

    print(
        f"Best Val F1      : "
        f"{best_f1:.4f}"
    )

    print(
        f"Test Accuracy    : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Fault Precision  : "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Fault Recall     : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"Fault F1         : "
        f"{test_metrics['f1']:.4f}"
    )

    print()
    print(
        "Best model:",
        OUTPUT_DIR / "best_model"
    )

    print(
        "History   :",
        OUTPUT_DIR / "training_history.json"
    )

    print(
        "Metrics   :",
        OUTPUT_DIR / "test_metrics.json"
    )

    print(
        "Predictions:",
        OUTPUT_DIR / "test_predictions.csv"
    )

    print()


if __name__ == "__main__":
    main()