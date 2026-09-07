from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# EAGLE-MARS EXPERIMENT 4
# VideoMAE Training
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    ROOT
    / "outputs"
    / "dataset_exp4"
    / "clips_manifest.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "training_exp4"
)

MODEL_NAME = (
    "MCG-NJU/videomae-small-finetuned-kinetics"
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_FRAMES = 16
IMAGE_SIZE = 224

BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01

EPOCHS = 8

PATIENCE = 2

NUM_WORKERS = 0

SEED = 42


# ============================================================
# LABELS
# ============================================================

LABEL2ID = {
    "NORMAL": 0,
    "FAULT": 1,
}

ID2LABEL = {
    0: "NORMAL",
    1: "FAULT",
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET
# ============================================================

class VideoMAEDataset(Dataset):

    def __init__(
        self,
        dataframe,
        processor,
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.processor = processor

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        file_path = (
            ROOT / row["file"]
        )

        frames = np.load(
            file_path,
            allow_pickle=False
        )

        # ----------------------------------------------------
        # Expected:
        #
        # (16, 224, 224, 3)
        # ----------------------------------------------------

        if frames.shape != (
            NUM_FRAMES,
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        ):

            raise RuntimeError(
                f"Unexpected clip shape "
                f"{frames.shape} in "
                f"{file_path}"
            )

        # VideoMAE processor expects a list/array
        # of frames.
        #
        # Each sample becomes:
        # pixel_values -> (T, C, H, W)
        # ----------------------------------------------------

        processed = self.processor(
            images=list(frames),
            return_tensors="pt",
        )

        pixel_values = (
            processed["pixel_values"]
            .squeeze(0)
        )

        label = LABEL2ID[
            row["label"]
        ]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(
                label,
                dtype=torch.long
            ),
        }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for batch in loader:

            pixel_values = (
                batch["pixel_values"]
                .to(
                    device,
                    non_blocking=True
                )
            )

            labels = (
                batch["labels"]
                .to(
                    device,
                    non_blocking=True
                )
            )

            outputs = model(
                pixel_values=pixel_values,
                labels=labels,
            )

            loss = outputs.loss

            logits = outputs.logits

            predictions = (
                torch.argmax(
                    logits,
                    dim=1
                )
            )

            batch_size = (
                labels.size(0)
            )

            total_loss += (
                loss.item() *
                batch_size
            )

            total_samples += batch_size

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    average_loss = (
        total_loss /
        total_samples
    )

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
    )

    return (
        average_loss,
        metrics,
        all_labels,
        all_predictions,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 90)
    print(
        "EAGLE-MARS VIDEOMAE "
        "EXPERIMENT 4 TRAINING"
    )
    print("=" * 90)

    print()
    print("DEVICE")
    print("-" * 90)

    print(
        f"Device : {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU    : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"VRAM   : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print()
    print("CONFIGURATION")
    print("-" * 90)

    print(
        f"Model              : {MODEL_NAME}"
    )

    print(
        f"Frames             : {NUM_FRAMES}"
    )

    print(
        f"Resolution         : "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Batch size         : {BATCH_SIZE}"
    )

    print(
        f"Gradient accum.    : "
        f"{GRADIENT_ACCUMULATION}"
    )

    print(
        f"Effective batch    : "
        f"{BATCH_SIZE * GRADIENT_ACCUMULATION}"
    )

    print(
        f"Learning rate      : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Epochs             : {EPOCHS}"
    )

    print(
        f"Early stopping     : "
        f"{PATIENCE} epochs"
    )

    print(
        f"Balanced sampler   : ENABLED"
    )

    print(
        f"Manifest           : "
        f"{MANIFEST}"
    )

    print(
        f"Output             : "
        f"{OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # LOAD MANIFEST
    # --------------------------------------------------------

    if not MANIFEST.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n"
            f"{MANIFEST}"
        )

    df = pd.read_csv(
        MANIFEST
    )

    print()
    print(
        f"Total clips: {len(df)}"
    )

    print()
    print("DATASET")
    print("-" * 90)

    print(
        df.groupby(
            ["split", "label"]
        ).size()
    )

    # --------------------------------------------------------
    # VERIFY FILES
    # --------------------------------------------------------

    print()
    print(
        "VERIFYING CLIP FILES..."
    )

    missing_files = []

    for file_path in df["file"]:

        full_path = (
            ROOT / file_path
        )

        if not full_path.exists():

            missing_files.append(
                str(full_path)
            )

    if missing_files:

        print(
            f"Missing files: "
            f"{len(missing_files)}"
        )

        for path in missing_files[:10]:
            print(path)

        raise FileNotFoundError(
            "One or more clip files are missing."
        )

    print(
        "All clip files verified."
    )

    # --------------------------------------------------------
    # SPLITS
    # --------------------------------------------------------

    train_df = df[
        df["split"] == "train"
    ].copy()

    val_df = df[
        df["split"] == "val"
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    print()
    print("SPLIT SIZES")
    print("-" * 90)

    print(
        f"TRAIN : {len(train_df)}"
    )

    print(
        f"VAL   : {len(val_df)}"
    )

    print(
        f"TEST  : {len(test_df)}"
    )

    # --------------------------------------------------------
    # PROCESSOR
    # --------------------------------------------------------

    print()
    print(
        "Loading VideoMAE processor..."
    )

    processor = (
        VideoMAEImageProcessor
        .from_pretrained(
            MODEL_NAME
        )
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = VideoMAEDataset(
        train_df,
        processor,
    )

    val_dataset = VideoMAEDataset(
        val_df,
        processor,
    )

    test_dataset = VideoMAEDataset(
        test_df,
        processor,
    )

    # --------------------------------------------------------
    # BALANCED SAMPLER
    # --------------------------------------------------------

    train_labels = [
        LABEL2ID[label]
        for label in train_df["label"]
    ]

    class_counts = np.bincount(
        train_labels,
        minlength=2
    )

    print()
    print("TRAINING CLASS COUNTS")
    print("-" * 90)

    print(
        f"NORMAL: "
        f"{class_counts[0]}"
    )

    print(
        f"FAULT : "
        f"{class_counts[1]}"
    )

    class_weights = (
        1.0 /
        np.maximum(
            class_counts,
            1
        )
    )

    sample_weights = np.array([
        class_weights[label]
        for label in train_labels
    ])

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(
            sample_weights
        ),
        num_samples=len(
            train_labels
        ),
        replacement=True,
    )

    print()
    print(
        "BALANCED SAMPLER ENABLED"
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print(
        "Loading VideoMAE-small..."
    )

    model = (
        VideoMAEForVideoClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
    )

    model.to(device)

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    use_amp = (
        device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "STARTING EXPERIMENT 4"
    )
    print("=" * 90)

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    history = []

    best_model_dir = (
        OUTPUT_DIR /
        "best_model"
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print()
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        running_loss = 0.0
        total_train_samples = 0

        for step, batch in enumerate(
            train_loader
        ):

            pixel_values = (
                batch["pixel_values"]
                .to(
                    device,
                    non_blocking=True
                )
            )

            labels = (
                batch["labels"]
                .to(
                    device,
                    non_blocking=True
                )
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp,
            ):

                outputs = model(
                    pixel_values=pixel_values,
                    labels=labels,
                )

                loss = outputs.loss

                loss_for_backward = (
                    loss /
                    GRADIENT_ACCUMULATION
                )

            scaler.scale(
                loss_for_backward
            ).backward()

            # ------------------------------------------------
            # Optimizer step
            # ------------------------------------------------

            if (
                (step + 1)
                % GRADIENT_ACCUMULATION
                == 0
            ):

                scaler.step(
                    optimizer
                )

                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

            batch_size = (
                labels.size(0)
            )

            running_loss += (
                loss.item() *
                batch_size
            )

            total_train_samples += (
                batch_size
            )

        # ----------------------------------------------------
        # Handle remaining gradients
        # ----------------------------------------------------

        if (
            len(train_loader)
            % GRADIENT_ACCUMULATION
            != 0
        ):

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        train_loss = (
            running_loss /
            total_train_samples
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        (
            val_loss,
            val_metrics,
            _,
            _,
        ) = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"Train Loss : "
            f"{train_loss:.4f}"
        )

        print(
            f"Val Loss   : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Acc    : "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"Val Prec   : "
            f"{val_metrics['precision']:.4f}"
        )

        print(
            f"Val Recall : "
            f"{val_metrics['recall']:.4f}"
        )

        print(
            f"Val F1     : "
            f"{val_metrics['f1']:.4f}"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy":
                val_metrics["accuracy"],
            "val_precision":
                val_metrics["precision"],
            "val_recall":
                val_metrics["recall"],
            "val_f1":
                val_metrics["f1"],
        }

        history.append(
            epoch_record
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if (
            val_metrics["f1"]
            > best_val_f1
        ):

            best_val_f1 = (
                val_metrics["f1"]
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            if best_model_dir.exists():

                import shutil

                shutil.rmtree(
                    best_model_dir
                )

            best_model_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            model.save_pretrained(
                best_model_dir
            )

            processor.save_pretrained(
                best_model_dir
            )

            print()
            print(
                "BEST MODEL SAVED"
            )

            print(
                f"Validation F1: "
                f"{best_val_f1:.4f}"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"No improvement "
                f"({epochs_without_improvement}/"
                f"{PATIENCE})"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()
            print(
                "EARLY STOPPING"
            )

            break

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_file = (
        OUTPUT_DIR /
        "training_history.json"
    )

    with open(
        history_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "LOADING BEST MODEL"
    )
    print("=" * 90)

    best_model = (
        VideoMAEForVideoClassification
        .from_pretrained(
            best_model_dir
        )
    )

    best_model.to(device)

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    (
        test_loss,
        test_metrics,
        test_labels,
        test_predictions,
    ) = evaluate(
        best_model,
        test_loader,
        device,
    )

    print()
    print("=" * 90)
    print(
        "EXPERIMENT 4 FINAL TEST"
    )
    print("=" * 90)

    print()
    print(
        f"Test Loss     : "
        f"{test_loss:.4f}"
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

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print()
    print(
        "CLASSIFICATION REPORT"
    )

    print("-" * 90)

    report = classification_report(
        test_labels,
        test_predictions,
        labels=[0, 1],
        target_names=[
            "NORMAL",
            "FAULT",
        ],
        zero_division=0,
    )

    print(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        test_labels,
        test_predictions,
        labels=[0, 1],
    )

    print(
        "CONFUSION MATRIX"
    )

    print("-" * 90)

    print(
        "                 Predicted"
    )

    print(
        "              NORMAL  FAULT"
    )

    print(
        f"Actual NORMAL"
        f"    {cm[0,0]:5d}"
        f" {cm[0,1]:6d}"
    )

    print(
        f"       FAULT"
        f"    {cm[1,0]:5d}"
        f" {cm[1,1]:6d}"
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions_df = test_df.copy()

    predictions_df[
        "true_id"
    ] = test_labels

    predictions_df[
        "predicted_id"
    ] = test_predictions

    predictions_df[
        "predicted_label"
    ] = [
        ID2LABEL[x]
        for x in test_predictions
    ]

    predictions_file = (
        OUTPUT_DIR /
        "test_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_file,
        index=False
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    test_metrics_output = {
        "best_epoch": best_epoch,
        "best_validation_f1":
            best_val_f1,
        "test_loss":
            test_loss,
        "test_accuracy":
            test_metrics["accuracy"],
        "test_precision":
            test_metrics["precision"],
        "test_recall":
            test_metrics["recall"],
        "test_f1":
            test_metrics["f1"],
        "confusion_matrix":
            cm.tolist(),
        "train_clips":
            len(train_df),
        "val_clips":
            len(val_df),
        "test_clips":
            len(test_df),
    }

    metrics_file = (
        OUTPUT_DIR /
        "test_metrics.json"
    )

    with open(
        metrics_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            test_metrics_output,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "EXPERIMENT 4 COMPLETE"
    )
    print("=" * 90)

    print()
    print(
        f"Best epoch       : "
        f"{best_epoch}"
    )

    print(
        f"Best Val F1      : "
        f"{best_val_f1:.4f}"
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
        f"Best model: "
        f"{best_model_dir}"
    )

    print(
        f"History   : "
        f"{history_file}"
    )

    print(
        f"Metrics   : "
        f"{metrics_file}"
    )

    print(
        f"Predictions: "
        f"{predictions_file}"
    )

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()