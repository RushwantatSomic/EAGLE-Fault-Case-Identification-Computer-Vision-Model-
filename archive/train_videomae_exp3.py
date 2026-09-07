import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


# ============================================================
# EAGLE-MARS — VIDEOMAE EXPERIMENT 3
# ============================================================
#
# Changes from Experiment 1:
#   - Uses cleaned Experiment 3 manifest
#   - Removes clips crossing temporal boundaries
#   - Lower learning rate
#   - No class weighting
#   - Early stopping
#   - Same test set as Experiments 1 and 2
#
# Changes from Experiment 2:
#   - NO WeightedRandomSampler
#   - Natural training distribution retained
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

MANIFEST = Path(
    "outputs/dataset/clips/clips_manifest_exp3.csv"
)

OUTPUT_DIR = Path(
    "outputs/training_exp3"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_NAME = (
    "MCG-NJU/videomae-small-finetuned-kinetics"
)

NUM_FRAMES = 16
IMAGE_SIZE = 224

BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4

EPOCHS = 8
PATIENCE = 2

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.05

NUM_WORKERS = 0

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

LABELS = {
    "NORMAL": 0,
    "FAULT": 1,
}

ID2LABEL = {
    0: "NORMAL",
    1: "FAULT",
}

LABEL2ID = {
    "NORMAL": 0,
    "FAULT": 1,
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATASET
# ============================================================

class EagleVideoDataset(Dataset):

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

    def __getitem__(
        self,
        index,
    ):

        row = self.df.iloc[index]

        path = Path(
            row["file"]
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Clip not found: {path}"
            )

        frames = np.load(path)

        expected_shape = (
            NUM_FRAMES,
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        )

        if frames.shape != expected_shape:

            raise RuntimeError(
                f"Unexpected clip shape "
                f"{frames.shape}: {path}"
            )

        # Convert uint8 frames into VideoMAE input.
        inputs = self.processor(
            list(frames),
            return_tensors="pt",
        )

        pixel_values = (
            inputs["pixel_values"]
            .squeeze(0)
        )

        label = LABELS[
            row["label"]
        ]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(
                label,
                dtype=torch.long,
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
        y_pred,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
    )

    return {
        "accuracy": float(
            accuracy
        ),
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1": float(
            f1
        ),
    }


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
):

    model.eval()

    all_labels = []
    all_predictions = []

    total_loss = 0.0
    total_samples = 0

    for batch in loader:

        pixel_values = (
            batch["pixel_values"]
            .to(
                DEVICE,
                non_blocking=True,
            )
        )

        labels = (
            batch["labels"]
            .to(
                DEVICE,
                non_blocking=True,
            )
        )

        outputs = model(
            pixel_values=pixel_values
        )

        loss = criterion(
            outputs.logits,
            labels,
        )

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        total_samples += (
            labels.size(0)
        )

        predictions = (
            outputs.logits.argmax(
                dim=-1
            )
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
    )

    metrics["loss"] = float(
        total_loss
        /
        max(
            total_samples,
            1,
        )
    )

    return (
        metrics,
        all_labels,
        all_predictions,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print(
        "EAGLE-MARS VIDEOMAE EXPERIMENT 3"
    )
    print("=" * 100)

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

        print(
            "VRAM:",
            round(
                torch.cuda
                .get_device_properties(
                    0
                )
                .total_memory
                / 1024**3,
                2,
            ),
            "GB",
        )

    print()
    print(
        "Configuration"
    )
    print("-" * 100)

    print(
        f"Model           : {MODEL_NAME}"
    )

    print(
        f"Frames          : {NUM_FRAMES}"
    )

    print(
        f"Resolution      : "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Batch size      : {BATCH_SIZE}"
    )

    print(
        f"Gradient accum. : "
        f"{GRADIENT_ACCUMULATION}"
    )

    print(
        f"Effective batch : "
        f"{BATCH_SIZE * GRADIENT_ACCUMULATION}"
    )

    print(
        f"Learning rate   : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Epochs          : {EPOCHS}"
    )

    print(
        f"Early stopping  : "
        f"{PATIENCE} epochs"
    )

    print(
        f"Manifest        : {MANIFEST}"
    )

    print(
        f"Output          : {OUTPUT_DIR}"
    )

    # ========================================================
    # LOAD MANIFEST
    # ========================================================

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
    print(
        "DATASET"
    )
    print("-" * 100)

    print(
        df.groupby(
            [
                "split",
                "label",
            ]
        ).size()
    )

    # ========================================================
    # VERIFY FILES
    # ========================================================

    missing = []

    for path in df["file"]:

        if not Path(path).exists():

            missing.append(
                path
            )

    if missing:

        print()
        print(
            f"ERROR: "
            f"{len(missing)} clips missing."
        )

        for path in missing[:20]:

            print(
                path
            )

        raise FileNotFoundError(
            "Missing training clips."
        )

    print()
    print(
        "All clip files verified."
    )

    # ========================================================
    # SPLITS
    # ========================================================

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
    print(
        "SPLIT SIZES"
    )
    print("-" * 100)

    print(
        f"TRAIN : {len(train_df)}"
    )

    print(
        f"VAL   : {len(val_df)}"
    )

    print(
        f"TEST  : {len(test_df)}"
    )

    # ========================================================
    # PROCESSOR
    # ========================================================

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

    # ========================================================
    # MODEL
    # ========================================================

    print(
        "Loading VideoMAE-small..."
    )

    model = (
        VideoMAEForVideoClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=2,
            label2id=LABEL2ID,
            id2label=ID2LABEL,
            ignore_mismatched_sizes=True,
        )
    )

    model.to(
        DEVICE
    )

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = EagleVideoDataset(
        train_df,
        processor,
    )

    val_dataset = EagleVideoDataset(
        val_df,
        processor,
    )

    test_dataset = EagleVideoDataset(
        test_df,
        processor,
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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

    # ========================================================
    # LOSS
    # ========================================================
    #
    # Experiment 2 showed that aggressive class balancing
    # damaged fault recall.
    #
    # Experiment 3 therefore uses ordinary CE loss.
    #
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ========================================================
    # MIXED PRECISION
    # ========================================================

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available(),
    )

    # ========================================================
    # TRAINING
    # ========================================================

    best_f1 = -1.0
    best_epoch = 0

    patience_counter = 0

    history = []

    print()
    print("=" * 100)
    print(
        "STARTING EXPERIMENT 3"
    )
    print("=" * 100)

    for epoch in range(
        EPOCHS
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        running_loss = 0.0
        sample_count = 0

        for step, batch in enumerate(
            train_loader
        ):

            pixel_values = (
                batch["pixel_values"]
                .to(
                    DEVICE,
                    non_blocking=True,
                )
            )

            labels = (
                batch["labels"]
                .to(
                    DEVICE,
                    non_blocking=True,
                )
            )

            # ------------------------------------------------
            # Mixed precision forward pass
            # ------------------------------------------------

            with torch.amp.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available(),
            ):

                outputs = model(
                    pixel_values=pixel_values
                )

                loss = criterion(
                    outputs.logits,
                    labels,
                )

                loss_for_backward = (
                    loss
                    /
                    GRADIENT_ACCUMULATION
                )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            scaler.scale(
                loss_for_backward
            ).backward()

            # ------------------------------------------------
            # Gradient accumulation
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

            running_loss += (
                loss.item()
                * labels.size(0)
            )

            sample_count += (
                labels.size(0)
            )

        # ----------------------------------------------------
        # Handle final incomplete accumulation group
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

        # ====================================================
        # TRAIN LOSS
        # ====================================================

        train_loss = (
            running_loss
            /
            max(
                sample_count,
                1,
            )
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        val_metrics, _, _ = evaluate(
            model,
            val_loader,
            criterion,
        )

        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print()
        print(
            f"Epoch "
            f"{epoch + 1}/{EPOCHS}"
        )

        print(
            f"Train Loss : "
            f"{train_loss:.4f}"
        )

        print(
            f"Val Loss   : "
            f"{val_metrics['loss']:.4f}"
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

        # ====================================================
        # HISTORY
        # ====================================================

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics[
                    "loss"
                ],
                "val_accuracy": val_metrics[
                    "accuracy"
                ],
                "val_precision": val_metrics[
                    "precision"
                ],
                "val_recall": val_metrics[
                    "recall"
                ],
                "val_f1": val_metrics[
                    "f1"
                ],
            }
        )

        # ====================================================
        # BEST MODEL
        # ====================================================

        if (
            val_metrics["f1"]
            >
            best_f1
        ):

            best_f1 = (
                val_metrics["f1"]
            )

            best_epoch = (
                epoch + 1
            )

            patience_counter = 0

            checkpoint_dir = (
                OUTPUT_DIR
                /
                "best_model"
            )

            checkpoint_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            model.save_pretrained(
                checkpoint_dir
            )

            processor.save_pretrained(
                checkpoint_dir
            )

            print()
            print(
                "BEST MODEL SAVED"
            )

            print(
                f"Validation F1: "
                f"{best_f1:.4f}"
            )

        else:

            patience_counter += 1

            print(
                f"No improvement "
                f"({patience_counter}/"
                f"{PATIENCE})"
            )

            if (
                patience_counter
                >= PATIENCE
            ):

                print()
                print(
                    "EARLY STOPPING"
                )

                print(
                    f"Best epoch: "
                    f"{best_epoch}"
                )

                print(
                    f"Best Val F1: "
                    f"{best_f1:.4f}"
                )

                break

        # ====================================================
        # GPU MEMORY CLEANUP
        # ====================================================

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    history_path = (
        OUTPUT_DIR
        /
        "training_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print()
    print("=" * 100)
    print(
        "LOADING BEST MODEL"
    )
    print("=" * 100)

    best_model_path = (
        OUTPUT_DIR
        /
        "best_model"
    )

    best_model = (
        VideoMAEForVideoClassification
        .from_pretrained(
            best_model_path
        )
    )

    best_model.to(
        DEVICE
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    print()
    print("=" * 100)
    print(
        "EXPERIMENT 3 FINAL TEST"
    )
    print("=" * 100)

    test_metrics, y_true, y_pred = (
        evaluate(
            best_model,
            test_loader,
            criterion,
        )
    )

    # ========================================================
    # TEST METRICS
    # ========================================================

    print()
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

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print()
    print(
        "CLASSIFICATION REPORT"
    )
    print("-" * 100)

    report = classification_report(
        y_true,
        y_pred,
        target_names=[
            "NORMAL",
            "FAULT",
        ],
        zero_division=0,
    )

    print(
        report
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    print(
        "CONFUSION MATRIX"
    )
    print("-" * 100)

    print(
        "                 Predicted"
    )

    print(
        "              NORMAL  FAULT"
    )

    print(
        f"Actual NORMAL  "
        f"{cm[0,0]:6d}  "
        f"{cm[0,1]:5d}"
    )

    print(
        f"       FAULT   "
        f"{cm[1,0]:6d}  "
        f"{cm[1,1]:5d}"
    )

    # ========================================================
    # SAVE CONFUSION MATRIX
    # ========================================================

    np.save(
        OUTPUT_DIR
        /
        "confusion_matrix.npy",
        cm,
    )

    # ========================================================
    # SAVE TEST METRICS
    # ========================================================

    test_metrics_path = (
        OUTPUT_DIR
        /
        "test_metrics.json"
    )

    with open(
        test_metrics_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "best_epoch": best_epoch,
                "best_validation_f1": best_f1,
                **test_metrics,
            },
            f,
            indent=2,
        )

    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    predictions = test_df.copy()

    predictions[
        "true_label_id"
    ] = [
        LABELS[x]
        for x in predictions[
            "label"
        ]
    ]

    predictions[
        "predicted_label_id"
    ] = y_pred

    predictions[
        "predicted_label"
    ] = [
        ID2LABEL[x]
        for x in y_pred
    ]

    predictions[
        "correct"
    ] = (
        predictions[
            "true_label_id"
        ]
        ==
        predictions[
            "predicted_label_id"
        ]
    )

    predictions.to_csv(
        OUTPUT_DIR
        /
        "test_predictions.csv",
        index=False,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 100)
    print(
        "EXPERIMENT 3 COMPLETE"
    )
    print("=" * 100)

    print()
    print(
        f"Best epoch       : "
        f"{best_epoch}"
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
        f"Best model: "
        f"{best_model_path}"
    )

    print(
        f"History: "
        f"{history_path}"
    )

    print(
        f"Metrics: "
        f"{test_metrics_path}"
    )

    print(
        f"Predictions: "
        f"{OUTPUT_DIR / 'test_predictions.csv'}"
    )

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()