import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
# CONFIG
# ============================================================

MANIFEST = Path("outputs/dataset/clips/clips_manifest.csv")
OUTPUT_DIR = Path("outputs/training_exp2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "MCG-NJU/videomae-small-finetuned-kinetics"

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
    "cuda" if torch.cuda.is_available() else "cpu"
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
# SEED
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

    def __init__(self, dataframe, processor):

        self.df = dataframe.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        frames = np.load(
            Path(row["file"])
        )

        if frames.shape != (
            NUM_FRAMES,
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        ):
            raise RuntimeError(
                f"Unexpected shape {frames.shape}: "
                f"{row['file']}"
            )

        inputs = self.processor(
            list(frames),
            return_tensors="pt",
        )

        pixel_values = (
            inputs["pixel_values"]
            .squeeze(0)
        )

        label = LABELS[row["label"]]

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

def metrics(y_true, y_pred):

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
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, loader):

    model.eval()

    labels_all = []
    predictions_all = []

    total_loss = 0.0
    total_samples = 0

    for batch in loader:

        pixel_values = batch[
            "pixel_values"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        labels = batch[
            "labels"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        outputs = model(
            pixel_values=pixel_values,
            labels=labels,
        )

        total_loss += (
            outputs.loss.item()
            * labels.size(0)
        )

        total_samples += labels.size(0)

        predictions = (
            outputs.logits.argmax(dim=-1)
        )

        labels_all.extend(
            labels.cpu().numpy()
        )

        predictions_all.extend(
            predictions.cpu().numpy()
        )

    result = metrics(
        labels_all,
        predictions_all,
    )

    result["loss"] = (
        total_loss /
        max(total_samples, 1)
    )

    return (
        result,
        labels_all,
        predictions_all,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("EAGLE-MARS VIDEOMAE EXPERIMENT 2")
    print("=" * 100)

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(
                    0
                ).total_memory / 1024**3,
                2,
            ),
            "GB",
        )

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    df = pd.read_csv(MANIFEST)

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
    print("DATASET")
    print("-" * 100)

    print(
        df.groupby(
            ["split", "label"]
        ).size()
    )

    # --------------------------------------------------------
    # Verify files
    # --------------------------------------------------------

    missing = [
        p for p in df["file"]
        if not Path(p).exists()
    ]

    if missing:

        print(
            f"Missing clips: {len(missing)}"
        )

        for p in missing[:10]:
            print(p)

        raise FileNotFoundError(
            "Training clips are missing."
        )

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    processor = (
        VideoMAEImageProcessor
        .from_pretrained(MODEL_NAME)
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("Loading VideoMAE-small...")

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

    model.to(DEVICE)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # BALANCED SAMPLER
    # --------------------------------------------------------

    train_labels = [
        LABELS[x]
        for x in train_df["label"]
    ]

    class_counts = np.bincount(
        train_labels,
        minlength=2,
    )

    class_weights = (
        len(train_labels)
        /
        (
            2.0
            * class_counts
        )
    )

    sample_weights = [
        class_weights[label]
        for label in train_labels
    ]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(
            sample_weights
        ),
        num_samples=len(train_labels),
        replacement=True,
    )

    print()
    print("TRAINING CLASS COUNTS")
    print(
        "NORMAL:",
        class_counts[0],
    )
    print(
        "FAULT :",
        class_counts[1],
    )

    print()
    print(
        "BALANCED SAMPLER ENABLED"
    )

    # --------------------------------------------------------
    # Loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
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

    # --------------------------------------------------------
    # Loss
    #
    # Do NOT also class-weight the loss.
    # The sampler already balances the classes.
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_f1 = -1.0
    patience_counter = 0

    history = []

    print()
    print("=" * 100)
    print("STARTING EXPERIMENT 2")
    print("=" * 100)

    for epoch in range(EPOCHS):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        running_loss = 0.0
        sample_count = 0

        for step, batch in enumerate(
            train_loader
        ):

            pixel_values = batch[
                "pixel_values"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            labels = batch[
                "labels"
            ].to(
                DEVICE,
                non_blocking=True,
            )

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

                loss = (
                    loss /
                    GRADIENT_ACCUMULATION
                )

            scaler.scale(loss).backward()

            if (
                (step + 1)
                % GRADIENT_ACCUMULATION
                == 0
            ):

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

            running_loss += (
                loss.item()
                * GRADIENT_ACCUMULATION
                * labels.size(0)
            )

            sample_count += labels.size(0)

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics, _, _ = evaluate(
            model,
            val_loader,
        )

        train_loss = (
            running_loss /
            max(sample_count, 1)
        )

        print()
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Train Loss : {train_loss:.4f}"
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

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **{
                f"val_{k}": v
                for k, v in val_metrics.items()
            },
        })

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if val_metrics["f1"] > best_f1:

            best_f1 = val_metrics["f1"]
            patience_counter = 0

            checkpoint_dir = (
                OUTPUT_DIR /
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

            print(
                f"BEST MODEL SAVED "
                f"(F1={best_f1:.4f})"
            )

        else:

            patience_counter += 1

            print(
                f"No improvement "
                f"({patience_counter}/{PATIENCE})"
            )

            if patience_counter >= PATIENCE:

                print(
                    "Early stopping."
                )

                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    with open(
        OUTPUT_DIR /
        "training_history.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # TEST BEST MODEL
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("EXPERIMENT 2 TEST")
    print("=" * 100)

    best_model = (
        VideoMAEForVideoClassification
        .from_pretrained(
            OUTPUT_DIR /
            "best_model"
        )
    )

    best_model.to(DEVICE)

    test_metrics, y_true, y_pred = (
        evaluate(
            best_model,
            test_loader,
        )
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

    report = classification_report(
        y_true,
        y_pred,
        target_names=[
            "NORMAL",
            "FAULT",
        ],
        zero_division=0,
    )

    print(report)

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    print("CONFUSION MATRIX")
    print(cm)

    np.save(
        OUTPUT_DIR /
        "confusion_matrix.npy",
        cm,
    )

    with open(
        OUTPUT_DIR /
        "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            test_metrics,
            f,
            indent=2,
        )

    print()
    print("=" * 100)
    print("EXPERIMENT 2 COMPLETE")
    print("=" * 100)

    print(
        f"Best validation F1: "
        f"{best_f1:.4f}"
    )

    print(
        "Output:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()