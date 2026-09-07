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
# CONFIG
# ============================================================

MANIFEST = Path("outputs/dataset/clips/clips_manifest.csv")
OUTPUT_DIR = Path("outputs/training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "MCG-NJU/videomae-small-finetuned-kinetics"

NUM_FRAMES = 16
IMAGE_SIZE = 224

BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4

EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.05

NUM_WORKERS = 0
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def __init__(self, dataframe, processor):

        self.df = dataframe.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        path = Path(row["file"])

        frames = np.load(path)

        # frames:
        # [16, 224, 224, 3]

        if frames.shape != (
            NUM_FRAMES,
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        ):
            raise RuntimeError(
                f"Unexpected clip shape {frames.shape}: {path}"
            )

        # VideoMAE processor expects a list/array of frames.
        inputs = self.processor(
            list(frames),
            return_tensors="pt",
        )

        # Remove processor batch dimension.
        pixel_values = inputs["pixel_values"].squeeze(0)

        label = LABELS[row["label"]]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
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

@torch.no_grad()
def evaluate(model, loader):

    model.eval()

    all_labels = []
    all_predictions = []

    total_loss = 0.0
    count = 0

    for batch in loader:

        pixel_values = batch["pixel_values"].to(
            DEVICE,
            non_blocking=True,
        )

        labels = batch["labels"].to(
            DEVICE,
            non_blocking=True,
        )

        outputs = model(
            pixel_values=pixel_values,
            labels=labels,
        )

        total_loss += outputs.loss.item() * labels.size(0)
        count += labels.size(0)

        predictions = outputs.logits.argmax(dim=-1)

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

    metrics["loss"] = total_loss / max(count, 1)

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
    print("EAGLE-MARS VIDEOMAE TRAINING")
    print("=" * 100)

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        print(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    if not MANIFEST.exists():

        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST}"
        )

    df = pd.read_csv(MANIFEST)

    print()
    print(f"Total clips: {len(df)}")

    print()
    print("DATASET:")
    print(
        df.groupby(
            ["split", "label"]
        ).size()
    )

    # --------------------------------------------------------
    # Verify files
    # --------------------------------------------------------

    missing = [
        f for f in df["file"]
        if not Path(f).exists()
    ]

    if missing:

        print()
        print(
            f"ERROR: {len(missing)} clip files are missing."
        )

        for path in missing[:10]:
            print(path)

        raise FileNotFoundError(
            "Missing training clips."
        )

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    print()
    print("Loading VideoMAE processor...")

    processor = VideoMAEImageProcessor.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("Loading VideoMAE model...")

    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
    )

    model.to(DEVICE)

    # --------------------------------------------------------
    # Dataset splits
    # --------------------------------------------------------

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

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

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    counts = train_df["label"].value_counts()

    normal_count = counts.get("NORMAL", 1)
    fault_count = counts.get("FAULT", 1)

    total = normal_count + fault_count

    weights = torch.tensor(
        [
            total / (2 * normal_count),
            total / (2 * fault_count),
        ],
        dtype=torch.float32,
        device=DEVICE,
    )

    print()
    print("CLASS WEIGHTS:")
    print(
        f"NORMAL: {weights[0].item():.4f}"
    )
    print(
        f"FAULT : {weights[1].item():.4f}"
    )

    criterion = nn.CrossEntropyLoss(
        weight=weights
    )

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

    history = []

    print()
    print("=" * 100)
    print("STARTING TRAINING")
    print("=" * 100)

    for epoch in range(EPOCHS):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        running_loss = 0.0
        sample_count = 0

        for step, batch in enumerate(train_loader):

            pixel_values = batch["pixel_values"].to(
                DEVICE,
                non_blocking=True,
            )

            labels = batch["labels"].to(
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
                "epoch": epoch + 1,
                "train_loss": train_loss,
                **{
                    f"val_{k}": v
                    for k, v in val_metrics.items()
                },
            }
        )

        # ----------------------------------------------------
        # Save ONLY best model
        # ----------------------------------------------------

        if val_metrics["f1"] > best_f1:

            best_f1 = val_metrics["f1"]

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
                f"(Val F1={best_f1:.4f})"
            )

        # Free GPU cache.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    with open(
        OUTPUT_DIR / "training_history.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("FINAL TEST")
    print("=" * 100)

    best_model = VideoMAEForVideoClassification.from_pretrained(
        OUTPUT_DIR / "best_model"
    )

    best_model.to(DEVICE)

    test_metrics, y_true, y_pred = evaluate(
        best_model,
        test_loader,
    )

    print(
        f"Test Loss     : {test_metrics['loss']:.4f}"
    )

    print(
        f"Test Accuracy : {test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test Precision: {test_metrics['precision']:.4f}"
    )

    print(
        f"Test Recall   : {test_metrics['recall']:.4f}"
    )

    print(
        f"Test F1       : {test_metrics['f1']:.4f}"
    )

    print()
    print("CLASSIFICATION REPORT")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "NORMAL",
                "FAULT",
            ],
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    print("CONFUSION MATRIX")
    print(cm)

    np.save(
        OUTPUT_DIR / "confusion_matrix.npy",
        cm,
    )

    with open(
        OUTPUT_DIR / "test_metrics.json",
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
    print("TRAINING COMPLETE")
    print("=" * 100)

    print(
        f"Best validation F1: {best_f1:.4f}"
    )

    print(
        f"Model: {OUTPUT_DIR / 'best_model'}"
    )


if __name__ == "__main__":
    main()