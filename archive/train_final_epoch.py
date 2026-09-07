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


# =============================================================================
# EAGLE-MARS FINAL CONSOLIDATION TRAINING
#
# Dataset:
#   5663 clips
#
# Sources:
#   Existing unified dataset
#   CVAT continual dataset
#
# Model:
#   MCG-NJU/videomae-small-finetuned-kinetics
#
# Training:
#   5 epochs
#   Balanced sampler
#   Batch size = 2
#   Gradient accumulation = 4
#   Effective batch = 8
# =============================================================================


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    ROOT
    / "outputs"
    / "unified_dataset"
    / "final_training"
    / "final_manifest.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "final_training"
)

BEST_MODEL_DIR = (
    OUTPUT_DIR
    / "best_model"
)

MODEL_NAME = (
    "MCG-NJU/"
    "videomae-small-finetuned-kinetics"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

NUM_FRAMES = 16

IMAGE_SIZE = 224

BATCH_SIZE = 2

GRADIENT_ACCUMULATION = 4

EFFECTIVE_BATCH = (
    BATCH_SIZE
    * GRADIENT_ACCUMULATION
)

LEARNING_RATE = 5e-6

EPOCHS = 5

EARLY_STOPPING = 2

NUM_WORKERS = 0

SEED = 42


LABEL2ID = {
    "NORMAL": 0,
    "FAULT": 1,
}

ID2LABEL = {
    0: "NORMAL",
    1: "FAULT",
}


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# =============================================================================
# DATASET
# =============================================================================

class VideoClipDataset(
    Dataset
):

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

        return len(
            self.df
        )

    def __getitem__(
        self,
        index,
    ):

        row = self.df.iloc[
            index
        ]

        path = row["file"]

        suffix = Path(
            path
        ).suffix.lower()

        if suffix == ".npy":

            frames = np.load(
                path,
                mmap_mode="r",
            )

        elif suffix == ".npz":

            data = np.load(
                path
            )

            try:

                keys = list(
                    data.keys()
                )

                preferred = [
                    "clip",
                    "frames",
                    "array",
                    "arr_0",
                ]

                selected = None

                for key in preferred:

                    if key in data:

                        selected = key
                        break

                if selected is None:

                    selected = keys[0]

                frames = np.asarray(
                    data[selected]
                )

            finally:

                data.close()

        else:

            raise RuntimeError(
                f"Unsupported clip: {path}"
            )

        frames = np.asarray(
            frames
        )

        # Safety check.
        if frames.shape != (
            NUM_FRAMES,
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        ):

            raise RuntimeError(
                f"Unexpected shape "
                f"{frames.shape} "
                f"for {path}"
            )

        # VideoMAE processor expects a list/array
        # of RGB frames.
        inputs = self.processor(
            list(frames),
            return_tensors="pt",
        )

        pixel_values = inputs[
            "pixel_values"
        ].squeeze(0)

        label = LABEL2ID[
            row["label"]
        ]

        return {
            "pixel_values":
                pixel_values,

            "labels":
                torch.tensor(
                    label,
                    dtype=torch.long,
                ),
        }


# =============================================================================
# COLLATE
# =============================================================================

def collate_fn(batch):

    pixel_values = torch.stack(
        [
            item[
                "pixel_values"
            ]
            for item in batch
        ]
    )

    labels = torch.stack(
        [
            item[
                "labels"
            ]
            for item in batch
        ]
    )

    return {
        "pixel_values":
            pixel_values,

        "labels":
            labels,
    }


# =============================================================================
# LOAD MODEL
# =============================================================================

def load_model():

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

    return model


# =============================================================================
# TRAIN ONE EPOCH
# =============================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0

    optimizer.zero_grad(
        set_to_none=True
    )

    for step, batch in enumerate(
        loader
    ):

        pixel_values = (
            batch[
                "pixel_values"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        labels = (
            batch[
                "labels"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        outputs = model(
            pixel_values=pixel_values,
            labels=labels,
        )

        loss = outputs.loss

        total_loss += (
            loss.item()
        )

        loss = (
            loss
            / GRADIENT_ACCUMULATION
        )

        loss.backward()

        if (
            (step + 1)
            % GRADIENT_ACCUMULATION
            == 0
        ):

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

    # Flush remaining gradients.
    if (
        len(loader)
        % GRADIENT_ACCUMULATION
        != 0
    ):

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        optimizer.zero_grad(
            set_to_none=True
        )

    return (
        total_loss
        / max(len(loader), 1)
    )


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    total_loss = 0.0

    y_true = []

    y_pred = []

    for batch in loader:

        pixel_values = (
            batch[
                "pixel_values"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        labels = (
            batch[
                "labels"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        outputs = model(
            pixel_values=pixel_values,
            labels=labels,
        )

        total_loss += (
            outputs.loss.item()
        )

        predictions = (
            torch.argmax(
                outputs.logits,
                dim=-1,
            )
        )

        y_true.extend(
            labels.cpu().numpy()
        )

        y_pred.extend(
            predictions.cpu().numpy()
        )

    loss = (
        total_loss
        / max(len(loader), 1)
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
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
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "y_true": y_true,
        "y_pred": y_pred,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    set_seed(
        SEED
    )

    print("=" * 90)
    print(
        "EAGLE-MARS FINAL 5-EPOCH "
        "VIDEOMAE TRAINING"
    )
    print("=" * 90)

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "DEVICE"
    )

    print("-" * 90)

    print(
        f"Device : {device}"
    )

    if torch.cuda.is_available():

        gpu = torch.cuda.get_device_name(
            0
        )

        vram = (
            torch.cuda.get_device_properties(
                0
            ).total_memory
            / 1024**3
        )

        print(
            f"GPU    : {gpu}"
        )

        print(
            f"VRAM   : {vram:.2f} GB"
        )

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    print()
    print(
        "CONFIGURATION"
    )

    print("-" * 90)

    print(
        f"Manifest             : {MANIFEST}"
    )

    print(
        f"Model                : {MODEL_NAME}"
    )

    print(
        f"Frames               : {NUM_FRAMES}"
    )

    print(
        f"Resolution           : "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Batch size           : {BATCH_SIZE}"
    )

    print(
        f"Gradient accumulation: "
        f"{GRADIENT_ACCUMULATION}"
    )

    print(
        f"Effective batch      : "
        f"{EFFECTIVE_BATCH}"
    )

    print(
        f"Learning rate        : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Epochs               : {EPOCHS}"
    )

    print(
        f"Balanced sampler     : ENABLED"
    )

    # -------------------------------------------------------------------------
    # Load manifest
    # -------------------------------------------------------------------------

    if not MANIFEST.exists():

        raise FileNotFoundError(
            f"\nManifest not found:\n"
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

    print("-" * 90)

    print(
        df.groupby(
            [
                "split",
                "label",
            ]
        ).size()
    )

    # -------------------------------------------------------------------------
    # Split
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Processor
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------

    train_dataset = VideoClipDataset(
        train_df,
        processor,
    )

    val_dataset = VideoClipDataset(
        val_df,
        processor,
    )

    test_dataset = VideoClipDataset(
        test_df,
        processor,
    )

    # -------------------------------------------------------------------------
    # Balanced sampler
    # -------------------------------------------------------------------------

    print()
    print(
        "TRAINING CLASS COUNTS"
    )

    print("-" * 90)

    counts = (
        train_df[
            "label"
        ]
        .value_counts()
    )

    print(
        counts
    )

    normal_count = counts.get(
        "NORMAL",
        0,
    )

    fault_count = counts.get(
        "FAULT",
        0,
    )

    if normal_count == 0 or fault_count == 0:

        raise RuntimeError(
            "Both NORMAL and FAULT "
            "classes are required."
        )

    class_weights = {
        "NORMAL":
            1.0 / normal_count,

        "FAULT":
            1.0 / fault_count,
    }

    sample_weights = train_df[
        "label"
    ].map(
        class_weights
    ).values

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            sample_weights,
            dtype=torch.double,
        ),
        num_samples=len(
            train_df
        ),
        replacement=True,
    )

    print()
    print(
        "BALANCED SAMPLER ENABLED"
    )

    # -------------------------------------------------------------------------
    # DataLoaders
    # -------------------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    model = load_model()

    model.to(
        device
    )

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if BEST_MODEL_DIR.exists():

        import shutil

        shutil.rmtree(
            BEST_MODEL_DIR
        )

    BEST_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    history = []

    best_f1 = -1.0

    best_epoch = 0

    no_improvement = 0

    print()
    print("=" * 90)
    print(
        "STARTING FINAL 5-EPOCH TRAINING"
    )
    print("=" * 90)

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print()
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        val_metrics = evaluate(
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

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy":
                    val_metrics["accuracy"],
                "val_precision":
                    val_metrics["precision"],
                "val_recall":
                    val_metrics["recall"],
                "val_f1":
                    val_metrics["f1"],
            }
        )

        # -------------------------------------------------------------
        # Best model
        # -------------------------------------------------------------

        if (
            val_metrics["f1"]
            > best_f1
        ):

            best_f1 = (
                val_metrics["f1"]
            )

            best_epoch = epoch

            no_improvement = 0

            model.save_pretrained(
                BEST_MODEL_DIR
            )

            processor.save_pretrained(
                BEST_MODEL_DIR
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

            no_improvement += 1

            print()
            print(
                f"No improvement "
                f"({no_improvement}/"
                f"{EARLY_STOPPING})"
            )

            if (
                no_improvement
                >= EARLY_STOPPING
            ):

                print()
                print(
                    "EARLY STOPPING"
                )

                break

    # -------------------------------------------------------------------------
    # Save history
    # -------------------------------------------------------------------------

    with open(
        OUTPUT_DIR
        / "training_history.json",
        "w",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Load best model
    # -------------------------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "LOADING BEST MODEL"
    )
    print("=" * 90)

    model = (
        VideoMAEForVideoClassification
        .from_pretrained(
            BEST_MODEL_DIR
        )
    )

    model.to(
        device
    )

    # -------------------------------------------------------------------------
    # Final test
    # -------------------------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "FINAL TEST"
    )
    print("=" * 90)

    test_metrics = evaluate(
        model,
        test_loader,
        device,
    )

    y_true = np.asarray(
        test_metrics["y_true"]
    )

    y_pred = np.asarray(
        test_metrics["y_pred"]
    )

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

    # -------------------------------------------------------------------------
    # Classification report
    # -------------------------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        target_names=[
            "NORMAL",
            "FAULT",
        ],
        zero_division=0,
    )

    print()
    print(
        "CLASSIFICATION REPORT"
    )

    print("-" * 90)

    print(
        report
    )

    # -------------------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
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
        f"Actual NORMAL "
        f"{cm[0,0]:8d}"
        f"{cm[0,1]:7d}"
    )

    print(
        f"       FAULT "
        f"{cm[1,0]:8d}"
        f"{cm[1,1]:7d}"
    )

    # -------------------------------------------------------------------------
    # Save predictions
    # -------------------------------------------------------------------------

    predictions = test_df.copy()

    predictions[
        "true_id"
    ] = y_true

    predictions[
        "predicted_id"
    ] = y_pred

    predictions[
        "true_label"
    ] = [
        ID2LABEL[int(x)]
        for x in y_true
    ]

    predictions[
        "predicted_label"
    ] = [
        ID2LABEL[int(x)]
        for x in y_pred
    ]

    predictions.to_csv(
        OUTPUT_DIR
        / "test_predictions.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save metrics
    # -------------------------------------------------------------------------

    metrics = {
        "best_epoch":
            best_epoch,

        "best_val_f1":
            best_f1,

        "test_loss":
            test_metrics["loss"],

        "test_accuracy":
            test_metrics["accuracy"],

        "test_precision_fault":
            test_metrics["precision"],

        "test_recall_fault":
            test_metrics["recall"],

        "test_f1_fault":
            test_metrics["f1"],

        "train_clips":
            len(train_df),

        "val_clips":
            len(val_df),

        "test_clips":
            len(test_df),

        "total_clips":
            len(df),

        "total_videos":
            df["video"].nunique(),

        "epochs_requested":
            EPOCHS,

        "model":
            MODEL_NAME,
    }

    with open(
        OUTPUT_DIR
        / "test_metrics.json",
        "w",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Complete
    # -------------------------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "FINAL TRAINING COMPLETE"
    )
    print("=" * 90)

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
        f"Best model:"
    )

    print(
        BEST_MODEL_DIR
    )

    print()
    print(
        f"History:"
    )

    print(
        OUTPUT_DIR
        / "training_history.json"
    )

    print()
    print(
        f"Metrics:"
    )

    print(
        OUTPUT_DIR
        / "test_metrics.json"
    )

    print()
    print(
        f"Predictions:"
    )

    print(
        OUTPUT_DIR
        / "test_predictions.csv"
    )

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()