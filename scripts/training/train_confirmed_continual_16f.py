# =============================================================================
# EAGLE-MARS
# CONFIRMED CONTINUAL LEARNING + 16-FRAME VIDEOMAE V2
#
# Purpose:
#   Train VideoMAE-small on the confirmed continual-learning dataset while
#   specifically addressing the NORMAL-class collapse observed in v1.
#
# Dataset:
#   - Original CVAT timeframe annotations
#   - Confirmed annotations2 timeframe annotations
#   - 50-video continual-learning annotations
#
# Important:
#   - Uses ONLY clips that physically exist.
#   - Uses the already-created confirmed continual manifest.
#   - Does NOT rebuild or alter the dataset.
#   - Does NOT use the test set for model selection.
#   - Optimizes checkpoint selection using FAULT F1.
#   - Searches validation threshold instead of assuming 0.50.
#
# Output:
#   outputs/final_training_confirmed_continual_16f_v2/
# =============================================================================

import os
import json
import math
import random
import warnings
from pathlib import Path

import cv2
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
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "unified_dataset"
    / "confirmed_continual"
    / "confirmed_continual_clips_manifest.csv"
)

CLIPS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "unified_dataset"
    / "confirmed_continual"
    / "clips"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "final_training_confirmed_continual_16f_v2"
)

BEST_MODEL_DIR = OUTPUT_DIR / "best_model"

HISTORY_PATH = OUTPUT_DIR / "training_history.json"
TEST_METRICS_PATH = OUTPUT_DIR / "test_metrics.json"
TEST_PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
TEST_VIDEO_METRICS_PATH = OUTPUT_DIR / "test_video_metrics.csv"
THRESHOLD_PATH = OUTPUT_DIR / "optimal_threshold.json"
CONFIG_PATH = OUTPUT_DIR / "training_config.json"


# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------

MODEL_NAME = "MCG-NJU/videomae-small-finetuned-kinetics"

NUM_FRAMES = 16
IMAGE_SIZE = 224

NUM_CLASSES = 2

LABEL2ID = {
    "NORMAL": 0,
    "FAULT": 1,
}

ID2LABEL = {
    0: "NORMAL",
    1: "FAULT",
}


# -----------------------------------------------------------------------------
# TRAINING
# -----------------------------------------------------------------------------

BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4

NUM_EPOCHS = 12

LEARNING_RATE = 2e-6
CLASSIFIER_LEARNING_RATE = 1e-5

WEIGHT_DECAY = 0.01

WARMUP_EPOCHS = 1

EARLY_STOPPING_PATIENCE = 4

NUM_WORKERS = 0

SEED = 42


# -----------------------------------------------------------------------------
# DATA SAMPLING
# -----------------------------------------------------------------------------

# Oversample FAULT samples.
FAULT_SAMPLING_MULTIPLIER = 2.5

# Keep each epoch manageable.
MAX_TRAIN_SAMPLES = 2400


# -----------------------------------------------------------------------------
# LOSS
# -----------------------------------------------------------------------------

# Weighted CE helps prevent the model from simply predicting NORMAL.
FAULT_CLASS_WEIGHT = 2.5

LABEL_SMOOTHING = 0.03


# -----------------------------------------------------------------------------
# THRESHOLD SEARCH
# -----------------------------------------------------------------------------

THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.75
THRESHOLD_STEP = 0.01


# =============================================================================
# UTILITY
# =============================================================================

def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dirs():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    BEST_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def print_header(title):

    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def print_section(title):

    print()
    print(title)
    print("-" * 90)


# =============================================================================
# VIDEO LOADING
# =============================================================================

def read_video_clip(
    path,
    num_frames=16,
    size=224,
):

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()

        raise RuntimeError(
            f"Video contains no frames: {path}"
        )

    if total_frames >= num_frames:

        indices = np.linspace(
            0,
            total_frames - 1,
            num_frames,
            dtype=np.int64,
        )

    else:

        indices = np.linspace(
            0,
            total_frames - 1,
            num_frames,
            dtype=np.int64,
        )

    frames = []

    for idx in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(idx),
        )

        ok, frame = cap.read()

        if not ok:

            cap.release()

            raise RuntimeError(
                f"Failed reading frame {idx} from {path}"
            )

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frame = cv2.resize(
            frame,
            (size, size),
            interpolation=cv2.INTER_AREA,
        )

        frames.append(frame)

    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return np.asarray(
        frames,
        dtype=np.uint8,
    )


# =============================================================================
# DATASET
# =============================================================================

class ClipDataset(Dataset):

    def __init__(
        self,
        dataframe,
        processor,
        clips_dir,
        training=False,
    ):

        self.df = dataframe.reset_index(drop=True)

        self.processor = processor

        self.clips_dir = Path(clips_dir)

        self.training = training

    def __len__(self):

        return len(self.df)

    def _resolve_clip(self, row):

        candidates = []

        if "clip_path" in row.index:

            value = str(
                row["clip_path"]
            ).strip()

            if value:
                candidates.append(
                    Path(value)
                )

        if "file" in row.index:

            value = str(
                row["file"]
            ).strip()

            if value:

                p = Path(value)

                if p.is_absolute():
                    candidates.append(p)
                else:
                    candidates.append(
                        self.clips_dir / p
                    )

        if "clip" in row.index:

            value = str(
                row["clip"]
            ).strip()

            if value:

                p = Path(value)

                if p.is_absolute():
                    candidates.append(p)
                else:
                    candidates.append(
                        self.clips_dir / p
                    )

        for p in candidates:

            if p.exists():
                return p

        raise FileNotFoundError(
            "Could not resolve clip for row:\n"
            + str(row.to_dict())
        )

    def __getitem__(self, index):

        row = self.df.iloc[index]

        clip_path = self._resolve_clip(row)

        frames = read_video_clip(
            clip_path,
            NUM_FRAMES,
            IMAGE_SIZE,
        )

        # -------------------------------------------------------------
        # Lightweight temporal/spatial augmentation
        # -------------------------------------------------------------

        if self.training:

            # Horizontal flip with low probability.
            if random.random() < 0.25:

                frames = frames[:, :, ::-1, :].copy()

            # Small brightness variation.
            if random.random() < 0.20:

                factor = random.uniform(
                    0.90,
                    1.10,
                )

                frames = np.clip(
                    frames.astype(np.float32)
                    * factor,
                    0,
                    255,
                ).astype(np.uint8)

        encoded = self.processor(
            list(frames),
            return_tensors="pt",
        )

        pixel_values = encoded[
            "pixel_values"
        ].squeeze(0)

        label = LABEL2ID[
            str(row["label"]).upper()
        ]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(
                label,
                dtype=torch.long,
            ),
            "index": index,
        }


# =============================================================================
# MANIFEST
# =============================================================================

def load_manifest():

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
        )

    df = pd.read_csv(
        MANIFEST_PATH
    )

    required = [
        "label",
        "split",
    ]

    for col in required:

        if col not in df.columns:

            raise RuntimeError(
                f"Required column missing: {col}"
            )

    df["label"] = (
        df["label"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["split"] = (
        df["split"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df = df[
        df["label"].isin(
            ["NORMAL", "FAULT"]
        )
    ].copy()

    df = df[
        df["split"].isin(
            ["train", "val", "test"]
        )
    ].copy()

    return df.reset_index(
        drop=True
    )


# =============================================================================
# CLIP PATH VALIDATION
# =============================================================================

def validate_clip_paths(df):

    print_section(
        "VALIDATING PHYSICAL CLIPS"
    )

    resolved = []

    missing = []

    for idx, row in df.iterrows():

        path = None

        candidates = []

        if "clip_path" in row.index:

            value = str(
                row["clip_path"]
            ).strip()

            if value:
                candidates.append(
                    Path(value)
                )

        if "file" in row.index:

            value = str(
                row["file"]
            ).strip()

            if value:

                p = Path(value)

                if p.is_absolute():
                    candidates.append(p)
                else:
                    candidates.append(
                        CLIPS_DIR / p
                    )

        if "clip" in row.index:

            value = str(
                row["clip"]
            ).strip()

            if value:

                p = Path(value)

                if p.is_absolute():
                    candidates.append(p)
                else:
                    candidates.append(
                        CLIPS_DIR / p
                    )

        for p in candidates:

            if p.exists():

                path = str(
                    p.resolve()
                )

                break

        if path is None:

            missing.append(
                idx
            )

            resolved.append("")

        else:

            resolved.append(
                path
            )

    df = df.copy()

    df["resolved_clip_path"] = resolved

    if missing:

        print(
            f"Missing clips: {len(missing)}"
        )

        df = df[
            df["resolved_clip_path"] != ""
        ].copy()

    else:

        print(
            "All clips resolved."
        )

    return df.reset_index(
        drop=True
    )


# =============================================================================
# DATA SPLIT
# =============================================================================

def prepare_splits(df):

    train_df = df[
        df["split"] == "train"
    ].copy()

    val_df = df[
        df["split"] == "val"
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    print_section(
        "DATASET DISTRIBUTION"
    )

    print(
        df.groupby(
            ["split", "label"]
        ).size()
    )

    print()

    print(
        f"TRAIN : {len(train_df)}"
    )

    print(
        f"VAL   : {len(val_df)}"
    )

    print(
        f"TEST  : {len(test_df)}"
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# =============================================================================
# VIDEO DIVERSITY
# =============================================================================

def print_video_statistics(
    train_df,
    val_df,
    test_df,
):

    print_section(
        "VIDEO-LEVEL DISTRIBUTION"
    )

    for name, data in [
        ("TRAIN", train_df),
        ("VAL", val_df),
        ("TEST", test_df),
    ]:

        if "video" in data.columns:

            videos = data[
                "video"
            ].nunique()

        elif "video_id" in data.columns:

            videos = data[
                "video_id"
            ].nunique()

        elif "physical_video" in data.columns:

            videos = data[
                "physical_video"
            ].nunique()

        else:

            videos = -1

        print(
            f"{name:<6}: "
            f"{videos} videos / "
            f"{len(data)} clips"
        )


# =============================================================================
# SAMPLER
# =============================================================================

def build_training_sampler(
    train_df,
):

    labels = (
        train_df["label"]
        .map(LABEL2ID)
        .astype(int)
        .values
    )

    fault_count = int(
        np.sum(labels == 1)
    )

    normal_count = int(
        np.sum(labels == 0)
    )

    print_section(
        "FAULT-BALANCED SAMPLER"
    )

    print(
        f"Original NORMAL : {normal_count}"
    )

    print(
        f"Original FAULT  : {fault_count}"
    )

    weights = np.ones(
        len(train_df),
        dtype=np.float64,
    )

    weights[
        labels == 1
    ] *= FAULT_SAMPLING_MULTIPLIER

    # Give every sample a minimum weight.
    weights = np.maximum(
        weights,
        1e-8,
    )

    if MAX_TRAIN_SAMPLES > 0:

        samples_per_epoch = min(
            MAX_TRAIN_SAMPLES,
            len(train_df),
        )

    else:

        samples_per_epoch = len(
            train_df
        )

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            weights,
            dtype=torch.double,
        ),
        num_samples=samples_per_epoch,
        replacement=True,
    )

    print(
        f"FAULT multiplier : "
        f"{FAULT_SAMPLING_MULTIPLIER}"
    )

    print(
        f"Samples / epoch  : "
        f"{samples_per_epoch}"
    )

    return sampler


# =============================================================================
# COLLATE
# =============================================================================

def collate_fn(batch):

    pixel_values = torch.stack(
        [
            item["pixel_values"]
            for item in batch
        ]
    )

    labels = torch.stack(
        [
            item["labels"]
            for item in batch
        ]
    )

    indices = torch.tensor(
        [
            item["index"]
            for item in batch
        ],
        dtype=torch.long,
    )

    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "index": indices,
    }


# =============================================================================
# MODEL
# =============================================================================

def build_model():

    print_section(
        "LOADING VIDEOMAE-SMALL"
    )

    model = (
        VideoMAEForVideoClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=NUM_CLASSES,
            label2id=LABEL2ID,
            id2label=ID2LABEL,
            ignore_mismatched_sizes=True,
        )
    )

    return model


# =============================================================================
# CLASSIFIER PARAMETERS
# =============================================================================

def get_classifier_parameters(
    model
):

    params = []

    for name, param in model.named_parameters():

        if (
            "classifier" in name
            or "score" in name
        ):

            params.append(
                param
            )

    return params


# =============================================================================
# OPTIMIZER
# =============================================================================

def build_optimizer(model):

    classifier_params = []

    backbone_params = []

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        if (
            "classifier" in name
            or "score" in name
        ):

            classifier_params.append(
                param
            )

        else:

            backbone_params.append(
                param
            )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": LEARNING_RATE,
            },
            {
                "params": classifier_params,
                "lr": CLASSIFIER_LEARNING_RATE,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    return optimizer


# =============================================================================
# SCHEDULER
# =============================================================================

def build_scheduler(
    optimizer
):

    total_steps = (
        NUM_EPOCHS
        * math.ceil(
            MAX_TRAIN_SAMPLES
            / BATCH_SIZE
            / GRADIENT_ACCUMULATION
        )
    )

    warmup_steps = (
        WARMUP_EPOCHS
        * math.ceil(
            MAX_TRAIN_SAMPLES
            / BATCH_SIZE
            / GRADIENT_ACCUMULATION
        )
    )

    def lr_lambda(step):

        if step < warmup_steps:

            if warmup_steps == 0:
                return 1.0

            return (
                float(step + 1)
                / float(warmup_steps)
            )

        progress = (
            step - warmup_steps
        ) / max(
            1,
            total_steps - warmup_steps,
        )

        return max(
            0.05,
            0.5
            * (
                1
                + math.cos(
                    math.pi
                    * progress
                )
            ),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )

    return scheduler


# =============================================================================
# LOSS
# =============================================================================

def build_loss():

    weights = torch.tensor(
        [
            1.0,
            FAULT_CLASS_WEIGHT,
        ],
        dtype=torch.float32,
    )

    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=LABEL_SMOOTHING,
    )

    return criterion


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    y_true,
    y_prob,
    threshold=0.5,
):

    y_true = np.asarray(
        y_true
    )

    y_prob = np.asarray(
        y_prob
    )

    y_pred = (
        y_prob >= threshold
    ).astype(int)

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


# =============================================================================
# THRESHOLD OPTIMIZATION
# =============================================================================

def find_best_threshold(
    y_true,
    y_prob,
):

    best = None

    thresholds = np.arange(
        THRESHOLD_MIN,
        THRESHOLD_MAX
        + THRESHOLD_STEP / 2,
        THRESHOLD_STEP,
    )

    for threshold in thresholds:

        metrics = calculate_metrics(
            y_true,
            y_prob,
            float(threshold),
        )

        # Primary objective: FAULT F1.
        # Secondary: FAULT recall.
        # Tertiary: accuracy.
        score = (
            metrics["f1"],
            metrics["recall"],
            metrics["accuracy"],
        )

        if (
            best is None
            or score > best["score"]
        ):

            best = {
                "threshold": float(
                    threshold
                ),
                "metrics": metrics,
                "score": score,
            }

    return best


# =============================================================================
# TRAIN ONE EPOCH
# =============================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
    epoch,
):

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    running_loss = 0.0

    num_batches = 0

    accumulation_counter = 0

    for batch_idx, batch in enumerate(
        loader
    ):

        pixel_values = (
            batch["pixel_values"]
            .to(
                device,
                non_blocking=True,
            )
        )

        labels = (
            batch["labels"]
            .to(
                device,
                non_blocking=True,
            )
        )

        outputs = model(
            pixel_values=pixel_values
        )

        logits = outputs.logits

        loss = criterion(
            logits,
            labels,
        )

        loss_for_backward = (
            loss
            / GRADIENT_ACCUMULATION
        )

        loss_for_backward.backward()

        accumulation_counter += 1

        if (
            accumulation_counter
            >= GRADIENT_ACCUMULATION
            or batch_idx
            == len(loader) - 1
        ):

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            scheduler.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            accumulation_counter = 0

        running_loss += float(
            loss.detach().cpu()
        )

        num_batches += 1

    return (
        running_loss
        / max(
            1,
            num_batches,
        )
    )


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0

    num_batches = 0

    all_labels = []

    all_probs = []

    all_indices = []

    for batch in loader:

        pixel_values = (
            batch["pixel_values"]
            .to(
                device,
                non_blocking=True,
            )
        )

        labels = (
            batch["labels"]
            .to(
                device,
                non_blocking=True,
            )
        )

        outputs = model(
            pixel_values=pixel_values
        )

        logits = outputs.logits

        loss = criterion(
            logits,
            labels,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[:, 1]

        running_loss += float(
            loss.detach().cpu()
        )

        num_batches += 1

        all_labels.extend(
            labels.cpu().numpy().tolist()
        )

        all_probs.extend(
            probabilities.cpu()
            .numpy()
            .tolist()
        )

        all_indices.extend(
            batch["index"]
            .cpu()
            .numpy()
            .tolist()
        )

    y_true = np.asarray(
        all_labels,
        dtype=np.int64,
    )

    y_prob = np.asarray(
        all_probs,
        dtype=np.float32,
    )

    best_threshold = find_best_threshold(
        y_true,
        y_prob,
    )

    default_metrics = calculate_metrics(
        y_true,
        y_prob,
        0.50,
    )

    optimal_metrics = best_threshold[
        "metrics"
    ]

    return {
        "loss": (
            running_loss
            / max(
                1,
                num_batches,
            )
        ),
        "labels": y_true,
        "probabilities": y_prob,
        "indices": np.asarray(
            all_indices,
            dtype=np.int64,
        ),
        "default_metrics": default_metrics,
        "optimal_threshold": best_threshold[
            "threshold"
        ],
        "optimal_metrics": optimal_metrics,
    }


# =============================================================================
# SAVE CHECKPOINT
# =============================================================================

def save_best_model(
    model,
    processor,
    metadata,
):

    if BEST_MODEL_DIR.exists():

        for p in BEST_MODEL_DIR.iterdir():

            if p.is_file():

                p.unlink()

    model.save_pretrained(
        BEST_MODEL_DIR
    )

    processor.save_pretrained(
        BEST_MODEL_DIR
    )

    with open(
        BEST_MODEL_DIR
        / "checkpoint_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )


# =============================================================================
# VIDEO METRICS
# =============================================================================

def get_video_column(df):

    for col in [
        "physical_video",
        "video",
        "video_id",
    ]:

        if col in df.columns:
            return col

    return None


def calculate_video_metrics(
    df,
    probabilities,
    threshold,
):

    df = df.copy()

    df = df.reset_index(
        drop=True
    )

    df["fault_probability"] = (
        probabilities
    )

    df["predicted_label"] = np.where(
        df["fault_probability"]
        >= threshold,
        "FAULT",
        "NORMAL",
    )

    video_col = get_video_column(
        df
    )

    if video_col is None:

        return pd.DataFrame()

    rows = []

    for video, group in df.groupby(
        video_col
    ):

        actual = (
            group["label"]
            .astype(str)
            .str.upper()
            == "FAULT"
        )

        predicted = (
            group["predicted_label"]
            == "FAULT"
        )

        actual_fault = int(
            actual.sum()
        )

        predicted_fault = int(
            predicted.sum()
        )

        detected_fault = int(
            (
                actual
                & predicted
            ).sum()
        )

        if actual_fault > 0:

            fault_recall = (
                detected_fault
                / actual_fault
            )

        else:

            fault_recall = np.nan

        if predicted_fault > 0:

            fault_precision = (
                detected_fault
                / predicted_fault
            )

        else:

            fault_precision = np.nan

        accuracy = float(
            (
                group["label"]
                .astype(str)
                .str.upper()
                == group[
                    "predicted_label"
                ]
            ).mean()
        )

        rows.append(
            {
                "video": video,
                "clips": len(group),
                "actual_fault": actual_fault,
                "predicted_fault": predicted_fault,
                "detected_fault": detected_fault,
                "fault_recall": fault_recall,
                "fault_precision": fault_precision,
                "accuracy": accuracy,
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "video"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print_header(
        "EAGLE-MARS CONFIRMED CONTINUAL + 16-FRAME VIDEOMAE V2"
    )

    seed_everything(
        SEED
    )

    ensure_dirs()

    # -------------------------------------------------------------------------
    # DEVICE
    # -------------------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print_section(
        "DEVICE"
    )

    print(
        f"Device : {device}"
    )

    if torch.cuda.is_available():

        gpu_name = (
            torch.cuda.get_device_name(
                0
            )
        )

        total_memory = (
            torch.cuda.get_device_properties(
                0
            ).total_memory
            / 1024**3
        )

        print(
            f"GPU    : {gpu_name}"
        )

        print(
            f"VRAM   : {total_memory:.2f} GB"
        )

    # -------------------------------------------------------------------------
    # CONFIG
    # -------------------------------------------------------------------------

    print_section(
        "CONFIGURATION"
    )

    print(
        f"Manifest                 : "
        f"{MANIFEST_PATH}"
    )

    print(
        f"Clips                    : "
        f"{CLIPS_DIR}"
    )

    print(
        f"Output                   : "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Model                    : "
        f"{MODEL_NAME}"
    )

    print(
        f"Frames                   : "
        f"{NUM_FRAMES}"
    )

    print(
        f"Resolution               : "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Batch size               : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Gradient accumulation    : "
        f"{GRADIENT_ACCUMULATION}"
    )

    print(
        f"Effective batch          : "
        f"{BATCH_SIZE * GRADIENT_ACCUMULATION}"
    )

    print(
        f"Backbone LR              : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Classifier LR            : "
        f"{CLASSIFIER_LEARNING_RATE}"
    )

    print(
        f"Epochs                   : "
        f"{NUM_EPOCHS}"
    )

    print(
        f"Early stopping           : "
        f"{EARLY_STOPPING_PATIENCE}"
    )

    print(
        f"Fault sampling multiplier: "
        f"{FAULT_SAMPLING_MULTIPLIER}"
    )

    print(
        f"Fault loss weight        : "
        f"{FAULT_CLASS_WEIGHT}"
    )

    # -------------------------------------------------------------------------
    # SAVE CONFIG
    # -------------------------------------------------------------------------

    config = {
        "manifest": str(
            MANIFEST_PATH
        ),
        "clips_dir": str(
            CLIPS_DIR
        ),
        "model": MODEL_NAME,
        "num_frames": NUM_FRAMES,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "effective_batch": (
            BATCH_SIZE
            * GRADIENT_ACCUMULATION
        ),
        "learning_rate": LEARNING_RATE,
        "classifier_learning_rate": (
            CLASSIFIER_LEARNING_RATE
        ),
        "epochs": NUM_EPOCHS,
        "early_stopping_patience": (
            EARLY_STOPPING_PATIENCE
        ),
        "fault_sampling_multiplier": (
            FAULT_SAMPLING_MULTIPLIER
        ),
        "fault_class_weight": (
            FAULT_CLASS_WEIGHT
        ),
        "label_smoothing": (
            LABEL_SMOOTHING
        ),
        "seed": SEED,
    }

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # MANIFEST
    # -------------------------------------------------------------------------

    df = load_manifest()

    print_section(
        "INPUT MANIFEST"
    )

    print(
        f"Rows: {len(df)}"
    )

    print()

    print(
        df.groupby(
            ["source", "label"]
        ).size()
    )

    # -------------------------------------------------------------------------
    # CLIP VALIDATION
    # -------------------------------------------------------------------------

    df = validate_clip_paths(
        df
    )

    if len(df) == 0:

        raise RuntimeError(
            "No valid clips available."
        )

    # -------------------------------------------------------------------------
    # SPLITS
    # -------------------------------------------------------------------------

    train_df, val_df, test_df = (
        prepare_splits(df)
    )

    print_video_statistics(
        train_df,
        val_df,
        test_df,
    )

    # -------------------------------------------------------------------------
    # PROCESSOR
    # -------------------------------------------------------------------------

    print_section(
        "LOADING VIDEOMAE PROCESSOR"
    )

    processor = (
        VideoMAEImageProcessor
        .from_pretrained(
            MODEL_NAME
        )
    )

    # -------------------------------------------------------------------------
    # DATASETS
    # -------------------------------------------------------------------------

    train_dataset = ClipDataset(
        train_df,
        processor,
        CLIPS_DIR,
        training=True,
    )

    val_dataset = ClipDataset(
        val_df,
        processor,
        CLIPS_DIR,
        training=False,
    )

    test_dataset = ClipDataset(
        test_df,
        processor,
        CLIPS_DIR,
        training=False,
    )

    # -------------------------------------------------------------------------
    # SAMPLER
    # -------------------------------------------------------------------------

    sampler = build_training_sampler(
        train_df
    )

    # -------------------------------------------------------------------------
    # LOADERS
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
    # MODEL
    # -------------------------------------------------------------------------

    model = build_model()

    model.to(
        device
    )

    # -------------------------------------------------------------------------
    # OPTIMIZER
    # -------------------------------------------------------------------------

    optimizer = build_optimizer(
        model
    )

    scheduler = build_scheduler(
        optimizer
    )

    criterion = build_loss()

    criterion = criterion.to(
        device
    )

    # -------------------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------------------

    print_header(
        "STARTING V2 TRAINING"
    )

    history = []

    best_f1 = -1.0

    best_recall = -1.0

    best_epoch = -1

    epochs_without_improvement = 0

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        print()
        print(
            f"Epoch {epoch}/{NUM_EPOCHS}"
        )

        print(
            "-" * 90
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            epoch,
        )

        val_result = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        val_metrics = (
            val_result[
                "optimal_metrics"
            ]
        )

        threshold = (
            val_result[
                "optimal_threshold"
            ]
        )

        print(
            f"Train Loss : "
            f"{train_loss:.4f}"
        )

        print(
            f"Val Loss   : "
            f"{val_result['loss']:.4f}"
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

        print(
            f"Threshold  : "
            f"{threshold:.2f}"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_result[
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
            "val_threshold": threshold,
            "default_threshold_f1": (
                val_result[
                    "default_metrics"
                ]["f1"]
            ),
            "learning_rates": [
                group["lr"]
                for group
                in optimizer.param_groups
            ],
        }

        history.append(
            epoch_record
        )

        # -------------------------------------------------------------
        # Checkpoint selection
        #
        # Primary:
        #   validation FAULT F1
        #
        # Secondary:
        #   validation FAULT recall
        # -------------------------------------------------------------

        improved = False

        if val_metrics["f1"] > best_f1:

            improved = True

        elif (
            abs(
                val_metrics["f1"]
                - best_f1
            )
            < 1e-8
            and val_metrics[
                "recall"
            ]
            > best_recall
        ):

            improved = True

        if improved:

            best_f1 = val_metrics[
                "f1"
            ]

            best_recall = val_metrics[
                "recall"
            ]

            best_epoch = epoch

            epochs_without_improvement = 0

            metadata = {
                "epoch": epoch,
                "val_f1": best_f1,
                "val_recall": best_recall,
                "val_precision": val_metrics[
                    "precision"
                ],
                "val_accuracy": val_metrics[
                    "accuracy"
                ],
                "threshold": threshold,
            }

            save_best_model(
                model,
                processor,
                metadata,
            )

            with open(
                THRESHOLD_PATH,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    {
                        "epoch": epoch,
                        "threshold": threshold,
                        "validation_metrics": val_metrics,
                    },
                    f,
                    indent=2,
                )

            print()
            print(
                "BEST MODEL SAVED"
            )

            print(
                f"Validation F1 : "
                f"{best_f1:.4f}"
            )

            print(
                f"Validation Recall : "
                f"{best_recall:.4f}"
            )

        else:

            epochs_without_improvement += 1

            print()
            print(
                f"No improvement "
                f"({epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE})"
            )

        # -------------------------------------------------------------
        # Save history every epoch.
        # -------------------------------------------------------------

        with open(
            HISTORY_PATH,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                history,
                f,
                indent=2,
            )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "EARLY STOPPING"
            )

            break

    # -------------------------------------------------------------------------
    # LOAD BEST MODEL
    # -------------------------------------------------------------------------

    print_header(
        "LOADING BEST MODEL"
    )

    if not BEST_MODEL_DIR.exists():

        raise RuntimeError(
            "Best model was never saved."
        )

    best_model = (
        VideoMAEForVideoClassification
        .from_pretrained(
            BEST_MODEL_DIR
        )
    )

    best_model.to(
        device
    )

    # -------------------------------------------------------------------------
    # BEST THRESHOLD
    # -------------------------------------------------------------------------

    if THRESHOLD_PATH.exists():

        with open(
            THRESHOLD_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            threshold_data = json.load(
                f
            )

        test_threshold = float(
            threshold_data[
                "threshold"
            ]
        )

    else:

        test_threshold = 0.50

    print(
        f"Selected validation threshold: "
        f"{test_threshold:.2f}"
    )

    # -------------------------------------------------------------------------
    # FINAL TEST
    # -------------------------------------------------------------------------

    print_header(
        "FINAL TEST"
    )

    test_result = evaluate(
        best_model,
        test_loader,
        criterion,
        device,
    )

    y_true = test_result[
        "labels"
    ]

    y_prob = test_result[
        "probabilities"
    ]

    # IMPORTANT:
    # Test threshold comes ONLY from validation.
    test_metrics = calculate_metrics(
        y_true,
        y_prob,
        test_threshold,
    )

    default_test_metrics = calculate_metrics(
        y_true,
        y_prob,
        0.50,
    )

    print(
        f"Test Loss      : "
        f"{test_result['loss']:.4f}"
    )

    print(
        f"Test Accuracy  : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test Precision : "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Test Recall    : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"Test F1        : "
        f"{test_metrics['f1']:.4f}"
    )

    print()
    print(
        f"Default 0.50 F1: "
        f"{default_test_metrics['f1']:.4f}"
    )

    # -------------------------------------------------------------------------
    # CLASSIFICATION REPORT
    # -------------------------------------------------------------------------

    y_pred = (
        y_prob
        >= test_threshold
    ).astype(int)

    print_section(
        "CLASSIFICATION REPORT"
    )

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[
                0,
                1,
            ],
            target_names=[
                "NORMAL",
                "FAULT",
            ],
            zero_division=0,
        )
    )

    # -------------------------------------------------------------------------
    # CONFUSION MATRIX
    # -------------------------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
    )

    print_section(
        "CONFUSION MATRIX"
    )

    print(
        "                 Predicted"
    )

    print(
        "              NORMAL  FAULT"
    )

    print(
        f"Actual NORMAL "
        f"{cm[0,0]:7d} "
        f"{cm[0,1]:6d}"
    )

    print(
        f"Actual FAULT  "
        f"{cm[1,0]:7d} "
        f"{cm[1,1]:6d}"
    )

    # -------------------------------------------------------------------------
    # TEST PREDICTIONS
    # -------------------------------------------------------------------------

    prediction_df = test_df.copy()

    prediction_df = (
        prediction_df
        .reset_index(drop=True)
    )

    prediction_df[
        "fault_probability"
    ] = y_prob

    prediction_df[
        "predicted_label"
    ] = np.where(
        y_prob
        >= test_threshold,
        "FAULT",
        "NORMAL",
    )

    prediction_df[
        "threshold_used"
    ] = test_threshold

    prediction_df.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False,
    )

    # -------------------------------------------------------------------------
    # VIDEO-LEVEL
    # -------------------------------------------------------------------------

    print_header(
        "VIDEO-LEVEL TEST PERFORMANCE"
    )

    video_metrics = calculate_video_metrics(
        prediction_df,
        y_prob,
        test_threshold,
    )

    if len(video_metrics) > 0:

        print(
            video_metrics.to_string(
                index=False
            )
        )

        video_metrics.to_csv(
            TEST_VIDEO_METRICS_PATH,
            index=False,
        )

        macro_video_accuracy = float(
            video_metrics[
                "accuracy"
            ].mean()
        )

        fault_recall_values = (
            video_metrics[
                "fault_recall"
            ]
            .dropna()
        )

        fault_precision_values = (
            video_metrics[
                "fault_precision"
            ]
            .dropna()
        )

        if len(
            fault_recall_values
        ):

            macro_video_fault_recall = float(
                fault_recall_values.mean()
            )

        else:

            macro_video_fault_recall = 0.0

        if len(
            fault_precision_values
        ):

            macro_video_fault_precision = float(
                fault_precision_values.mean()
            )

        else:

            macro_video_fault_precision = 0.0

        print()

        print(
            f"Macro Video Accuracy        : "
            f"{macro_video_accuracy:.4f}"
        )

        print(
            f"Macro Video Fault Recall    : "
            f"{macro_video_fault_recall:.4f}"
        )

        print(
            f"Macro Video Fault Precision : "
            f"{macro_video_fault_precision:.4f}"
        )

    else:

        macro_video_accuracy = None

        macro_video_fault_recall = None

        macro_video_fault_precision = None

        print(
            "Video column not available."
        )

    # -------------------------------------------------------------------------
    # METRICS JSON
    # -------------------------------------------------------------------------

    metrics_output = {

        "model": MODEL_NAME,

        "frames": NUM_FRAMES,

        "resolution": IMAGE_SIZE,

        "best_epoch": best_epoch,

        "best_validation_f1": best_f1,

        "best_validation_recall": best_recall,

        "test_threshold": test_threshold,

        "test_loss": test_result[
            "loss"
        ],

        "test_accuracy": test_metrics[
            "accuracy"
        ],

        "test_precision": test_metrics[
            "precision"
        ],

        "test_recall": test_metrics[
            "recall"
        ],

        "test_f1": test_metrics[
            "f1"
        ],

        "default_threshold_0_50": {

            "accuracy": default_test_metrics[
                "accuracy"
            ],

            "precision": default_test_metrics[
                "precision"
            ],

            "recall": default_test_metrics[
                "recall"
            ],

            "f1": default_test_metrics[
                "f1"
            ],
        },

        "confusion_matrix": cm.tolist(),

        "macro_video_accuracy":
            macro_video_accuracy,

        "macro_video_fault_recall":
            macro_video_fault_recall,

        "macro_video_fault_precision":
            macro_video_fault_precision,

        "train_samples": len(
            train_df
        ),

        "validation_samples": len(
            val_df
        ),

        "test_samples": len(
            test_df
        ),
    }

    with open(
        TEST_METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics_output,
            f,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------

    print_header(
        "V2 TRAINING COMPLETE"
    )

    print(
        f"Best epoch       : "
        f"{best_epoch}"
    )

    print(
        f"Best Val F1      : "
        f"{best_f1:.4f}"
    )

    print(
        f"Best Val Recall  : "
        f"{best_recall:.4f}"
    )

    print(
        f"Test threshold   : "
        f"{test_threshold:.2f}"
    )

    print(
        f"Test Accuracy    : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"FAULT Precision  : "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"FAULT Recall     : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"FAULT F1         : "
        f"{test_metrics['f1']:.4f}"
    )

    print()

    print(
        "Best model:"
    )

    print(
        BEST_MODEL_DIR
    )

    print()

    print(
        "History:"
    )

    print(
        HISTORY_PATH
    )

    print()

    print(
        "Metrics:"
    )

    print(
        TEST_METRICS_PATH
    )

    print()

    print(
        "Predictions:"
    )

    print(
        TEST_PREDICTIONS_PATH
    )

    print()

    print(
        "Video metrics:"
    )

    print(
        TEST_VIDEO_METRICS_PATH
    )

    print()

    print(
        "Threshold:"
    )

    print(
        THRESHOLD_PATH
    )

    print("=" * 90)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()