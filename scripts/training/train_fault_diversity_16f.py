from pathlib import Path
import json
import random
import shutil

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader, Sampler
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
# EAGLE-MARS
# FAULT-DIVERSITY + 16-FRAME VIDEOMAE TRAINING
# =============================================================================
#
# Purpose:
#
#   Train VideoMAE-small while ensuring that every training epoch contains
#   samples from ALL videos that contain at least one FAULT.
#
# Training strategy:
#
#   1. Identify all fault-containing training videos.
#   2. Use ALL fault-containing videos every epoch.
#   3. Randomly select the same number of NORMAL-only videos.
#   4. Use at most 32 clips from each selected video.
#   5. Shuffle all selected clips.
#
# Important:
#
#   The pretrained MCG-NJU VideoMAE-small Kinetics checkpoint used here
#   expects 16-frame input.
#
#   Therefore this experiment uses 16 frames, NOT 32.
#
# =============================================================================


# =============================================================================
# PATHS
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
    / "final_training_fault_diversity_16f"
)

BEST_MODEL_DIR = (
    OUTPUT_DIR
    / "best_model"
)

MODEL_NAME = (
    "MCG-NJU/videomae-small-finetuned-kinetics"
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

MAX_CLIPS_PER_VIDEO = 32

NUM_WORKERS = 0

SEED = 42


# =============================================================================
# LABELS
# =============================================================================

LABEL2ID = {
    "NORMAL": 0,
    "FAULT": 1,
}

ID2LABEL = {
    0: "NORMAL",
    1: "FAULT",
}


# =============================================================================
# RANDOM SEED
# =============================================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# =============================================================================
# LOAD NPZ / NPY CLIP
# =============================================================================

def load_clip_file(file_path):

    file_path = Path(file_path)

    if not file_path.exists():

        raise FileNotFoundError(
            "\nClip file not found:\n"
            f"{file_path}"
        )

    data = np.load(
        file_path,
        allow_pickle=False,
    )

    # -------------------------------------------------------------------------
    # NPZ
    # -------------------------------------------------------------------------

    if isinstance(
        data,
        np.lib.npyio.NpzFile,
    ):

        keys = list(
            data.files
        )

        if len(keys) == 0:

            data.close()

            raise RuntimeError(
                f"Empty NPZ file:\n{file_path}"
            )

        preferred_keys = [
            "frames",
            "clip",
            "array",
            "data",
            "x",
        ]

        selected_key = None

        for key in preferred_keys:

            if key in keys:

                selected_key = key

                break

        if selected_key is None:

            selected_key = keys[0]

        array = data[
            selected_key
        ]

        data.close()

        return np.asarray(array)

    # -------------------------------------------------------------------------
    # NPY
    # -------------------------------------------------------------------------

    return np.asarray(data)


# =============================================================================
# NORMALIZE CLIP FORMAT
# =============================================================================

def normalize_clip(array):

    if array.ndim != 4:

        raise RuntimeError(
            "Expected a 4D video clip. "
            f"Received shape: {array.shape}"
        )

    # -------------------------------------------------------------------------
    # Expected format:
    #
    # (T, H, W, C)
    #
    # Example:
    #
    # (16, 224, 224, 3)
    # -------------------------------------------------------------------------

    if (
        array.shape[0] == NUM_FRAMES
        and array.shape[-1] == 3
    ):

        frames = array

    # -------------------------------------------------------------------------
    # Alternative:
    #
    # (T, C, H, W)
    # -------------------------------------------------------------------------

    elif (
        array.shape[0] == NUM_FRAMES
        and array.shape[1] == 3
    ):

        frames = np.transpose(
            array,
            (
                0,
                2,
                3,
                1,
            ),
        )

    else:

        raise RuntimeError(
            "Unsupported clip shape: "
            f"{array.shape}\n"
            f"Expected ({NUM_FRAMES}, H, W, 3) "
            f"or ({NUM_FRAMES}, 3, H, W)."
        )

    if frames.shape[0] != NUM_FRAMES:

        raise RuntimeError(
            f"Expected {NUM_FRAMES} frames, "
            f"got {frames.shape[0]}"
        )

    # -------------------------------------------------------------------------
    # Convert to uint8 if required
    # -------------------------------------------------------------------------

    if frames.dtype != np.uint8:

        if np.issubdtype(
            frames.dtype,
            np.floating,
        ):

            if (
                frames.min() >= 0.0
                and frames.max() <= 1.0
            ):

                frames = frames * 255.0

        frames = np.clip(
            frames,
            0,
            255,
        ).astype(
            np.uint8
        )

    return frames


# =============================================================================
# DATASET
# =============================================================================

class ClipDataset(Dataset):

    def __init__(
        self,
        dataframe,
        processor,
    ):

        self.df = (
            dataframe
            .reset_index(
                drop=True
            )
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

        file_path = Path(
            str(
                row["file"]
            )
        )

        # Resolve relative paths.

        if not file_path.is_absolute():

            file_path = (
                ROOT
                / file_path
            )

        array = load_clip_file(
            file_path
        )

        frames = normalize_clip(
            array
        )

        # ---------------------------------------------------------------------
        # VideoMAE preprocessing
        # ---------------------------------------------------------------------

        inputs = self.processor(
            list(frames),
            return_tensors="pt",
        )

        pixel_values = (
            inputs[
                "pixel_values"
            ]
            .squeeze(0)
        )

        label_name = str(
            row["label"]
        ).upper()

        if label_name not in LABEL2ID:

            raise RuntimeError(
                f"Unknown label: {label_name}"
            )

        label = LABEL2ID[
            label_name
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
# FAULT-DIVERSITY SAMPLER
# =============================================================================

class FaultDiversitySampler(Sampler):

    """
    Every epoch:

        ALL fault-containing videos
        +
        equal number of NORMAL-only videos.

    At most MAX_CLIPS_PER_VIDEO clips
    are selected from each video.
    """

    def __init__(
        self,
        dataframe,
        max_clips_per_video,
        seed,
    ):

        self.df = (
            dataframe
            .reset_index(
                drop=True
            )
        )

        self.max_clips = (
            max_clips_per_video
        )

        self.seed = seed

        self.epoch = 0

        # ---------------------------------------------------------------------
        # Determine video-level labels
        # ---------------------------------------------------------------------

        video_labels = {}

        for _, row in (
            self.df.iterrows()
        ):

            video = str(
                row["video"]
            )

            label = str(
                row["label"]
            ).upper()

            if video not in video_labels:

                video_labels[
                    video
                ] = set()

            video_labels[
                video
            ].add(
                label
            )

        self.fault_videos = sorted(
            [
                video
                for video, labels
                in video_labels.items()
                if "FAULT" in labels
            ]
        )

        self.normal_videos = sorted(
            [
                video
                for video, labels
                in video_labels.items()
                if "FAULT" not in labels
            ]
        )

    def set_epoch(
        self,
        epoch,
    ):

        self.epoch = epoch

    def __iter__(self):

        rng = random.Random(
            self.seed
            + self.epoch
        )

        # ---------------------------------------------------------------------
        # ALL fault-containing videos
        # ---------------------------------------------------------------------

        selected_fault_videos = list(
            self.fault_videos
        )

        # ---------------------------------------------------------------------
        # Same number of NORMAL-only videos
        # ---------------------------------------------------------------------

        normal_count = min(
            len(
                self.normal_videos
            ),
            len(
                selected_fault_videos
            ),
        )

        selected_normal_videos = (
            rng.sample(
                self.normal_videos,
                normal_count,
            )
        )

        selected_videos = (
            selected_fault_videos
            + selected_normal_videos
        )

        indices = []

        # ---------------------------------------------------------------------
        # Select clips
        # ---------------------------------------------------------------------

        for video in selected_videos:

            video_indices = (
                self.df[
                    self.df[
                        "video"
                    ].astype(str)
                    == str(video)
                ]
                .index
                .tolist()
            )

            if len(video_indices) == 0:

                continue

            sample_count = min(
                len(video_indices),
                self.max_clips,
            )

            selected_indices = rng.sample(
                video_indices,
                sample_count,
            )

            indices.extend(
                selected_indices
            )

        # ---------------------------------------------------------------------
        # Shuffle all selected samples
        # ---------------------------------------------------------------------

        rng.shuffle(
            indices
        )

        return iter(
            indices
        )

    def __len__(self):

        normal_count = min(
            len(
                self.normal_videos
            ),
            len(
                self.fault_videos
            ),
        )

        return (
            normal_count
            * 2
            * self.max_clips
        )


# =============================================================================
# MODEL
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
    sampler,
    epoch,
):

    model.train()

    sampler.set_epoch(
        epoch
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    total_loss = 0.0

    steps = 0

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

        steps += 1

        scaled_loss = (
            loss
            / GRADIENT_ACCUMULATION
        )

        scaled_loss.backward()

        if (
            (step + 1)
            % GRADIENT_ACCUMULATION
            == 0
        ):

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

    # -------------------------------------------------------------------------
    # Flush remaining gradients
    # -------------------------------------------------------------------------

    if (
        steps
        % GRADIENT_ACCUMULATION
        != 0
    ):

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        optimizer.zero_grad(
            set_to_none=True
        )

    return (
        total_loss
        / max(
            steps,
            1,
        )
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

        predictions = torch.argmax(
            outputs.logits,
            dim=-1,
        )

        y_true.extend(
            labels
            .cpu()
            .numpy()
            .tolist()
        )

        y_pred.extend(
            predictions
            .cpu()
            .numpy()
            .tolist()
        )

    loss = (
        total_loss
        / max(
            len(loader),
            1,
        )
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
        "loss":
            float(loss),

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "y_true":
            y_true,

        "y_pred":
            y_pred,
    }


# =============================================================================
# VIDEO-LEVEL METRICS
# =============================================================================

def create_video_metrics(
    test_df,
    y_true,
    y_pred,
):

    results = (
        test_df
        .reset_index(
            drop=True
        )
        .copy()
    )

    results[
        "true_label"
    ] = [
        ID2LABEL[
            int(x)
        ]
        for x in y_true
    ]

    results[
        "predicted_label"
    ] = [
        ID2LABEL[
            int(x)
        ]
        for x in y_pred
    ]

    results[
        "correct"
    ] = (
        results[
            "true_label"
        ]
        == results[
            "predicted_label"
        ]
    )

    rows = []

    for video, group in (
        results.groupby(
            "video"
        )
    ):

        actual_fault = (
            group[
                "true_label"
            ]
            == "FAULT"
        )

        predicted_fault = (
            group[
                "predicted_label"
            ]
            == "FAULT"
        )

        actual_fault_count = int(
            actual_fault.sum()
        )

        predicted_fault_count = int(
            predicted_fault.sum()
        )

        detected_fault_count = int(
            (
                actual_fault
                & predicted_fault
            ).sum()
        )

        if actual_fault_count > 0:

            fault_recall = (
                detected_fault_count
                / actual_fault_count
            )

        else:

            fault_recall = np.nan

        if predicted_fault_count > 0:

            fault_precision = (
                detected_fault_count
                / predicted_fault_count
            )

        else:

            fault_precision = np.nan

        rows.append(
            {
                "video":
                    video,

                "clips":
                    len(group),

                "actual_fault":
                    actual_fault_count,

                "predicted_fault":
                    predicted_fault_count,

                "detected_fault":
                    detected_fault_count,

                "fault_recall":
                    fault_recall,

                "fault_precision":
                    fault_precision,

                "accuracy":
                    group[
                        "correct"
                    ].mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    set_seed(
        SEED
    )

    print("=" * 90)
    print(
        "EAGLE-MARS FAULT-DIVERSITY + "
        "16-FRAME VIDEOMAE"
    )
    print("=" * 90)

    # =========================================================================
    # DEVICE
    # =========================================================================

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

        gpu_name = (
            torch.cuda
            .get_device_name(
                0
            )
        )

        gpu_memory = (
            torch.cuda
            .get_device_properties(
                0
            )
            .total_memory
            / 1024**3
        )

        print(
            f"GPU    : {gpu_name}"
        )

        print(
            f"VRAM   : {gpu_memory:.2f} GB"
        )

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    print()
    print(
        "CONFIGURATION"
    )

    print("-" * 90)

    print(
        f"Manifest              : {MANIFEST}"
    )

    print(
        f"Model                 : {MODEL_NAME}"
    )

    print(
        f"Frames                : {NUM_FRAMES}"
    )

    print(
        f"Resolution            : "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Batch size            : {BATCH_SIZE}"
    )

    print(
        f"Gradient accumulation : "
        f"{GRADIENT_ACCUMULATION}"
    )

    print(
        f"Effective batch       : "
        f"{EFFECTIVE_BATCH}"
    )

    print(
        f"Learning rate         : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Epochs                : {EPOCHS}"
    )

    print(
        f"Early stopping        : "
        f"{EARLY_STOPPING}"
    )

    print(
        f"Max clips/video       : "
        f"{MAX_CLIPS_PER_VIDEO}"
    )

    # =========================================================================
    # CHECK MANIFEST
    # =========================================================================

    if not MANIFEST.exists():

        raise FileNotFoundError(
            "\nManifest not found:\n"
            f"{MANIFEST}"
        )

    df = pd.read_csv(
        MANIFEST
    )

    print()
    print(
        f"Total clips: {len(df)}"
    )

    # =========================================================================
    # DATASET
    # =========================================================================

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

    # =========================================================================
    # SPLITS
    # =========================================================================
    #
    # IMPORTANT:
    #
    # There is NO global deduplication here.
    #
    # The final manifest contains 5,663 clip references.
    # The sampler decides which clips are used per epoch.
    #
    # =========================================================================

    train_df = (
        df[
            df["split"]
            == "train"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    val_df = (
        df[
            df["split"]
            == "val"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    test_df = (
        df[
            df["split"]
            == "test"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        "SPLIT SIZES"
    )

    print("-" * 90)

    print(
        f"TRAIN : {len(train_df)} clips / "
        f"{train_df.video.nunique()} videos"
    )

    print(
        f"VAL   : {len(val_df)} clips / "
        f"{val_df.video.nunique()} videos"
    )

    print(
        f"TEST  : {len(test_df)} clips / "
        f"{test_df.video.nunique()} videos"
    )

    # =========================================================================
    # TRAINING VIDEO DIVERSITY
    # =========================================================================

    video_labels = {}

    for _, row in (
        train_df.iterrows()
    ):

        video = str(
            row["video"]
        )

        label = str(
            row["label"]
        ).upper()

        if video not in video_labels:

            video_labels[
                video
            ] = set()

        video_labels[
            video
        ].add(
            label
        )

    fault_videos = sorted(
        [
            video
            for video, labels
            in video_labels.items()
            if "FAULT" in labels
        ]
    )

    normal_videos = sorted(
        [
            video
            for video, labels
            in video_labels.items()
            if "FAULT" not in labels
        ]
    )

    print()
    print(
        "TRAINING VIDEO DIVERSITY"
    )

    print("-" * 90)

    print(
        f"Fault-containing videos : "
        f"{len(fault_videos)}"
    )

    print(
        f"Normal-only videos      : "
        f"{len(normal_videos)}"
    )

    print()
    print(
        "FAULT-CONTAINING TRAINING VIDEOS"
    )

    for video in fault_videos:

        video_rows = train_df[
            train_df[
                "video"
            ].astype(str)
            == video
        ]

        fault_count = int(
            (
                video_rows[
                    "label"
                ].astype(str).str.upper()
                == "FAULT"
            ).sum()
        )

        normal_count = int(
            (
                video_rows[
                    "label"
                ].astype(str).str.upper()
                == "NORMAL"
            ).sum()
        )

        print(
            f"  {video:<30} "
            f"FAULT={fault_count:4d} "
            f"NORMAL={normal_count:4d}"
        )

    # =========================================================================
    # PROCESSOR
    # =========================================================================

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

    # =========================================================================
    # DATASETS
    # =========================================================================

    train_dataset = ClipDataset(
        train_df,
        processor,
    )

    val_dataset = ClipDataset(
        val_df,
        processor,
    )

    test_dataset = ClipDataset(
        test_df,
        processor,
    )

    # =========================================================================
    # SAMPLER
    # =========================================================================

    sampler = (
        FaultDiversitySampler(
            train_df,
            MAX_CLIPS_PER_VIDEO,
            SEED,
        )
    )

    normal_videos_per_epoch = min(
        len(
            sampler.normal_videos
        ),
        len(
            sampler.fault_videos
        ),
    )

    maximum_samples_per_epoch = (
        normal_videos_per_epoch
        * 2
        * MAX_CLIPS_PER_VIDEO
    )

    print()
    print(
        "FAULT-DIVERSITY SAMPLER"
    )

    print("-" * 90)

    print(
        f"Fault videos used every epoch : "
        f"{len(sampler.fault_videos)}"
    )

    print(
        f"Normal videos available       : "
        f"{len(sampler.normal_videos)}"
    )

    print(
        f"Normal videos sampled/epoch   : "
        f"{normal_videos_per_epoch}"
    )

    print(
        f"Max clips/video               : "
        f"{MAX_CLIPS_PER_VIDEO}"
    )

    print(
        f"Maximum samples/epoch        : "
        f"{maximum_samples_per_epoch}"
    )

    # =========================================================================
    # DATALOADERS
    # =========================================================================

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

    # =========================================================================
    # MODEL
    # =========================================================================

    model = load_model()

    model.to(
        device
    )

    # =========================================================================
    # OPTIMIZER
    # =========================================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    # =========================================================================
    # OUTPUT
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if BEST_MODEL_DIR.exists():

        shutil.rmtree(
            BEST_MODEL_DIR
        )

    BEST_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # TRAINING
    # =========================================================================

    print()
    print("=" * 90)
    print(
        "STARTING FAULT-DIVERSITY + "
        "16-FRAME TRAINING"
    )
    print("=" * 90)

    history = []

    best_f1 = -1.0

    best_epoch = 0

    no_improvement = 0

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
            sampler,
            epoch,
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
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "val_loss":
                    val_metrics[
                        "loss"
                    ],

                "val_accuracy":
                    val_metrics[
                        "accuracy"
                    ],

                "val_precision":
                    val_metrics[
                        "precision"
                    ],

                "val_recall":
                    val_metrics[
                        "recall"
                    ],

                "val_f1":
                    val_metrics[
                        "f1"
                    ],
            }
        )

        # ---------------------------------------------------------------------
        # SAVE BEST MODEL
        # ---------------------------------------------------------------------

        if (
            val_metrics["f1"]
            > best_f1
        ):

            best_f1 = (
                val_metrics["f1"]
            )

            best_epoch = epoch

            no_improvement = 0

            if BEST_MODEL_DIR.exists():

                shutil.rmtree(
                    BEST_MODEL_DIR
                )

            BEST_MODEL_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

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

    # =========================================================================
    # SAVE HISTORY
    # =========================================================================

    history_path = (
        OUTPUT_DIR
        / "training_history.json"
    )

    with open(
        history_path,
        "w",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    # =========================================================================
    # LOAD BEST MODEL
    # =========================================================================

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

    # =========================================================================
    # FINAL TEST
    # =========================================================================

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
        test_metrics[
            "y_true"
        ]
    )

    y_pred = np.asarray(
        test_metrics[
            "y_pred"
        ]
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

    # =========================================================================
    # CLASSIFICATION REPORT
    # =========================================================================

    print()
    print(
        "CLASSIFICATION REPORT"
    )

    print("-" * 90)

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

    # =========================================================================
    # CONFUSION MATRIX
    # =========================================================================

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
        f"{cm[0, 0]:8d}"
        f"{cm[0, 1]:7d}"
    )

    print(
        f"       FAULT "
        f"{cm[1, 0]:8d}"
        f"{cm[1, 1]:7d}"
    )

    # =========================================================================
    # TEST PREDICTIONS
    # =========================================================================

    predictions = (
        test_df
        .reset_index(
            drop=True
        )
        .copy()
    )

    predictions[
        "true_label"
    ] = [
        ID2LABEL[
            int(x)
        ]
        for x in y_true
    ]

    predictions[
        "predicted_label"
    ] = [
        ID2LABEL[
            int(x)
        ]
        for x in y_pred
    ]

    predictions[
        "correct"
    ] = (
        predictions[
            "true_label"
        ]
        == predictions[
            "predicted_label"
        ]
    )

    predictions_path = (
        OUTPUT_DIR
        / "test_predictions.csv"
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    # =========================================================================
    # VIDEO-LEVEL PERFORMANCE
    # =========================================================================

    video_metrics = (
        create_video_metrics(
            test_df,
            y_true,
            y_pred,
        )
    )

    print()
    print(
        "VIDEO-LEVEL TEST PERFORMANCE"
    )

    print("-" * 90)

    print(
        video_metrics.to_string(
            index=False
        )
    )

    video_metrics_path = (
        OUTPUT_DIR
        / "test_video_metrics.csv"
    )

    video_metrics.to_csv(
        video_metrics_path,
        index=False,
    )

    # =========================================================================
    # VIDEO-LEVEL SUMMARY
    # =========================================================================

    valid_fault_recall = (
        video_metrics[
            "fault_recall"
        ]
        .dropna()
    )

    valid_fault_precision = (
        video_metrics[
            "fault_precision"
        ]
        .dropna()
    )

    if len(
        valid_fault_recall
    ) > 0:

        macro_video_fault_recall = (
            valid_fault_recall.mean()
        )

    else:

        macro_video_fault_recall = 0.0

    if len(
        valid_fault_precision
    ) > 0:

        macro_video_fault_precision = (
            valid_fault_precision.mean()
        )

    else:

        macro_video_fault_precision = 0.0

    macro_video_accuracy = (
        video_metrics[
            "accuracy"
        ].mean()
    )

    print()
    print(
        f"Macro Video Accuracy       : "
        f"{macro_video_accuracy:.4f}"
    )

    print(
        f"Macro Video Fault Recall   : "
        f"{macro_video_fault_recall:.4f}"
    )

    print(
        f"Macro Video Fault Precision: "
        f"{macro_video_fault_precision:.4f}"
    )

    # =========================================================================
    # SAVE METRICS
    # =========================================================================

    metrics = {

        "experiment":
            "fault_diversity_16_frames",

        "model":
            MODEL_NAME,

        "frames":
            NUM_FRAMES,

        "resolution":
            IMAGE_SIZE,

        "batch_size":
            BATCH_SIZE,

        "gradient_accumulation":
            GRADIENT_ACCUMULATION,

        "effective_batch":
            EFFECTIVE_BATCH,

        "learning_rate":
            LEARNING_RATE,

        "epochs_requested":
            EPOCHS,

        "early_stopping":
            EARLY_STOPPING,

        "max_clips_per_video":
            MAX_CLIPS_PER_VIDEO,

        "best_epoch":
            best_epoch,

        "best_val_fault_f1":
            best_f1,

        "test_loss":
            test_metrics[
                "loss"
            ],

        "test_accuracy":
            test_metrics[
                "accuracy"
            ],

        "test_fault_precision":
            test_metrics[
                "precision"
            ],

        "test_fault_recall":
            test_metrics[
                "recall"
            ],

        "test_fault_f1":
            test_metrics[
                "f1"
            ],

        "macro_video_accuracy":
            float(
                macro_video_accuracy
            ),

        "macro_video_fault_recall":
            float(
                macro_video_fault_recall
            ),

        "macro_video_fault_precision":
            float(
                macro_video_fault_precision
            ),

        "train_clips":
            len(train_df),

        "val_clips":
            len(val_df),

        "test_clips":
            len(test_df),

        "train_videos":
            train_df.video.nunique(),

        "val_videos":
            val_df.video.nunique(),

        "test_videos":
            test_df.video.nunique(),

        "fault_training_videos":
            len(fault_videos),

        "normal_training_videos":
            len(normal_videos),

        "normal_videos_per_epoch":
            normal_videos_per_epoch,

        "maximum_samples_per_epoch":
            maximum_samples_per_epoch,
    }

    metrics_path = (
        OUTPUT_DIR
        / "test_metrics.json"
    )

    with open(
        metrics_path,
        "w",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print()
    print("=" * 90)
    print(
        "FAULT-DIVERSITY TRAINING COMPLETE"
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
        f"Macro Video "
        f"Fault Recall    : "
        f"{macro_video_fault_recall:.4f}"
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
        history_path
    )

    print()
    print(
        "Metrics:"
    )

    print(
        metrics_path
    )

    print()
    print(
        "Predictions:"
    )

    print(
        predictions_path
    )

    print()
    print(
        "Video metrics:"
    )

    print(
        video_metrics_path
    )

    print()
    print("=" * 90)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()