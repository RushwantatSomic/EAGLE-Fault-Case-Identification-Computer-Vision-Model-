import os
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
)


# ============================================================
# EAGLE-MARS
# CONTINUAL LEARNING - 50 VIDEO EXPERIMENT
# ============================================================

# ------------------------------------------------------------
# STARTING MODEL
# ------------------------------------------------------------

START_MODEL = r"outputs\training_exp4\best_model"


# ------------------------------------------------------------
# VIDEO DIRECTORY
#
# Put the 50 videos you want to annotate here.
# ------------------------------------------------------------

VIDEO_DIR = r"data\continual_videos"


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT_DIR = r"outputs\continual_learning"

ANNOTATION_FILE = os.path.join(
    OUTPUT_DIR,
    "continual_annotations.json"
)

MODEL_DIR = os.path.join(
    OUTPUT_DIR,
    "current_model"
)

CHECKPOINT_DIR = os.path.join(
    OUTPUT_DIR,
    "checkpoints"
)

USED_VIDEO_FILE = os.path.join(
    OUTPUT_DIR,
    "used_videos.txt"
)


# ------------------------------------------------------------
# TRAINING CONFIGURATION
# ------------------------------------------------------------

NUM_FRAMES = 16

# Take a new training window every 8 frames.
STRIDE = 8

BATCH_SIZE = 2

GRADIENT_ACCUMULATION = 4

LEARNING_RATE = 5e-6

EPOCHS_PER_VIDEO = 1

# Replay samples from old videos.
REPLAY_PER_OLD_VIDEO = 2

# Maximum total replay samples.
MAX_REPLAY_SAMPLES = 30

# Save a checkpoint after every video.
SAVE_EVERY_VIDEO = True

# Reproducibility.
SEED = 42


# ------------------------------------------------------------
# LABELS
# ------------------------------------------------------------

NORMAL = 0
FAULT = 1

LABEL_NAMES = {
    0: "NORMAL",
    1: "FAULT",
}


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HELPERS
# ============================================================

def ensure_directories():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )


def save_json(data, path):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2
        )


def load_annotations():

    if not os.path.exists(
        ANNOTATION_FILE
    ):

        return []

    with open(
        ANNOTATION_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_annotations(data):

    save_json(
        data,
        ANNOTATION_FILE
    )


def load_used_videos():

    if not os.path.exists(
        USED_VIDEO_FILE
    ):

        return set()

    with open(
        USED_VIDEO_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return {
            line.strip()
            for line in f
            if line.strip()
        }


def mark_video_used(video_path):

    with open(
        USED_VIDEO_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            os.path.abspath(video_path)
            + "\n"
        )


# ============================================================
# VIDEO INFORMATION
# ============================================================

def get_video_info(video_path):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    cap.release()

    if fps <= 0:

        fps = 30.0

    duration = (
        frame_count / fps
    )

    return (
        fps,
        frame_count,
        width,
        height,
        duration
    )


# ============================================================
# TEMPORAL ANNOTATION
# ============================================================

def parse_time(value):

    value = value.strip()

    if ":" in value:

        parts = value.split(":")

        if len(parts) == 2:

            minutes = float(
                parts[0]
            )

            seconds = float(
                parts[1]
            )

            return (
                minutes * 60
                + seconds
            )

        elif len(parts) == 3:

            hours = float(
                parts[0]
            )

            minutes = float(
                parts[1]
            )

            seconds = float(
                parts[2]
            )

            return (
                hours * 3600
                + minutes * 60
                + seconds
            )

    return float(value)


def parse_fault_intervals(
    text,
    duration
):

    text = text.strip()

    if text.upper() == "N":

        return []


    intervals = []


    # Examples accepted:
    #
    # 5.2-8.7
    # 00:05.2-00:08.7
    # 5-8, 12-14
    #
    chunks = text.split(",")


    for chunk in chunks:

        chunk = chunk.strip()

        if not chunk:
            continue

        if "-" not in chunk:

            raise ValueError(
                f"Invalid interval: {chunk}"
            )

        start_text, end_text = (
            chunk.split(
                "-",
                1
            )
        )

        start = parse_time(
            start_text
        )

        end = parse_time(
            end_text
        )


        if start < 0:

            raise ValueError(
                "Start time cannot be negative."
            )


        if end <= start:

            raise ValueError(
                "End time must be greater than start time."
            )


        if start >= duration:

            raise ValueError(
                f"Start time {start:.2f}s "
                f"is outside the video."
            )


        end = min(
            end,
            duration
        )


        intervals.append(
            (
                start,
                end
            )
        )


    intervals.sort()

    return intervals


def frame_is_fault(
    frame_number,
    fps,
    fault_intervals
):

    time = (
        frame_number / fps
    )

    for start, end in fault_intervals:

        if (
            time >= start
            and
            time <= end
        ):

            return True

    return False


# ============================================================
# GENERATE WINDOWS
# ============================================================

def generate_windows(
    video_path,
    fps,
    frame_count,
    fault_intervals
):

    windows = []


    for start in range(
        0,
        frame_count - NUM_FRAMES + 1,
        STRIDE
    ):

        end = (
            start
            + NUM_FRAMES
            - 1
        )


        fault_count = 0


        for frame_number in range(
            start,
            end + 1
        ):

            if frame_is_fault(
                frame_number,
                fps,
                fault_intervals
            ):

                fault_count += 1


        # ----------------------------------------------------
        # Window labeling
        #
        # A window is FAULT if at least 50% of its frames
        # are inside the annotated fault interval.
        # ----------------------------------------------------

        if fault_count >= (
            NUM_FRAMES * 0.5
        ):

            label = FAULT

        else:

            label = NORMAL


        windows.append(
            {
                "video": os.path.abspath(
                    video_path
                ),
                "start_frame": start,
                "end_frame": end,
                "label": label,
            }
        )


    return windows


# ============================================================
# READ 16 FRAMES
# ============================================================

def read_clip(
    video_path,
    start_frame
):

    cap = cv2.VideoCapture(
        video_path
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame
    )


    frames = []


    for _ in range(
        NUM_FRAMES
    ):

        ret, frame = cap.read()

        if not ret:

            break

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frames.append(
            frame
        )


    cap.release()


    if len(frames) != NUM_FRAMES:

        return None


    return np.asarray(
        frames,
        dtype=np.uint8
    )


# ============================================================
# DATASET
# ============================================================

class VideoWindowDataset(
    Dataset
):

    def __init__(
        self,
        samples,
        processor
    ):

        self.samples = samples

        self.processor = processor


    def __len__(self):

        return len(
            self.samples
        )


    def __getitem__(
        self,
        index
    ):

        sample = self.samples[
            index
        ]


        frames = read_clip(
            sample["video"],
            sample["start_frame"]
        )


        if frames is None:

            raise RuntimeError(
                "Could not read clip:\n"
                f"{sample}"
            )


        processed = self.processor(
            images=list(frames),
            return_tensors="pt"
        )


        pixel_values = (
            processed[
                "pixel_values"
            ][0]
        )


        label = torch.tensor(
            sample["label"],
            dtype=torch.long
        )


        return {
            "pixel_values":
                pixel_values,

            "labels":
                label
        }


# ============================================================
# MODEL TRAINING
# ============================================================

def train_on_samples(
    model,
    processor,
    samples
):

    if not samples:

        print(
            "No training samples."
        )

        return


    counts = {
        NORMAL: 0,
        FAULT: 0
    }


    for sample in samples:

        counts[
            sample["label"]
        ] += 1


    print()
    print(
        "TRAINING DATA"
    )
    print("-" * 80)

    print(
        f"NORMAL: {counts[NORMAL]}"
    )

    print(
        f"FAULT : {counts[FAULT]}"
    )


    dataset = VideoWindowDataset(
        samples,
        processor
    )


    # --------------------------------------------------------
    # Balanced sampler
    # --------------------------------------------------------

    class_weights = {}

    for label in [
        NORMAL,
        FAULT
    ]:

        if counts[label] > 0:

            class_weights[label] = (
                1.0 /
                counts[label]
            )

        else:

            class_weights[label] = 0.0


    weights = [
        class_weights[
            sample["label"]
        ]
        for sample in samples
    ]


    sampler = (
        WeightedRandomSampler(
            weights=torch.DoubleTensor(
                weights
            ),
            num_samples=len(
                samples
            ),
            replacement=True
        )
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )


    model.train()


    for epoch in range(
        EPOCHS_PER_VIDEO
    ):

        total_loss = 0.0

        optimizer.zero_grad(
            set_to_none=True
        )


        for batch_index, batch in enumerate(
            loader
        ):

            pixel_values = (
                batch[
                    "pixel_values"
                ].to(
                    device,
                    non_blocking=True
                )
            )


            labels = (
                batch[
                    "labels"
                ].to(
                    device,
                    non_blocking=True
                )
            )


            outputs = model(
                pixel_values=pixel_values,
                labels=labels
            )


            loss = (
                outputs.loss /
                GRADIENT_ACCUMULATION
            )


            loss.backward()


            if (
                (batch_index + 1)
                % GRADIENT_ACCUMULATION
                == 0
            ):

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )


                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )


            total_loss += (
                loss.item()
                *
                GRADIENT_ACCUMULATION
            )


        # Handle final incomplete
        # accumulation group.
        if (
            len(loader)
            %
            GRADIENT_ACCUMULATION
            != 0
        ):

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )


        average_loss = (
            total_loss /
            max(
                1,
                len(loader)
            )
        )


        print(
            f"Epoch {epoch + 1}: "
            f"Loss = {average_loss:.4f}"
        )


    model.eval()


# ============================================================
# REPLAY DATA
# ============================================================

def build_replay_samples(
    all_annotations
):

    candidates = []


    for annotation in (
        all_annotations
    ):

        windows = annotation.get(
            "windows",
            []
        )


        for window in windows:

            candidates.append(
                window
            )


    if not candidates:

        return []


    # --------------------------------------------------------
    # Separate classes
    # --------------------------------------------------------

    normal = [
        x for x in candidates
        if x["label"] == NORMAL
    ]

    fault = [
        x for x in candidates
        if x["label"] == FAULT
    ]


    random.shuffle(
        normal
    )

    random.shuffle(
        fault
    )


    # Take roughly balanced replay.
    half = MAX_REPLAY_SAMPLES // 2


    selected = (
        normal[:half]
        +
        fault[:half]
    )


    random.shuffle(
        selected
    )


    return selected


# ============================================================
# SAVE MODEL
# ============================================================

def save_current_model(
    model,
    processor,
    video_number
):

    if os.path.exists(
        MODEL_DIR
    ):

        shutil.rmtree(
            MODEL_DIR
        )


    model.save_pretrained(
        MODEL_DIR
    )

    processor.save_pretrained(
        MODEL_DIR
    )


    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"video_{video_number:02d}"
    )


    if os.path.exists(
        checkpoint_path
    ):

        shutil.rmtree(
            checkpoint_path
        )


    model.save_pretrained(
        checkpoint_path
    )

    processor.save_pretrained(
        checkpoint_path
    )


# ============================================================
# VIDEO SELECTION
# ============================================================

def get_available_videos():

    extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv"
    }


    if not os.path.exists(
        VIDEO_DIR
    ):

        os.makedirs(
            VIDEO_DIR,
            exist_ok=True
        )


        print()
        print(
            "Created video directory:"
        )

        print(
            os.path.abspath(
                VIDEO_DIR
            )
        )

        print(
            "\nPut your videos there and run again."
        )

        return []


    videos = []


    for path in Path(
        VIDEO_DIR
    ).rglob("*"):

        if (
            path.is_file()
            and
            path.suffix.lower()
            in extensions
        ):

            videos.append(
                str(path)
            )


    videos.sort(
        key=lambda x:
        os.path.basename(x).lower()
    )


    return videos


# ============================================================
# VIDEO PREVIEW
# ============================================================

def preview_video(
    video_path
):

    print()
    print(
        "Opening video preview..."
    )

    print(
        "SPACE = pause/resume"
    )

    print(
        "Q / ESC = close preview"
    )


    cap = cv2.VideoCapture(
        video_path
    )


    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30


    while True:

        ret, frame = cap.read()

        if not ret:
            break


        cv2.imshow(
            "Annotation Preview",
            frame
        )


        key = (
            cv2.waitKey(
                max(
                    1,
                    int(
                        1000 / fps
                    )
                )
            )
            & 0xFF
        )


        if key == ord("q") or key == 27:

            break


        if key == ord(" "):

            while True:

                pause_key = (
                    cv2.waitKey(
                        30
                    )
                    & 0xFF
                )


                if pause_key == ord(" "):

                    break


                if (
                    pause_key == ord("q")
                    or pause_key == 27
                ):

                    cap.release()

                    cv2.destroyAllWindows()

                    return


    cap.release()

    cv2.destroyAllWindows()


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_directories()


    print("=" * 90)
    print(
        "EAGLE-MARS CONTINUAL LEARNING"
    )
    print(
        "50 VIDEO INCREMENTAL TRAINING"
    )
    print("=" * 90)


    print()
    print(
        f"Device : {device}"
    )


    if torch.cuda.is_available():

        print(
            f"GPU    : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"VRAM   : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )


    # --------------------------------------------------------
    # Load videos
    # --------------------------------------------------------

    videos = get_available_videos()


    if not videos:

        return


    used_videos = load_used_videos()


    available = [
        video
        for video in videos
        if os.path.abspath(video)
        not in used_videos
    ]


    print()
    print(
        f"Videos available : {len(videos)}"
    )

    print(
        f"Already used     : {len(used_videos)}"
    )

    print(
        f"Remaining        : {len(available)}"
    )


    if len(available) < 50:

        print()
        print(
            "WARNING:"
        )

        print(
            f"Only {len(available)} unused "
            "videos are currently available."
        )

        print(
            "The experiment can continue "
            "until the available videos are exhausted."
        )


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print(
        "MODEL"
    )

    print("-" * 90)

    print(
        f"Starting model: {START_MODEL}"
    )


    model_path = (
        MODEL_DIR
        if os.path.exists(
            MODEL_DIR
        )
        else START_MODEL
    )


    print(
        f"Loading model: {model_path}"
    )


    processor = (
        VideoMAEImageProcessor
        .from_pretrained(
            model_path
        )
    )


    model = (
        VideoMAEForVideoClassification
        .from_pretrained(
            model_path
        )
    )


    model.to(device)

    model.eval()


    # --------------------------------------------------------
    # Existing annotations
    # --------------------------------------------------------

    annotations = load_annotations()


    print()
    print(
        f"Previously annotated videos: "
        f"{len(annotations)}"
    )


    # --------------------------------------------------------
    # Main 50-video loop
    # --------------------------------------------------------

    for video_number in range(
        1,
        51
    ):

        available = [
            video
            for video in videos
            if os.path.abspath(video)
            not in load_used_videos()
        ]


        if not available:

            print(
                "No unused videos remaining."
            )

            break


        # ----------------------------------------------------
        # Automatically take first video.
        # ----------------------------------------------------

        video_path = available[0]


        print()
        print("=" * 90)

        print(
            f"VIDEO {video_number} / 50"
        )

        print("=" * 90)


        print(
            f"File: {os.path.basename(video_path)}"
        )

        print(
            f"Path: {os.path.abspath(video_path)}"
        )


        (
            fps,
            frame_count,
            width,
            height,
            duration
        ) = get_video_info(
            video_path
        )


        print()
        print(
            f"FPS       : {fps:.2f}"
        )

        print(
            f"Frames    : {frame_count}"
        )

        print(
            f"Resolution: {width}x{height}"
        )

        print(
            f"Duration  : {duration:.2f}s"
        )


        # ----------------------------------------------------
        # Preview
        # ----------------------------------------------------

        preview_answer = input(
            "\nPreview this video first? "
            "[Y/n]: "
        ).strip().lower()


        if (
            preview_answer == ""
            or
            preview_answer == "y"
        ):

            preview_video(
                video_path
            )


        # ----------------------------------------------------
        # Annotation
        # ----------------------------------------------------

        print()
        print(
            "=" * 90
        )

        print(
            "ENTER FAULT TIME INTERVALS"
        )

        print("=" * 90)


        print(
            "Examples:"
        )

        print(
            "  N"
        )

        print(
            "  5.2-8.7"
        )

        print(
            "  5.2-8.7, 12.4-15.8"
        )

        print(
            "  00:05.2-00:08.7"
        )


        while True:

            user_input = input(
                "\nFault timeframe(s): "
            ).strip()


            try:

                fault_intervals = (
                    parse_fault_intervals(
                        user_input,
                        duration
                    )
                )

                break

            except Exception as e:

                print(
                    f"Invalid input: {e}"
                )


        # ----------------------------------------------------
        # Generate training windows
        # ----------------------------------------------------

        windows = generate_windows(
            video_path,
            fps,
            frame_count,
            fault_intervals
        )


        normal_count = sum(
            1
            for x in windows
            if x["label"] == NORMAL
        )


        fault_count = sum(
            1
            for x in windows
            if x["label"] == FAULT
        )


        print()
        print(
            "GENERATED WINDOWS"
        )

        print("-" * 90)

        print(
            f"Total  : {len(windows)}"
        )

        print(
            f"Normal : {normal_count}"
        )

        print(
            f"Fault  : {fault_count}"
        )


        # ----------------------------------------------------
        # Save annotation BEFORE training
        # ----------------------------------------------------

        annotation_record = {

            "video_number":
                video_number,

            "video":
                os.path.abspath(
                    video_path
                ),

            "fps":
                fps,

            "frame_count":
                frame_count,

            "duration":
                duration,

            "fault_intervals":
                fault_intervals,

            "windows":
                windows,
        }


        annotations.append(
            annotation_record
        )


        save_annotations(
            annotations
        )


        mark_video_used(
            video_path
        )


        # ----------------------------------------------------
        # Current video + replay
        # ----------------------------------------------------

        current_samples = windows


        replay_samples = (
            build_replay_samples(
                annotations[:-1]
            )
        )


        training_samples = (
            current_samples
            +
            replay_samples
        )


        print()
        print(
            "CONTINUAL LEARNING UPDATE"
        )

        print("-" * 90)

        print(
            f"New video samples    : "
            f"{len(current_samples)}"
        )

        print(
            f"Replay samples       : "
            f"{len(replay_samples)}"
        )

        print(
            f"Total update samples : "
            f"{len(training_samples)}"
        )


        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_on_samples(
            model,
            processor,
            training_samples
        )


        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_current_model(
            model,
            processor,
            video_number
        )


        print()
        print(
            "=" * 90
        )

        print(
            f"VIDEO {video_number} COMPLETE"
        )

        print("=" * 90)

        print(
            f"Current model: {MODEL_DIR}"
        )

        print(
            f"Annotations  : {ANNOTATION_FILE}"
        )


        # ----------------------------------------------------
        # Stop / continue
        # ----------------------------------------------------

        if video_number < 50:

            answer = input(
                "\nPress ENTER for next video "
                "or type Q to stop: "
            ).strip().lower()


            if answer == "q":

                print(
                    "\nExperiment paused."
                )

                print(
                    "Run the script again to continue."
                )

                break


    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "CONTINUAL LEARNING EXPERIMENT COMPLETE"
    )
    print("=" * 90)


    print(
        f"Videos annotated : "
        f"{len(annotations)}"
    )

    print(
        f"Final model      : "
        f"{MODEL_DIR}"
    )

    print(
        f"Annotations      : "
        f"{ANNOTATION_FILE}"
    )

    print(
        f"Checkpoints      : "
        f"{CHECKPOINT_DIR}"
    )


if __name__ == "__main__":

    main()