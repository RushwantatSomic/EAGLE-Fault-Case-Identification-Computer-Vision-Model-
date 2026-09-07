# =============================================================================
# EAGLE-MARS
# CONFIRMED ALL-CONTINUAL 16-FRAME CLIP GENERATOR
#
# Sources:
#   1. Original CVAT timeframe annotations
#   2. CONFIRMED annotations2 timeframe annotations
#   3. Existing 50-video continual-learning annotations
#
# IMPORTANT:
#   - continual_50_videos uses the manifest's video_path directly.
#   - annotations2 uses ONLY confirmed mappings.
#   - No UNKNOWN_TASK guessing.
#   - Existing valid clips are preserved/reused.
# =============================================================================

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_MANIFEST = (
    ROOT
    / "outputs"
    / "unified_dataset"
    / "all_continual"
    / "all_continual_manifest.csv"
)

OUTPUT_ROOT = (
    ROOT
    / "outputs"
    / "unified_dataset"
    / "confirmed_continual"
)

CLIP_DIR = OUTPUT_ROOT / "clips"

OUTPUT_MANIFEST = (
    OUTPUT_ROOT
    / "confirmed_continual_clips_manifest.csv"
)

SUMMARY_JSON = (
    CLIP_DIR
    / "clip_generation_summary.json"
)

ERRORS_JSON = (
    CLIP_DIR
    / "clip_generation_errors.json"
)

MAPPING_JSON = (
    ROOT
    / "outputs"
    / "unified_dataset"
    / "all_continual"
    / "annotations2_physical_mapping.json"
)

ANNOTATED_VIDEO_DIR = (
    ROOT
    / "data"
    / "annotated_videos"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

NUM_FRAMES = 16
IMAGE_SIZE = 224

OVERWRITE = False

JPEG_QUALITY = 95

PROGRESS_EVERY = 100

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".MP4",
    ".AVI",
    ".MOV",
    ".MKV",
}


# =============================================================================
# DISPLAY
# =============================================================================

def header(title: str) -> None:
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def section(title: str) -> None:
    print()
    print(title)
    print("-" * 90)


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_name(value: object) -> str:
    if value is None:
        return ""

    s = str(value).strip()

    s = s.replace("\\", "/")

    s = Path(s).name

    if s.lower().endswith(".mp4"):
        s = s[:-4]

    s = s.lower()

    s = re.sub(r"\s+", " ", s)
    s = s.strip()

    return s


# =============================================================================
# VIDEO INFO
# =============================================================================

def read_video_info(path: Path) -> Optional[Dict]:
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        return None

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if frames <= 0:
        return None

    if not math.isfinite(fps) or fps <= 0:
        fps = 0.0

    return {
        "path": str(path.resolve()),
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
    }


def build_physical_video_index(directory: Path) -> Dict[str, Dict]:
    section("BUILDING PHYSICAL VIDEO INDEX")

    print(f"Directory : {directory}")

    if not directory.exists():
        print("ERROR: directory does not exist.")
        return {}

    files = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix in VIDEO_EXTENSIONS
    ]

    files = sorted(files, key=lambda x: x.name.lower())

    print(f"Videos    : {len(files)}")
    print()

    index = {}

    for p in files:
        info = read_video_info(p)

        if info is None:
            print(f"FAILED   {p.name}")
            continue

        key = normalize_name(p.name)

        index[key] = {
            "name": p.name,
            **info,
        }

        print(
            f"{p.name:<36}"
            f"frames={info['frames']:5d} "
            f"fps={info['fps']:7.2f} "
            f"{info['width']}x{info['height']}"
        )

    return index


# =============================================================================
# CONFIRMED annotations2 MAPPING
# =============================================================================

def load_confirmed_annotations2_mapping(
    mapping_path: Path,
    physical_index: Dict[str, Dict],
) -> Dict[str, Dict]:

    section("LOADING CONFIRMED annotations2 MAPPINGS")

    if not mapping_path.exists():
        print("Mapping file not found:")
        print(mapping_path)
        print()
        print("No annotations2 videos will be used.")
        return {}

    try:
        data = json.loads(
            mapping_path.read_text(
                encoding="utf-8"
            )
        )
    except Exception as exc:
        print(
            f"ERROR reading mapping JSON: {exc}"
        )
        return {}

    confirmed = {}

    # -------------------------------------------------------------------------
    # The mapping file can have different structures depending on the
    # version of the mapper.
    #
    # We ONLY accept records that explicitly contain a physical video.
    # -------------------------------------------------------------------------

    records = []

    if isinstance(data, list):

        # Example:
        # [
        #   {
        #       "task_id": "2523742",
        #       "physical_video": "Lamella Chain 2.mp4"
        #   }
        # ]

        records = [
            x
            for x in data
            if isinstance(x, dict)
        ]

    elif isinstance(data, dict):

        # -------------------------------------------------------------
        # Preferred structure:
        #
        # {
        #   "2523742": {
        #       "physical_video": "Lamella Chain 2.mp4"
        #   }
        # }
        # -------------------------------------------------------------

        for key, value in data.items():

            if not isinstance(value, dict):
                continue

            item = dict(value)

            item.setdefault(
                "task_id",
                str(key),
            )

            records.append(item)

    # -------------------------------------------------------------------------
    # Process only dictionary records.
    # -------------------------------------------------------------------------

    for item in records:

        if not isinstance(item, dict):
            continue

        task_id = str(
            item.get("task_id")
            or item.get("id")
            or item.get("task")
            or ""
        ).strip()

        if not task_id:
            continue

        physical_video = (
            item.get("physical_video")
            or item.get("physical_video_name")
            or item.get("mapped_video")
            or item.get("video")
        )

        if not physical_video:
            continue

        physical_video = str(
            physical_video
        ).strip()

        physical_key = normalize_name(
            physical_video
        )

        # -------------------------------------------------------------
        # STRICT physical-video verification.
        # -------------------------------------------------------------

        if physical_key not in physical_index:

            print(
                f"IGNORED task={task_id:<12} "
                f"physical video not found: "
                f"{physical_video}"
            )

            continue

        physical = physical_index[
            physical_key
        ]

        confirmed[task_id] = {
            "task_id": task_id,
            "physical_video": physical[
                "name"
            ],
            "physical_path": physical[
                "path"
            ],
        }

    print()

    print(
        f"Confirmed annotations2 tasks: "
        f"{len(confirmed)}"
    )

    if confirmed:

        print()
        print(
            "CONFIRMED MAPPINGS"
        )
        print("-" * 90)

        for task_id, item in sorted(
            confirmed.items(),
            key=lambda x: x[0],
        ):

            print(
                f"{task_id:<12} -> "
                f"{item['physical_video']}"
            )

    else:

        print()
        print(
            "No confirmed annotations2 mappings "
            "were found."
        )

    return confirmed


# =============================================================================
# SOURCE RESOLUTION
# =============================================================================

def resolve_video_path(
    row: pd.Series,
    physical_index: Dict[str, Dict],
    confirmed_annotations2: Dict[str, Dict],
) -> Tuple[Optional[Path], str]:

    source = str(row.get("source", "")).strip()

    # -------------------------------------------------------------------------
    # 1. CONTINUAL 50-VIDEO SOURCE
    #
    # TRUST THE MANIFEST'S ABSOLUTE video_path.
    # -------------------------------------------------------------------------

    if source == "continual_50_videos":

        raw_path = row.get("video_path")

        if pd.notna(raw_path):
            path = Path(str(raw_path).strip())

            if path.exists() and path.is_file():
                return path.resolve(), "manifest_video_path"

        # Secondary exact lookup by filename.
        video_name = str(
            row.get("video", "")
        ).strip()

        key = normalize_name(video_name)

        if key in physical_index:
            return (
                Path(
                    physical_index[key]["path"]
                ),
                "physical_index_exact",
            )

        return None, "continual_video_not_found"

    # -------------------------------------------------------------------------
    # 2. annotations2
    #
    # ONLY CONFIRMED TASKS.
    # -------------------------------------------------------------------------

    if source == "cvat_annotations2_timeframe":

        task_id = str(
            row.get("task_id", "")
        ).strip()

        if task_id not in confirmed_annotations2:
            return None, "annotations2_mapping_not_confirmed"

        item = confirmed_annotations2[task_id]

        path = Path(item["physical_path"])

        if not path.exists():
            return None, "confirmed_physical_video_missing"

        return path, "confirmed_annotations2_mapping"

    # -------------------------------------------------------------------------
    # 3. ORIGINAL CVAT
    #
    # Exact physical-video filename matching only.
    # -------------------------------------------------------------------------

    if source == "cvat_original_timeframe":

        video_name = str(
            row.get("video", "")
        ).strip()

        key = normalize_name(video_name)

        # Reject UNKNOWN_TASK outright.
        if key.startswith("unknown_task"):
            return None, "unknown_task_rejected"

        if key in physical_index:
            return (
                Path(
                    physical_index[key]["path"]
                ),
                "original_cvat_exact",
            )

        return None, "original_cvat_video_not_found"

    return None, "unknown_source"


# =============================================================================
# FRAME VALIDATION
# =============================================================================

def get_integer(row: pd.Series, column: str) -> Optional[int]:
    value = row.get(column)

    if value is None:
        return None

    if pd.isna(value):
        return None

    try:
        return int(round(float(value)))
    except Exception:
        return None


def determine_frame_range(
    row: pd.Series,
    video_frames: int,
) -> Tuple[Optional[int], Optional[int], str]:

    start = get_integer(row, "start_frame")
    end = get_integer(row, "end_frame")

    if start is None:
        start = get_integer(row, "start_frame_local")

    if end is None:
        end = get_integer(row, "end_frame_local")

    if start is None:
        start = 0

    if end is None:
        end = start + NUM_FRAMES - 1

    # Make sure start/end are sensible.
    start = max(0, start)
    end = max(start, end)

    # We need NUM_FRAMES actual frames.
    if start + NUM_FRAMES > video_frames:
        return None, None, "window_exceeds_physical_video"

    # Do not silently shift windows.
    # The requested temporal location must actually exist.
    if end >= video_frames:
        return None, None, "annotation_end_exceeds_physical_video"

    if end - start + 1 < NUM_FRAMES:
        return None, None, "annotation_window_shorter_than_16_frames"

    return start, start + NUM_FRAMES - 1, "valid"


# =============================================================================
# CLIP NAME
# =============================================================================

def safe_filename(value: object) -> str:
    s = str(value)

    s = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        s,
    )

    s = re.sub(
        r"\s+",
        "_",
        s,
    )

    return s.strip(" ._")


def make_clip_filename(
    row: pd.Series,
    row_index: int,
) -> str:

    source = safe_filename(
        row.get("source", "unknown")
    )

    video = safe_filename(
        Path(
            str(row.get("video", "video"))
        ).stem
    )

    label = safe_filename(
        row.get("label", "UNKNOWN")
    )

    start = get_integer(
        row,
        "start_frame",
    )

    if start is None:
        start = get_integer(
            row,
            "start_frame_local",
        )

    if start is None:
        start = 0

    task_id = safe_filename(
        row.get("task_id", "")
    )

    task_part = (
        f"_task{task_id}"
        if task_id
        and task_id.lower() != "nan"
        else ""
    )

    return (
        f"{source}"
        f"__{video}"
        f"{task_part}"
        f"__{label}"
        f"__f{start:06d}"
        f"__r{row_index:06d}.mp4"
    )


# =============================================================================
# CLIP WRITER
# =============================================================================

def write_clip(
    video_path: Path,
    start_frame: int,
    output_path: Path,
) -> Tuple[bool, str]:

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return False, "cannot_open_video"

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if width <= 0 or height <= 0:
        cap.release()
        return False, "invalid_video_dimensions"

    # -------------------------------------------------------------------------
    # MP4 writer
    # -------------------------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = output_path.with_suffix(
        ".tmp.mp4"
    )

    if temp_path.exists():
        try:
            temp_path.unlink()
        except Exception:
            pass

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(temp_path),
        fourcc,
        fps,
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
    )

    if not writer.isOpened():
        cap.release()
        return False, "cannot_open_video_writer"

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame,
    )

    written = 0

    try:

        for _ in range(NUM_FRAMES):

            ok, frame = cap.read()

            if not ok or frame is None:
                writer.release()
                cap.release()

                if temp_path.exists():
                    temp_path.unlink()

                return False, "failed_reading_frame"

            frame = cv2.resize(
                frame,
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                interpolation=cv2.INTER_AREA,
            )

            writer.write(frame)

            written += 1

    finally:
        writer.release()
        cap.release()

    if written != NUM_FRAMES:
        if temp_path.exists():
            temp_path.unlink()

        return False, (
            f"wrong_frame_count_{written}"
        )

    # Replace atomically.
    try:

        if output_path.exists():
            output_path.unlink()

        temp_path.replace(
            output_path
        )

    except Exception as exc:

        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass

        return False, (
            f"rename_failed_{exc}"
        )

    return True, "generated"


# =============================================================================
# EXISTING CLIP VALIDATION
# =============================================================================

def existing_clip_is_valid(
    path: Path,
) -> bool:

    if not path.exists():
        return False

    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():
        return False

    frames = int(
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

    return (
        frames == NUM_FRAMES
        and width == IMAGE_SIZE
        and height == IMAGE_SIZE
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    header(
        "EAGLE-MARS CONFIRMED ALL-CONTINUAL 16-FRAME CLIP GENERATOR"
    )

    print()
    print("RULES")
    print("-" * 90)
    print("1. continual_50_videos -> manifest video_path")
    print("2. annotations2 -> confirmed mappings only")
    print("3. original CVAT -> exact physical-video match only")
    print("4. UNKNOWN_TASK_* -> rejected")
    print("5. Existing valid clips -> reused")
    print("6. No fuzzy video matching")
    print()

    print("CONFIGURATION")
    print("-" * 90)
    print(
        f"Input manifest : {INPUT_MANIFEST}"
    )
    print(
        f"Annotated vids : {ANNOTATED_VIDEO_DIR}"
    )
    print(
        f"Output clips   : {CLIP_DIR}"
    )
    print(
        f"Output manifest: {OUTPUT_MANIFEST}"
    )
    print(
        f"Frames         : {NUM_FRAMES}"
    )
    print(
        f"Resolution     : {IMAGE_SIZE}x{IMAGE_SIZE}"
    )
    print(
        f"Overwrite      : {OVERWRITE}"
    )

    # -------------------------------------------------------------------------
    # Validate input
    # -------------------------------------------------------------------------

    if not INPUT_MANIFEST.exists():
        print()
        print("ERROR: input manifest does not exist.")
        print(INPUT_MANIFEST)
        sys.exit(1)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    CLIP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Load manifest
    # -------------------------------------------------------------------------

    df = pd.read_csv(
        INPUT_MANIFEST
    )

    print()
    print(
        f"Manifest rows: {len(df)}"
    )

    print()
    print("Columns:")
    print(df.columns.tolist())

    required = {
        "video_id",
        "video",
        "source",
        "task_id",
        "split",
        "label",
        "start_frame",
        "end_frame",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        print()
        print(
            "ERROR: missing required columns:"
        )
        print(sorted(missing))
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Physical videos
    # -------------------------------------------------------------------------

    physical_index = (
        build_physical_video_index(
            ANNOTATED_VIDEO_DIR
        )
    )

    # -------------------------------------------------------------------------
    # Confirmed annotations2
    # -------------------------------------------------------------------------

    confirmed_annotations2 = (
        load_confirmed_annotations2_mapping(
            MAPPING_JSON,
            physical_index,
        )
    )

    # -------------------------------------------------------------------------
    # Filter source rows
    # -------------------------------------------------------------------------

    section(
        "SELECTING SAFE INPUT DATASET"
    )

    source_order = [
        "continual_50_videos",
        "cvat_original_timeframe",
        "cvat_annotations2_timeframe",
    ]

    selected_rows = []
    rejected_rows = []

    for idx, row in df.iterrows():

        source = str(
            row.get("source", "")
        ).strip()

        # Only the three intended sources.
        if source not in source_order:
            rejected_rows.append(
                {
                    "row": int(idx),
                    "source": source,
                    "video": str(
                        row.get("video", "")
                    ),
                    "reason": "source_not_allowed",
                }
            )
            continue

        # annotations2: only confirmed tasks.
        if source == "cvat_annotations2_timeframe":

            task_id = str(
                row.get("task_id", "")
            ).strip()

            if (
                task_id
                not in confirmed_annotations2
            ):
                rejected_rows.append(
                    {
                        "row": int(idx),
                        "source": source,
                        "video": str(
                            row.get("video", "")
                        ),
                        "task_id": task_id,
                        "reason": (
                            "annotations2_mapping_not_confirmed"
                        ),
                    }
                )
                continue

        selected_rows.append(
            (
                int(idx),
                row,
            )
        )

    print()
    print(
        f"Original manifest rows : {len(df)}"
    )
    print(
        f"Selected safe rows     : {len(selected_rows)}"
    )
    print(
        f"Rejected rows          : {len(rejected_rows)}"
    )

    # -------------------------------------------------------------------------
    # Selected source statistics
    # -------------------------------------------------------------------------

    if selected_rows:

        selected_df = pd.DataFrame(
            [
                row.to_dict()
                for _, row in selected_rows
            ]
        )

        print()
        print(
            "SAFE SOURCE × LABEL"
        )
        print("-" * 90)
        print(
            selected_df
            .groupby(
                ["source", "label"]
            )
            .size()
        )

        print()
        print(
            "SAFE SPLIT × LABEL"
        )
        print("-" * 90)
        print(
            selected_df
            .groupby(
                ["split", "label"]
            )
            .size()
        )

    # -------------------------------------------------------------------------
    # Generate
    # -------------------------------------------------------------------------

    section(
        "GENERATING / REUSING SAFE CLIPS"
    )

    generated = []
    errors = []

    counts = {
        "generated": 0,
        "reused": 0,
        "failed": 0,
    }

    source_counts = {}
    source_label_counts = {}

    total = len(selected_rows)

    for position, (
        original_idx,
        row,
    ) in enumerate(
        selected_rows,
        start=1,
    ):

        source = str(
            row.get("source", "")
        ).strip()

        label = str(
            row.get("label", "")
        ).strip()

        video_name = str(
            row.get("video", "")
        ).strip()

        # ---------------------------------------------------------------------
        # Resolve physical video
        # ---------------------------------------------------------------------

        video_path, resolution_method = (
            resolve_video_path(
                row,
                physical_index,
                confirmed_annotations2,
            )
        )

        if video_path is None:

            counts["failed"] += 1

            errors.append(
                {
                    "row": int(original_idx),
                    "video": video_name,
                    "source": source,
                    "label": label,
                    "reason": resolution_method,
                }
            )

            continue

        # ---------------------------------------------------------------------
        # Physical video metadata
        # ---------------------------------------------------------------------

        info = read_video_info(
            video_path
        )

        if info is None:

            counts["failed"] += 1

            errors.append(
                {
                    "row": int(original_idx),
                    "video": video_name,
                    "source": source,
                    "label": label,
                    "physical_video": str(
                        video_path
                    ),
                    "reason": (
                        "cannot_read_physical_video"
                    ),
                }
            )

            continue

        # ---------------------------------------------------------------------
        # Frame range
        # ---------------------------------------------------------------------

        start_frame, end_frame, frame_status = (
            determine_frame_range(
                row,
                info["frames"],
            )
        )

        if (
            start_frame is None
            or end_frame is None
        ):

            counts["failed"] += 1

            errors.append(
                {
                    "row": int(original_idx),
                    "video": video_name,
                    "source": source,
                    "label": label,
                    "physical_video": str(
                        video_path
                    ),
                    "physical_frames": info[
                        "frames"
                    ],
                    "requested_start": row.get(
                        "start_frame"
                    ),
                    "requested_end": row.get(
                        "end_frame"
                    ),
                    "reason": frame_status,
                }
            )

            continue

        # ---------------------------------------------------------------------
        # Output path
        # ---------------------------------------------------------------------

        clip_filename = (
            make_clip_filename(
                row,
                original_idx,
            )
        )

        clip_path = (
            CLIP_DIR
            / clip_filename
        )

        # ---------------------------------------------------------------------
        # Existing clip
        # ---------------------------------------------------------------------

        if (
            not OVERWRITE
            and existing_clip_is_valid(
                clip_path
            )
        ):

            counts["reused"] += 1

            record = row.to_dict()

            record.update(
                {
                    "physical_video": video_path.name,
                    "physical_video_path": str(
                        video_path
                    ),
                    "clip_path": str(
                        clip_path
                    ),
                    "clip_frames": NUM_FRAMES,
                    "clip_width": IMAGE_SIZE,
                    "clip_height": IMAGE_SIZE,
                    "resolved_by": resolution_method,
                    "clip_status": "reused",
                    "original_manifest_row": int(
                        original_idx
                    ),
                    "actual_start_frame": int(
                        start_frame
                    ),
                    "actual_end_frame": int(
                        end_frame
                    ),
                }
            )

            generated.append(record)

            key = (
                source,
                label,
            )

            source_label_counts[key] = (
                source_label_counts.get(
                    key,
                    0,
                )
                + 1
            )

        else:

            # ---------------------------------------------------------------
            # Generate actual clip
            # ---------------------------------------------------------------

            ok, reason = write_clip(
                video_path,
                start_frame,
                clip_path,
            )

            if not ok:

                counts["failed"] += 1

                errors.append(
                    {
                        "row": int(
                            original_idx
                        ),
                        "video": video_name,
                        "source": source,
                        "label": label,
                        "physical_video": str(
                            video_path
                        ),
                        "start_frame": int(
                            start_frame
                        ),
                        "end_frame": int(
                            end_frame
                        ),
                        "reason": reason,
                    }
                )

                continue

            counts["generated"] += 1

            record = row.to_dict()

            record.update(
                {
                    "physical_video": video_path.name,
                    "physical_video_path": str(
                        video_path
                    ),
                    "clip_path": str(
                        clip_path
                    ),
                    "clip_frames": NUM_FRAMES,
                    "clip_width": IMAGE_SIZE,
                    "clip_height": IMAGE_SIZE,
                    "resolved_by": resolution_method,
                    "clip_status": "generated",
                    "original_manifest_row": int(
                        original_idx
                    ),
                    "actual_start_frame": int(
                        start_frame
                    ),
                    "actual_end_frame": int(
                        end_frame
                    ),
                }
            )

            generated.append(record)

            key = (
                source,
                label,
            )

            source_label_counts[key] = (
                source_label_counts.get(
                    key,
                    0,
                )
                + 1
            )

        # ---------------------------------------------------------------------
        # Progress
        # ---------------------------------------------------------------------

        if (
            position % PROGRESS_EVERY == 0
            or position == total
        ):

            percent = (
                100.0
                * position
                / max(total, 1)
            )

            print(
                f"Progress: "
                f"{position}/{total} "
                f"({percent:.1f}%) | "
                f"generated={counts['generated']} | "
                f"reused={counts['reused']} | "
                f"failed={counts['failed']}"
            )

    # -------------------------------------------------------------------------
    # Final manifest
    # -------------------------------------------------------------------------

    final_df = pd.DataFrame(
        generated
    )

    if not final_df.empty:

        final_df = final_df.sort_values(
            [
                "split",
                "source",
                "physical_video",
                "actual_start_frame",
            ],
            kind="stable",
        ).reset_index(
            drop=True
        )

    final_df.to_csv(
        OUTPUT_MANIFEST,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------------------

    header(
        "FINAL CLIP VALIDATION"
    )

    print(
        f"Selected input rows : {len(selected_rows)}"
    )

    print(
        f"Clips generated     : {counts['generated']}"
    )

    print(
        f"Existing reused     : {counts['reused']}"
    )

    print(
        f"Failed              : {counts['failed']}"
    )

    print(
        f"Final clips         : {len(final_df)}"
    )

    if not final_df.empty:

        print()
        print(
            "FINAL SOURCE × LABEL"
        )
        print("-" * 90)

        print(
            final_df
            .groupby(
                ["source", "label"]
            )
            .size()
        )

        print()
        print(
            "FINAL SPLIT × LABEL"
        )
        print("-" * 90)

        print(
            final_df
            .groupby(
                ["split", "label"]
            )
            .size()
        )

        print()
        print(
            "FINAL VIDEOS BY SPLIT"
        )
        print("-" * 90)

        print(
            final_df
            .groupby("split")
            ["physical_video"]
            .nunique()
        )

        print()
        print(
            "FINAL SOURCE × VIDEO"
        )
        print("-" * 90)

        print(
            final_df
            .groupby(
                [
                    "source",
                    "physical_video",
                ]
            )
            .size()
            .to_string()
        )

        # -------------------------------------------------------------
        # UNKNOWN TASK safety check
        # -------------------------------------------------------------

        unknown_mask = (
            final_df[
                "physical_video"
            ]
            .astype(str)
            .str.upper()
            .str.startswith(
                "UNKNOWN_TASK"
            )
        )

        unknown_count = int(
            unknown_mask.sum()
        )

        print()
        print(
            "UNKNOWN_TASK PHYSICAL VIDEO CHECK"
        )
        print("-" * 90)

        print(
            f"UNKNOWN_TASK clips: "
            f"{unknown_count}"
        )

        # -------------------------------------------------------------
        # Missing physical paths
        # -------------------------------------------------------------

        missing_physical = 0

        for p in final_df[
            "physical_video_path"
        ].astype(str):

            if not Path(p).exists():
                missing_physical += 1

        print()
        print(
            "PHYSICAL FILE CHECK"
        )
        print("-" * 90)

        print(
            f"Missing physical files: "
            f"{missing_physical}"
        )

        # -------------------------------------------------------------
        # Clip file check
        # -------------------------------------------------------------

        missing_clips = 0
        invalid_clips = 0

        for p in final_df[
            "clip_path"
        ].astype(str):

            clip = Path(p)

            if not clip.exists():
                missing_clips += 1
                continue

            if not existing_clip_is_valid(
                clip
            ):
                invalid_clips += 1

        print()
        print(
            "CLIP FILE CHECK"
        )
        print("-" * 90)

        print(
            f"Missing clips : {missing_clips}"
        )

        print(
            f"Invalid clips : {invalid_clips}"
        )

    # -------------------------------------------------------------------------
    # Save errors
    # -------------------------------------------------------------------------

    ERRORS_JSON.write_text(
        json.dumps(
            errors,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Save summary
    # -------------------------------------------------------------------------

    summary = {
        "input_manifest": str(
            INPUT_MANIFEST
        ),
        "output_manifest": str(
            OUTPUT_MANIFEST
        ),
        "clip_directory": str(
            CLIP_DIR
        ),
        "num_frames": NUM_FRAMES,
        "image_size": IMAGE_SIZE,
        "overwrite": OVERWRITE,
        "input_rows": int(
            len(df)
        ),
        "selected_rows": int(
            len(selected_rows)
        ),
        "rejected_rows": int(
            len(rejected_rows)
        ),
        "generated": int(
            counts["generated"]
        ),
        "reused": int(
            counts["reused"]
        ),
        "failed": int(
            counts["failed"]
        ),
        "final_clips": int(
            len(final_df)
        ),
        "confirmed_annotations2_tasks": int(
            len(
                confirmed_annotations2
            )
        ),
        "errors_file": str(
            ERRORS_JSON
        ),
        "rejected_rows": rejected_rows,
    }

    if not final_df.empty:

        summary[
            "final_source_label_counts"
        ] = {
            f"{source}|{label}": int(
                count
            )
            for (
                source,
                label,
            ), count in (
                final_df
                .groupby(
                    [
                        "source",
                        "label",
                    ]
                )
                .size()
                .items()
            )
        }

        summary[
            "final_split_label_counts"
        ] = {
            f"{split}|{label}": int(
                count
            )
            for (
                split,
                label,
            ), count in (
                final_df
                .groupby(
                    [
                        "split",
                        "label",
                    ]
                )
                .size()
                .items()
            )
        }

        summary[
            "final_video_counts"
        ] = {
            str(split): int(
                count
            )
            for split, count in (
                final_df
                .groupby(
                    "split"
                )[
                    "physical_video"
                ]
                .nunique()
                .items()
            )
        }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Rejected rows summary
    # -------------------------------------------------------------------------

    section(
        "REJECTED ROWS"
    )

    if rejected_rows:

        rejected_df = pd.DataFrame(
            rejected_rows
        )

        if (
            "source" in rejected_df
            and "reason" in rejected_df
        ):

            print(
                rejected_df
                .groupby(
                    [
                        "source",
                        "reason",
                    ]
                )
                .size()
                .to_string()
            )

    else:
        print(
            "No rows rejected during source selection."
        )

    # -------------------------------------------------------------------------
    # Errors summary
    # -------------------------------------------------------------------------

    section(
        "GENERATION ERRORS"
    )

    if errors:

        errors_df = pd.DataFrame(
            errors
        )

        if (
            "source" in errors_df
            and "reason" in errors_df
        ):

            print(
                errors_df
                .groupby(
                    [
                        "source",
                        "reason",
                    ]
                )
                .size()
                .sort_values(
                    ascending=False
                )
                .to_string()
            )

    else:
        print(
            "No generation errors."
        )

    # -------------------------------------------------------------------------
    # Final
    # -------------------------------------------------------------------------

    header(
        "ALL CONTINUAL CLIP GENERATION COMPLETE"
    )

    print()
    print(
        "Clips directory:"
    )
    print(CLIP_DIR)

    print()
    print(
        "Manifest:"
    )
    print(OUTPUT_MANIFEST)

    print()
    print(
        "Summary:"
    )
    print(SUMMARY_JSON)

    print()
    print(
        "Errors:"
    )
    print(ERRORS_JSON)

    print()
    print(
        "FINAL RESULT"
    )
    print("-" * 90)
    print(
        f"Input rows       : {len(df)}"
    )
    print(
        f"Safe rows        : {len(selected_rows)}"
    )
    print(
        f"Generated        : {counts['generated']}"
    )
    print(
        f"Reused           : {counts['reused']}"
    )
    print(
        f"Failed           : {counts['failed']}"
    )
    print(
        f"Final clips      : {len(final_df)}"
    )

    if counts["failed"] > 0:
        print()
        print(
            "WARNING: some rows could not be converted."
        )
        print(
            "Review:"
        )
        print(
            ERRORS_JSON
        )
    else:
        print()
        print(
            "SUCCESS: every selected row has a valid clip."
        )

    print()
    print(
        "The resulting manifest is ready for training "
        "only after checking the final source/split counts."
    )


if __name__ == "__main__":
    main()