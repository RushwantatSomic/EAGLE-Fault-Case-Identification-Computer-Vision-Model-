from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET

import cv2


# =============================================================================
# EAGLE-MARS
# CVAT -> MANUAL TIMEFRAME ANNOTATION
# =============================================================================
#
# PURPOSE
# -------
# Take the videos that were already annotated in the CVAT projects and allow
# manual temporal fault annotation using timestamps.
#
# SOURCES
# -------
#   Annotations/cvat_export.xml
#   Annotations/annotations2.xml
#
# OUTPUT
# ------
#   outputs/unified_dataset/timeframe_annotations/
#
#       cvat_original_timeframes.json
#       cvat_annotations2_timeframes.json
#
# These will later be combined with:
#
#   outputs/continual_learning/continual_annotations.json
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

ANNOTATIONS_DIR = ROOT / "Annotations"

ORIGINAL_XML = ANNOTATIONS_DIR / "cvat_export.xml"
ANNOTATIONS2_XML = ANNOTATIONS_DIR / "annotations2.xml"

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "unified_dataset"
    / "timeframe_annotations"
)

ORIGINAL_OUTPUT = (
    OUTPUT_DIR
    / "cvat_original_timeframes.json"
)

ANNOTATIONS2_OUTPUT = (
    OUTPUT_DIR
    / "cvat_annotations2_timeframes.json"
)


# =============================================================================
# VIDEO SEARCH
# =============================================================================

VIDEO_SEARCH_ROOTS = [
    ROOT,
]


# =============================================================================
# TEMPORAL CONFIGURATION
# =============================================================================

WINDOW_FRAMES = 16
WINDOW_STRIDE = 8

FAULT_THRESHOLD = 0.50


# =============================================================================
# REAL CVAT TASK -> VIDEO MAPPING
# =============================================================================
#
# IMPORTANT:
#
# These are the actual videos from your CVAT project.
#
# We explicitly map the task IDs because the XML export you have does not
# reliably expose the original video filename at the location the previous
# script expected.
#
# =============================================================================

ORIGINAL_TASK_VIDEO_MAP = {

    "2523742": "Carton1.mp4",
    "2523747": "Carton2.mp4",
    "2523748": "Carton3.mp4",

    "2524033": "Cartoning.mp4",

    "2524047": "Cartoning 2.mp4",

    "2524049": "Collecting Table 2.mp4",
    "2524050": "Collecting Table 3.mp4",
    "2524051": "Collecting Table 4.mp4",
    "2524052": "Collecting Table 5.mp4",
    "2524053": "Collecting Table.mp4",

    "2524054": "Covering 2.mp4",
    "2524056": "Covering 3.mp4",
    "2524057": "Covering 4.mp4",

    "2524059": "Lamella Chain 2.mp4",
    "2524061": "Lamella Chain 3.mp4",
    "2524062": "Lamella Chain 4.mp4",
    "2524063": "Lamella Chain.mp4",

    "2524064": "Magazine 2.mp4",
    "2524065": "Magazine 3.mp4",
    "2524068": "Magazine 4.mp4",

    "2526254": "Magazine.mp4",
}


# =============================================================================
# annotations2.xml MAPPING
# =============================================================================
#
# These are the videos you showed earlier from annotations2.xml.
#
# We normally obtain the task IDs from the XML and map them to the actual
# filename using the CVAT task metadata / XML information.
#
# If a filename is already correctly present in the XML, that is used.
#
# =============================================================================


# =============================================================================
# PRINT HELPERS
# =============================================================================

def line():
    print("=" * 90)


def subline():
    print("-" * 90)


# =============================================================================
# TIMESTAMP PARSING
# =============================================================================

def parse_timestamp(value):

    value = str(value).strip()

    if not value:
        raise ValueError("Empty timestamp.")

    # Seconds
    if re.fullmatch(r"\d+(\.\d+)?", value):
        return float(value)

    parts = value.split(":")

    # MM:SS
    if len(parts) == 2:

        minutes = float(parts[0])
        seconds = float(parts[1])

        if seconds >= 60:
            raise ValueError(
                "Seconds must be less than 60."
            )

        return minutes * 60 + seconds

    # HH:MM:SS
    if len(parts) == 3:

        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])

        if minutes >= 60:
            raise ValueError(
                "Minutes must be less than 60."
            )

        if seconds >= 60:
            raise ValueError(
                "Seconds must be less than 60."
            )

        return (
            hours * 3600
            + minutes * 60
            + seconds
        )

    raise ValueError(
        f"Invalid timestamp: {value}"
    )


# =============================================================================
# FORMAT TIME
# =============================================================================

def format_time(seconds):

    seconds = max(
        0.0,
        float(seconds)
    )

    hours = int(seconds // 3600)

    minutes = int(
        (seconds % 3600) // 60
    )

    remaining = seconds % 60

    if hours > 0:

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{remaining:05.2f}"
        )

    return (
        f"{minutes:02d}:"
        f"{remaining:05.2f}"
    )


# =============================================================================
# LOAD XML TASKS
# =============================================================================

def load_xml_tasks(
    xml_path,
    source_name,
):

    print()
    print(
        f"Loading XML:\n  {xml_path}"
    )

    if not xml_path.exists():

        raise FileNotFoundError(
            f"XML file not found:\n{xml_path}"
        )

    root = ET.parse(
        xml_path
    ).getroot()

    tracks = root.findall(
        ".//track"
    )

    tasks = {}

    for track in tracks:

        task_id = str(
            track.attrib.get(
                "task_id",
                "UNKNOWN"
            )
        )

        tasks.setdefault(
            task_id,
            []
        ).append(
            track
        )

    results = []

    for task_id, task_tracks in tasks.items():

        frames = []
        labels = set()

        for track in task_tracks:

            labels.add(
                track.attrib.get(
                    "label",
                    "UNKNOWN"
                )
            )

            for box in track.findall(
                "./box"
            ):

                frame_value = box.attrib.get(
                    "frame"
                )

                if frame_value is None:
                    continue

                try:
                    frames.append(
                        int(frame_value)
                    )
                except ValueError:
                    pass

        if not frames:
            continue

        min_frame = min(frames)
        max_frame = max(frames)

        # ---------------------------------------------------------------
        # Find task metadata
        # ---------------------------------------------------------------

        task_element = None

        for task in root.findall(
            ".//task"
        ):

            if str(
                task.attrib.get("id", "")
            ) == task_id:

                task_element = task
                break

        video_name = None

        if task_element is not None:

            name_element = task_element.find(
                "./name"
            )

            if (
                name_element is not None
                and name_element.text
            ):

                video_name = (
                    Path(
                        name_element.text.strip()
                    ).name
                )

        # ---------------------------------------------------------------
        # Look for original image filename
        # ---------------------------------------------------------------

        if not video_name:

            for image in root.findall(
                ".//image"
            ):

                name = image.attrib.get(
                    "name"
                )

                if name:

                    video_name = (
                        Path(name).name
                    )

                    break

        # ---------------------------------------------------------------
        # ORIGINAL CVAT:
        # use our verified task mapping
        # ---------------------------------------------------------------

        if (
            source_name
            == "cvat_original_timeframe"
        ):

            if task_id in ORIGINAL_TASK_VIDEO_MAP:

                video_name = (
                    ORIGINAL_TASK_VIDEO_MAP[
                        task_id
                    ]
                )

        # ---------------------------------------------------------------
        # Fallback
        # ---------------------------------------------------------------

        if not video_name:

            video_name = (
                f"UNKNOWN_TASK_{task_id}.mp4"
            )

        results.append(
            {
                "source": source_name,

                "task_id": task_id,

                "video": video_name,

                "labelled_start_frame": min_frame,

                "labelled_end_frame": max_frame,

                "labelled_frames": len(
                    set(frames)
                ),

                "tracks": len(
                    task_tracks
                ),

                "cvat_labels": sorted(
                    labels
                ),
            }
        )

    # -----------------------------------------------------------------
    # Sort
    # -----------------------------------------------------------------

    results.sort(
        key=lambda item: (
            item["labelled_start_frame"],
            item["task_id"],
        )
    )

    return results


# =============================================================================
# FIND ACTUAL VIDEO
# =============================================================================

def find_video(
    video_name
):

    # ---------------------------------------------------------------
    # Direct path
    # ---------------------------------------------------------------

    candidate = Path(
        video_name
    )

    if candidate.exists():

        return candidate.resolve()

    # ---------------------------------------------------------------
    # Search project
    # ---------------------------------------------------------------

    for root in VIDEO_SEARCH_ROOTS:

        if not root.exists():
            continue

        matches = list(
            root.rglob(
                video_name
            )
        )

        if matches:

            return matches[0].resolve()

    return None


# =============================================================================
# VIDEO INFORMATION
# =============================================================================

def read_video_info(
    video_path
):

    if video_path is None:

        return None

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        return None

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
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

        return None

    duration = (
        total_frames / fps
    )

    return {
        "fps": float(fps),
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_seconds": duration,
    }


# =============================================================================
# ASK FPS WHEN VIDEO IS NOT FOUND
# =============================================================================

def ask_fps():

    while True:

        value = input(
            "Enter FPS "
            "(for example 25 or 30): "
        ).strip()

        try:

            fps = float(value)

            if fps <= 0:
                raise ValueError

            return fps

        except ValueError:

            print(
                "Please enter a positive FPS."
            )


# =============================================================================
# ASK TIMEFRAME
# =============================================================================

def ask_fault_interval(
    duration
):

    while True:

        start_text = input(
            "\nFault START "
            "(or N for no fault): "
        ).strip()

        if start_text.upper() == "N":

            return None

        try:

            start = parse_timestamp(
                start_text
            )

        except ValueError as exc:

            print(
                f"Invalid start time: {exc}"
            )
            continue

        if start < 0 or start >= duration:

            print(
                "Start time is outside "
                "the video."
            )
            continue

        break

    while True:

        end_text = input(
            "Fault END: "
        ).strip()

        try:

            end = parse_timestamp(
                end_text
            )

        except ValueError as exc:

            print(
                f"Invalid end time: {exc}"
            )
            continue

        if end <= start:

            print(
                "End must be greater than start."
            )
            continue

        if end > duration:

            print(
                "End time is beyond "
                "the video duration."
            )
            continue

        break

    return {
        "start_seconds": start,
        "end_seconds": end,
    }


# =============================================================================
# YES / NO
# =============================================================================

def yes_no(
    question
):

    while True:

        value = input(
            f"{question} [Y/N]: "
        ).strip().upper()

        if value in ("Y", "YES"):
            return True

        if value in ("N", "NO"):
            return False

        print(
            "Please enter Y or N."
        )


# =============================================================================
# CREATE WINDOWS
# =============================================================================

def generate_windows(
    total_frames,
    fps,
    fault_intervals,
):

    def is_fault_frame(
        frame
    ):

        for interval in fault_intervals:

            if (
                interval["start_frame"]
                <= frame
                <= interval["end_frame"]
            ):

                return True

        return False

    windows = []

    start = 0

    while (
        start + WINDOW_FRAMES
        <= total_frames
    ):

        end = (
            start
            + WINDOW_FRAMES
            - 1
        )

        fault_frames = 0

        for frame in range(
            start,
            end + 1
        ):

            if is_fault_frame(
                frame
            ):

                fault_frames += 1

        fault_ratio = (
            fault_frames
            / WINDOW_FRAMES
        )

        if (
            fault_ratio
            >= FAULT_THRESHOLD
        ):

            label = "FAULT"

        else:

            label = "NORMAL"

        windows.append(
            {
                "start_frame": start,

                "end_frame": end,

                "start_seconds":
                    start / fps,

                "end_seconds":
                    (end + 1) / fps,

                "fault_frames":
                    fault_frames,

                "fault_ratio":
                    fault_ratio,

                "label":
                    label,
            }
        )

        start += WINDOW_STRIDE

    return windows


# =============================================================================
# ANNOTATE ONE VIDEO
# =============================================================================

def annotate_video(
    task,
    index,
    total
):

    task_id = task[
        "task_id"
    ]

    video_name = task[
        "video"
    ]

    source = task[
        "source"
    ]

    line()

    print(
        f"VIDEO {index}/{total}"
    )

    line()

    print(
        f"Source : {source}"
    )

    print(
        f"Task   : {task_id}"
    )

    print(
        f"VIDEO  : {video_name}"
    )

    print()

    print(
        "CVAT INFORMATION"
    )

    subline()

    print(
        f"Labelled frames : "
        f"{task['labelled_start_frame']}"
        f" - "
        f"{task['labelled_end_frame']}"
    )

    print(
        f"CVAT tracks     : "
        f"{task['tracks']}"
    )

    print(
        "CVAT labels     : "
        + ", ".join(
            task["cvat_labels"]
        )
    )

    # -----------------------------------------------------------------
    # Find actual video
    # -----------------------------------------------------------------

    video_path = find_video(
        video_name
    )

    if video_path:

        print()
        print(
            "ACTUAL VIDEO FILE"
        )

        subline()

        print(
            video_path
        )

    else:

        print()
        print(
            "WARNING: VIDEO FILE NOT FOUND"
        )

        print(
            f"Expected filename: {video_name}"
        )

        print()
        print(
            "The annotation can still be "
            "created, but FPS must be "
            "entered manually."
        )

    # -----------------------------------------------------------------
    # Read video
    # -----------------------------------------------------------------

    info = read_video_info(
        video_path
    )

    if info:

        fps = info[
            "fps"
        ]

        total_frames = info[
            "total_frames"
        ]

        duration = info[
            "duration_seconds"
        ]

        width = info[
            "width"
        ]

        height = info[
            "height"
        ]

        print()
        print(
            "VIDEO INFORMATION"
        )

        subline()

        print(
            f"FPS        : {fps:.3f}"
        )

        print(
            f"Frames     : {total_frames}"
        )

        print(
            f"Duration   : "
            f"{format_time(duration)}"
        )

        print(
            f"Resolution : "
            f"{width}x{height}"
        )

    else:

        fps = ask_fps()

        total_frames = (
            task[
                "labelled_end_frame"
            ] + 1
        )

        duration = (
            total_frames / fps
        )

        width = None
        height = None

    # -----------------------------------------------------------------
    # Manual annotation
    # -----------------------------------------------------------------

    print()
    print(
        "MANUAL FAULT TIMEFRAMES"
    )

    subline()

    print(
        "Enter the time period(s) where "
        "a real fault is visible."
    )

    print()
    print(
        "Example:"
    )

    print(
        "  Fault START: 01:20"
    )

    print(
        "  Fault END  : 01:35"
    )

    print()
    print(
        "For a completely normal video:"
    )

    print(
        "  Enter N at the first prompt."
    )

    fault_intervals = []

    while True:

        interval = ask_fault_interval(
            duration
        )

        if interval is None:

            break

        start_frame = int(
            round(
                interval[
                    "start_seconds"
                ]
                * fps
            )
        )

        end_frame = int(
            round(
                interval[
                    "end_seconds"
                ]
                * fps
            )
        ) - 1

        start_frame = max(
            0,
            start_frame
        )

        end_frame = min(
            total_frames - 1,
            end_frame
        )

        interval[
            "start_frame"
        ] = start_frame

        interval[
            "end_frame"
        ] = end_frame

        fault_intervals.append(
            interval
        )

        print()
        print(
            "ADDED FAULT:"
        )

        print(
            f"  "
            f"{format_time(interval['start_seconds'])}"
            f" -> "
            f"{format_time(interval['end_seconds'])}"
        )

        print(
            f"  frames "
            f"{start_frame}"
            f" - "
            f"{end_frame}"
        )

        if not yes_no(
            "Add another fault timeframe?"
        ):

            break

    # -----------------------------------------------------------------
    # Generate windows
    # -----------------------------------------------------------------

    windows = generate_windows(
        total_frames,
        fps,
        fault_intervals
    )

    fault_windows = sum(
        1
        for window in windows
        if window["label"]
        == "FAULT"
    )

    normal_windows = (
        len(windows)
        - fault_windows
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    print()
    print(
        "GENERATED DATASET"
    )

    subline()

    print(
        f"16-frame windows : "
        f"{len(windows)}"
    )

    print(
        f"NORMAL windows   : "
        f"{normal_windows}"
    )

    print(
        f"FAULT windows    : "
        f"{fault_windows}"
    )

    print()

    if fault_intervals:

        print(
            "FAULT INTERVALS"
        )

        for interval in fault_intervals:

            print(
                f"  "
                f"{format_time(interval['start_seconds'])}"
                f" -> "
                f"{format_time(interval['end_seconds'])}"
                f" | frames "
                f"{interval['start_frame']}"
                f"-"
                f"{interval['end_frame']}"
            )

    else:

        print(
            "FAULT INTERVALS: NONE"
        )

    print()

    if not yes_no(
        "Save this annotation?"
    ):

        print(
            "Annotation discarded."
        )

        return None

    return {
        "source": source,

        "task_id": task_id,

        "video": video_name,

        "video_path":
            str(video_path)
            if video_path
            else None,

        "fps": float(fps),

        "total_frames":
            int(total_frames),

        "duration_seconds":
            float(duration),

        "width":
            width,

        "height":
            height,

        "cvat_labelled_start_frame":
            task[
                "labelled_start_frame"
            ],

        "cvat_labelled_end_frame":
            task[
                "labelled_end_frame"
            ],

        "cvat_labels":
            task[
                "cvat_labels"
            ],

        "fault_intervals":
            fault_intervals,

        "window_frames":
            WINDOW_FRAMES,

        "window_stride":
            WINDOW_STRIDE,

        "fault_window_threshold":
            FAULT_THRESHOLD,

        "windows":
            windows,
    }


# =============================================================================
# PROCESS SOURCE
# =============================================================================

def process_source(
    xml_path,
    source_name,
    output_path
):

    tasks = load_xml_tasks(
        xml_path,
        source_name
    )

    if not tasks:

        raise RuntimeError(
            f"No CVAT tasks found in:\n"
            f"{xml_path}"
        )

    print()
    line()

    print(
        f"{source_name.upper()}"
    )

    line()

    print(
        f"Videos detected: {len(tasks)}"
    )

    print()

    for index, task in enumerate(
        tasks,
        start=1
    ):

        print(
            f"{index:2d}. "
            f"{task['video']:<32} "
            f"Task={task['task_id']:<10} "
            f"frames="
            f"{task['labelled_start_frame']}-"
            f"{task['labelled_end_frame']}"
        )

    # -----------------------------------------------------------------
    # Existing annotations
    # -----------------------------------------------------------------

    existing = []

    if output_path.exists():

        try:

            with open(
                output_path,
                "r",
                encoding="utf-8"
            ) as f:

                existing = json.load(
                    f
                )

        except Exception:

            existing = []

    existing_keys = {
        (
            str(item.get("task_id")),
            item.get("video")
        )
        for item in existing
    }

    annotations = list(
        existing
    )

    # -----------------------------------------------------------------
    # Process
    # -----------------------------------------------------------------

    for index, task in enumerate(
        tasks,
        start=1
    ):

        key = (
            str(task["task_id"]),
            task["video"]
        )

        if key in existing_keys:

            print()
            print(
                f"[{index}/{len(tasks)}] "
                f"Already annotated: "
                f"{task['video']}"
            )

            continue

        result = annotate_video(
            task,
            index,
            len(tasks)
        )

        if result is None:

            continue

        annotations.append(
            result
        )

        existing_keys.add(
            key
        )

        # -------------------------------------------------------------
        # SAVE AFTER EVERY VIDEO
        # -------------------------------------------------------------

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                annotations,
                f,
                indent=2
            )

        print()
        print(
            f"Progress saved:"
        )

        print(
            f"  {output_path}"
        )

    return annotations


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary(
    title,
    annotations
):

    total_videos = len(
        annotations
    )

    total_windows = sum(
        len(
            item.get(
                "windows",
                []
            )
        )
        for item in annotations
    )

    fault_windows = sum(
        sum(
            1
            for window in item.get(
                "windows",
                []
            )
            if window.get(
                "label"
            ) == "FAULT"
        )
        for item in annotations
    )

    normal_windows = (
        total_windows
        - fault_windows
    )

    fault_videos = sum(
        1
        for item in annotations
        if item.get(
            "fault_intervals"
        )
    )

    print()
    line()

    print(
        title
    )

    line()

    print(
        f"Videos          : "
        f"{total_videos}"
    )

    print(
        f"Fault videos    : "
        f"{fault_videos}"
    )

    print(
        f"Normal-only     : "
        f"{total_videos - fault_videos}"
    )

    print()

    print(
        f"Total windows   : "
        f"{total_windows}"
    )

    print(
        f"NORMAL windows  : "
        f"{normal_windows}"
    )

    print(
        f"FAULT windows   : "
        f"{fault_windows}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    line()

    print(
        "EAGLE-MARS CVAT TIMEFRAME ANNOTATION"
    )

    line()

    print()
    print(
        "This will annotate the ACTUAL videos "
        "from your existing CVAT projects."
    )

    print()
    print(
        "It will NOT modify:"
    )

    print(
        "  Annotations/cvat_export.xml"
    )

    print(
        "  Annotations/annotations2.xml"
    )

    print()
    print(
        "Temporal configuration:"
    )

    print(
        f"  Window : {WINDOW_FRAMES} frames"
    )

    print(
        f"  Stride : {WINDOW_STRIDE} frames"
    )

    print(
        f"  Fault window threshold : "
        f"{FAULT_THRESHOLD:.0%}"
    )

    print()
    print(
        "Output directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    # =========================================================================
    # ORIGINAL CVAT
    # =========================================================================

    original = process_source(
        ORIGINAL_XML,
        "cvat_original_timeframe",
        ORIGINAL_OUTPUT
    )

    print_summary(
        "CVAT ORIGINAL TIMEFRAME SUMMARY",
        original
    )

    # =========================================================================
    # ANNOTATIONS2
    # =========================================================================

    print()
    print(
        "The first CVAT source is complete."
    )

    print()
    input(
        "Press ENTER to continue to annotations2.xml..."
    )

    annotations2 = process_source(
        ANNOTATIONS2_XML,
        "cvat_annotations2_timeframe",
        ANNOTATIONS2_OUTPUT
    )

    print_summary(
        "CVAT ANNOTATIONS2 TIMEFRAME SUMMARY",
        annotations2
    )

    # =========================================================================
    # FINAL
    # =========================================================================

    print()
    line()

    print(
        "CVAT TIMEFRAME ANNOTATION COMPLETE"
    )

    line()

    print()
    print(
        "OUTPUT FILES"
    )

    subline()

    print(
        ORIGINAL_OUTPUT
    )

    print(
        ANNOTATIONS2_OUTPUT
    )

    print()
    print(
        "These will later be combined with:"
    )

    print(
        ROOT
        / "outputs"
        / "continual_learning"
        / "continual_annotations.json"
    )

    print()
    line()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()