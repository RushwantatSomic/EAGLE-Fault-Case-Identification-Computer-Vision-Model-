from pathlib import Path
import json
import xml.etree.ElementTree as ET
import cv2


# =============================================================================
# EAGLE-MARS
# STRICT annotations2 -> PHYSICAL VIDEO MAPPER
#
# IMPORTANT:
#
# annotations2.xml contains two naming conventions:
#
#   1. Normal names
#      "Lamella Chain 2"
#      "Magazine 4"
#      "Cartoning 2"
#
#   2. TASK2 names
#      "TASK2 Lamella Chain"
#      "TASK2 Lamella Chain2"
#      "TASK2 Mag1"
#      ...
#
# TASK2 names are explicitly mapped below.
#
# We DO NOT infer video identity from frame counts.
# We DO NOT use UNKNOWN_TASK files.
# =============================================================================


ROOT = Path(__file__).resolve().parents[1]

XML_PATH = (
    ROOT
    / "Annotations"
    / "annotations2.xml"
)

TIMEFRAME_JSON = (
    ROOT
    / "outputs"
    / "unified_dataset"
    / "timeframe_annotations"
    / "cvat_annotations2_timeframes.json"
)

VIDEO_DIR = (
    ROOT
    / "data"
    / "annotated_videos"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "unified_dataset"
    / "all_continual"
)

MAPPING_FILE = (
    OUTPUT_DIR
    / "annotations2_physical_mapping.json"
)

REVIEW_FILE = (
    OUTPUT_DIR
    / "annotations2_mapping_review.json"
)


# =============================================================================
# EXPLICIT TASK2 MAPPING
# =============================================================================

TASK2_MAPPING = {

    "TASK2 Lamella Chain":
        "Lamella Chain.mp4",

    "TASK2 Lamella Chain2":
        "LamChain2.mp4",

    "TASK2 Lamella Chain3":
        "LamChain3.mp4",

    "TASK2 Mag1":
        "Mag1.mp4",

    "TASK2 Mag2":
        "Mag2.mp4",

    "TASK2 Mag3":
        "Mag3.mp4",

    "TASK2 Table1":
        "Table1.mp4",

    "TASK2 Table2":
        "Table2.mp4",

    "TASK2 Table3":
        "Table3.mp4",

    "TASK2 Carton1":
        "Carton1.mp4",

    "TASK2 Carton2":
        "Carton2.mp4",

    "TASK2 Carton3":
        "Carton3.mp4",

    "TASK2 Cover1":
        "Cover1.mp4",

    "TASK2 Cover2":
        "Cover2.mp4",

    "TASK2 Cover3":
        "Cover3.mp4",
}


# =============================================================================
# HELPERS
# =============================================================================

def normalize_name(value):

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
    )


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def save_json(
    path,
    data,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def get_video_info(path):

    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():

        return None

    frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
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

    return {
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
    }


# =============================================================================
# PHYSICAL VIDEO INDEX
# =============================================================================

def build_physical_video_index():

    print()
    print("=" * 90)
    print(
        "BUILDING PHYSICAL VIDEO INDEX"
    )
    print("=" * 90)

    videos = sorted(
        VIDEO_DIR.glob("*.mp4"),
        key=lambda p: p.name.lower(),
    )

    print(
        f"Directory : {VIDEO_DIR}"
    )

    print(
        f"Videos    : {len(videos)}"
    )

    index = {}

    for path in videos:

        info = get_video_info(
            path
        )

        if info is None:

            print(
                f"WARNING: Cannot open {path.name}"
            )

            continue

        index[
            normalize_name(
                path.stem
            )
        ] = {

            "file":
                path.name,

            "path":
                str(path),

            "frames":
                info["frames"],

            "fps":
                info["fps"],

            "width":
                info["width"],

            "height":
                info["height"],
        }

        print(
            f"{path.name:<32}"
            f" frames={info['frames']:5d}"
            f" fps={info['fps']:.4f}"
        )

    return index


# =============================================================================
# READ XML
# =============================================================================

def read_cvat_tasks():

    print()
    print("=" * 90)
    print(
        "READING annotations2.xml"
    )
    print("=" * 90)

    tree = ET.parse(
        XML_PATH
    )

    root = tree.getroot()

    tasks = []

    for task in root.iter():

        if task.tag.lower() != "task":
            continue

        fields = {}

        for child in task:

            tag = child.tag.lower()

            text = (
                child.text.strip()
                if child.text
                else ""
            )

            fields[tag] = text

        task_id = fields.get(
            "id",
            "",
        )

        name = fields.get(
            "name",
            "",
        )

        size_text = fields.get(
            "size",
            "",
        )

        start_text = fields.get(
            "start_frame",
            "",
        )

        stop_text = fields.get(
            "stop_frame",
            "",
        )

        try:
            size = int(
                size_text
            )
        except Exception:
            size = None

        try:
            start_frame = int(
                start_text
            )
        except Exception:
            start_frame = None

        try:
            stop_frame = int(
                stop_text
            )
        except Exception:
            stop_frame = None

        tasks.append(
            {
                "task_id":
                    str(task_id),

                "name":
                    name,

                "size":
                    size,

                "start_frame":
                    start_frame,

                "stop_frame":
                    stop_frame,
            }
        )

    print(
        f"Tasks found: {len(tasks)}"
    )

    return tasks


# =============================================================================
# READ TIMEFRAME RECORDS
# =============================================================================

def read_timeframes():

    print()
    print("=" * 90)
    print(
        "READING TIMEFRAME JSON"
    )
    print("=" * 90)

    records = load_json(
        TIMEFRAME_JSON
    )

    print(
        f"Records: {len(records)}"
    )

    return records


# =============================================================================
# MAPPING
# =============================================================================

def create_mapping(
    tasks,
    timeframe_records,
    physical_index,
):

    print()
    print("=" * 90)
    print(
        "MAPPING annotations2 TASKS"
    )
    print("=" * 90)

    xml_by_id = {
        str(task["task_id"]):
            task
        for task in tasks
    }

    accepted = []

    review = []

    for record in timeframe_records:

        task_id = str(
            record.get(
                "task_id",
                "",
            )
        )

        timeframe_video = record.get(
            "video",
            "",
        )

        task = xml_by_id.get(
            task_id
        )

        if task is None:

            review.append(
                {
                    "task_id":
                        task_id,

                    "timeframe_video":
                        timeframe_video,

                    "reason":
                        "task_not_found_in_xml",
                }
            )

            continue

        cvat_name = task.get(
            "name",
            "",
        )

        # ---------------------------------------------------------------------
        # CASE 1
        # Explicit TASK2 mapping
        # ---------------------------------------------------------------------

        physical_filename = (
            TASK2_MAPPING.get(
                cvat_name
            )
        )

        mapping_method = (
            "explicit_task2_mapping"
        )

        # ---------------------------------------------------------------------
        # CASE 2
        # Normal exact name
        # ---------------------------------------------------------------------

        if physical_filename is None:

            physical_filename = (
                cvat_name
                + ".mp4"
            )

            mapping_method = (
                "exact_task_name"
            )

        physical_key = normalize_name(
            Path(
                physical_filename
            ).stem
        )

        physical = physical_index.get(
            physical_key
        )

        # ---------------------------------------------------------------------
        # Physical video not found
        # ---------------------------------------------------------------------

        if physical is None:

            review.append(
                {
                    "task_id":
                        task_id,

                    "cvat_name":
                        cvat_name,

                    "timeframe_video":
                        timeframe_video,

                    "expected_physical_video":
                        physical_filename,

                    "reason":
                        "physical_video_not_found",
                }
            )

            print(
                f"REVIEW "
                f"task={task_id:<10} "
                f"{cvat_name:<32}"
                f" -> {physical_filename}"
            )

            continue

        # ---------------------------------------------------------------------
        # IMPORTANT:
        #
        # We intentionally DO NOT compare CVAT size with physical frame count.
        #
        # CVAT may contain an annotated/exported segment while the physical
        # video contains the complete recording.
        # ---------------------------------------------------------------------

        cvat_frames = task.get(
            "size"
        )

        physical_frames = physical[
            "frames"
        ]

        # ---------------------------------------------------------------------
        # Check requested timeframe frame range.
        # ---------------------------------------------------------------------

        max_requested_frame = None

        for key in [
            "total_frames",
            "stop_frame",
            "end_frame",
        ]:

            value = record.get(
                key
            )

            if isinstance(
                value,
                int,
            ):

                max_requested_frame = (
                    value - 1
                    if key == "total_frames"
                    else value
                )

                break

        range_warning = None

        if (
            max_requested_frame
            is not None
            and physical_frames
            is not None
            and max_requested_frame
            >= physical_frames
        ):

            range_warning = (
                "timeframe_exceeds_physical_video"
            )

        # ---------------------------------------------------------------------
        # ACCEPT
        # ---------------------------------------------------------------------

        result = {

            "task_id":
                task_id,

            "cvat_name":
                cvat_name,

            "cvat_video":
                timeframe_video,

            "physical_video":
                physical["file"],

            "physical_path":
                physical["path"],

            "cvat_frames":
                cvat_frames,

            "physical_frames":
                physical_frames,

            "fps":
                physical["fps"],

            "width":
                physical["width"],

            "height":
                physical["height"],

            "mapping_method":
                mapping_method,
        }

        if range_warning:

            result[
                "range_warning"
            ] = range_warning

            review.append(
                {
                    "task_id":
                        task_id,

                    "cvat_name":
                        cvat_name,

                    "physical_video":
                        physical["file"],

                    "cvat_frames":
                        cvat_frames,

                    "physical_frames":
                        physical_frames,

                    "reason":
                        range_warning,
                }
            )

            print(
                f"REVIEW "
                f"task={task_id:<10} "
                f"{cvat_name:<32} "
                f"-> {physical['file']} "
                f"[FRAME RANGE WARNING]"
            )

            continue

        accepted.append(
            result
        )

        print(
            f"ACCEPTED "
            f"task={task_id:<10} "
            f"{cvat_name:<32} "
            f"-> {physical['file']}"
        )

    return accepted, review


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 90)
    print(
        "EAGLE-MARS STRICT annotations2 "
        "PHYSICAL VIDEO MAPPER"
    )
    print("=" * 90)

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    if not XML_PATH.exists():

        raise FileNotFoundError(
            f"Missing XML:\n{XML_PATH}"
        )

    if not TIMEFRAME_JSON.exists():

        raise FileNotFoundError(
            f"Missing timeframe JSON:\n"
            f"{TIMEFRAME_JSON}"
        )

    if not VIDEO_DIR.exists():

        raise FileNotFoundError(
            f"Missing video directory:\n"
            f"{VIDEO_DIR}"
        )

    # -------------------------------------------------------------------------
    # Build indexes
    # -------------------------------------------------------------------------

    physical_index = (
        build_physical_video_index()
    )

    tasks = read_cvat_tasks()

    timeframe_records = (
        read_timeframes()
    )

    # -------------------------------------------------------------------------
    # Mapping
    # -------------------------------------------------------------------------

    accepted, review = (
        create_mapping(
            tasks,
            timeframe_records,
            physical_index,
        )
    )

    # -------------------------------------------------------------------------
    # Save accepted mappings
    # -------------------------------------------------------------------------

    mapping_output = {

        "project":
            "EAGLE-MARS",

        "mapping_method":
            "exact_names_plus_explicit_task2_mapping",

        "physical_video_directory":
            str(VIDEO_DIR),

        "total_tasks":
            len(timeframe_records),

        "accepted":
            len(accepted),

        "review":
            len(review),

        "mappings":
            accepted,
    }

    save_json(
        MAPPING_FILE,
        mapping_output,
    )

    # -------------------------------------------------------------------------
    # Save review
    # -------------------------------------------------------------------------

    review_output = {

        "project":
            "EAGLE-MARS",

        "total_tasks":
            len(timeframe_records),

        "accepted":
            len(accepted),

        "review":
            len(review),

        "items":
            review,
    }

    save_json(
        REVIEW_FILE,
        review_output,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "MAPPING COMPLETE"
    )
    print("=" * 90)

    print()

    print(
        f"annotations2 tasks : "
        f"{len(timeframe_records)}"
    )

    print(
        f"Accepted           : "
        f"{len(accepted)}"
    )

    print(
        f"Needs review       : "
        f"{len(review)}"
    )

    print()
    print(
        "OUTPUT"
    )

    print("-" * 90)

    print(
        "Accepted mappings:"
    )

    print(
        MAPPING_FILE
    )

    print()

    print(
        "Review file:"
    )

    print(
        REVIEW_FILE
    )

    print()

    # -------------------------------------------------------------------------
    # Mapping table
    # -------------------------------------------------------------------------

    if accepted:

        print(
            "FINAL MAPPINGS"
        )

        print("-" * 90)

        for item in accepted:

            print(
                f"{item['task_id']:<10} "
                f"{item['cvat_name']:<32} "
                f"-> "
                f"{item['physical_video']}"
            )

    print()

    if review:

        print(
            "WARNING:"
        )

        print(
            f"{len(review)} mapping(s) "
            f"require review."
        )

    else:

        print(
            "SUCCESS:"
        )

        print(
            "All 36 annotations2 tasks "
            "were mapped to physical videos."
        )

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()