import random
import sqlite3
from pathlib import Path

import cv2
import numpy as np
from decord import VideoReader, cpu


DB_PATH = Path("database/eagle_dataset.db")
OUTPUT_ROOT = Path("outputs/dataset_samples")

SAMPLES_PER_MACHINE = 5
FRAMES_PER_VIDEO = 5

MACHINE_IDS = [
    "24320003",
    "24320015",
    "24320021",
    "24400017",
    "24400019",
]


def get_machine_id(path):
    """
    Extract machine ID from the path.
    """

    parts = path.replace("/", "\\").split("\\")

    try:
        index = [
            part.lower()
            for part in parts
        ].index("unsorted")

        return parts[index + 1]

    except (ValueError, IndexError):
        return "UNKNOWN"


def make_contact_sheet(frames, title, output_path):
    """
    Create a horizontal contact sheet.
    """

    labeled = []

    for index, frame in enumerate(frames):

        image = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR,
        )

        height, width = image.shape[:2]

        cv2.putText(
            image,
            f"{index * 25}%",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        labeled.append(image)

    target_height = 360

    resized = []

    for image in labeled:

        scale = target_height / image.shape[0]

        new_width = int(
            image.shape[1] * scale
        )

        image = cv2.resize(
            image,
            (new_width, target_height),
        )

        resized.append(image)

    sheet = np.hstack(resized)

    title_bar = np.zeros(
        (60, sheet.shape[1], 3),
        dtype=np.uint8,
    )

    cv2.putText(
        title_bar,
        title,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    final = np.vstack(
        [title_bar, sheet]
    )

    cv2.imwrite(
        str(output_path),
        final,
    )


def main():

    print("=" * 80)
    print("EAGLE-MARS VISUAL DATASET SAMPLER")
    print("=" * 80)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DB_PATH
    )

    cursor = connection.cursor()

    for machine_id in MACHINE_IDS:

        print()
        print(
            f"Processing machine: {machine_id}"
        )

        cursor.execute(
            """
            SELECT full_path
            FROM videos
            WHERE readable = 1
            AND full_path LIKE ?
            """,
            (
                f"%\\{machine_id}\\%",
            ),
        )

        paths = [
            row[0]
            for row in cursor.fetchall()
        ]

        if not paths:

            print(
                "  No readable videos found."
            )

            continue

        random.seed(42)

        selected = random.sample(
            paths,
            min(
                SAMPLES_PER_MACHINE,
                len(paths),
            ),
        )

        machine_output = (
            OUTPUT_ROOT / machine_id
        )

        machine_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        for video_index, video_path in enumerate(
            selected,
            start=1,
        ):

            print(
                f"  [{video_index}/{len(selected)}] "
                f"{Path(video_path).parent.name}"
            )

            try:

                vr = VideoReader(
                    video_path,
                    ctx=cpu(0),
                )

                total_frames = len(vr)

                indices = np.linspace(
                    0,
                    total_frames - 1,
                    FRAMES_PER_VIDEO,
                    dtype=int,
                )

                frames = vr.get_batch(
                    indices
                ).asnumpy()

                output_path = (
                    machine_output
                    / f"sample_{video_index:02d}.jpg"
                )

                title = (
                    f"{machine_id} | "
                    f"Sample {video_index} | "
                    f"{Path(video_path).parent.name}"
                )

                make_contact_sheet(
                    frames,
                    title,
                    output_path,
                )

            except Exception as exc:

                print(
                    f"    ERROR: {exc}"
                )

    connection.close()

    print()
    print("=" * 80)
    print("SAMPLING COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Samples saved to: "
        f"{OUTPUT_ROOT.resolve()}"
    )


if __name__ == "__main__":
    main()