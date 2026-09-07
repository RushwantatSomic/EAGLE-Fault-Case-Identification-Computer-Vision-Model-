import os
import cv2
import torch
import numpy as np

from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
)


# ============================================================================
# EAGLE-MARS
# VISUAL VIDEOMAE INFERENCE
#
# CURRENT MODEL:
# final_training_fault_diversity_16f
#
# Architecture:
#   VideoMAE-small
#   16 frames
#   224 x 224
#   stride = 8
#
# Analytics panel:
#   SAME GRAPHICS AS ORIGINAL
#   MOVED TO THE BOTTOM OF THE SCREEN
# ============================================================================


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = (
    r"outputs\final_training_fault_diversity_16f\best_model"
)


# --------------------------------------------------------------------------
# VIDEO
# --------------------------------------------------------------------------

VIDEO_PATH = (
    r"J:\Datenaustausch\Verkauf\Barm\Eagle Somic Connect Daten Line 1 Maschine Bolton Mars Canada July 26\Eagle Somic Connect Bolton Data 28 July 26 from Lee\metteagle\unsorted\24320021\20250717_13-55-41\event.mp4"
)


# --------------------------------------------------------------------------
# OUTPUT
# --------------------------------------------------------------------------

OUTPUT_DIR = r"outputs\inference"

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "fault_diversity_16f_visual_prediction_bottom.mp4",
)


# ============================================================================
# VIDEOMAE CONFIGURATION
# ============================================================================

NUM_FRAMES = 16

# Analyze one window every 8 frames.
INFERENCE_STRIDE = 8

# Decision threshold.
#
# FAULT is displayed only when:
#
#     P(FAULT) >= 0.30
#
FAULT_THRESHOLD = 0.3

DISPLAY_WIDTH = 1100


LABELS = {
    0: "NORMAL",
    1: "FAULT",
}


# ============================================================================
# DEVICE
# ============================================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 90)
print("EAGLE-MARS VISUAL VIDEOMAE INFERENCE")
print("=" * 90)


print()
print("DEVICE")
print("-" * 90)

print(f"Device : {device}")


if torch.cuda.is_available():

    print(
        f"GPU    : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"VRAM   : "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )


# ============================================================================
# CHECK FILES
# ============================================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "\nModel not found:\n"
        f"{MODEL_PATH}\n\n"
        "Expected current best model:\n"
        "outputs\\final_training_fault_diversity_16f\\best_model"
    )


if not os.path.exists(VIDEO_PATH):

    raise FileNotFoundError(
        "\nTest video not found:\n"
        f"{VIDEO_PATH}\n\n"
        "Change VIDEO_PATH at the top of this script."
    )


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================================
# LOAD MODEL
# ============================================================================

print()
print("MODEL")
print("-" * 90)

print(
    f"Path       : {MODEL_PATH}"
)

print(
    "Experiment : FAULT-DIVERSITY + 16-FRAME"
)

print(
    f"Frames     : {NUM_FRAMES}"
)

print(
    f"Threshold  : {FAULT_THRESHOLD:.2f}"
)


print()
print("Loading processor...")

processor = VideoMAEImageProcessor.from_pretrained(
    MODEL_PATH
)


print("Loading VideoMAE model...")

model = VideoMAEForVideoClassification.from_pretrained(
    MODEL_PATH
)


model.to(device)

model.eval()


# ============================================================================
# OPEN VIDEO
# ============================================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)


if not cap.isOpened():

    raise RuntimeError(
        "\nCould not open video:\n"
        f"{VIDEO_PATH}"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)


if fps <= 0:

    fps = 30.0


total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


original_width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)


original_height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)


duration = (
    total_frames / fps
    if fps > 0
    else 0
)


print()
print("VIDEO")
print("-" * 90)

print(
    f"File      : {os.path.basename(VIDEO_PATH)}"
)

print(
    f"FPS       : {fps:.2f}"
)

print(
    f"Frames    : {total_frames}"
)

print(
    f"Resolution: "
    f"{original_width}x{original_height}"
)

print(
    f"Duration  : {duration:.2f}s"
)


# ============================================================================
# DISPLAY DIMENSIONS
# ============================================================================

if original_width > DISPLAY_WIDTH:

    scale = (
        DISPLAY_WIDTH
        / original_width
    )

    display_width = DISPLAY_WIDTH

    display_height = int(
        original_height * scale
    )

else:

    display_width = original_width

    display_height = original_height


# ============================================================================
# VIDEO WRITER
# ============================================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (
        display_width,
        display_height,
    ),
)


if not writer.isOpened():

    raise RuntimeError(
        "\nCould not create output video:\n"
        f"{OUTPUT_VIDEO}"
    )


print()
print("OUTPUT")
print("-" * 90)

print(
    f"Saved to : {OUTPUT_VIDEO}"
)


# ============================================================================
# INFERENCE
# ============================================================================

def run_inference(frames):

    rgb_frames = [
        cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )
        for frame in frames
    ]


    inputs = processor(
        images=rgb_frames,
        return_tensors="pt",
    )


    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }


    with torch.no_grad():

        outputs = model(
            **inputs
        )


        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )[0]


    normal_probability = float(
        probabilities[0].item()
    )


    fault_probability = float(
        probabilities[1].item()
    )


    # ------------------------------------------------------------
    # THRESHOLD DECISION
    # ------------------------------------------------------------

    if (
        fault_probability
        >= FAULT_THRESHOLD
    ):

        prediction = "FAULT"

        confidence = fault_probability

    else:

        prediction = "NORMAL"

        confidence = normal_probability


    return (
        prediction,
        confidence,
        normal_probability,
        fault_probability,
    )


# ============================================================================
# TEXT DRAWING
# ============================================================================

def draw_text(
    image,
    text,
    position,
    font_scale=0.7,
    thickness=2,
    color=(255, 255, 255),
):

    x, y = position


    # Black outline

    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )


    # Main text

    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# ============================================================================
# PROBABILITY BAR
# ============================================================================

def draw_probability_bar(
    image,
    x,
    y,
    width,
    label,
    probability,
):

    height = 28


    # Background

    cv2.rectangle(
        image,
        (x, y),
        (
            x + width,
            y + height,
        ),
        (50, 50, 50),
        -1,
    )


    filled = int(
        width * probability
    )


    if filled > 0:

        cv2.rectangle(
            image,
            (x, y),
            (
                x + filled,
                y + height,
            ),
            (80, 180, 80),
            -1,
        )


    draw_text(
        image,
        f"{label}: {probability:.3f}",
        (
            x + 8,
            y + 21,
        ),
        font_scale=0.55,
        thickness=1,
    )


# ============================================================================
# INITIAL STATE
# ============================================================================

frame_buffer = []

frame_index = 0

last_inference_frame = -1

current_prediction = "WAITING"

current_confidence = 0.0

current_normal_probability = 0.0

current_fault_probability = 0.0

current_window_start = 0

current_window_end = 0

window_count = 0

paused = False


# ============================================================================
# CONTROLS
# ============================================================================

print()
print("CONTROLS")
print("-" * 90)

print(
    "SPACE = Pause / Resume"
)

print(
    "R     = Restart"
)

print(
    "Q     = Quit"
)

print(
    "ESC   = Quit"
)


# ============================================================================
# DISPLAY WINDOW
# ============================================================================

window_name = (
    "EAGLE-MARS - "
    "Fault-Diversity VideoMAE 16F"
)


cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL,
)


# ============================================================================
# MAIN LOOP
# ============================================================================

while True:

    # ------------------------------------------------------------
    # READ FRAME
    # ------------------------------------------------------------

    ret, frame = cap.read()


    if not ret:

        break


    current_frame = frame_index

    frame_index += 1


    # ------------------------------------------------------------
    # FRAME BUFFER
    # ------------------------------------------------------------

    frame_buffer.append(
        frame.copy()
    )


    if len(frame_buffer) > NUM_FRAMES:

        frame_buffer.pop(0)


    # ------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------

    if (
        len(frame_buffer)
        == NUM_FRAMES
        and (
            last_inference_frame < 0
            or (
                current_frame
                - last_inference_frame
            )
            >= INFERENCE_STRIDE
        )
    ):

        (
            current_prediction,
            current_confidence,
            current_normal_probability,
            current_fault_probability,
        ) = run_inference(
            frame_buffer
        )


        current_window_start = (
            current_frame
            - NUM_FRAMES
            + 1
        )


        current_window_end = (
            current_frame
        )


        last_inference_frame = (
            current_frame
        )


        window_count += 1


    # ------------------------------------------------------------
    # RESIZE
    # ------------------------------------------------------------

    if original_width > DISPLAY_WIDTH:

        display = cv2.resize(
            frame,
            (
                display_width,
                display_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

    else:

        display = frame.copy()


    # ============================================================
    # BOTTOM ANALYTICS PANEL
    #
    # SAME GRAPHICS AS THE ORIGINAL
    # ONLY THE POSITION HAS BEEN CHANGED.
    # ============================================================

    panel_height = 220

    panel_top = max(
        0,
        display_height - panel_height,
    )


    overlay = display.copy()


    cv2.rectangle(
        overlay,
        (0, panel_top),
        (
            display_width,
            display_height,
        ),
        (0, 0, 0),
        -1,
    )


    display = cv2.addWeighted(
        overlay,
        0.72,
        display,
        0.28,
        0,
    )


    # ------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------

    if current_prediction == "FAULT":

        prediction_color = (
            0,
            0,
            255,
        )

    elif current_prediction == "NORMAL":

        prediction_color = (
            0,
            220,
            0,
        )

    else:

        prediction_color = (
            255,
            255,
            255,
        )


    draw_text(
        display,
        f"PREDICTION: {current_prediction}",
        (
            25,
            panel_top + 48,
        ),
        font_scale=1.05,
        thickness=3,
        color=prediction_color,
    )


    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------

    draw_text(
        display,
        "MODEL: VideoMAE-small | 16 FRAMES",
        (
            25,
            panel_top + 78,
        ),
        font_scale=0.55,
        thickness=2,
    )


    # ------------------------------------------------------------
    # THRESHOLD
    # ------------------------------------------------------------

    draw_text(
        display,
        (
            f"FAULT THRESHOLD: "
            f"{FAULT_THRESHOLD:.2f}"
        ),
        (
            25,
            panel_top + 105,
        ),
        font_scale=0.55,
        thickness=2,
    )


    # ------------------------------------------------------------
    # PROBABILITY BARS
    # ------------------------------------------------------------

    draw_probability_bar(
        display,
        25,
        panel_top + 120,
        300,
        "NORMAL",
        current_normal_probability,
    )


    draw_probability_bar(
        display,
        25,
        panel_top + 155,
        300,
        "FAULT",
        current_fault_probability,
    )


    # ------------------------------------------------------------
    # MODEL WINDOW
    # ------------------------------------------------------------

    draw_text(
        display,
        (
            f"Window: "
            f"{current_window_start}-"
            f"{current_window_end}"
        ),
        (
            350,
            panel_top + 125,
        ),
        font_scale=0.58,
        thickness=2,
    )


    # ------------------------------------------------------------
    # TIME
    # ------------------------------------------------------------

    current_time = (
        current_frame / fps
    )


    draw_text(
        display,
        (
            f"Time: "
            f"{current_time:.2f}s / "
            f"{duration:.2f}s"
        ),
        (
            350,
            panel_top + 155,
        ),
        font_scale=0.58,
        thickness=2,
    )


    # ------------------------------------------------------------
    # CONFIDENCE
    # ------------------------------------------------------------

    draw_text(
        display,
        (
            f"Confidence: "
            f"{current_confidence:.3f}"
        ),
        (
            350,
            panel_top + 185,
        ),
        font_scale=0.58,
        thickness=2,
    )


    # ------------------------------------------------------------
    # INFERENCE COUNT
    # ------------------------------------------------------------

    draw_text(
        display,
        (
            f"Windows tested: "
            f"{window_count}"
        ),
        (
            600,
            panel_top + 125,
        ),
        font_scale=0.58,
        thickness=2,
    )


    # ------------------------------------------------------------
    # STRIDE
    # ------------------------------------------------------------

    draw_text(
        display,
        (
            f"Stride: "
            f"{INFERENCE_STRIDE} frames"
        ),
        (
            600,
            panel_top + 155,
        ),
        font_scale=0.58,
        thickness=2,
    )


    # ------------------------------------------------------------
    # WRITE OUTPUT
    # ------------------------------------------------------------

    writer.write(
        display
    )


    # ------------------------------------------------------------
    # SHOW
    # ------------------------------------------------------------

    cv2.imshow(
        window_name,
        display,
    )


    # ------------------------------------------------------------
    # KEYBOARD
    # ------------------------------------------------------------

    key = (
        cv2.waitKey(1)
        & 0xFF
    )


    # ------------------------------------------------------------
    # QUIT
    # ------------------------------------------------------------

    if (
        key == ord("q")
        or key == 27
    ):

        break


    # ------------------------------------------------------------
    # PAUSE
    # ------------------------------------------------------------

    elif key == ord(" "):

        paused = True


        while paused:

            pause_key = (
                cv2.waitKey(30)
                & 0xFF
            )


            # ----------------------------------------------------
            # RESUME
            # ----------------------------------------------------

            if pause_key == ord(" "):

                paused = False


            # ----------------------------------------------------
            # QUIT
            # ----------------------------------------------------

            elif (
                pause_key == ord("q")
                or pause_key == 27
            ):

                paused = False

                frame_index = (
                    total_frames
                )

                break


            # ----------------------------------------------------
            # RESTART
            # ----------------------------------------------------

            elif pause_key == ord("r"):

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    0,
                )


                frame_buffer = []

                frame_index = 0

                last_inference_frame = -1

                current_prediction = (
                    "WAITING"
                )

                current_confidence = 0.0

                current_normal_probability = (
                    0.0
                )

                current_fault_probability = (
                    0.0
                )

                current_window_start = 0

                current_window_end = 0

                window_count = 0

                paused = False

                break


            cv2.imshow(
                window_name,
                display,
            )


    # ------------------------------------------------------------
    # RESTART
    # ------------------------------------------------------------

    elif key == ord("r"):

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            0,
        )


        frame_buffer = []

        frame_index = 0

        last_inference_frame = -1

        current_prediction = (
            "WAITING"
        )

        current_confidence = 0.0

        current_normal_probability = (
            0.0
        )

        current_fault_probability = (
            0.0
        )

        current_window_start = 0

        current_window_end = 0

        window_count = 0


# ============================================================================
# CLEANUP
# ============================================================================

cap.release()

writer.release()

cv2.destroyAllWindows()


# ============================================================================
# FINAL
# ============================================================================

print()
print("=" * 90)
print("VISUAL INFERENCE COMPLETE")
print("=" * 90)

print(
    f"Model        : {MODEL_PATH}"
)

print(
    f"Frames/window: {NUM_FRAMES}"
)

print(
    f"Stride       : {INFERENCE_STRIDE}"
)

print(
    f"Threshold    : {FAULT_THRESHOLD:.2f}"
)

print(
    f"Windows tested: {window_count}"
)

print(
    f"Output video : {OUTPUT_VIDEO}"
)

print("=" * 90)