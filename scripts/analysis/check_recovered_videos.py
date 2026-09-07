import cv2
from pathlib import Path

VIDEO_DIR = Path("data/annotated_videos")

videos = sorted(VIDEO_DIR.glob("*.mp4"))

print("=" * 90)
print("EAGLE-MARS RECOVERED VIDEO CHECK")
print("=" * 90)

print(f"Videos found: {len(videos)}")

for i, path in enumerate(videos, 1):

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        print(f"{i:02d} ERROR: {path.name}")
        continue

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frames / fps if fps else 0

    cap.release()

    print(
        f"{i:02d} "
        f"{path.name:<20} "
        f"{frames:5d} frames "
        f"{fps:7.2f} FPS "
        f"{width}x{height} "
        f"{duration:6.2f}s"
    )

print("=" * 90)