import cv2
from pathlib import Path

VIDEO = Path("data/annotated_videos/Cartoning 2.mp4")
OUTPUT = Path("outputs/timing_check")
OUTPUT.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open: {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("=" * 80)
print("TIMING CHECK")
print("=" * 80)
print(f"Video  : {VIDEO}")
print(f"FPS    : {fps}")
print(f"Frames : {total}")
print(f"Duration: {total / fps:.2f}s")
print()

for i in range(11):
    frame_no = round((total - 1) * i / 10)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()

    if not ok:
        print(f"ERROR reading frame {frame_no}")
        continue

    filename = OUTPUT / f"frame_{i:02d}_{frame_no}.jpg"
    cv2.imwrite(str(filename), frame)

    print(f"{i:02d} -> frame {frame_no:4d} -> {frame_no / fps:6.2f}s")

cap.release()

print()
print(f"Frames saved to: {OUTPUT}")