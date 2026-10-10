"""Score an image or a video for violence.

    python cli.py photo.jpg
    python cli.py clip.mp4 --out annotated.mp4 --threshold 0.5
"""

import argparse
import os

import cv2
import numpy as np
from PIL import Image

from detector import ViolenceDetector

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def annotate(path, out, times, probs, threshold):
    """write a copy of the video with the smoothed score and a red border while it is above threshold"""
    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        p = float(np.interp(index / fps, times, probs))      # sampled scores, interpolated to every frame
        colour = (0, 0, 220) if p >= threshold else (0, 160, 0)
        if p >= threshold:
            cv2.rectangle(frame, (0, 0), (size[0] - 1, size[1] - 1), colour, 8)
        cv2.putText(frame, f"violence {p:.2f}", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2, cv2.LINE_AA)
        writer.write(frame)
        index += 1
    capture.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--every", type=float, default=0.5, help="seconds between sampled video frames")
    parser.add_argument("--window", type=int, default=5, help="moving-average window, in sampled frames")
    parser.add_argument("--out", help="write an annotated copy of the video here")
    args = parser.parse_args()

    detector = ViolenceDetector()
    if os.path.splitext(args.path)[1].lower() not in VIDEO_EXTENSIONS:
        p = detector.predict_image(Image.open(args.path).convert("RGB"))
        print(f"{'VIOLENT' if p >= args.threshold else 'non-violent'}  (p = {p:.3f})")
        return
    times, probs = detector.predict_video(args.path, args.every, args.window)
    flagged = times[probs >= args.threshold]
    print(f"clip score {probs.mean():.3f}, peak {probs.max():.3f}; "
          f"{len(flagged)}/{len(times)} sampled frames above {args.threshold}")
    if len(flagged):
        print(f"first flagged at {flagged[0]:.1f}s, last at {flagged[-1]:.1f}s")
    if args.out:
        annotate(args.path, args.out, times, probs, args.threshold)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
