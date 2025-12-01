import cv2
import numpy as np
from pathlib import Path

def night_score(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[..., 2]
    mean_v = float(V.mean())
    dark_ratio = float((V < 40).sum() / V.size)
    return mean_v, dark_ratio

img_path = Path("pytorch-CycleGAN-and-pix2pix/datasets/cyclegan_yolo_clear_d2n/trainB/images/0000f77c-6257be58.jpg")
if not img_path.exists():
    print(f"File not found: {img_path}")
    exit(1)

img = cv2.imread(str(img_path))
if img is None:
    print("Failed to load image")
    exit(1)

mean_v, dark_ratio = night_score(img)
print(f"Mean V: {mean_v:.2f} (Threshold < 55 is Night)")
print(f"Dark Ratio: {dark_ratio:.2f} (Threshold > 0.35 is Night)")

if mean_v < 55 and dark_ratio > 0.35:
    print("Result: NIGHT")
else:
    print("Result: DAY")
