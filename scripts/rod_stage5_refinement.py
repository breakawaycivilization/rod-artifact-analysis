import cv2
import numpy as np
import os
from collections import defaultdict

# =============================
# CONFIG
# =============================
ANNOTATED_DIR = "annotated"
OUTPUT_DIR = "isolated_symbols"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper functions
def preprocess_frame(frame):
    """Reduce reflections and isolate engravings"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # Adaptive threshold for reflective surfaces
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 5
    )
    # Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    return clean

def find_persistent_symbols(prev_symbols, new_symbols, tolerance=10):
    """Track shapes that persist across frames"""
    persistent = []
    for (x1,y1,w1,h1) in new_symbols:
        for (x2,y2,w2,h2) in prev_symbols:
            if abs(x1-x2)<tolerance and abs(y1-y2)<tolerance:
                persistent.append((x1,y1,w1,h1))
                break
    return persistent

# =============================
# MAIN ANALYSIS
# =============================
symbol_counts = defaultdict(int)
prev_contours = []

for idx, fname in enumerate(sorted(os.listdir(ANNOTATED_DIR))):
    if not fname.endswith(".jpg"):
        continue

    path = os.path.join(ANNOTATED_DIR, fname)
    frame = cv2.imread(path)
    processed = preprocess_frame(frame)

    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    symbols = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 10 < w < 200 and 10 < h < 200:  # filter noise and large regions
            symbols.append((x, y, w, h))

    persistent = find_persistent_symbols(prev_contours, symbols)
    prev_contours = symbols

    for (x,y,w,h) in persistent:
        crop = frame[y:y+h, x:x+w]
        name = f"symbol_{idx:04d}_{x}_{y}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, name), crop)
        symbol_counts[name] += 1

# =============================
# REPORT
# =============================
report_path = os.path.join(OUTPUT_DIR, "isolated_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("🔍 Refined Symbol Detection Report\n")
    f.write("="*40 + "\n\n")
    f.write(f"Total frames analyzed: {len(os.listdir(ANNOTATED_DIR))}\n")
    f.write(f"Total isolated symbol crops: {len(symbol_counts)}\n\n")

    f.write("Top persistent symbol regions:\n")
    for name, count in sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True):
        f.write(f"{name}: {count}\n")

print("✅ Stage 5 complete.")
print(f"📁 Isolated symbols saved in '{OUTPUT_DIR}/'")
print(f"📄 Report saved as '{report_path}'")
