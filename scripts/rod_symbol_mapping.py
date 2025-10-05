import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from collections import Counter

# Paths
ENHANCED_DIR = "enhanced"
CLUSTER_DIR = "symbol_clusters"
ANNOTATED_DIR = "annotated"
REPORT_FILE = "symbol_summary.txt"

os.makedirs(CLUSTER_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 80, 200)
    return edges

def extract_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if 10 < cv2.contourArea(c) < 1000]

def shape_signature(crop):
    resized = cv2.resize(crop, (32, 32))
    return resized.flatten() / 255.0

def load_symbols(folder):
    all_signatures = []
    for f in os.listdir(folder):
        if f.endswith((".jpg", ".png")):
            img = cv2.imread(os.path.join(folder, f))
            edges = preprocess(img)
            boxes = extract_contours(edges)
            for (x, y, w, h) in boxes:
                roi = edges[y:y+h, x:x+w]
                sig = shape_signature(roi)
                all_signatures.append(sig)
    return np.array(all_signatures)

print("🧩 Loading enhanced frames...")
symbols = load_symbols(ENHANCED_DIR)
print(f"🔹 {len(symbols)} symbol candidates detected")

if len(symbols) > 5:
    print("🔍 Clustering symbols...")
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(symbols)
    labels = kmeans.labels_

    counter = Counter(labels)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("Symbol Cluster Summary:\n\n")
        for cluster, count in counter.items():
            f.write(f"Cluster {cluster}: {count} instances\n")

    # Optional pattern naming (very basic heuristic)
    pattern_names = {
        0: "△ Triangle / Delta / Fire",
        1: "⚡ Lines / Frequency",
        2: "✡ Star / Intersect",
        3: "🌀 Spiral / Energy Flow",
        4: "◯ Circle / Magnetic Field"
    }

    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write("\nPossible Correlations:\n")
        for cluster, name in pattern_names.items():
            f.write(f"Cluster {cluster}: {name}\n")

    print(f"✅ Clustering complete. Report saved to '{REPORT_FILE}'.")
else:
    print("⚠️ Not enough symbol samples for clustering.")

print("🎨 Annotating frames...")
for f in os.listdir(ENHANCED_DIR):
    if f.endswith((".jpg", ".png")):
        path = os.path.join(ENHANCED_DIR, f)
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 200)
        boxes = extract_contours(edges)
        for (x, y, w, h) in boxes:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.imwrite(os.path.join(ANNOTATED_DIR, f), img)

print("📁 Annotated frames saved in 'annotated/' folder.")
print("📄 Symbol mapping complete!")
