import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from PIL import Image

# Paths
ANNOTATED_DIR = "annotated"
REFINED_DIR = "refined"
COLLAGE_FILE = "symbol_collage.jpg"
REPORT_FILE = "symbol_refined_summary.txt"

os.makedirs(REFINED_DIR, exist_ok=True)

def isolate_reflective_regions(img):
    """Detect reflective/metallic areas and mask the rest."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # High brightness + low saturation = reflective
    lower = np.array([0, 0, 180])
    upper = np.array([180, 80, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_and(img, img, mask=mask)

def extract_symbol_crops(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 70, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crops = []
    for c in contours:
        area = cv2.contourArea(c)
        if 200 < area < 2500:
            x, y, w, h = cv2.boundingRect(c)
            crop = gray[y:y+h, x:x+w]
            if crop.size > 0:
                crops.append(cv2.resize(crop, (32, 32)).flatten())
    return crops

print("🧩 Scanning annotated frames...")
samples = []
for file in os.listdir(ANNOTATED_DIR):
    if not file.lower().endswith(".jpg"):
        continue
    img = cv2.imread(os.path.join(ANNOTATED_DIR, file))
    masked = isolate_reflective_regions(img)
    cv2.imwrite(os.path.join(REFINED_DIR, file), masked)
    samples.extend(extract_symbol_crops(masked))

print(f"🔹 {len(samples)} cropped symbol samples extracted")

if len(samples) < 10:
    print("⚠️ Not enough samples for reclustering.")
    exit()

print("🔍 Reclustering symbol crops...")
X = np.array(samples)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X)
labels = kmeans.labels_
counter = Counter(labels)

# Write report
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("Refined Symbol Cluster Summary\n\n")
    for k, v in counter.items():
        f.write(f"Cluster {k}: {v} instances\n")

# Generate collage (top 10 samples per cluster)
print("🎨 Generating collage...")
collage_rows = []
for cluster in range(5):
    idxs = np.where(labels == cluster)[0][:10]
    tiles = []
    for i in idxs:
        tile = (X[i].reshape(32, 32) * 255).astype(np.uint8)
        tiles.append(tile)
    if tiles:
        row = np.hstack(tiles)
        collage_rows.append(row)
if collage_rows:
    collage = np.vstack(collage_rows)
    cv2.imwrite(COLLAGE_FILE, collage)
    print(f"🖼️ Collage saved as {COLLAGE_FILE}")

print("📄 Refined analysis complete. Report + collage ready.")
