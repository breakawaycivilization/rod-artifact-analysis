import cv2, os, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm

# Input & output
FRAME_DIR = r"frames"          # or "enhanced" if you prefer
OUT_DIR   = r"material_scan"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR,"heatmaps"), exist_ok=True)

def analyze_frame(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    stddev = np.std(gray)
    hist = cv2.calcHist([img],[0,1,2],None,[16,16,16],[0,256,0,256,0,256])
    hist = cv2.normalize(hist,hist).flatten()
    return brightness, stddev, hist

def make_heatmap(img, name):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.absolute(lap)
    lap = cv2.normalize(lap,None,0,255,cv2.NORM_MINMAX)
    lap = lap.astype(np.uint8)
    heat = cv2.applyColorMap(lap, cv2.COLORMAP_PLASMA)
    cv2.imwrite(os.path.join(OUT_DIR,"heatmaps",name),heat)
    return np.mean(lap)

frames = sorted([f for f in os.listdir(FRAME_DIR) if f.lower().endswith(('.jpg','.png'))])
print(f"Scanning {len(frames)} frames…")

brightness_series, texture_series = [], []
for f in tqdm(frames):
    path = os.path.join(FRAME_DIR,f)
    img = cv2.imread(path)
    if img is None: continue
    b,std,hist = analyze_frame(img)
    t = make_heatmap(img,f)
    brightness_series.append(b)
    texture_series.append(t)

# Plot brightness over time
plt.figure(figsize=(10,4))
plt.plot(brightness_series,label="Mean brightness")
plt.plot(texture_series,label="Surface texture (Laplacian mean)")
plt.legend(); plt.xlabel("Frame"); plt.ylabel("Value")
plt.title("Brightness & Surface Energy Profile")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"brightness_texture.png"),dpi=160); plt.close()

# Detect anomalies
b = np.array(brightness_series)
t = np.array(texture_series)
b_dev = (b - b.mean())/b.std()
t_dev = (t - t.mean())/t.std()
spikes = np.where((np.abs(b_dev)>3)|(np.abs(t_dev)>3))[0]

# Write report
with open(os.path.join(OUT_DIR,"material_report.txt"),"w",encoding="utf-8") as f:
    f.write("🧪 Material & Spectral Signature Scan\n\n")
    f.write(f"Frames analyzed: {len(frames)}\n")
    f.write(f"Brightness mean ±σ: {b.mean():.2f} ± {b.std():.2f}\n")
    f.write(f"Texture mean ±σ: {t.mean():.2f} ± {t.std():.2f}\n")
    f.write(f"Detected {len(spikes)} anomalous frames (>3σ):\n")
    for s in spikes:
        f.write(f"  Frame {s}: Brightness z={b_dev[s]:.2f}, Texture z={t_dev[s]:.2f}\n")
print("✅ Material scan complete.")
