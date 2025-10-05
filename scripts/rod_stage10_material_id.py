import os, cv2, numpy as np, matplotlib.pyplot as plt
from collections import defaultdict
from scipy.fft import rfft, rfftfreq

# ---------- CONFIG ----------
FRAME_DIR = r"frames"              # or "enhanced"
OUT_DIR   = r"material_id"
ROI_MODE  = "auto"                 # "auto" or "manual"
# If manual, set a central ROI (x,y,w,h):
MANUAL_ROI = (0,0,0,0)

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- helpers ----------
def pick_roi(img):
    """Pick a bright, specular-ish ROI automatically."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # brightest 2% mask
    thresh = np.percentile(gray, 98)
    mask = (gray >= thresh).astype(np.uint8)
    # bounding rect of the largest bright blob
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        # fallback to center patch
        cx, cy = w//2, h//2
        return (max(cx-40,0), max(cy-40,0), 80, 80)
    c = max(cnts, key=cv2.contourArea)
    x,y,w2,h2 = cv2.boundingRect(c)
    # pad a little
    pad = 8
    x = max(x-pad,0); y = max(y-pad,0)
    w2 = min(w2+2*pad, w-x); h2 = min(h2+2*pad, h-y)
    return (x,y,w2,h2)

def rgb_stats(roi_bgr):
    b,g,r = cv2.split(roi_bgr)
    # means (linear-ish proxy)
    mB, mG, mR = float(np.mean(b)), float(np.mean(g)), float(np.mean(r))
    # highlight sharpness = std dev of gray (lower std in bright region => tighter specular)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    sharp = 255.0 - float(np.std(gray))
    return (mR, mG, mB), sharp

def uv_chromaticity(roi_bgr):
    # convert to XYZ then u'v' (CIE 1976)
    roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    # sRGB to XYZ (D65)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    XYZ = np.tensordot(roi, M.T, axes=1)
    X = XYZ[...,0].mean(); Y = XYZ[...,1].mean(); Z = XYZ[...,2].mean()
    denom = X + 15*Y + 3*Z + 1e-9
    u_prime = (4*X)/denom
    v_prime = (9*Y)/denom
    return float(u_prime), float(v_prime)

def classify_material(avg_rgb, sharpness, uv):
    """Very rough heuristics based on literature & lab practice."""
    R,G,B = avg_rgb
    # channel ratios (larger -> warmer tint)
    rg = (R+1e-6)/(G+1e-6)
    rb = (R+1e-6)/(B+1e-6)
    gb = (G+1e-6)/(B+1e-6)
    u,v = uv

    # baseline guesses
    score = defaultdict(float)

    # Polished Aluminum: neutral/blueish highlight, high specular tightness
    if rb < 1.05 and gb < 1.05: score["polished_aluminum"] += 1.0
    if sharpness > 230: score["polished_aluminum"] += 0.8
    if u<0.22 and v<0.54: score["polished_aluminum"] += 0.4

    # Stainless/NiCr: slightly warm-neutral, broadish specular, high reflectance
    if 1.00<=rb<=1.10 and 1.00<=rg<=1.08: score["stainless_or_nickel_chrome"] += 0.9
    if 210<sharpness<235: score["stainless_or_nickel_chrome"] += 0.6

    # Titanium/oxide film: subtle warm tint; interference can bias R>G>=B
    if rb>1.05 and rg>1.02: score["titanium_with_oxide"] += 1.0
    if 200<sharpness<235: score["titanium_with_oxide"] += 0.4

    # Anodized aluminum (clear/very light): neutral body, slightly reduced specular
    if rb<1.06 and rg<1.06 and sharpness<230: score["anodized_aluminum_clear"] += 0.7

    # Painted/coated polymer (gloss): warm/cool bias, lower specular tightness
    if sharpness<205: score["gloss_coated_polymer"] += 0.8

    # Normalize to 0..1
    total = sum(score.values()) + 1e-9
    for k in list(score.keys()):
        score[k] = score[k]/total

    # Return as sorted list
    ranked = sorted(score.items(), key=lambda x: -x[1])
    return ranked[:3]  # top 3

# ---------- main ----------
frames = sorted([f for f in os.listdir(FRAME_DIR) if f.lower().endswith((".jpg",".png"))])
if not frames:
    print("No frames found. Set FRAME_DIR correctly."); raise SystemExit

series = []
roi_series = []
fft_series = []

for idx, f in enumerate(frames):
    img = cv2.imread(os.path.join(FRAME_DIR,f))
    if img is None: continue

    if ROI_MODE=="auto":
        x,y,w,h = pick_roi(img)
    else:
        x,y,w,h = MANUAL_ROI
        if w*h==0:
            h0,w0 = img.shape[:2]
            x,y,w,h = (w0//2-60, h0//2-60, 120, 120)

    roi = img[y:y+h, x:x+w]
    if roi.size==0:
        continue

    (mR,mG,mB), sharp = rgb_stats(roi)
    uv = uv_chromaticity(roi)
    series.append({
        "frame": idx, "file": f, "mR": mR, "mG": mG, "mB": mB,
        "sharp": sharp, "u'": uv[0], "v'": uv[1]
    })
    roi_series.append(np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)))

# FFT on ROI brightness to check for modulation beyond camera/exposure drift
y = np.array(roi_series, dtype=np.float32)
y = y - y.mean()
Y = np.abs(rfft(y))
freqs = rfftfreq(len(y), d=1.0/30.0)  # assume ~30 fps; change if needed

# Aggregate material guess across frames
from collections import Counter
votes = Counter()
for row in series:
    ranked = classify_material((row["mR"],row["mG"],row["mB"]), row["sharp"], (row["u'"],row["v'"]))
    for k,weight in ranked:
        votes[k] += weight

top_ranked = votes.most_common(5)

# Save charts
import pandas as pd
df = pd.DataFrame(series)
df.to_csv(os.path.join(OUT_DIR,"material_timeseries.csv"), index=False)

plt.figure(figsize=(10,4))
plt.plot(df["frame"], df["mR"], label="R"); plt.plot(df["frame"], df["mG"], label="G"); plt.plot(df["frame"], df["mB"], label="B")
plt.title("ROI channel means over time"); plt.xlabel("Frame"); plt.ylabel("Mean (0-255)")
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"rgb_over_time.png"), dpi=160); plt.close()

plt.figure(figsize=(10,4))
plt.plot(df["frame"], df["sharp"])
plt.title("Specular sharpness proxy (higher = tighter highlight)"); plt.xlabel("Frame"); plt.ylabel("Sharpness")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"specular_sharpness.png"), dpi=160); plt.close()

plt.figure(figsize=(6,6))
plt.scatter(df["u'"], df["v'"], s=10, alpha=0.6)
plt.title("u'v' chromaticity of ROI highlight/body")
plt.xlabel("u'"); plt.ylabel("v'")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"uv_scatter.png"), dpi=160); plt.close()

plt.figure(figsize=(10,4))
plt.semilogy(freqs, Y + 1e-6)
plt.title("FFT of ROI brightness (check for non-mains modulation)")
plt.xlabel("Frequency (Hz)"); plt.ylabel("|Amplitude| (log)")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"roi_fft.png"), dpi=160); plt.close()

# Write summary
with open(os.path.join(OUT_DIR,"material_summary.txt"), "w", encoding="utf-8") as f:
    f.write("Material/Coating Inference (heuristic)\n")
    f.write("======================================\n\n")
    f.write("Top votes:\n")
    for k,v in top_ranked:
        f.write(f"  - {k}: {v:.2f}\n")
    f.write("\nNotes:\n")
    f.write("- Results are heuristic; confirm with XRF if possible.\n")
    f.write("- Specular sharpness trends + u'v' drift help distinguish polished Al vs NiCr vs Ti-oxide.\n")
    f.write("- FFT peaks around 50/60 Hz suggest mains flicker; other stable peaks may indicate internal modulation.\n")

print("✅ Stage 10 complete. See the 'material_id' folder for results.")
