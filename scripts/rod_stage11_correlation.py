import os, re, math, json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.optimize import curve_fit
from scipy.fft import rfft, rfftfreq

# ------------------ CONFIG ------------------
FRAME_DIR   = "frames"   # or "enhanced"
ISOLATEDDIR = "isolated_symbols"
DECODEDCSV  = "decoded_matches/decoded_symbols.csv"

OUTDIR      = "stage11_out"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(os.path.join(OUTDIR, "stage11_heat_overlays"), exist_ok=True)

# windows (px)
LOCAL_R = 20            # half-size for local energy window around symbol
SMOOTH_K = (5,5)        # smoothing before Laplacian

# FFT settings
FPS = 30.0              # adjust if your video FPS differs

# -------------- helpers ---------------------
def lap_energy(gray):
    g = cv2.GaussianBlur(gray, SMOOTH_K, 0)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    lap = np.abs(lap)
    return lap

def exp_decay(t, A, tau, C):
    return A*np.exp(-t/tau) + C

def parse_symbol_name(fname):
    # symbol_0123_456_789.jpg
    m = re.search(r"symbol_(\d+)_(\d+)_(\d+)\.(?:jpg|png)$", fname)
    if not m: return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))

# -------------- load decoded ---------------
if not os.path.exists(DECODEDCSV):
    raise SystemExit("decoded_symbols.csv not found. Run Stage 6 first.")

decoded = pd.read_csv(DECODEDCSV)  # file,class,score,tm_corr,hu_sim,width,height
decoded["parsed"] = decoded["file"].apply(parse_symbol_name)
decoded = decoded[decoded["parsed"].notnull()].copy()
decoded[["frame","x","y"]] = pd.DataFrame(decoded["parsed"].tolist(), index=decoded.index)
decoded.drop(columns=["parsed"], inplace=True)

# -------------- scan frames, build energy series ---------------
frames = sorted([f for f in os.listdir(FRAME_DIR) if f.lower().endswith((".jpg",".png"))])
if not frames:
    raise SystemExit("No frames found in FRAME_DIR.")

glob_brightness = []
glob_texture = []
fft_roi_series = []  # brightness of a fixed ROI to track modulation
roi_xywh = None

# We also store per-frame energy maps for local sampling
frame_gray = {}
frame_lap  = {}

for idx, fn in enumerate(frames):
    path = os.path.join(FRAME_DIR, fn)
    img = cv2.imread(path)
    if img is None: continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = lap_energy(gray)

    frame_gray[idx] = gray
    frame_lap[idx]  = lap

    glob_brightness.append(float(gray.mean()))
    glob_texture.append(float(lap.mean()))

    # pick a fixed ROI on first frame: brightest 2% blob bounding box
    if roi_xywh is None:
        thr = np.percentile(gray, 98)
        mask = (gray >= thr).astype(np.uint8)
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            pad=8; x=max(x-pad,0); y=max(y-pad,0)
            w=min(w+2*pad, img.shape[1]-x); h=min(h+2*pad, img.shape[0]-y)
            roi_xywh = (x,y,w,h)
        else:
            h0,w0 = gray.shape; roi_xywh = (w0//2-40, h0//2-40, 80, 80)

    x,y,w,h = roi_xywh
    roi = gray[y:y+h, x:x+w]
    fft_roi_series.append(float(roi.mean()))

# FFT of ROI series
y = np.array(fft_roi_series, dtype=np.float32)
y = y - y.mean()
Y = np.abs(rfft(y))
freqs = rfftfreq(len(y), d=1.0/FPS)

# try decay fit on global brightness (proxy for “powering down”)
t = np.arange(len(glob_brightness), dtype=np.float32)
gb = np.array(glob_brightness)
try:
    p0 = (gb.max()-gb.min(), len(gb)/4.0, gb.min())
    popt, pcov = curve_fit(exp_decay, t, gb, p0=p0, maxfev=10000)
    decay_params = {"A": float(popt[0]), "tau_frames": float(popt[1]), "C": float(popt[2])}
except Exception as e:
    decay_params = {"A": None, "tau_frames": None, "C": None}

# -------------- per-symbol local energy/z-scores -----------------
rows = []
for i, r in decoded.iterrows():
    f = int(r["frame"])
    if f not in frame_gray: 
        continue
    gray = frame_gray[f]; lap = frame_lap[f]
    H,W = gray.shape
    cx, cy = int(r["x"]), int(r["y"])
    x0 = max(cx-LOCAL_R, 0); x1 = min(cx+LOCAL_R, W-1)
    y0 = max(cy-LOCAL_R, 0); y1 = min(cy+LOCAL_R, H-1)
    patch_g = gray[y0:y1, x0:x1]
    patch_l = lap[y0:y1, x0:x1]

    if patch_g.size == 0 or patch_l.size == 0:
        continue

    loc_b = float(patch_g.mean())
    loc_t = float(patch_l.mean())

    # z vs frame baseline
    z_b = (loc_b - gray.mean()) / (gray.std()+1e-9)
    z_t = (loc_t - lap.mean())  / (lap.std()+1e-9)

    rows.append({
        "file": r["file"],
        "class": r["class"],
        "score": float(r["score"]),
        "frame": f,
        "x": cx, "y": cy,
        "local_brightness": loc_b,
        "local_texture": loc_t,
        "z_local_brightness": z_b,
        "z_local_texture": z_t
    })

sym = pd.DataFrame(rows)
sym.to_csv(os.path.join(OUTDIR, "stage11_correlations.csv"), index=False)

# -------------- stats per class -----------------
summary_lines = []
summary_lines.append("Stage 11 — Symbol ↔ Energy Correlation\n")
summary_lines.append(f"Frames analyzed: {len(frames)}")
summary_lines.append(f"Symbols analyzed: {len(sym)}")
summary_lines.append(f"Global brightness mean ±σ: {np.mean(glob_brightness):.2f} ± {np.std(glob_brightness):.2f}")
summary_lines.append(f"Global texture mean ±σ:    {np.mean(glob_texture):.2f} ± {np.std(glob_texture):.2f}")
summary_lines.append("")

if len(sym):
    for cls, g in sym.groupby("class"):
        zt = g["z_local_texture"].values
        zb = g["z_local_brightness"].values
        # compare to zero-mean null (are they significantly > 0?)
        t_t, p_t = ttest_ind(zt, np.zeros_like(zt), equal_var=False)
        t_b, p_b = ttest_ind(zb, np.zeros_like(zb), equal_var=False)
        summary_lines.append(f"[{cls}] n={len(g)}  ⟨z_text⟩={zt.mean():.2f}, p={p_t:.3g}  |  ⟨z_bright⟩={zb.mean():.2f}, p={p_b:.3g}")

# decay notes
summary_lines.append("")
summary_lines.append("Decay fit (global brightness vs. frame):")
summary_lines.append(f"  A={decay_params['A']}, tau_frames={decay_params['tau_frames']}, C={decay_params['C']}")

# FFT notes
summary_lines.append("")
peaks = []
if len(freqs) > 5:
    # crude peak pick: top 5 non-zero bins
    idxs = np.argsort(Y[1:])[-5:]+1
    peaks = [(float(freqs[i]), float(Y[i])) for i in sorted(idxs, key=lambda k: -Y[k])]
summary_lines.append("Top FFT peaks (Hz, amplitude):")
for f0, amp in peaks:
    summary_lines.append(f"  {f0:.2f} Hz  |  {amp:.2f}")

with open(os.path.join(OUTDIR, "stage11_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

# -------------- overlays & plots ---------------
# overlays on a few representative frames
for f in sorted(set(sym["frame"]))[:40]:
    path = os.path.join(FRAME_DIR, frames[f])
    img = cv2.imread(path); gray = frame_gray[f]; lap = frame_lap[f]
    # make heatmap from normalized laplacian
    lapn = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(lapn, cv2.COLORMAP_INFERNO)
    mix  = cv2.addWeighted(img, 0.6, heat, 0.4, 0)

    for _, r in sym[sym["frame"]==f].iterrows():
        x,y = int(r["x"]), int(r["y"])
        cv2.rectangle(mix, (x-LOCAL_R,y-LOCAL_R), (x+LOCAL_R,y+LOCAL_R), (0,255,0), 2)
        cv2.putText(mix, r["class"][:6], (x+4,y-LOCAL_R-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imwrite(os.path.join(OUTDIR, "stage11_heat_overlays", f"overlay_{f:04d}.jpg"), mix)

# global decay plot
plt.figure(figsize=(10,4))
plt.plot(glob_brightness, label="Global brightness")
if decay_params["tau_frames"] is not None:
    tfit = exp_decay(t, *list(decay_params.values()))
    plt.plot(tfit, label=f"Exp fit τ={decay_params['tau_frames']:.1f} frames")
plt.legend(); plt.title("Global energy (brightness) & decay fit")
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "stage11_decay_curve.png"), dpi=160); plt.close()

print("✅ Stage 11 complete.")
print("  - CSV: stage11_out/stage11_correlations.csv")
print("  - Summary: stage11_out/stage11_summary.txt")
print("  - Overlays: stage11_out/stage11_heat_overlays/")
print("  - Decay plot: stage11_out/stage11_decay_curve.png")
