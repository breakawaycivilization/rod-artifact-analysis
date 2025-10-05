import os, cv2, math, csv, shutil
import numpy as np
from PIL import Image

# --------- CONFIG ----------
ISOLATED_DIR = "isolated_symbols"          # from Stage 5
MATCH_DIR    = "decoded_matches"
SHEETS_DIR   = os.path.join(MATCH_DIR, "_contact_sheets")
CSV_FILE     = os.path.join(MATCH_DIR, "decoded_symbols.csv")

TOP_K_PER_CLASS = 150      # how many top examples to save per class
TEMPLATE_SIZE   = 64       # working size for templates & crops
MIN_SIZE        = 18       # ignore teeny crops (w or h < MIN_SIZE)
TM_THRESH       = 0.48     # template-match threshold (raise to be stricter)
HU_WEIGHT       = 0.35     # weight for Hu similarity in combined score

os.makedirs(MATCH_DIR, exist_ok=True)
os.makedirs(SHEETS_DIR, exist_ok=True)

# --------- helpers ----------
def to_square(img, size=128):
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    s = max(h, w)
    out = np.zeros((s, s, 3), dtype=np.uint8)
    y0 = (s - h) // 2
    x0 = (s - w) // 2
    out[y0:y0+h, x0:x0+w] = img
    return cv2.resize(out, (size, size))

def normalize_gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    g = cv2.resize(g, (TEMPLATE_SIZE, TEMPLATE_SIZE), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (3,3), 0)
    return g

def hu_moments(gray):
    # binary for moment stability
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 35, 5)
    m = cv2.moments(thr)
    hu = cv2.HuMoments(m).flatten()
    # log transform for scale
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu

def ncc(a, b):
    # normalized cross-correlation
    a = (a.astype(np.float32) - a.mean()) / (a.std() + 1e-6)
    b = (b.astype(np.float32) - b.mean()) / (b.std() + 1e-6)
    return float((a*b).mean())

def combine_scores(tm, hu_sim, hu_weight=HU_WEIGHT):
    return (1.0 - hu_weight)*tm + hu_weight*hu_sim

# --------- make procedural templates ----------
def template_triangle():
    img = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), np.uint8)
    pts = np.array([[TEMPLATE_SIZE*0.15, TEMPLATE_SIZE*0.85],
                    [TEMPLATE_SIZE*0.85, TEMPLATE_SIZE*0.85],
                    [TEMPLATE_SIZE*0.5,  TEMPLATE_SIZE*0.15]], dtype=np.int32)
    cv2.polylines(img, [pts], True, 255, 3)
    return img

def template_star6():
    img = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), np.uint8)
    # two triangles
    a = np.array([[16,44],[48,44],[32,12]], np.int32)
    b = np.array([[16,20],[48,20],[32,52]], np.int32)
    cv2.polylines(img,[a],True,255,3); cv2.polylines(img,[b],True,255,3)
    return img

def template_spiral():
    img = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), np.uint8)
    c = (TEMPLATE_SIZE//2, TEMPLATE_SIZE//2)
    r = TEMPLATE_SIZE*0.4
    theta = 0.0
    pts = []
    while r>4:
        x = int(c[0] + r*math.cos(theta))
        y = int(c[1] + r*math.sin(theta))
        pts.append((x,y))
        theta += 0.25
        r -= 0.6
    for i in range(len(pts)-1):
        cv2.line(img, pts[i], pts[i+1], 255, 3)
    return img

def template_parallel():
    img = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), np.uint8)
    for x in [18, 32, 46]:
        cv2.line(img, (x,14), (x,50), 255, 3)
    return img

def template_circle():
    img = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), np.uint8)
    cv2.circle(img, (TEMPLATE_SIZE//2, TEMPLATE_SIZE//2), 20, 255, 3)
    return img

TEMPLATES = {
    "triangle": template_triangle(),
    "star6":    template_star6(),
    "spiral":   template_spiral(),
    "lines":    template_parallel(),
    "circle":   template_circle(),
}

# precompute Hu for templates
T_HU = {k: hu_moments(v) for k,v in TEMPLATES.items()}

# --------- scan crops ----------
rows = []
kept = {k: [] for k in TEMPLATES.keys()}

crops = sorted([f for f in os.listdir(ISOLATED_DIR) if f.lower().endswith((".jpg",".png"))])
total = len(crops)
print(f"Scanning {total} crops from '{ISOLATED_DIR}' ...")

for i, fname in enumerate(crops, 1):
    path = os.path.join(ISOLATED_DIR, fname)
    img = cv2.imread(path)
    if img is None: 
        continue
    h, w = img.shape[:2]
    if h < MIN_SIZE or w < MIN_SIZE: 
        continue

    g = normalize_gray(to_square(img))
    hu = hu_moments(g)

    best_class, best_score, best_tm, best_hu = None, -1.0, 0.0, 0.0

    for cls, tmpl in TEMPLATES.items():
        tm = ncc(g, tmpl)  # [-1..1] higher is better
        # Hu similarity as negative L2 distance in log-space, normalized
        dist = np.linalg.norm(hu - T_HU[cls])
        hu_sim = 1.0 / (1.0 + dist)        # [0..1], higher is better
        score = combine_scores(tm, hu_sim)

        if score > best_score:
            best_score, best_class, best_tm, best_hu = score, cls, tm, hu_sim

    if best_score >= TM_THRESH:
        rows.append([fname, best_class, round(best_score,4), round(best_tm,4), round(best_hu,4), w, h])
        kept[best_class].append((best_score, fname))

print("Scoring complete. Writing CSV and copying exemplars...")

# CSV
os.makedirs(MATCH_DIR, exist_ok=True)
with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["file","class","score","tm_corr","hu_sim","width","height"])
    for r in sorted(rows, key=lambda x: (-x[2], x[1])):
        writer.writerow(r)

# Per-class folders with top-K exemplars
for cls, samples in kept.items():
    outdir = os.path.join(MATCH_DIR, cls)
    os.makedirs(outdir, exist_ok=True)
    for score, fname in sorted(samples, reverse=True)[:TOP_K_PER_CLASS]:
        src = os.path.join(ISOLATED_DIR, fname)
        dst = os.path.join(outdir, f"{score:.3f}__{fname}")
        if os.path.exists(src):
            shutil.copy2(src, dst)

# Contact sheets
def make_contact_sheet(folder, out_path, tile=96, cols=20, max_imgs=200):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith((".jpg",".png"))]
    files = sorted(files, key=lambda p: -float(os.path.basename(p).split("__")[0]))[:max_imgs]
    if not files: 
        return
    rows = math.ceil(len(files)/cols)
    sheet = Image.new("L", (cols*tile, rows*tile), color=0)
    for i, fp in enumerate(files):
        g = Image.open(fp).convert("L").resize((tile,tile))
        sheet.paste(g, ((i%cols)*tile, (i//cols)*tile))
    sheet.save(out_path)

for cls in TEMPLATES.keys():
    folder = os.path.join(MATCH_DIR, cls)
    out = os.path.join(SHEETS_DIR, f"{cls}_sheet.jpg")
    make_contact_sheet(folder, out)

print(f"✅ Done. CSV: {CSV_FILE}")
print(f"📁 Matches per class in: {MATCH_DIR}")
print(f"🖼️ Contact sheets in: {SHEETS_DIR}")
