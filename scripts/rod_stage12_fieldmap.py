import cv2
import numpy as np
import os
from tqdm import tqdm

# --- Paths -------------------------------------------------------------
ANNOTATED_DIR = "annotated"
BRIGHTNESS_FILE = r"C:\Users\abuka\Downloads\material_scan\brightness_texture.png"
OUT_DIR = "field_output"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Helper functions --------------------------------------------------
def frame_files(path):
    exts = (".png", ".jpg", ".jpeg")
    return sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(exts)])

def load_brightness_map(path):
    if not os.path.exists(path):
        return None
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def normalize(img):
    img = img.astype(np.float32)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)

# --- Main analysis -----------------------------------------------------
def main():
    print("🧭 Loading annotated frames and brightness data...")
    brightness_map = load_brightness_map(BRIGHTNESS_FILE)
    if brightness_map is None:
        print("⚠️ Could not find brightness map.")
        return

    frames = frame_files(ANNOTATED_DIR)
    if not frames:
        print("⚠️ No annotated frames found.")
        return

    energies, variances = [], []
    print(f"📸 Found {len(frames)} frames. Processing...")

    for frame_path in tqdm(frames):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

        if 'symbol_mask' not in locals():
            symbol_mask = np.zeros_like(mask)

        # --- Align dimensions before combining ---
        if mask.shape != symbol_mask.shape:
            mask = cv2.resize(mask, (symbol_mask.shape[1], symbol_mask.shape[0]))

        symbol_mask = cv2.bitwise_or(symbol_mask, mask)

        # --- Ensure brightness map matches current frame ---
        if brightness_map.shape != gray.shape:
            brightness_resized = cv2.resize(brightness_map, (gray.shape[1], gray.shape[0]))
        else:
            brightness_resized = brightness_map

        roi = cv2.bitwise_and(gray, brightness_resized)
        energy = np.mean(roi)
        variance = np.var(roi)

        energies.append(energy)
        variances.append(variance)

    energies = np.array(energies)
    variances = np.array(variances)

    # --- Write report ---
    report = os.path.join(OUT_DIR, "coherence_report.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write("=== Field Coherence Report ===\n")
        f.write(f"Frames analyzed: {len(frames)}\n")
        f.write(f"Average energy: {np.mean(energies):.2f}\n")
        f.write(f"Energy variance: {np.var(energies):.4f}\n")
        f.write(f"Average brightness variance: {np.mean(variances):.2f}\n")

    # --- Ensure we always create the spectrogram ---
    if len(energies) == 0 or len(variances) == 0:
        print("⚠️ No data to plot. Creating blank spectrogram...")
        spec_img = np.zeros((200, 200, 3), dtype=np.uint8)
    else:
        # --- Align lengths ---
        min_len = min(len(energies), len(variances))
        energies = np.array(energies[:min_len]).reshape(-1, 1)
        variances = np.array(variances[:min_len]).reshape(-1, 1)

        # --- Generate spectrogram-like visualization ---
        spec_img = np.zeros((200, len(energies), 3), dtype=np.uint8)

        for i, val in enumerate(energies.flatten()):
            y = 199 - int(val) if val < 200 else 0
            cv2.line(spec_img, (i, 199), (i, y), (0, 255, 0), 1)

        for i, val in enumerate(variances.flatten()):
            y = 199 - int(val) if val < 200 else 0
            cv2.line(spec_img, (i, 199), (i, y), (255, 0, 0), 1)

    out_path = os.path.join(OUT_DIR, "field_spectrogram.png")
    cv2.imwrite(out_path, spec_img)
    print(f"✅ Field spectrogram saved to {out_path}")

if __name__ == "__main__":
    main()
