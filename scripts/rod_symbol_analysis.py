import cv2
import os
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw

# Input and output folders
frames_dir = "frames"
enhanced_dir = "enhanced"
os.makedirs(enhanced_dir, exist_ok=True)

report_lines = ["🔎 Symbol and Marking Analysis Report\n"]

# Function to enhance image clarity and contrast
def enhance_image(image_path):
    img = Image.open(image_path).convert("RGB")
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.5)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)
    return img

# Process all frames
for i, filename in enumerate(sorted(os.listdir(frames_dir))):
    if not filename.lower().endswith(".jpg"):
        continue

    path = os.path.join(frames_dir, filename)
    img = enhance_image(path)

    # Convert to numpy array for OpenCV
    img_cv = np.array(img)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 60, 180)

    # Morphological ops to enhance markings
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Find contours (possible markings)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    marked_img = img.copy()
    draw = ImageDraw.Draw(marked_img)

    symbol_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if 200 < area < 3000:  # filter noise and too-large shapes
            x, y, w, h = cv2.boundingRect(contour)
            draw.rectangle([x, y, x+w, y+h], outline="red", width=2)
            symbol_count += 1

    if symbol_count > 0:
        marked_img.save(os.path.join(enhanced_dir, f"symbol_{filename}"))
        report_lines.append(f"{filename}: {symbol_count} possible markings detected.")

    if i % 100 == 0:
        print(f"Processed {i} frames...")

# Write report
with open("symbols_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("✅ Symbol detection complete.")
print("📁 Enhanced frames saved in 'enhanced/' folder.")
print("📄 Report saved to 'symbols_report.txt'.")
