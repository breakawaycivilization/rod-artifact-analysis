import cv2
import numpy as np
import os

# ==== FILE PATHS ====
video_files = [
    r"C:\Users\abuka\Downloads\signal-2025-10-04-130927.mp4",
    r"C:\Users\abuka\Downloads\signal-2025-10-04-130927_002.mp4"
]

# ==== SETTINGS ====
OUTPUT_DIR = "frames"
REPORT_FILE = "analysis_report.txt"
MIN_CHANGE_THRESHOLD = 40  # sensitivity for light flicker
MOTION_THRESHOLD = 50000   # sensitivity for motion detection

os.makedirs(OUTPUT_DIR, exist_ok=True)

report_lines = []
frame_counter = 0

def analyze_video(video_path):
    global frame_counter
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    frame_index = 0
    last_save_index = -50

    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"🎥 Analyzing {video_path} ({fps:.2f} fps)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)

        # Detect flicker
        mean_brightness = np.mean(gray)
        if prev_frame is not None:
            brightness_diff = abs(mean_brightness - np.mean(prev_frame))
            if brightness_diff > MIN_CHANGE_THRESHOLD:
                timestamp = frame_index / fps
                report_lines.append(f"[{timestamp:.2f}s] ⚡ Flicker Detected (Δ={brightness_diff:.2f})")
                if frame_index - last_save_index > 50:
                    save_frame(frame, frame_counter, "flicker")
                    frame_counter += 1
                    last_save_index = frame_index

        # Detect motion
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, blur)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_score = np.sum(thresh)
            if motion_score > MOTION_THRESHOLD:
                timestamp = frame_index / fps
                report_lines.append(f"[{timestamp:.2f}s] 🚀 Motion Detected (Score={motion_score})")
                if frame_index - last_save_index > 50:
                    save_frame(frame, frame_counter, "motion")
                    frame_counter += 1
                    last_save_index = frame_index

        prev_frame = blur
        frame_index += 1

    cap.release()
    print(f"✅ Finished {os.path.basename(video_path)}")


def save_frame(frame, index, label):
    """Save the extracted frame with label and timestamp."""
    filename = os.path.join(OUTPUT_DIR, f"{label}_{index:04d}.jpg")
    cv2.imwrite(filename, frame)
    print(f"🖼️ Saved frame: {filename}")


def main():
    for video_path in video_files:
        analyze_video(video_path)

    with open("rod_analysis_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n📄 Analysis complete! Results saved to '{REPORT_FILE}'")
    print(f"🖼️ Frames saved in '{OUTPUT_DIR}/' folder.")


if __name__ == "__main__":
    main()
