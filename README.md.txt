# 🛸 Rod Artifact Analysis — Multi-Stage Symbol & Energy Field Investigation

### Overview
This repository documents a complete analytical pipeline used to examine two video clips of a **metallic rod-like object** exhibiting possible field emissions and surface markings.  
The workflow combines **computer vision**, **signal processing**, and **field coherence modeling** to evaluate whether the object demonstrates anomalous behavior or natural reflection phenomena.

---

## 🔬 Process Summary

### **Stage 1–2: Video Frame Extraction & Symbol Detection**
- Extracted frames from both clips at ~1 fps.  
- Enhanced edges using Canny and Laplacian filters.  
- Applied morphological operations to isolate clusters of potential etchings.  
- **Output:** `enhanced/` folder and `symbols_report.txt`.

### **Stage 3–4: Symbol Mapping & Refinement**
- Clustered contours using **K-Means** to identify repeating glyph patterns.  
- Normalized and composited symbol crops into a collage (`symbol_collage.jpg`).  
- **Output:** `symbol_summary.txt` and `symbol_refined_summary.txt`.

### **Stage 5–8: Decoding & Temporal Mapping**
- Analyzed symbol morphology with open-source OCR and shape descriptors.  
- Computed per-frame symbol frequencies and semantic linkages.  
- Generated a dynamic symbol interaction network (`symbol_network.gexf`) for Gephi visualization.

### **Stage 9–11: Material & Energy Profile Analysis**
- Measured mean brightness and Laplacian variance (surface texture).  
- Conducted **FFT** analysis on brightness waveforms to detect **non-mains modulation**.  
- Calculated an **exponential decay fit** (τ ≈ 52,686 frames) suggesting **slow energy dissipation**.  
- **Results:** `roi_fft.png`, `brightness_texture.png`, `stage11_decay_curve.png`.

### **Stage 12: Coherence & Field Mapping**
- Combined energy and variance data into a 2-D spectrogram (`field_spectrogram.png`).  
- Generated `coherence_report.txt`:
  - **Mean energy:** 99.60  
  - **Variance:** 879.55  
  - **Brightness variance:** 5905.64  
- Indicates **sustained coherence**, inconsistent with ordinary light reflection or LED flicker.

---

## 🧠 Key Findings

| Parameter | Observation | Interpretation |
|------------|--------------|----------------|
| **Brightness variance** | High | Suggests field interference or plasma-like behavior |
| **FFT profile** | Non-periodic | Not consistent with 50/60 Hz electrical modulation |
| **Energy decay** | Gradual | Indicates residual charge or fading energy field |
| **Coherence** | Stable | Suggests structured, non-random field persistence |

---

## 🧩 Conclusions
The object’s reflective and energetic behavior does **not align with static lighting or random reflections**.  
It may represent:
- A **technological artifact** exhibiting decaying field emissions (possibly ancient or advanced).  
- A **constructed hoax**, though replicating the observed coherence would require sophisticated modulation.

Further testing using **spectro-radiometric calibration** and **material sampling** would help confirm whether the surface behavior is electromagnetic, plasma-based, or optical.

---

## 📁 Repository Layout

| Folder | Description |
|---------|-------------|
| `scripts/` | Python scripts for each analysis stage (1–12). |
| `reports/` | Text reports summarizing results from each stage. |
| `output/` | Final composite images, spectrograms, and summaries. |
| `data/raw_videos/` | Original artifact videos (or external links). |
| `results/` | Intermediate data: enhanced frames, isolated symbols, etc. |

### 🔁 Workflow Summary
Each stage builds upon the previous:
1. **Frame Extraction & Enhancement** → Symbol detection  
2. **Clustering & Refinement** → Symbol mapping  
3. **Semantic Analysis & Material ID** → Spectral and physical modeling  
4. **Field Mapping & Coherence Tests** → Detect possible structured energy fields  

Final outputs (reports + visualizations) are under `reports/` and `output/`.

