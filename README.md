# 🩺 Diabetic Foot Ulcer (DFU) AI Detection Studio (OpenCV)

An AI-powered computer vision desktop application for automated Diabetic Foot Ulcer (DFU) detection, wound localization, and clinical risk stratification using deep convolutional neural networks (**EfficientNetB3**) and a native **OpenCV** graphical user interface.

---

## 🌟 Key Features
- **Pure OpenCV Interface**: Runs as a fast native desktop application with no web server or browser dependencies.
- **Multi-Image Support**: Native Windows file selection dialog supports selecting single images or multiple photos simultaneously.
- **Side-by-Side Clinical HUD**:
  - **Left Canvas**: Patient foot photo annotated with color-coded bounding boxes and attention heatmaps over the wound area.
  - **Right Canvas**: Real-time diagnostic HUD with class probability distributions and triage recommendations.
- **3 Diagnostic Categories**:
  - 🔴 **Diabetic Foot Ulcer (DFU)** (Critical Risk)
  - 🟢 **Healthy Skin** (Normal Integrity)
  - 🟡 **Superficial Wound** (Moderate Risk)
- **Interactive Keyboard Navigation**:
  - `[N]` or `[→]` : Next Image
  - `[P]` or `[←]` : Previous Image
  - `[S]` : Save annotated detection image to `output_detections/`
  - `[O]` : Open native file dialog to load more images
  - `[Q]` or `[ESC]` : Close window and exit

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SrivatsanMK/Diabetic-Foot-Ulcer-Detection.git
   cd Diabetic-Foot-Ulcer-Detection
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv dfuenv
   # Windows:
   .\dfuenv\Scripts\activate
   # Linux/Mac:
   source dfuenv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Application:**
   ```bash
   python app.py
   ```
