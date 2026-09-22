# 🩺 Diabetic Foot Ulcer (DFU) Detection & Clinical Diagnostic Studio

An AI-powered clinical decision-support application for automated Diabetic Foot Ulcer (DFU) detection, lesion localization, and risk stratification using deep convolutional neural networks (**EfficientNetB3**) and **Streamlit**.

---

## 🌟 Key Features
- **Multi-Image & Batch Upload**: Upload single or multiple patient foot images simultaneously.
- **Deep Learning Classifier**: Powered by fine-tuned **EfficientNetB3** trained on the DFU dataset.
- **Lesion Localization Overlay**: Generates bounding boxes and attention heatmaps highlighting suspected ulcer sites.
- **3 Clinical Classes**:
  - 🚨 **Diabetic Foot Ulcer (DFU)** (Critical chronic ulceration)
  - ⚠️ **Superficial Wound** (Moderate risk abrasion)
  - ✅ **Healthy Skin** (Normal epithelial integrity)
- **Clinical Triage Protocols**: Actionable medical guidelines including pressure offloading (Total Contact Casting) and infection surveillance.
- **Export Diagnostic Reports**: Download consolidated patient examination summaries in JSON.

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

4. **Run the application:**
    ```bash
    streamlit run app.py
    ```


---
