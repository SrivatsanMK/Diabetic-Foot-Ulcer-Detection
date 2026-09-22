# 🩺 Diabetic Foot Ulcer (DFU) Detection Studio

An AI-powered clinical decision-support application for automated Diabetic Foot Ulcer (DFU) classification, lesion localization, and triage using **EfficientNetB3** and **Streamlit**.

---

## 🌟 Features

- **Multi-Image & Benchmark Intake**: Upload patient foot images or load pre-curated test cases with one click.
- **EfficientNetB3 Classifier**: High-precision classification across 3 diagnostic classes:
  - 🚨 **Diabetic Foot Ulcer (DFU)**: Critical chronic ulceration requiring immediate attention.
  - ⚠️ **Superficial Wound**: Moderate risk abrasion requiring monitoring.
  - ✅ **Healthy Skin**: Intact skin integrity.
- **Lesion Localization**: Automated bounding box and heatmap overlay highlighting suspected ulcer sites.
- **Executive KPI Dashboard**: Live summary cards showing total cases evaluated and triage breakdowns.
- **Side-by-Side Deep-Dive**: Comparative view of baseline photo vs. AI detection overlay with confidence breakdowns and clinical protocols.
- **Report Export**: Download diagnostic summaries in **CSV** and **JSON** formats.

---

## 📁 Project Structure

```text
Diabetic-Foot-Ulcer-Detection/
├── Model/
│   └── dfu_best_model.keras                 # Trained EfficientNetB3 model weights
├── Real World Testing/                      # Sample benchmark images for testing
├── Notebook/
│   └── Diabetic_Foot_Ulcer_Detection.ipynb  # Model training notebook
├── app.py                                   # Streamlit application
├── requirements.txt                         # Dependencies
├── .gitignore                               # Git ignore rules
└── README.md                                # Project documentation
```

---

## 🛠️ How to Run

### 1. Activate Virtual Environment
```powershell
.\dfuenv\Scripts\activate
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Launch the Application
```powershell
streamlit run app.py
```

Access the app at `http://localhost:8501`.

---

## 🚀 Push to GitHub

```powershell
git add .
git commit -m "Update DFU detection app with professional Streamlit UI"
git push -u origin main
```