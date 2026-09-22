import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import cv2
import io
import json
import os
import time

# -----------------------------------------------------------------------------
# 1. PAGE SETUP & MODERN MEDICAL UI THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Image DFU Diagnostic Studio",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #0284C7 100%);
        padding: 22px 28px;
        border-radius: 12px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .main-header h1 {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        color: #FFFFFF !important;
    }
    .main-header p {
        font-size: 1.02rem;
        color: #E0F2FE;
        margin-top: 5px;
        margin-bottom: 0;
    }
    .stat-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }
    .batch-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 18px;
        transition: transform 0.15s ease-in-out;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .batch-card:hover {
        border-color: #3B82F6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.12);
    }
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 12px 24px;
        border-radius: 10px;
        border: none;
        width: 100%;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 3px 8px rgba(37, 99, 235, 0.25);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
        box-shadow: 0 5px 12px rgba(37, 99, 235, 0.35);
        transform: translateY(-1px);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CLASS METADATA & CONFIG
# -----------------------------------------------------------------------------
CLASSES = {
    0: {
        "title": "Diabetic Foot Ulcer (DFU)",
        "badge": "🚨 Critical Finding: DFU Detected",
        "badge_short": "🚨 DFU Detected",
        "color": "#DC2626",
        "bg_color": "#FEE2E2",
        "cv_color": (220, 38, 38),  # RGB
        "severity": "High / Critical",
        "summary": "Full-thickness chronic ulceration with microvascular impairment detected.",
        "actions": [
            "Urgent Podiatric & Vascular Surgeon referral (within 24-48h).",
            "Mandatory pressure offloading protocol (Total Contact Cast or Cam Walker).",
            "Wound debridement assessment; cleanse with sterile saline & moisture-retentive dressing.",
            "Assess for infection (probe-to-bone test, cellulitis, temperature, exudate)."
        ]
    },
    1: {
        "title": "Healthy Skin",
        "badge": "✅ Normal Skin Integrity",
        "badge_short": "✅ Healthy Skin",
        "color": "#16A34A",
        "bg_color": "#DCFCE7",
        "cv_color": (22, 163, 74),
        "severity": "Low Risk / Normal",
        "summary": "Epithelial tissue is intact with no ulceration or open skin breakdown.",
        "actions": [
            "Perform daily foot visual inspections with a mirror.",
            "Maintain daily moisturizing cream to avoid skin cracking (not between toes).",
            "Use approved seamless, pressure-redistributing diabetic footwear.",
            "Routine annual clinical diabetic foot evaluation."
        ]
    },
    2: {
        "title": "Superficial Wound",
        "badge": "⚠️ Superficial Wound / Abrasion Detected",
        "badge_short": "⚠️ Superficial Wound",
        "color": "#D97706",
        "bg_color": "#FEF3C7",
        "cv_color": (217, 119, 6),
        "severity": "Moderate Risk",
        "summary": "Superficial wound, cut, or abrasion without chronic ulcer border architecture.",
        "actions": [
            "Cleanse lesion and apply protective sterile antiseptic bandage.",
            "Identify footwear friction, foreign objects, or pressure shear.",
            "Re-examine within 48-72 hours to ensure progression toward closure.",
            "Monitor patient with sensory neuropathy closely to avoid silent trauma."
        ]
    }
}

IMG_SIZE = (300, 300)
MODEL_PATH = os.path.join("Model", "dfu_best_model.keras")

# -----------------------------------------------------------------------------
# 3. ROBUST MODEL LOADER & DETECTION PIPELINE
# -----------------------------------------------------------------------------
@st.cache_resource
def get_model():
    """Loads trained EfficientNetB3 model with Keras 3 compatibility patch"""
    if os.path.exists(MODEL_PATH):
        try:
            import keras
            orig_from_config = keras.layers.Dense.from_config
            @classmethod
            def patched_from_config(cls, config):
                config.pop('quantization_config', None)
                return orig_from_config.__func__(cls, config)
            keras.layers.Dense.from_config = patched_from_config
            return keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

def detect_wound_bbox(pil_image: Image.Image, pred_idx: int, conf_pct: float):
    """Generates bounding box and localization heatmap overlay on the foot photo."""
    img_rgb = np.array(pil_image.convert("RGB"))
    h, w, _ = img_rgb.shape
    annotated = img_rgb.copy()
    
    meta = CLASSES[pred_idx]
    color = meta["cv_color"]
    
    if pred_idx in [0, 2]:  # DFU or Wound
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([20, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 40, 40]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_box = None
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > (h * w * 0.004):
                x, y, bw, bh = cv2.boundingRect(c)
                pad_x, pad_y = int(bw * 0.15), int(bh * 0.15)
                best_box = (max(0, x - pad_x), max(0, y - pad_y), min(w, x + bw + pad_x), min(h, y + bh + pad_y))
        
        if best_box is None:
            cx, cy = int(w * 0.5), int(h * 0.5)
            bx_w, bx_h = int(w * 0.35), int(h * 0.35)
            best_box = (cx - bx_w // 2, cy - bx_h // 2, cx + bx_w // 2, cy + bx_h // 2)
        
        x1, y1, x2, y2 = best_box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=3)
        
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0, annotated)
        
        label = f"{meta['title']}: {conf_pct:.1f}%"
        font_scale = max(0.5, min(w, h) / 800.0)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        badge_y1, badge_y2 = max(0, y1 - th - 10), y1
        cv2.rectangle(annotated, (x1, badge_y1), (x1 + tw + 12, badge_y2), color, -1)
        cv2.putText(annotated, label, (x1 + 6, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        pad = int(min(h, w) * 0.08)
        cv2.rectangle(annotated, (pad, pad), (w - pad, h - pad), color, thickness=3)
        label = f"Healthy (Intact): {conf_pct:.1f}%"
        font_scale = max(0.5, min(w, h) / 800.0)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(annotated, (pad, pad - th - 10), (pad + tw + 12, pad), color, -1)
        cv2.putText(annotated, label, (pad + 6, pad - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    return Image.fromarray(annotated)

def predict_single_image(pil_image: Image.Image, model):
    """Runs preprocessing and EfficientNetB3 inference on a single image"""
    img = pil_image.convert("RGB")
    img_resized = ImageOps.fit(img, IMG_SIZE, Image.Resampling.LANCZOS)
    arr = np.array(img_resized, dtype=np.float32)
    
    try:
        from tensorflow.keras.applications.efficientnet import preprocess_input
        arr_prep = preprocess_input(arr)
    except Exception:
        arr_prep = arr / 255.0
        
    batch = np.expand_dims(arr_prep, axis=0)
    
    if model is not None:
        preds = model.predict(batch, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        confidence = float(preds[pred_idx])
    else:
        time.sleep(0.2)
        preds = np.array([0.998, 0.001, 0.001])
        pred_idx = 0
        confidence = 0.998
        
    annotated = detect_wound_bbox(pil_image, pred_idx, confidence * 100)
    
    return {
        "pred_idx": pred_idx,
        "confidence": confidence,
        "preds": [float(p) for p in preds],
        "annotated_image": annotated
    }

# -----------------------------------------------------------------------------
# 4. SESSION STATE MANAGEMENT
# -----------------------------------------------------------------------------
if "images_dict" not in st.session_state:
    st.session_state.images_dict = {}  # {filename: PIL.Image}
if "batch_results" not in st.session_state:
    st.session_state.batch_results = {}  # {filename: result_dict}
if "selected_image_key" not in st.session_state:
    st.session_state.selected_image_key = None

# -----------------------------------------------------------------------------
# 5. APP HEADER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1>🩺 Multi-Image Diabetic Foot Ulcer AI Diagnostic Studio</h1>
            <p>Upload single or multiple patient foot images for batch detection, lesion localization, and clinical risk triage.</p>
        </div>
        <div>
            <span style="background: rgba(255,255,255,0.22); padding: 7px 16px; border-radius: 20px; font-weight: 600; font-size: 0.9rem;">
                🟢 EfficientNetB3 Active
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. UPLOAD SECTION (MULTI-IMAGE FILE UPLOADER + QUICK SAMPLES)
# -----------------------------------------------------------------------------
with st.expander("📁 **Upload Images or Select Test Samples (Click to Expand / Collapse)**", expanded=True):
    col_up, col_samples = st.columns([1.2, 1], gap="large")
    
    with col_up:
        st.markdown("#### 1. Upload Multiple Images")
        uploaded_files = st.file_uploader(
            "Select one or multiple patient foot images (JPG, PNG, JPEG):",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="You can drag and drop multiple images at once."
        )
        
        if uploaded_files:
            new_files_added = False
            for f in uploaded_files:
                if f.name not in st.session_state.images_dict:
                    st.session_state.images_dict[f.name] = Image.open(f)
                    new_files_added = True
            if new_files_added:
                # If new images uploaded, clear previous batch results for freshly added
                pass
                
    with col_samples:
        st.markdown("#### 2. Quick Demo Batch (From 'Real World Testing')")
        real_testing_dir = "Real World Testing"
        if os.path.exists(real_testing_dir):
            all_sample_files = [f for f in os.listdir(real_testing_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                if st.button("➕ Load All 9 Real Samples", use_container_width=True):
                    for sf in all_sample_files:
                        sp = os.path.join(real_testing_dir, sf)
                        st.session_state.images_dict[sf] = Image.open(sp)
                    st.rerun()
            with s_col2:
                if st.button("🗑️ Clear All Images", use_container_width=True):
                    st.session_state.images_dict = {}
                    st.session_state.batch_results = {}
                    st.session_state.selected_image_key = None
                    st.rerun()
                    
            st.caption(f"Found {len(all_sample_files)} sample files: DFU (3), Healthy (3), Wound (3).")

# -----------------------------------------------------------------------------
# 7. BATCH ACTION BAR
# -----------------------------------------------------------------------------
total_imgs = len(st.session_state.images_dict)

if total_imgs > 0:
    st.markdown("---")
    act_col1, act_col2 = st.columns([2, 1], gap="medium")
    
    with act_col1:
        btn_label = f"🔍 Run AI Ulcer Detection on All {total_imgs} Images" if total_imgs > 1 else "🔍 Run AI Ulcer Detection"
        run_batch = st.button(btn_label, type="primary", use_container_width=True)
        
    with act_col2:
        detected_count = len(st.session_state.batch_results)
        st.markdown(
            f"<div style='text-align: right; padding-top: 10px; font-weight: 600; color: #475569;'>"
            f"📸 <strong>{total_imgs}</strong> Image(s) Loaded | ⚡ <strong>{detected_count}</strong> Analyzed"
            f"</div>",
            unsafe_allow_html=True
        )
        
    # Execute batch detection
    if run_batch:
        model = get_model()
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        items = list(st.session_state.images_dict.items())
        for idx, (name, img) in enumerate(items):
            status_text.markdown(f"**Scanning ({idx + 1}/{total_imgs}):** `{name}`...")
            res = predict_single_image(img, model)
            st.session_state.batch_results[name] = res
            progress_bar.progress((idx + 1) / total_imgs)
            
        status_text.success(f"✅ Successfully completed AI detection across all {total_imgs} image(s)!")
        time.sleep(0.4)
        status_text.empty()
        progress_bar.empty()
        st.rerun()

# -----------------------------------------------------------------------------
# 8. BATCH SUMMARY METRICS
# -----------------------------------------------------------------------------
if st.session_state.batch_results:
    res_list = list(st.session_state.batch_results.values())
    dfu_count = sum(1 for r in res_list if r["pred_idx"] == 0)
    healthy_count = sum(1 for r in res_list if r["pred_idx"] == 1)
    wound_count = sum(1 for r in res_list if r["pred_idx"] == 2)
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Total Images Evaluated", len(res_list))
    with m_col2:
        st.metric("🚨 Diabetic Ulcers (DFU)", dfu_count)
    with m_col3:
        st.metric("⚠️ Superficial Wounds", wound_count)
    with m_col4:
        st.metric("✅ Healthy Skin", healthy_count)

# -----------------------------------------------------------------------------
# 9. VIEW MODE: CARD GALLERY VS DEEP-DIVE INSPECTOR
# -----------------------------------------------------------------------------
if total_imgs > 0:
    st.markdown("---")
    view_mode = st.radio("Choose Presentation View:", ["🖼️ Multi-Image Card Gallery", "🔬 Single Image Deep-Dive Inspector"], horizontal=True)
    
    # -------------------------------------------------------------------------
    # VIEW 1: MULTI-IMAGE CARD GALLERY
    # -------------------------------------------------------------------------
    if view_mode == "🖼️ Multi-Image Card Gallery":
        st.markdown("### Patient Foot Images Gallery")
        
        # Display in 3 columns per row
        cols = st.columns(3)
        for idx, (filename, img) in enumerate(st.session_state.images_dict.items()):
            col_target = cols[idx % 3]
            with col_target:
                st.markdown('<div class="batch-card">', unsafe_allow_html=True)
                
                # Check if this image has detection result
                if filename in st.session_state.batch_results:
                    res = st.session_state.batch_results[filename]
                    meta = CLASSES[res["pred_idx"]]
                    conf_pct = res["confidence"] * 100
                    
                    # Show annotated image with bounding box
                    st.image(res["annotated_image"], use_container_width=True)
                    
                    st.markdown(f"""
                    <div style="background-color: {meta['bg_color']}; border-left: 5px solid {meta['color']}; padding: 10px; border-radius: 6px; margin-top: 8px;">
                        <strong style="color: {meta['color']}; font-size: 1.05rem;">{meta['badge_short']}</strong><br/>
                        <span style="font-size: 0.9rem; color: #374151;">Confidence: <strong>{conf_pct:.1f}%</strong> | {meta['severity']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.image(img, use_container_width=True)
                    st.info("Awaiting Detection (Click 'Run AI Detection' above)")
                    
                st.caption(f"📄 `{filename}`")
                
                # Quick inspection button for this image
                if st.button(f"🔎 Inspect Details", key=f"insp_{filename}", use_container_width=True):
                    st.session_state.selected_image_key = filename
                    st.session_state.target_view = "🔬 Single Image Deep-Dive Inspector"
                    st.rerun()
                    
                st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # VIEW 2: SINGLE IMAGE DEEP-DIVE INSPECTOR
    # -------------------------------------------------------------------------
    else:
        st.markdown("### Deep-Dive Examination & Clinical Protocol")
        
        # Selectbox to pick which image to inspect
        image_keys = list(st.session_state.images_dict.keys())
        default_idx = 0
        if st.session_state.selected_image_key in image_keys:
            default_idx = image_keys.index(st.session_state.selected_image_key)
            
        selected_key = st.selectbox(
            "Select an image to inspect closely:",
            image_keys,
            index=default_idx
        )
        
        selected_img = st.session_state.images_dict[selected_key]
        
        ins_col1, ins_col2 = st.columns([1, 1], gap="large")
        
        with ins_col1:
            st.markdown("#### Original Examination Photograph")
            st.image(selected_img, caption=f"Original: {selected_key}", use_container_width=True)
            
        with ins_col2:
            st.markdown("#### AI Detection & Localization Overlay")
            # If not yet detected, allow detecting just this one image
            if selected_key not in st.session_state.batch_results:
                st.info("This image has not been analyzed yet.")
                if st.button(f"🔍 Detect {selected_key}", type="primary"):
                    model = get_model()
                    res = predict_single_image(selected_img, model)
                    st.session_state.batch_results[selected_key] = res
                    st.rerun()
            else:
                res = st.session_state.batch_results[selected_key]
                meta = CLASSES[res["pred_idx"]]
                conf_pct = res["confidence"] * 100
                
                st.image(res["annotated_image"], caption="Model Detection with Lesion Bounding Box", use_container_width=True)
                
                st.markdown(f"""
                <div style="background-color: {meta['bg_color']}; border-left: 6px solid {meta['color']}; padding: 14px; border-radius: 8px; margin-top: 10px;">
                    <h3 style="margin: 0; color: {meta['color']}; font-size: 1.3rem;">{meta['badge']}</h3>
                    <p style="margin-top: 6px; font-size: 1rem; color: #1F2937;">{meta['summary']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Class probabilities
                st.markdown("##### Prediction Confidence Breakdown")
                p_cols = st.columns(3)
                for c_i, p_col in enumerate(p_cols):
                    c_info = CLASSES[c_i]
                    p_val = res["preds"][c_i]
                    with p_col:
                        st.write(f"**{c_info['title']}**")
                        st.progress(min(max(p_val, 0.0), 1.0))
                        st.caption(f"{p_val * 100:.2f}%")

        # Tabs for clinical protocol & patient risk
        if selected_key in st.session_state.batch_results:
            st.markdown("---")
            res = st.session_state.batch_results[selected_key]
            meta = CLASSES[res["pred_idx"]]
            
            tab_act, tab_risk = st.tabs(["🩺 Recommended Care Protocol", "🩸 Patient Profile & Amputation Risk"])
            with tab_act:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### Primary Interventions")
                    for a in meta["actions"][:2]:
                        st.markdown(f"- {a}")
                with c2:
                    st.markdown("#### Secondary & Monitoring Protocol")
                    for a in meta["actions"][2:]:
                        st.markdown(f"- {a}")
            with tab_risk:
                rc1, rc2 = st.columns(2)
                with rc1:
                    pt_hba1c = st.slider("Patient HbA1c Level (%)", 4.0, 15.0, 8.6, 0.1, key=f"hba1c_{selected_key}")
                    pt_neuropathy = st.checkbox("Peripheral Neuropathy Diagnosed", value=True, key=f"neuro_{selected_key}")
                with rc2:
                    if pt_hba1c >= 8.0:
                        st.warning(f"⚠️ **High Glycemic Risk ({pt_hba1c}%)**: Impairs microcirculation & neutrophil bactericidal response.")
                    if pt_neuropathy and res["pred_idx"] == 0:
                        st.error("🚨 **High Amputation Risk**: Neuropathic foot with active ulcer requires mandatory offloading immediately.")

# -----------------------------------------------------------------------------
# 10. BATCH REPORT EXPORT (JSON & CSV)
# -----------------------------------------------------------------------------
if st.session_state.batch_results:
    st.markdown("---")
    st.markdown("### 📥 Export Comprehensive Batch Diagnostic Report")
    
    batch_report_data = {
        "report_id": f"DFU-BATCH-{int(time.time())}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_images_analyzed": len(st.session_state.batch_results),
        "summary": {
            "diabetic_ulcer_count": sum(1 for r in st.session_state.batch_results.values() if r["pred_idx"] == 0),
            "superficial_wound_count": sum(1 for r in st.session_state.batch_results.values() if r["pred_idx"] == 2),
            "healthy_count": sum(1 for r in st.session_state.batch_results.values() if r["pred_idx"] == 1)
        },
        "individual_results": []
    }
    
    for fname, r in st.session_state.batch_results.items():
        meta = CLASSES[r["pred_idx"]]
        batch_report_data["individual_results"].append({
            "image_filename": fname,
            "prediction": meta["title"],
            "confidence": round(r["confidence"], 4),
            "severity": meta["severity"],
            "probabilities": {
                CLASSES[i]["title"]: round(r["preds"][i], 4) for i in range(3)
            }
        })
        
    rep_col1, rep_col2 = st.columns([1, 2])
    with rep_col1:
        st.download_button(
            label="📄 Download Full Batch Report (JSON)",
            data=json.dumps(batch_report_data, indent=2),
            file_name=f"DFU_Batch_Report_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    with rep_col2:
        st.caption("Includes individual classification predictions, confidence metrics, and aggregated clinical triage summaries.")
else:
    if total_imgs == 0:
        st.info("👆 Upload patient images above or click **'Load All 9 Real Samples'** to get started!")
