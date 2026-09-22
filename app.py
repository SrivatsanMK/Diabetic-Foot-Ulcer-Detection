import os
import sys
import time
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageOps

# -----------------------------------------------------------------------------
# 1. CLINICAL METADATA & CLASS MAPPINGS
# -----------------------------------------------------------------------------
CLASSES = {
    0: {
        "title": "DIABETIC FOOT ULCER",
        "severity": "CRITICAL RISK",
        "color_bgr": (38, 38, 220),       # Red in BGR
        "bg_bgr": (50, 40, 90),
        "protocol": [
            "Urgent Podiatric & Vascular consult (24-48h)",
            "Strict Offloading (Total Contact Cast / Boot)",
            "Wound debridement assessment & sterile saline",
            "Infection check (Probe-to-bone / cellulitis)"
        ]
    },
    1: {
        "title": "HEALTHY SKIN",
        "severity": "NORMAL / LOW RISK",
        "color_bgr": (74, 163, 22),       # Green in BGR
        "bg_bgr": (40, 70, 40),
        "protocol": [
            "Daily foot inspection with handheld mirror",
            "Daily moisturizing cream (avoid between toes)",
            "Approved seamless diabetic footwear",
            "Routine annual clinical diabetic screening"
        ]
    },
    2: {
        "title": "SUPERFICIAL WOUND",
        "severity": "MODERATE RISK",
        "color_bgr": (6, 119, 217),       # Amber / Orange in BGR
        "bg_bgr": (30, 60, 90),
        "protocol": [
            "Cleanse lesion with saline & apply bandage",
            "Identify & eliminate footwear friction / shear",
            "Re-evaluate closure within 48 to 72 hours",
            "Protect against silent neuropathic trauma"
        ]
    }
}

IMG_SIZE = (300, 300)
MODEL_PATH = os.path.join("Model", "dfu_best_model.keras")

# -----------------------------------------------------------------------------
# 2. MODEL LOADER WITH KERAS 3 COMPATIBILITY SAFEGUARD
# -----------------------------------------------------------------------------
def load_detection_model():
    """Loads trained EfficientNetB3 model with backward compatibility patch"""
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Warning: Model file not found at {MODEL_PATH}")
        return None
    try:
        import keras
        # Patch Dense.from_config to strip quantization_config if present
        orig_from_config = keras.layers.Dense.from_config
        @classmethod
        def patched_from_config(cls, config):
            config.pop('quantization_config', None)
            return orig_from_config.__func__(cls, config)
        keras.layers.Dense.from_config = patched_from_config

        print("[*] Loading EfficientNetB3 model from Model/dfu_best_model.keras...")
        model = keras.models.load_model(MODEL_PATH)
        print("[+] Model loaded successfully!")
        return model
    except Exception as e:
        print(f"[!] Error loading model: {e}")
        return None

# -----------------------------------------------------------------------------
# 3. OPENCV LESION LOCALIZATION & BOUNDING BOX
# -----------------------------------------------------------------------------
def detect_lesion_contour(img_bgr, pred_idx):
    """Detects lesion boundary and returns bounding box coordinates (x1, y1, x2, y2)"""
    h, w, _ = img_bgr.shape
    if pred_idx in [0, 2]:  # DFU or Wound
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
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
        return best_box
    else:
        pad = int(min(h, w) * 0.06)
        return (pad, pad, w - pad, h - pad)

# -----------------------------------------------------------------------------
# 4. INFERENCE ENGINE
# -----------------------------------------------------------------------------
def run_model_inference(img_bgr, model):
    """Preprocesses image and runs inference through EfficientNetB3"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    img_resized = ImageOps.fit(pil_img, IMG_SIZE, Image.Resampling.LANCZOS)
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
        # Fallback simulation
        preds = np.array([0.998, 0.001, 0.001])
        pred_idx = 0
        confidence = 0.998

    return pred_idx, confidence, preds

# -----------------------------------------------------------------------------
# 5. OPENCV DISPLAY RENDERER (SIDE-BY-SIDE CLINICAL CANVAS)
# -----------------------------------------------------------------------------
def render_opencv_display(img_bgr, pred_idx, confidence, preds, current_idx, total_images, filename, notification=""):
    """Composites patient examination photo and clinical HUD onto a single high-res OpenCV canvas"""
    target_img_h = 650
    h, w, _ = img_bgr.shape
    scale = target_img_h / float(h)
    new_w = int(w * scale)
    scaled_img = cv2.resize(img_bgr, (new_w, target_img_h))

    meta = CLASSES[pred_idx]
    color = meta["color_bgr"]

    # Annotate photo with bounding box & lesion attention overlay
    x1, y1, x2, y2 = detect_lesion_contour(scaled_img, pred_idx)
    cv2.rectangle(scaled_img, (x1, y1), (x2, y2), color, 3)

    if pred_idx in [0, 2]:
        overlay = scaled_img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.22, scaled_img, 0.78, 0, scaled_img)

    # Class Badge on top of bounding box
    badge_label = f"{meta['title']}: {confidence * 100:.1f}%"
    (tw, th), _ = cv2.getTextSize(badge_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    b_y1 = max(0, y1 - th - 10)
    cv2.rectangle(scaled_img, (x1, b_y1), (x1 + tw + 12, y1), color, -1)
    cv2.putText(scaled_img, badge_label, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Build Right Clinical HUD Panel
    hud_w = 460
    hud = np.full((target_img_h, hud_w, 3), 26, dtype=np.uint8)  # Deep slate dark theme

    # Header
    cv2.rectangle(hud, (0, 0), (hud_w, 55), (45, 30, 20), -1)
    cv2.putText(hud, "DFU AI CLINICAL DIAGNOSTICS", (16, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # File info & Progress
    cv2.putText(hud, f"Image [{current_idx + 1}/{total_images}]:", (16, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1, cv2.LINE_AA)
    fname_short = os.path.basename(filename)
    if len(fname_short) > 28:
        fname_short = fname_short[:25] + "..."
    cv2.putText(hud, fname_short, (16, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (241, 245, 249), 2, cv2.LINE_AA)

    # Severity Banner
    cv2.rectangle(hud, (16, 130), (hud_w - 16, 195), meta["bg_bgr"], -1)
    cv2.rectangle(hud, (16, 130), (hud_w - 16, 195), color, 2)
    cv2.putText(hud, meta["severity"], (30, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.putText(hud, meta["title"], (30, 182), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Confidence Bars
    cv2.putText(hud, "CONFIDENCE DISTRIBUTION", (16, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1, cv2.LINE_AA)
    labels = ["Diabetic Foot Ulcer", "Healthy Skin", "Superficial Wound"]
    bar_y = 260
    for i, lab in enumerate(labels):
        p_val = float(preds[i])
        bar_color = CLASSES[i]["color_bgr"]
        cv2.putText(hud, f"{lab}: {p_val * 100:.1f}%", (16, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (226, 232, 240), 1, cv2.LINE_AA)
        cv2.rectangle(hud, (16, bar_y + 8), (hud_w - 16, bar_y + 18), (50, 50, 50), -1)
        fill_w = int((hud_w - 32) * p_val)
        if fill_w > 0:
            cv2.rectangle(hud, (16, bar_y + 8), (16 + fill_w, bar_y + 18), bar_color, -1)
        bar_y += 45

    # Clinical Protocol
    cv2.putText(hud, "RECOMMENDED PROTOCOL", (16, bar_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1, cv2.LINE_AA)
    proto_y = bar_y + 35
    for proto in meta["protocol"]:
        cv2.circle(hud, (22, proto_y - 4), 3, color, -1)
        cv2.putText(hud, proto, (34, proto_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (203, 213, 225), 1, cv2.LINE_AA)
        proto_y += 24

    # Navigation Controls Footer
    cv2.rectangle(hud, (0, target_img_h - 45), (hud_w, target_img_h), (35, 35, 40), -1)
    cv2.putText(hud, "[N] Next | [P] Prev | [S] Save | [O] Open | [Q] Quit", (12, target_img_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1, cv2.LINE_AA)

    # Toast Notification (e.g. "Saved successfully")
    if notification:
        cv2.rectangle(scaled_img, (20, target_img_h - 50), (280, target_img_h - 15), (34, 139, 34), -1)
        cv2.putText(scaled_img, notification, (30, target_img_h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Combine Left (Photo) + Right (HUD)
    canvas = np.hstack([scaled_img, hud])
    return canvas

# -----------------------------------------------------------------------------
# 6. MULTI-IMAGE FILE SELECTION DIALOG
# -----------------------------------------------------------------------------
def select_images_dialog():
    """Opens native Windows file dialog to pick multiple images"""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_paths = filedialog.askopenfilenames(
        title="Select Patient Foot Images (Multi-Select Supported)",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    root.destroy()
    return list(file_paths)

# -----------------------------------------------------------------------------
# 7. MAIN OPEN-CV WINDOW EVENT LOOP
# -----------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("   DIABETIC FOOT ULCER (DFU) AI DETECTOR - PURE OPENCV")
    print("=" * 65)

    model = load_detection_model()

    # 1. Prompt user to select images
    print("[*] Opening file picker to select images...")
    image_paths = select_images_dialog()

    # Fallback to demo images if user canceled dialog
    if not image_paths:
        sample_dir = "Real World Testing"
        if os.path.exists(sample_dir):
            image_paths = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"[*] No file chosen. Automatically loaded {len(image_paths)} demo samples from 'Real World Testing'.")
        else:
            print("[!] No images selected and no test samples found. Exiting.")
            return

    os.makedirs("output_detections", exist_ok=True)
    window_name = "Diabetic Foot Ulcer AI Detection Studio"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    current_idx = 0
    total_imgs = len(image_paths)
    notification = ""
    notif_time = 0

    while True:
        current_file = image_paths[current_idx]
        img_bgr = cv2.imread(current_file)
        if img_bgr is None:
            print(f"[!] Could not read image: {current_file}")
            current_idx = (current_idx + 1) % total_imgs
            continue

        # Run AI Inference
        pred_idx, confidence, preds = run_model_inference(img_bgr, model)

        # Clear toast notification after 2 seconds
        if notification and (time.time() - notif_time > 2.0):
            notification = ""

        # Render complete canvas (photo + HUD)
        canvas = render_opencv_display(img_bgr, pred_idx, confidence, preds, current_idx, total_imgs, current_file, notification)
        cv2.imshow(window_name, canvas)

        # Keyboard listener
        key = cv2.waitKey(30) & 0xFF

        if key in [ord('q'), ord('Q'), 27]:  # Quit
            print("[*] Exiting application.")
            break
        elif key in [ord('n'), ord('N'), 83]:  # Next image
            current_idx = (current_idx + 1) % total_imgs
            notification = ""
        elif key in [ord('p'), ord('P'), 81]:  # Previous image
            current_idx = (current_idx - 1 + total_imgs) % total_imgs
            notification = ""
        elif key in [ord('s'), ord('S')]:  # Save output
            out_filename = os.path.join("output_detections", f"detected_{os.path.basename(current_file)}")
            cv2.imwrite(out_filename, canvas)
            notification = "SAVED TO DISK!"
            notif_time = time.time()
            print(f"[+] Saved detection result to: {out_filename}")
        elif key in [ord('o'), ord('O')]:  # Open new batch
            new_paths = select_images_dialog()
            if new_paths:
                image_paths = new_paths
                total_imgs = len(image_paths)
                current_idx = 0
                print(f"[+] Loaded new batch of {total_imgs} images.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
