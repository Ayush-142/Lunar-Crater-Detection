"""
Lunar Crater Detection — Streamlit demo
Upload a lunar image (Chandrayaan-2 OHRC / TMC-2 tile or similar),
get crater detections from a YOLOv11 model with a ResNet101 backbone.

Weights are downloaded at startup from the Hugging Face model repo
Ayush-1452/lunar-crater-yolov11-resnet101 (file: best.pt).
"""

import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

WEIGHTS_REPO = "Ayush-1452/lunar-crater-yolov11-resnet101"
WEIGHTS_FILE = "best.pt"

# Set per-camera ground resolution to convert pixels -> meters.
# OHRC ~0.25 m/px, TMC-2 ~5 m/px. Used only for the size table.
RESOLUTIONS = {"OHRC (0.25 m/px)": 0.25, "TMC-2 (5 m/px)": 5.0, "Unknown": None}


@st.cache_resource
def load_model():
    weights_path = hf_hub_download(repo_id=WEIGHTS_REPO, filename=WEIGHTS_FILE)
    return YOLO(weights_path)


def detect(model, image: Image.Image, conf: float, camera: str):
    results = model.predict(image, conf=conf, imgsz=640, verbose=False)
    r = results[0]

    # Annotated image (BGR numpy -> RGB PIL)
    annotated = Image.fromarray(r.plot()[..., ::-1])

    n = len(r.boxes)
    if n == 0:
        return annotated, None

    m_per_px = RESOLUTIONS.get(camera)
    rows = []
    for i, box in enumerate(r.boxes, 1):
        c = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        w, h = x2 - x1, y2 - y1
        row = {"#": i, "Confidence": round(c, 2), "Box size (px)": f"{w:.0f}×{h:.0f}"}
        if m_per_px:
            diam_m = ((w + h) / 2) * m_per_px
            row["Approx. diameter"] = (
                f"~{diam_m:,.0f} m" if diam_m < 1000 else f"~{diam_m / 1000:.2f} km"
            )
        rows.append(row)

    return annotated, rows


st.set_page_config(page_title="Lunar Crater Detection", page_icon="🌕", layout="wide")

st.title("🌕 Lunar Crater Detection — Chandrayaan-2")
st.markdown(
    "YOLOv11 object detector with an ImageNet-pretrained ResNet101 (timm) backbone, "
    "trained on ISRO Chandrayaan-2 OHRC and TMC-2 imagery. "
    "Upload a lunar image and adjust the confidence slider."
)

model = load_model()

col_controls, col_output = st.columns([1, 2])

with col_controls:
    uploaded = st.file_uploader("Lunar surface image", type=["png", "jpg", "jpeg"])
    conf = st.slider("Confidence threshold", min_value=0.05, max_value=0.9, value=0.25, step=0.05)
    camera = st.radio("Source camera (for size estimates)", list(RESOLUTIONS.keys()), index=0)

with col_output:
    if uploaded is None:
        st.info("Upload an image to run detection.")
    else:
        image = Image.open(uploaded).convert("RGB")
        annotated, rows = detect(model, image, conf, camera)
        st.image(annotated, caption="Detections", use_container_width=True)

        if rows is None:
            st.warning("No craters detected at this confidence threshold.")
        else:
            st.markdown(f"**{len(rows)} crater(s) detected**")
            st.table(rows)
