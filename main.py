import io
import os
import math
import base64
import concurrent.futures

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODELS_DIR = os.getenv("MODELS_DIR", "models")

MODEL_PATHS = {
    "yolo":           os.path.join(MODELS_DIR, "yolo-model.tflite"),
    "segmentation":   os.path.join(MODELS_DIR, "u2net_160_float32.tflite"),
    "grain_vs_other": os.path.join(MODELS_DIR, "rice_vs_other_v5_mobilenetv3.tflite"),
    "brokenness":     os.path.join(MODELS_DIR, "brokenness_v3_mobilenetv3.tflite"),
    "other_matter":   os.path.join(MODELS_DIR, "v3_other_matter_mobilenetv3.tflite"),
    "contrasting":    os.path.join(MODELS_DIR, "contrasting_type_v3_mobilenetv3.tflite"),
    "defective":      os.path.join(MODELS_DIR, "defectives-v3_mobilenetv3.tflite"),
}

YOLO_INPUT_SIZE    = (1024, 1024)
SEG_INPUT_SIZE     = (160, 160)
CLS_INPUT_SIZE     = (224, 224)
NMS_CONF_THRESHOLD = 0.8
NMS_IOU_THRESHOLD  = 0.9
SEG_THRESHOLD      = 0.35

try:
    import tensorflow.lite as tflite
except ImportError:
    from tflite_runtime import interpreter as tflite

app = FastAPI(title="RiceWise Analysis API")

# ──────────────────────────────────────────────
# MODEL CACHE  (load once, reuse across requests)
# ──────────────────────────────────────────────
_interpreter_cache: dict[str, tflite.Interpreter] = {}

def get_interpreter(model_key: str) -> tflite.Interpreter:
    if model_key not in _interpreter_cache:
        path = MODEL_PATHS[model_key]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        interp = tflite.Interpreter(model_path=path, num_threads=4)
        interp.allocate_tensors()
        _interpreter_cache[model_key] = interp
    return _interpreter_cache[model_key]


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ──────────────────────────────────────────────
# DRAWING HELPERS  (OpenCV)
# ──────────────────────────────────────────────
_PALETTE_BGR = [
    (  0, 200,   0),   # green
    (  0,   0, 220),   # blue
    (  0, 150, 220),   # amber  → BGR
    (180,   0, 180),   # purple
    (200, 200,   0),   # cyan   → BGR
    (  0,  80, 255),   # orange → BGR
    (220, 100, 100),   # cornflower → BGR
    (100, 160,   0),   # teal   → BGR
    (100,  50, 220),   # rose   → BGR
    ( 50, 160, 160),   # olive  → BGR
]
_label_color_map: dict[str, tuple] = {}

def _get_color(label: str) -> tuple:
    if label not in _label_color_map:
        _label_color_map[label] = _PALETTE_BGR[len(_label_color_map) % len(_PALETTE_BGR)]
    return _label_color_map[label]


def draw_boxes_cv2(
    image: np.ndarray,
    boxes_px: list[tuple[int, int, int, int]],   # [(x1,y1,x2,y2), …]
    labels: list[str],
    confidences: list[float],
) -> np.ndarray:
    """
    Draw bounding boxes + label chips on a BGR OpenCV image.
    Matches draw_results / draw_results_for_stage from the notebook.
    """
    vis = image.copy()
    for (x1, y1, x2, y2), label, conf in zip(boxes_px, labels, confidences):
        color = _get_color(label)
        text  = f"{label} {conf:.2f}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def _cv2_to_base64(image: np.ndarray, ext: str = ".jpg") -> str:
    """Encode a BGR OpenCV image to a base64 string."""
    success, buf = cv2.imencode(ext, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf).decode("utf-8")


# ──────────────────────────────────────────────
# STAGE 1 — YOLO DETECTION
# ──────────────────────────────────────────────
def run_detection(image: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int,int,int,int]]]:
    """
    image : BGR uint8 ndarray
    Returns:
        crops — list of BGR crops, one per detected grain
        boxes — list of (x1, y1, x2, y2) pixel coords in the *original* image
    """
    interp  = get_interpreter("yolo")
    inp_det = interp.get_input_details()
    out_det = interp.get_output_details()

    h, w = image.shape[:2]

    # Pre-process — matches detect_and_crop_yolo in notebook
    rgb_resized = cv2.cvtColor(
        cv2.resize(image, YOLO_INPUT_SIZE), cv2.COLOR_BGR2RGB
    ).astype(np.float32) / 255.0
    tensor = np.expand_dims(rgb_resized, axis=0)

    interp.set_tensor(inp_det[0]["index"], tensor)
    interp.invoke()

    yolo_output = interp.get_tensor(out_det[0]["index"])
    output = np.transpose(yolo_output[0])   # (N, 5) — cx,cy,bw,bh,conf

    raw_boxes, scores = [], []
    for det in output:
        cx, cy, bw, bh, conf = det
        if conf > NMS_CONF_THRESHOLD:
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            raw_boxes.append([x1, y1, x2, y2])
            scores.append(float(conf))

    if not raw_boxes:
        return [], []

    # NMS via OpenCV — matches notebook
    indices = cv2.dnn.NMSBoxes(raw_boxes, scores, NMS_CONF_THRESHOLD, NMS_IOU_THRESHOLD)

    crops, boxes = [], []
    if len(indices) > 0:
        for i in indices.flatten():
            x1, y1, x2, y2 = raw_boxes[i]
            x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
            y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
                boxes.append((x1, y1, x2, y2))

    return crops, boxes


# ──────────────────────────────────────────────
# STAGE 2 — U2NET SEGMENTATION
# ──────────────────────────────────────────────
def segment_crop(crop: np.ndarray) -> np.ndarray:
    """
    crop : BGR uint8 ndarray
    Returns a BGR uint8 ndarray with background zeroed out.
    Matches segment_image from the notebook.
    """
    interp  = get_interpreter("segmentation")
    inp_det = interp.get_input_details()
    out_det = interp.get_output_details()

    # Pre-process — resize to 160×160, normalize to [0,1]
    rgb     = cv2.cvtColor(cv2.resize(crop, SEG_INPUT_SIZE), cv2.COLOR_BGR2RGB)
    tensor  = np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)

    interp.set_tensor(inp_det[0]["index"], tensor)
    interp.invoke()

    pred = interp.get_tensor(out_det[0]["index"])[0, :, :, 0]

    pred -= pred.min()
    if pred.max() != 0:
        pred /= pred.max()

    # Resize mask back to original crop size and threshold
    oh, ow = crop.shape[:2]
    mask   = cv2.resize(pred, (ow, oh))
    mask   = (mask > SEG_THRESHOLD).astype(np.uint8) * 255

    # Keep largest connected component — matches notebook
    _, labels_cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if len(stats) > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean_mask    = np.zeros_like(mask)
        clean_mask[labels_cc == largest_label] = 255
    else:
        clean_mask = mask

    kernel     = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
    clean_mask = cv2.dilate(clean_mask, kernel, iterations=1)

    segmented = cv2.bitwise_and(crop, crop, mask=clean_mask)
    return segmented


def run_segmentation(crops: list[np.ndarray]) -> list[np.ndarray]:
    segmented = []
    for crop in crops:
        try:
            segmented.append(segment_crop(crop))
        except Exception as e:
            print(f"Segmentation failed for crop, using raw: {e}")
            segmented.append(crop)
    return segmented


# ──────────────────────────────────────────────
# STAGE 3+ — CLASSIFIERS
# ──────────────────────────────────────────────
def _classify_single(crop: np.ndarray, interp, labels: list[str]) -> tuple[str, float]:
    """
    Classify one crop. Matches classify_image from the notebook.
    Returns (label, confidence).
    """
    inp_det = interp.get_input_details()
    out_det = interp.get_output_details()
    dtype   = inp_det[0]["dtype"]

    rgb     = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, CLS_INPUT_SIZE)

    # Models expect raw [0,255] — preprocessing is baked in
    tensor = resized.astype(np.uint8 if dtype == np.uint8 else np.float32)
    tensor = np.expand_dims(tensor, axis=0)

    interp.set_tensor(inp_det[0]["index"], tensor)
    interp.invoke()

    logits = interp.get_tensor(out_det[0]["index"]).flatten()

    # Sigmoid binary (single output)
    if logits.shape[0] == 1:
        raw  = float(logits[0])
        prob = 1.0 / (1.0 + math.exp(-raw)) if (raw < 0 or raw > 1) else raw
        idx        = 1 if prob >= 0.5 else 0
        confidence = prob if idx == 1 else 1.0 - prob
    # Softmax multi-class
    else:
        if logits.min() < 0 or logits.max() > 1:
            e      = np.exp(logits - logits.max())
            logits = e / e.sum()
        idx        = int(np.argmax(logits))
        confidence = float(logits[idx])

    label = labels[idx] if idx < len(labels) else f"class_{idx}"
    return label, confidence


def run_classifier(
    model_key: str,
    model_id: str,
    crops: list[np.ndarray],
    labels: list[str],
) -> dict:
    interp = get_interpreter(model_key)

    counts       = {l: 0 for l in labels}
    conf_sums    = {l: 0.0 for l in labels}
    per_grain_labels      = []
    per_grain_confidences = []

    for crop in crops:
        label, confidence = _classify_single(crop, interp, labels)
        counts[label]    += 1
        conf_sums[label] += confidence
        per_grain_labels.append(label)
        per_grain_confidences.append(confidence)

    avg_confidences = {
        l: (conf_sums[l] / counts[l] if counts[l] > 0 else 0.0)
        for l in labels
    }

    return {
        "modelId":               model_id,
        "totalKernelsAnalyzed":  len(crops),
        "classCounts":           counts,
        "classConfidences":      avg_confidences,
        "perGrainLabels":        per_grain_labels,
        "perGrainConfidences":   per_grain_confidences,
    }


# ──────────────────────────────────────────────
# ANNOTATED IMAGE BUILDER
# ──────────────────────────────────────────────
def _build_annotated_image(
    image: np.ndarray,
    boxes: list[tuple[int,int,int,int]],
    grain_indices: list[int],
    per_grain_labels: list[str],
    per_grain_confidences: list[float],
) -> str:
    """Draw boxes only for grains this model classified. Returns base64 JPEG."""
    selected_boxes = [boxes[i] for i in grain_indices]
    annotated = draw_boxes_cv2(image, selected_boxes, per_grain_labels, per_grain_confidences)
    return _cv2_to_base64(annotated)


# ──────────────────────────────────────────────
# FULL PIPELINE
# ──────────────────────────────────────────────
def full_pipeline(image: np.ndarray) -> dict:
    """
    image : BGR uint8 ndarray (decoded from upload)
    """
    results          = []
    annotated_images = {}

    # ── Stage 1: Detection ──────────────────────
    grain_crops, boxes = run_detection(image)
    print(f"[pipeline] Detection: {len(grain_crops)} grains")

    if not grain_crops:
        return {"results": results, "annotatedImages": annotated_images}

    all_indices = list(range(len(grain_crops)))

    # ── Stage 2: Segmentation ───────────────────
    grain_crops = run_segmentation(grain_crops)
    print("[pipeline] Segmentation done")

    # ── Stage 3: CLS1 — Rice vs Other ───────────
    cls1 = run_classifier(
        model_key="grain_vs_other",
        model_id="grain_vs_other",
        crops=grain_crops,
        labels=["other matter", "rice grain"],
    )
    results.append(cls1)
    print(f"[pipeline] CLS1 rice={cls1['classCounts']['rice grain']}  "
          f"other={cls1['classCounts']['other matter']}")

    annotated_images["model1_grain_vs_other"] = _build_annotated_image(
        image, boxes, all_indices,
        cls1["perGrainLabels"], cls1["perGrainConfidences"],
    )

    if not cls1["perGrainLabels"]:
        return {"results": results, "annotatedImages": annotated_images}

    rice_indices  = [i for i, l in enumerate(cls1["perGrainLabels"]) if l == "rice grain"]
    other_indices = [i for i, l in enumerate(cls1["perGrainLabels"]) if l == "other matter"]
    rice_crops    = [grain_crops[i] for i in rice_indices]
    other_crops   = [grain_crops[i] for i in other_indices]

    # ── Stages 4-6: CLS2 + CLS5 + CLS3 in parallel ──
    cls2 = cls5 = cls3 = None

    def _cls2():
        return run_classifier("brokenness",   "brokenness",       rice_crops,
                              ["brewer", "broken", "not broken"])
    def _cls5():
        return run_classifier("other_matter", "other_matter",     other_crops,
                              ["foreign matter", "paddy rice"])
    def _cls3():
        return run_classifier("contrasting",  "contrasting_type", rice_crops,
                              ["contrasting type", "Indica rice"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        f2 = ex.submit(_cls2) if rice_crops  else None
        f5 = ex.submit(_cls5) if other_crops else None
        f3 = ex.submit(_cls3) if rice_crops  else None

        if f2: cls2 = f2.result()
        if f5: cls5 = f5.result()
        if f3: cls3 = f3.result()

    print("[pipeline] CLS2/CLS3/CLS5 done")

    if cls2:
        results.append(cls2)
        annotated_images["model2_brokenness"] = _build_annotated_image(
            image, boxes, rice_indices,
            cls2["perGrainLabels"], cls2["perGrainConfidences"],
        )

    if cls5:
        results.append(cls5)
        annotated_images["model5_other_matter"] = _build_annotated_image(
            image, boxes, other_indices,
            cls5["perGrainLabels"], cls5["perGrainConfidences"],
        )

    if cls3:
        results.append(cls3)
        annotated_images["model3_contrasting_type"] = _build_annotated_image(
            image, boxes, rice_indices,
            cls3["perGrainLabels"], cls3["perGrainConfidences"],
        )

    # ── Stage 7: CLS4 — Defectives (Indica only) ──
    indica_indices: list[int] = []
    if cls3:
        for i, lbl in enumerate(cls3["perGrainLabels"]):
            if lbl == "Indica rice":
                indica_indices.append(rice_indices[i])

    cls4 = None
    if indica_indices:
        indica_crops = [grain_crops[i] for i in indica_indices]
        cls4 = run_classifier(
            model_key="defective",
            model_id="defective",
            crops=indica_crops,
            labels=["chalky", "damaged", "discolored", "immature", "no defective", "red kernels"],
        )
        results.append(cls4)
        print(f"[pipeline] CLS4 done: {cls4['classCounts']}")

        annotated_images["model4_defectives"] = _build_annotated_image(
            image, boxes, indica_indices,
            cls4["perGrainLabels"], cls4["perGrainConfidences"],
        )

    # ── Final composite — deepest label wins per grain ──
    final_label_map: dict[int, str]   = {}
    final_conf_map:  dict[int, float] = {}

    # Seed with CLS1
    for i, (lbl, conf) in enumerate(zip(cls1["perGrainLabels"], cls1["perGrainConfidences"])):
        final_label_map[i] = lbl
        final_conf_map[i]  = conf

    # Override with CLS2 / CLS5
    if cls2:
        for local_i, (lbl, conf) in enumerate(zip(cls2["perGrainLabels"], cls2["perGrainConfidences"])):
            g = rice_indices[local_i]
            final_label_map[g] = lbl
            final_conf_map[g]  = conf
    if cls5:
        for local_i, (lbl, conf) in enumerate(zip(cls5["perGrainLabels"], cls5["perGrainConfidences"])):
            g = other_indices[local_i]
            final_label_map[g] = lbl
            final_conf_map[g]  = conf

    # Override with CLS3
    if cls3:
        for local_i, (lbl, conf) in enumerate(zip(cls3["perGrainLabels"], cls3["perGrainConfidences"])):
            g = rice_indices[local_i]
            final_label_map[g] = lbl
            final_conf_map[g]  = conf

    # Override with CLS4 (deepest — wins last)
    if indica_indices and cls4:
        for local_i, (lbl, conf) in enumerate(zip(cls4["perGrainLabels"], cls4["perGrainConfidences"])):
            g = indica_indices[local_i]
            final_label_map[g] = lbl
            final_conf_map[g]  = conf

    final_labels = [final_label_map[i] for i in range(len(grain_crops))]
    final_confs  = [final_conf_map[i]  for i in range(len(grain_crops))]

    annotated_images["final"] = _build_annotated_image(
        image, boxes, all_indices, final_labels, final_confs,
    )

    return {"results": results, "annotatedImages": annotated_images}


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        data = await file.read()
        arr  = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)   # BGR uint8
        if image is None:
            raise ValueError("imdecode returned None")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    try:
        pipeline_output = full_pipeline(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    return JSONResponse(content=pipeline_output)