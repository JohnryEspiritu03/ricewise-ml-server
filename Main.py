import io
import os
import math
import concurrent.futures
from typing import Optional

import numpy as np
from PIL import Image
import tensorflow.lite as tflite
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODELS_DIR = os.getenv("MODELS_DIR", "models")

MODEL_PATHS = {
    "yolo":           os.path.join(MODELS_DIR, "yolo-model-96.tflite"),
    "segmentation":   os.path.join(MODELS_DIR, "u2net_160_float32.tflite"),
    "grain_vs_other": os.path.join(MODELS_DIR, "rice_vs_other_v2_mobilenetv3.tflite"),
    "brokenness":     os.path.join(MODELS_DIR, "brokenness_v2_mobilenetv3.tflite"),
    "other_matter":   os.path.join(MODELS_DIR, "v2_other_matter_mobilenetv3.tflite"),
    "contrasting":    os.path.join(MODELS_DIR, "contrasting_type_mobilenetv3.tflite"),
    "defective":      os.path.join(MODELS_DIR, "defectives-v2_mobilenetv3.tflite"),
}

YOLO_INPUT_SIZE    = 1024
SEG_INPUT_SIZE     = 160
CLS_INPUT_SIZE     = 224
NMS_CONF_THRESHOLD = 0.8
NMS_IOU_THRESHOLD  = 0.9
SEG_THRESHOLD      = 0.35

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
def _iou(a: list, b: list) -> float:
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    denom  = a_area + b_area - inter
    return inter / denom if denom > 0 else 0.0


def apply_nms(raw_output: np.ndarray, conf_thresh: float, iou_thresh: float) -> list:
    """raw_output shape: [5, 21504]"""
    boxes = []
    for i in range(21504):
        conf = float(raw_output[4][i])
        if conf > conf_thresh:
            boxes.append([
                float(raw_output[0][i]),
                float(raw_output[1][i]),
                float(raw_output[2][i]),
                float(raw_output[3][i]),
                conf,
            ])
    boxes.sort(key=lambda x: x[4], reverse=True)
    result = []
    while boxes:
        best = boxes.pop(0)
        result.append(best)
        boxes = [b for b in boxes if _iou(best, b) <= iou_thresh]
    return result


def pil_to_nhwc(image: Image.Image, size: int, normalize: bool = True) -> np.ndarray:
    """Resize → RGB float32 [1, H, W, 3].  normalize=True → /255, False → raw 0-255."""
    img = image.resize((size, size), Image.BILINEAR).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    if normalize:
        arr /= 255.0
    return arr[np.newaxis]  # [1, H, W, 3]


def softmax(x: list) -> list:
    m = max(x)
    exps = [math.exp(v - m) for v in x]
    s = sum(exps)
    return [v / s for v in exps]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ──────────────────────────────────────────────
# STAGE 1 — YOLO DETECTION
# ──────────────────────────────────────────────
def run_detection(image: Image.Image) -> list[Image.Image]:
    interp  = get_interpreter("yolo")
    inp_det = interp.get_input_details()
    out_det = interp.get_output_details()

    tensor = pil_to_nhwc(image, YOLO_INPUT_SIZE)
    interp.set_tensor(inp_det[0]['index'], tensor)
    interp.invoke()

    raw = interp.get_tensor(out_det[0]['index'])[0]  # [5, 21504]
    boxes = apply_nms(raw, NMS_CONF_THRESHOLD, NMS_IOU_THRESHOLD)

    w, h = image.size
    crops = []
    for box in boxes:
        x = int((box[0] - box[2] / 2) * w)
        y = int((box[1] - box[3] / 2) * h)
        bw = int(box[2] * w)
        bh = int(box[3] * h)
        x  = max(0, min(x, w - 1))
        y  = max(0, min(y, h - 1))
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))
        crops.append(image.crop((x, y, x + bw, y + bh)))

    return crops


# ──────────────────────────────────────────────
# STAGE 2 — U2NET SEGMENTATION
# ──────────────────────────────────────────────
def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """BFS to keep the largest white blob. mask is H×W uint8."""
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    best_pixels = []

    for start_y in range(h):
        for start_x in range(w):
            if mask[start_y, start_x] == 0 or visited[start_y, start_x]:
                continue
            # BFS
            queue = [(start_x, start_y)]
            visited[start_y, start_x] = 1
            component = []
            while queue:
                cx, cy = queue.pop()
                component.append((cx, cy))
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            if not visited[ny, nx] and mask[ny, nx] > 0:
                                visited[ny, nx] = 1
                                queue.append((nx, ny))
            if len(component) > len(best_pixels):
                best_pixels = component

    result = np.zeros((h, w), dtype=np.uint8)
    for (px, py) in best_pixels:
        result[py, px] = 255
    return result


def _morphological_op(mask: np.ndarray, op: str, k: int = 2) -> np.ndarray:
    h, w  = mask.shape
    out   = np.zeros_like(mask)
    for y in range(k, h - k):
        for x in range(k, w - k):
            patch = mask[y - k:y + k + 1, x - k:x + k + 1]
            out[y, x] = patch.max() if op == "dilate" else patch.min()
    return out


def segment_crop(crop: Image.Image) -> Image.Image:
    interp  = get_interpreter("segmentation")
    inp_det = interp.get_input_details()
    out_det = interp.get_output_details()

    tensor = pil_to_nhwc(crop, SEG_INPUT_SIZE)
    interp.set_tensor(inp_det[0]['index'], tensor)
    interp.invoke()

    pred = interp.get_tensor(out_det[0]['index'])[0, :, :, 0]  # [160, 160]

    # Normalize
    mn, mx = pred.min(), pred.max()
    if mx > mn:
        pred = (pred - mn) / (mx - mn)

    # Threshold → binary mask at SEG_INPUT_SIZE
    mask_small = (pred > SEG_THRESHOLD).astype(np.uint8) * 255

    # Resize to original crop size
    orig_w, orig_h = crop.size
    mask_img = Image.fromarray(mask_small).resize((orig_w, orig_h), Image.NEAREST)
    mask_np  = np.array(mask_img)

    # Morphological ops
    largest = _keep_largest_component(mask_np)
    closed  = _morphological_op(_morphological_op(largest, "dilate", 2), "erode", 2)
    final   = _morphological_op(closed, "dilate", 2)

    # Apply mask to crop
    crop_arr   = np.array(crop.convert("RGB"))
    masked_arr = np.where(final[:, :, np.newaxis] > 0, crop_arr, 0).astype(np.uint8)
    return Image.fromarray(masked_arr)


def run_segmentation(crops: list[Image.Image]) -> list[Image.Image]:
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
def run_classifier(
    model_key: str,
    model_id: str,
    crops: list[Image.Image],
    labels: list[str],
) -> dict:
    interp  = get_interpreter(model_key)
    inp_det = interp.get_input_details()
    out_det = interp.get_output_details()
    output_size = out_det[0]['shape'][1]

    counts       = {l: 0 for l in labels}
    conf_sums    = {l: 0.0 for l in labels}
    per_grain_labels      = []
    per_grain_confidences = []

    for crop in crops:
        # Preprocess — no /255 normalization for classifiers (raw 0-255 like Dart)
        tensor = pil_to_nhwc(crop, CLS_INPUT_SIZE, normalize=False)

        interp.set_tensor(inp_det[0]['index'], tensor)
        interp.invoke()
        raw = interp.get_tensor(out_det[0]['index'])[0].tolist()

        if output_size == 1:
            raw_val = raw[0]
            prob    = raw_val if 0.0 <= raw_val <= 1.0 else sigmoid(raw_val)
            best_idx   = 1 if prob >= 0.5 else 0
            confidence = prob if best_idx == 1 else 1.0 - prob
        else:
            mn, mx = min(raw), max(raw)
            probs  = softmax(raw) if (mn < 0 or mx > 1) else raw
            best_idx   = int(np.argmax(probs))
            confidence = probs[best_idx]

        label = labels[best_idx]
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
# FULL PIPELINE
# ──────────────────────────────────────────────
def full_pipeline(image: Image.Image) -> list[dict]:
    results = []

    # ── Stage 1: Detection ──────────────────────
    grain_crops = run_detection(image)
    print(f"[pipeline] Detection: {len(grain_crops)} grains")

    if not grain_crops:
        return results

    # ── Stage 2: Segmentation ───────────────────
    grain_crops = run_segmentation(grain_crops)
    print(f"[pipeline] Segmentation done")

    # ── Stage 3: CLS1 — Rice vs Other ───────────
    cls1 = run_classifier(
        model_key="grain_vs_other",
        model_id="grain_vs_other",
        crops=grain_crops,
        labels=["other matter", "rice grain"],
    )
    results.append(cls1)
    print(f"[pipeline] CLS1 rice={cls1['classCounts']['rice grain']}  other={cls1['classCounts']['other matter']}")

    if not cls1["perGrainLabels"]:
        return results

    rice_indices  = [i for i, l in enumerate(cls1["perGrainLabels"]) if l == "rice grain"]
    other_indices = [i for i, l in enumerate(cls1["perGrainLabels"]) if l == "other matter"]
    rice_crops    = [grain_crops[i] for i in rice_indices]
    other_crops   = [grain_crops[i] for i in other_indices]

    # ── Stages 4-6: CLS2 + CLS5 + CLS3 in parallel ──
    cls2 = cls5 = cls3 = None

    def _cls2():
        return run_classifier("brokenness",   "brokenness",      rice_crops,  ["brewer", "broken", "not broken"])
    def _cls5():
        return run_classifier("other_matter", "other_matter",    other_crops, ["foreign matter", "paddy rice"])
    def _cls3():
        return run_classifier("contrasting",  "contrasting_type", rice_crops, ["contrasting type", "Indica rice"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        f2 = ex.submit(_cls2) if rice_crops  else None
        f5 = ex.submit(_cls5) if other_crops else None
        f3 = ex.submit(_cls3) if rice_crops  else None

        if f2: cls2 = f2.result()
        if f5: cls5 = f5.result()
        if f3: cls3 = f3.result()

    if cls2: results.append(cls2)
    if cls5: results.append(cls5)
    if cls3: results.append(cls3)
    print(f"[pipeline] CLS2/CLS3/CLS5 done")

    # ── Stage 7: CLS4 — Defectives (Indica only) ─
    indica_indices: list[int] = []
    if cls3:
        for i, lbl in enumerate(cls3["perGrainLabels"]):
            if lbl == "Indica rice":
                indica_indices.append(rice_indices[i])

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

    return results


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
        data  = await file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    try:
        results = full_pipeline(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    return JSONResponse(content={"results": results})