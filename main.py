"""
PyQt6 webcam app: Face -> Emotion (7 classes) + Age (9 bins) + Gender (2 classes) using three ViT classifiers.

How it works:
- OpenCV reads webcam frames.
- YOLO (Ultralytics) detects faces; if YOLO is unavailable or weights are not provided, Haar cascade is used as fallback.
- Each face is cropped -> HuggingFace ViT processor -> three torch models -> labels.
- Results are drawn on the frame and displayed in PyQt6.

You need:
  pip install pyqt6 opencv-python torch torchvision transformers ultralytics

Run:
  python main.py --emotion-weights /path/to/vit_emotion/best.pt --age-weights /path/to/vit_age/best.pt --gender-weights /path/to/vit_gender/best.pt

Notes:
- If you used a different ViT backbone during training, pass it via --backbone (must match training).
- First run may download the backbone weights from HuggingFace.
"""

from __future__ import annotations

import argparse
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QLineEdit, QFileDialog, QComboBox, QCheckBox, QSpinBox, QMessageBox, QGroupBox, QFormLayout
)


# -------------------------
# Labels (must match training)
# -------------------------

EMO_CLASSES_7 = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

AGE_BINS_9 = [(0, 2), (3, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 120)]
AGE_NAMES_9 = [f"{a}-{b}" for a, b in AGE_BINS_9]

# Gender labels — порядок має збігатися з тренуванням ноутбука:
# UTK_GENDER_TO_ID = {0: 1, 1: 0}, GENDER_NAMES = ["woman", "man"]
GEN_CLASSES_2 = ["woman", "man"]


# -------------------------
# Model definition (same as in your notebook)
# -------------------------

class ViTClassifier(nn.Module):
    def __init__(self, backbone: str, n_classes: int):
        super().__init__()
        self.vit = AutoModel.from_pretrained(backbone)
        h = self.vit.config.hidden_size
        self.head = nn.Linear(h, n_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.vit(pixel_values=pixel_values)
        cls = out.last_hidden_state[:, 0]
        return self.head(cls)


@dataclass
class ModelsBundle:
    device: torch.device
    processor: AutoImageProcessor
    emotion_model: ViTClassifier
    age_model: ViTClassifier
    gender_model: Optional[ViTClassifier]
    backbone: str
    emotion_labels: List[str]
    age_labels: List[str]
    gender_labels: List[str]



def _unwrap_state_dict(obj):
    """Best-effort unwrap for checkpoints saved in various formats."""
    # Most common: raw state_dict (OrderedDict[str, Tensor])
    if isinstance(obj, dict):
        # If it looks like a raw state_dict (tensor-like values)
        if obj and all(hasattr(v, "shape") for v in obj.values()):
            return obj
        # Common wrappers
        for k in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if k in obj and isinstance(obj[k], dict):
                inner = obj[k]
                if inner and all(hasattr(v, "shape") for v in inner.values()):
                    return inner
        # Sometimes saved as {"epoch":..., "model": {"state_dict": ...}}
        for v in obj.values():
            if isinstance(v, dict):
                inner = _unwrap_state_dict(v)
                if inner is not None:
                    return inner
    return None



def _strip_if_all(state_dict, prefix: str):
    """Strip prefix only if *all* keys start with it."""
    if state_dict and all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict

def _remap_classifier_to_head(state_dict):
    """Support checkpoints saved with classifier/fc head names."""
    # classifier -> head
    if "classifier.weight" in state_dict and "head.weight" not in state_dict:
        state_dict["head.weight"] = state_dict.pop("classifier.weight")
        if "classifier.bias" in state_dict:
            state_dict["head.bias"] = state_dict.pop("classifier.bias")
    # fc -> head
    if "fc.weight" in state_dict and "head.weight" not in state_dict:
        state_dict["head.weight"] = state_dict.pop("fc.weight")
        if "fc.bias" in state_dict:
            state_dict["head.bias"] = state_dict.pop("fc.bias")
    return state_dict

def _strip_prefix(state_dict, prefix: str):
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _infer_head_out_features(state_dict):
    """Infer number of classes from common classifier head names."""
    for key in ("head.weight", "classifier.weight", "fc.weight"):
        if key in state_dict and hasattr(state_dict[key], "shape"):
            return int(state_dict[key].shape[0])
    # fallback: look for something that endswith '.weight' and contains 'head'
    for k, v in state_dict.items():
        if k.endswith(".weight") and ("head" in k or "classifier" in k) and hasattr(v, "shape"):
            return int(v.shape[0])
    return None


def load_vit_classifier(
    weights_path: Path,
    backbone: str,
    n_classes: Optional[int],
    device: torch.device,
) -> ViTClassifier:
    """Load a ViTClassifier from a checkpoint.

    - If n_classes is None (or <= 0), it will be inferred from the checkpoint head.
    - If the checkpoint was saved with DataParallel, 'module.' prefix is removed.
    """
    ckpt = torch.load(str(weights_path), map_location="cpu")
    sd = _unwrap_state_dict(ckpt)
    if sd is None:
        raise RuntimeError(
            f"Checkpoint format not recognized for: {weights_path}. "
            "Expected a PyTorch state_dict or a dict containing state_dict/model_state_dict."
        )

    sd = _strip_prefix(sd, "module.")
    sd = _strip_if_all(sd, "_orig_mod.")
    sd = _strip_if_all(sd, "model.")
    sd = _strip_if_all(sd, "net.")
    sd = _remap_classifier_to_head(sd)

    inferred = _infer_head_out_features(sd)
    if n_classes is None or int(n_classes) <= 0:
        if inferred is None:
            raise RuntimeError(
                f"Cannot infer number of classes from checkpoint head for: {weights_path}. "
                "Pass --emotion-num-classes / --age-num-classes."
            )
        n_classes = inferred
    else:
        n_classes = int(n_classes)
        if inferred is not None and inferred != n_classes:
            # Most likely user swapped weights (e.g., age weights loaded as emotion).
            # We prefer the checkpoint value to avoid shape mismatch.
            print(
                f"[WARN] {weights_path.name}: checkpoint head has {inferred} outputs, "
                f"but you requested {n_classes}. Using {inferred} (checkpoint)."
            )
            n_classes = inferred

    model = ViTClassifier(backbone, n_classes=n_classes)

    # strict=True to catch backbone mismatch early
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()

    # small speed-up on GPU
    if device.type == "cuda":
        model.half()

    return model



def load_models(
    emotion_weights: Path,
    age_weights: Path,
    backbone: str,
    gender_weights: Optional[Path] = None,
    emotion_num_classes: Optional[int] = None,
    age_num_classes: Optional[int] = None,
    gender_num_classes: Optional[int] = None,
) -> ModelsBundle:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        processor = AutoImageProcessor.from_pretrained(backbone)
    except Exception as e:
        raise RuntimeError(
            f"Cannot load image processor/backbone: {backbone}. "
            "If you are offline, download the model once or set --backbone to a local folder."
        ) from e

    emo = load_vit_classifier(emotion_weights, backbone, n_classes=emotion_num_classes, device=device)
    age = load_vit_classifier(age_weights, backbone, n_classes=age_num_classes, device=device)
    gen: Optional[ViTClassifier] = None
    if gender_weights is not None:
        gen = load_vit_classifier(gender_weights, backbone, n_classes=gender_num_classes, device=device)

    # Build label maps safely
    emo_n = int(getattr(emo.head, "out_features", len(EMO_CLASSES_7)))
    age_n = int(getattr(age.head, "out_features", len(AGE_NAMES_9)))
    gen_n = int(getattr(gen.head, "out_features", len(GEN_CLASSES_2))) if gen is not None else 0

    # Auto-route: розкладаємо моделі по слотах за кількістю виходів head
    # (7 -> emotion, 9 -> age, 2 -> gender), якщо користувач переплутав файли.
    expected = {len(EMO_CLASSES_7): "emo", len(AGE_NAMES_9): "age", len(GEN_CLASSES_2): "gen"}
    loaded = [(emo, emo_n, "emo"), (age, age_n, "age")]
    if gen is not None:
        loaded.append((gen, gen_n, "gen"))

    # розподіл за виявленим розміром head
    routed = {"emo": None, "age": None, "gen": None}
    used = [False] * len(loaded)
    for i, (m, n, slot) in enumerate(loaded):
        target = expected.get(n)
        if target is not None and routed[target] is None:
            routed[target] = (m, n)
            used[i] = True
    # моделі, що не змогли «знайти свій слот», лишаються там, де їх передав користувач
    for i, (m, n, slot) in enumerate(loaded):
        if not used[i] and routed[slot] is None:
            routed[slot] = (m, n)

    new_emo = routed["emo"] or (emo, emo_n)
    new_age = routed["age"] or (age, age_n)
    new_gen = routed["gen"]

    if (new_emo[0] is not emo) or (new_age[0] is not age) or (gen is not None and new_gen and new_gen[0] is not gen):
        print("[WARN] Виявлено невідповідність слотів і ваг — авто-перерозподіл за розміром head.")

    emo, emo_n = new_emo
    age, age_n = new_age
    if gen is not None and new_gen is not None:
        gen, gen_n = new_gen

    emo_labels = EMO_CLASSES_7 if emo_n == len(EMO_CLASSES_7) else [f"emo_{i}" for i in range(emo_n)]
    age_labels = AGE_NAMES_9 if age_n == len(AGE_NAMES_9) else [f"age_{i}" for i in range(age_n)]
    if gen is not None:
        gen_labels = GEN_CLASSES_2 if gen_n == len(GEN_CLASSES_2) else [f"gen_{i}" for i in range(gen_n)]
    else:
        gen_labels = []

    if emo_n != len(EMO_CLASSES_7):
        print(
            f"[WARN] Emotion model outputs {emo_n} classes, but default label list has {len(EMO_CLASSES_7)}. "
            "Using generic labels emo_0.."
        )
    if age_n != len(AGE_NAMES_9):
        print(
            f"[WARN] Age model outputs {age_n} classes, but default label list has {len(AGE_NAMES_9)}. "
            "Using generic labels age_0.."
        )
    if gen is not None and gen_n != len(GEN_CLASSES_2):
        print(
            f"[WARN] Gender model outputs {gen_n} classes, but default label list has {len(GEN_CLASSES_2)}. "
            "Using generic labels gen_0.."
        )

    return ModelsBundle(
        device=device,
        processor=processor,
        emotion_model=emo,
        age_model=age,
        gender_model=gen,
        backbone=backbone,
        emotion_labels=emo_labels,
        age_labels=age_labels,
        gender_labels=gen_labels,
    )


def make_haar_face_detector() -> cv2.CascadeClassifier:
    """Create and validate OpenCV Haar face detector (fallback)."""
    xml_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(xml_path)
    if detector.empty():
        raise RuntimeError(f"Failed to load Haar cascade from: {xml_path}")
    return detector


class YoloFaceDetector:
    """YOLO-based face detector (Ultralytics).

    Wraps an Ultralytics YOLO model so it can be used as the primary face
    detector for the application. Falls back gracefully: if the `ultralytics`
    package or model weights are not available, this class raises on init and
    the caller is expected to fall back to Haar.
    """

    def __init__(self, weights: str = "yolo26n.pt", conf: float = 0.35,
                 iou: float = 0.5, device: Optional[str] = None) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Ultralytics is not installed. Install via `pip install ultralytics`."
            ) from e

        self.model = YOLO(weights)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device  # None → auto (cuda if available else cpu)

    def detect(self, frame_bgr: np.ndarray,
               min_size: int = 60) -> List[Tuple[int, int, int, int]]:
        """Return list of (x, y, w, h) face boxes (same format as Haar)."""
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        boxes_out: List[Tuple[int, int, int, int]] = []
        if not results:
            return boxes_out
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return boxes_out
        xyxy = r.boxes.xyxy.detach().cpu().numpy()
        for (x1, y1, x2, y2) in xyxy:
            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)
            if w >= min_size and h >= min_size:
                boxes_out.append((x, y, w, h))
        return boxes_out


def make_face_detector(yolo_weights: Optional[str] = None,
                       yolo_conf: float = 0.35):
    """Factory: try to build YOLO detector; on failure fall back to Haar.

    Returns the detector object. The caller should use `detect_faces` which
    knows how to dispatch to either backend.
    """
    if yolo_weights:
        try:
            det = YoloFaceDetector(weights=yolo_weights, conf=yolo_conf)
            print(f"[INFO] Using YOLO face detector: {yolo_weights}")
            return det
        except Exception as e:
            print(f"[WARN] YOLO init failed ({e}); falling back to Haar.")
    det = make_haar_face_detector()
    print("[INFO] Using Haar cascade face detector (fallback).")
    return det


def detect_faces(detector, frame_bgr: np.ndarray,
                 min_size: int = 60) -> List[Tuple[int, int, int, int]]:
    """Unified detection entry point. Works with both YOLO and Haar."""
    # YOLO path
    if isinstance(detector, YoloFaceDetector):
        return detector.detect(frame_bgr, min_size=min_size)
    # Haar path (cv2.CascadeClassifier)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def clamp_box(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def expand_box(x: int, y: int, w: int, h: int, W: int, H: int, frac: float = 0.18) -> Tuple[int, int, int, int]:
    dx = int(w * frac)
    dy = int(h * frac)
    return clamp_box(x - dx, y - dy, w + 2 * dx, h + 2 * dy, W, H)


# -------------------------
# Inference helpers
# -------------------------

@torch.no_grad()
def infer_face(models: ModelsBundle, face_bgr: np.ndarray) -> Tuple[str, float, str, float, Optional[str], Optional[float]]:
    """
    Returns:
      (emotion_label, emotion_prob, age_label, age_prob, gender_label, gender_prob)
      gender_label / gender_prob будуть None, якщо гендерну модель не завантажено.
    """
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    px = models.processor(images=rgb, return_tensors="pt")["pixel_values"]

    # match half precision on GPU
    if models.device.type == "cuda":
        px = px.half()
    px = px.to(models.device)

    emo_logits = models.emotion_model(px)
    age_logits = models.age_model(px)

    emo_prob = torch.softmax(emo_logits, dim=1).squeeze(0)
    age_prob = torch.softmax(age_logits, dim=1).squeeze(0)

    emo_id = int(torch.argmax(emo_prob).item())
    age_id = int(torch.argmax(age_prob).item())

    emo_label = models.emotion_labels[emo_id] if emo_id < len(models.emotion_labels) else f"emo_{emo_id}"
    age_label = models.age_labels[age_id] if age_id < len(models.age_labels) else f"age_{age_id}"

    gen_label: Optional[str] = None
    gen_p: Optional[float] = None
    if models.gender_model is not None:
        gen_logits = models.gender_model(px)
        gen_prob = torch.softmax(gen_logits, dim=1).squeeze(0)
        gen_id = int(torch.argmax(gen_prob).item())
        gen_label = models.gender_labels[gen_id] if gen_id < len(models.gender_labels) else f"gen_{gen_id}"
        gen_p = float(gen_prob[gen_id].item())

    return emo_label, float(emo_prob[emo_id].item()), age_label, float(age_prob[age_id].item()), gen_label, gen_p


def draw_label(frame: np.ndarray, x: int, y: int, lines: List[str]) -> None:
    # background box
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 2

    # compute text size
    sizes = [cv2.getTextSize(t, font, scale, thick)[0] for t in lines]
    w = max(s[0] for s in sizes) if sizes else 0
    h = sum(s[1] + 6 for s in sizes) + 6

    x0 = x
    y0 = max(0, y - h - 6)

    cv2.rectangle(frame, (x0, y0), (x0 + w + 10, y0 + h), (0, 0, 0), -1)

    yy = y0 + 18
    for t in lines:
        cv2.putText(frame, t, (x0 + 5, yy), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        yy += 22


def process_frame(
    frame_bgr: np.ndarray,
    models: Optional[ModelsBundle],
    detector: cv2.CascadeClassifier,
    do_infer: bool,
    max_faces: int,
    min_face_size: int,
    target_width: int = 0,
) -> np.ndarray:
    """Один кадр -> детекція облич -> інференс -> малюємо рамки/підписи. Повертає BGR."""
    frame = frame_bgr
    if target_width and frame.shape[1] > target_width:
        r = target_width / frame.shape[1]
        frame = cv2.resize(frame, (target_width, int(frame.shape[0] * r)), interpolation=cv2.INTER_AREA)

    if not do_infer or models is None:
        return frame

    faces = detect_faces(detector, frame, min_size=min_face_size)
    faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[:max_faces]
    H, W = frame.shape[:2]
    for (x, y, w, h) in faces:
        x, y, w, h = expand_box(x, y, w, h, W, H, frac=0.18)
        if w <= 0 or h <= 0:
            continue
        face = frame[y:y + h, x:x + w]
        if face.size == 0:
            continue
        try:
            emo, emo_p, age, age_p, gen, gen_p = infer_face(models, face)
            label_lines = [
                f"emo: {emo} ({emo_p:.2f})",
                f"age: {age} ({age_p:.2f})",
            ]
            if gen is not None and gen_p is not None:
                label_lines.append(f"gen: {gen} ({gen_p:.2f})")
        except Exception as e:
            label_lines = [f"model error: {type(e).__name__}"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        draw_label(frame, x, y, label_lines)
    return frame


def bgr_to_qimage(frame_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


# -------------------------
# Worker thread (camera + inference)
# -------------------------

class VideoWorker(QThread):
    frame_ready = pyqtSignal(QImage)
    fps_ready = pyqtSignal(float)
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._running = False

        # Джерело: "camera" або "video"
        self.source_type: str = "camera"
        self.camera_index = 0
        self.video_path: Optional[str] = None

        self.target_width = 960
        self.min_face_size = 60
        self.max_faces = 3
        self.do_infer = True

        self.models: Optional[ModelsBundle] = None
        # Try YOLO first; fall back to Haar if unavailable.
        # Use lightweight YOLO weights by default; can be overridden via --yolo-weights.
        self.detector = make_face_detector(yolo_weights="yolo26n.pt")

    def configure(
        self,
        source_type: str,
        target_width: int,
        min_face_size: int,
        max_faces: int,
        do_infer: bool,
        camera_index: int = 0,
        video_path: Optional[str] = None,
    ) -> None:
        self.source_type = source_type
        self.camera_index = camera_index
        self.video_path = video_path
        self.target_width = target_width
        self.min_face_size = min_face_size
        self.max_faces = max_faces
        self.do_infer = do_infer

    def set_models(self, models: Optional[ModelsBundle]) -> None:
        self.models = models

    def stop(self) -> None:
        self._running = False


    def _open_camera_with_fallback(self) -> Optional[cv2.VideoCapture]:
        """Open camera with OS-appropriate backends and index fallback.

        Returns an opened cv2.VideoCapture or None.
        """
        candidates = [int(self.camera_index)] + [i for i in range(0, 6) if i != int(self.camera_index)]

        plat = sys.platform.lower()
        backends: List[int] = []

        def add_backend(name: str):
            if hasattr(cv2, name):
                backends.append(getattr(cv2, name))

        if plat.startswith("linux"):
            add_backend("CAP_V4L2")
            add_backend("CAP_GSTREAMER")
            add_backend("CAP_ANY")
        elif plat.startswith("win"):
            add_backend("CAP_DSHOW")
            add_backend("CAP_MSMF")
            add_backend("CAP_ANY")
        elif plat == "darwin":
            add_backend("CAP_AVFOUNDATION")
            add_backend("CAP_ANY")
        else:
            add_backend("CAP_ANY")

        seen = set()
        backends = [b for b in backends if not (b in seen or seen.add(b))]

        last_err: Optional[Exception] = None

        for idx in candidates:
            for be in backends:
                try:
                    cap = cv2.VideoCapture(int(idx), int(be))
                    if not cap.isOpened():
                        cap.release()
                        continue

                    # Try to reduce latency (best-effort)
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass

                    ok, frame = cap.read()
                    if ok and frame is not None and frame.size > 0:
                        self.camera_index = int(idx)
                        return cap

                    cap.release()
                except Exception as e:
                    last_err = e

        if last_err is not None:
            print(f"[WARN] Camera open failed: {type(last_err).__name__}: {last_err}")
        return None

    def _open_video_file(self) -> Optional[cv2.VideoCapture]:
        if not self.video_path:
            return None
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None
        return cap

    def run(self) -> None:
        if self.source_type == "video":
            cap = self._open_video_file()
            if cap is None:
                self.error.emit(f"Не можу відкрити відеофайл: {self.video_path}")
                return
            # FPS відеофайлу — щоб не програвати швидше за реальне.
            # GIF-контейнери часто не повідомляють FPS — fallback ~15.
            src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if not (1.0 <= src_fps <= 120.0):
                src_fps = 15.0
            frame_period = 1.0 / src_fps
        else:
            cap = self._open_camera_with_fallback()
            if cap is None or not cap.isOpened():
                self.error.emit(f"Не можу відкрити камеру index={self.camera_index}.")
                return
            frame_period = 0.0  # камера сама диктує темп

        self._running = True
        last_t = time.time()
        fps_ema = 0.0

        try:
            while self._running:
                t_read = time.time()
                ok, frame = cap.read()
                if not ok or frame is None:
                    if self.source_type == "video":
                        # Кінець відео — це не помилка
                        break
                    else:
                        self.error.emit("Камера не віддає кадри.")
                        break

                frame = process_frame(
                    frame_bgr=frame,
                    models=self.models,
                    detector=self.detector,
                    do_infer=self.do_infer,
                    max_faces=self.max_faces,
                    min_face_size=self.min_face_size,
                    target_width=self.target_width,
                )

                # FPS
                t = time.time()
                dt = t - last_t
                last_t = t
                inst = 1.0 / dt if dt > 0 else 0.0
                fps_ema = inst if fps_ema == 0.0 else (0.9 * fps_ema + 0.1 * inst)
                self.fps_ready.emit(float(fps_ema))

                self.frame_ready.emit(bgr_to_qimage(frame))

                # Темп відтворення відео (щоб не швидше за оригінал)
                if frame_period > 0:
                    sleep_left = frame_period - (time.time() - t_read)
                    if sleep_left > 0:
                        time.sleep(sleep_left)

        finally:
            cap.release()
            self._running = False


# -------------------------
# UI
# -------------------------

class MainWindow(QMainWindow):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.setWindowTitle("Face Analytics — Камера / Відео / Фото")

        self.worker = VideoWorker()
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.fps_ready.connect(self.on_fps)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_worker_finished)

        self.models: Optional[ModelsBundle] = None

        # Video view
        self.video = QLabel("Натисни Start")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(960, 540)
        self.video.setStyleSheet("background:#111; color:#bbb; border-radius:12px;")

        # Controls
        self.btn_camera = QPushButton("📷 Камера")
        self.btn_video = QPushButton("🎬 Відео")
        self.btn_image = QPushButton("🖼 Фото")
        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setEnabled(False)

        self.btn_camera.clicked.connect(self.start_camera)
        self.btn_video.clicked.connect(self.open_video)
        self.btn_image.clicked.connect(self.open_image)
        self.btn_stop.clicked.connect(self.stop_worker)

        self.fps_lbl = QLabel("FPS: -")
        self.fps_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.chk_infer = QCheckBox("Робити розпізнавання")
        self.chk_infer.setChecked(True)

        self.cam_combo = QComboBox()
        for i in range(0, 5):
            self.cam_combo.addItem(f"Camera {i}", i)
        self.cam_combo.setCurrentIndex(int(args.camera))

        self.spin_width = QSpinBox()
        self.spin_width.setRange(320, 1920)
        self.spin_width.setValue(int(args.width))

        self.spin_minface = QSpinBox()
        self.spin_minface.setRange(30, 400)
        self.spin_minface.setValue(int(args.min_face))

        self.spin_maxfaces = QSpinBox()
        self.spin_maxfaces.setRange(1, 10)
        self.spin_maxfaces.setValue(int(args.max_faces))

        # Живі апдейти параметрів — щоб зміни діяли під час програвання відео/камери
        self.spin_width.valueChanged.connect(
            lambda v: setattr(self.worker, "target_width", int(v))
        )
        self.spin_minface.valueChanged.connect(
            lambda v: setattr(self.worker, "min_face_size", int(v))
        )
        self.spin_maxfaces.valueChanged.connect(
            lambda v: setattr(self.worker, "max_faces", int(v))
        )
        self.chk_infer.toggled.connect(
            lambda v: setattr(self.worker, "do_infer", bool(v))
        )

        # Weights paths
        self.ed_emotion = QLineEdit(str(args.emotion_weights or ""))
        self.ed_age = QLineEdit(str(args.age_weights or ""))
        self.ed_gender = QLineEdit(str(args.gender_weights or ""))
        self.ed_backbone = QLineEdit(str(args.backbone))

        self.btn_browse_emo = QPushButton("...")
        self.btn_browse_age = QPushButton("...")
        self.btn_browse_gen = QPushButton("...")
        self.btn_load = QPushButton("Load models")

        self.btn_browse_emo.clicked.connect(lambda: self.browse_file(self.ed_emotion))
        self.btn_browse_age.clicked.connect(lambda: self.browse_file(self.ed_age))
        self.btn_browse_gen.clicked.connect(lambda: self.browse_file(self.ed_gender))
        self.btn_load.clicked.connect(self.load_models_clicked)

        # Layout
        ctrl_box = QGroupBox("Налаштування")
        form = QFormLayout()
        form.addRow("Камера:", self.cam_combo)
        form.addRow("Ширина кадру (для швидкості):", self.spin_width)
        form.addRow("Min face size:", self.spin_minface)
        form.addRow("Max faces:", self.spin_maxfaces)
        form.addRow("", self.chk_infer)
        form.addRow("Backbone:", self.ed_backbone)

        row_emo = QWidget()
        hl1 = QHBoxLayout(row_emo); hl1.setContentsMargins(0,0,0,0)
        hl1.addWidget(self.ed_emotion, 1); hl1.addWidget(self.btn_browse_emo)
        form.addRow("Emotion weights:", row_emo)

        row_age = QWidget()
        hl2 = QHBoxLayout(row_age); hl2.setContentsMargins(0,0,0,0)
        hl2.addWidget(self.ed_age, 1); hl2.addWidget(self.btn_browse_age)
        form.addRow("Age weights:", row_age)

        row_gen = QWidget()
        hl3 = QHBoxLayout(row_gen); hl3.setContentsMargins(0,0,0,0)
        hl3.addWidget(self.ed_gender, 1); hl3.addWidget(self.btn_browse_gen)
        form.addRow("Gender weights:", row_gen)

        form.addRow("", self.btn_load)
        ctrl_box.setLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_camera)
        btn_row.addWidget(self.btn_video)
        btn_row.addWidget(self.btn_image)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch(1)
        btn_row.addWidget(self.fps_lbl)

        left = QVBoxLayout()
        left.addWidget(self.video, 1)
        left.addLayout(btn_row)

        right = QVBoxLayout()
        right.addWidget(ctrl_box)
        right.addStretch(1)

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.addLayout(left, 3)
        layout.addLayout(right, 2)
        self.setCentralWidget(root)

        # Menu (Quit)
        quit_act = QAction("Quit", self)
        quit_act.triggered.connect(self.close)
        self.menuBar().addAction(quit_act)

        # Auto-load if paths given
        if args.emotion_weights and args.age_weights:
            self.load_models_clicked()

    def browse_file(self, line_edit: QLineEdit) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Select weights (.pt)", "", "PyTorch weights (*.pt *.pth);;All files (*)")
        if p:
            line_edit.setText(p)

    def load_models_clicked(self) -> None:
        emo_p = Path(self.ed_emotion.text()).expanduser()
        age_p = Path(self.ed_age.text()).expanduser()
        gen_text = self.ed_gender.text().strip()
        gen_p: Optional[Path] = Path(gen_text).expanduser() if gen_text else None
        backbone = self.ed_backbone.text().strip()

        if not emo_p.exists():
            QMessageBox.warning(self, "Weights not found", f"Emotion weights not found:\n{emo_p}")
            return
        if not age_p.exists():
            QMessageBox.warning(self, "Weights not found", f"Age weights not found:\n{age_p}")
            return
        if gen_p is not None and not gen_p.exists():
            QMessageBox.warning(self, "Weights not found", f"Gender weights not found:\n{gen_p}")
            return
        if not backbone:
            QMessageBox.warning(self, "Backbone missing", "Backbone is empty.")
            return

        try:
            self.models = load_models(
                emo_p,
                age_p,
                backbone=backbone,
                gender_weights=gen_p,
                emotion_num_classes=(self.args.emotion_num_classes or None),
                age_num_classes=(self.args.age_num_classes or None),
                gender_num_classes=(self.args.gender_num_classes or None),
            )
            self.worker.set_models(self.models)
            dev = str(self.models.device)
            gen_info = (
                f"\nGender classes: {len(self.models.gender_labels)}"
                if self.models.gender_model is not None else "\nGender: not loaded"
            )
            QMessageBox.information(
                self,
                "OK",
                f"Models loaded.\nDevice: {dev}\nBackbone: {self.models.backbone}\n"
                f"Emotion classes: {len(self.models.emotion_labels)}\n"
                f"Age classes: {len(self.models.age_labels)}"
                f"{gen_info}",
            )
        except Exception as e:
            self.models = None
            self.worker.set_models(None)
            QMessageBox.critical(self, "Load error", f"Failed to load models:\n{type(e).__name__}: {e}")

    def _set_source_buttons_enabled(self, enabled: bool) -> None:
        self.btn_camera.setEnabled(enabled)
        self.btn_video.setEnabled(enabled)
        self.btn_image.setEnabled(enabled)
        self.btn_stop.setEnabled(not enabled)

    def _common_worker_params(self) -> dict:
        return dict(
            target_width=int(self.spin_width.value()),
            min_face_size=int(self.spin_minface.value()),
            max_faces=int(self.spin_maxfaces.value()),
            do_infer=bool(self.chk_infer.isChecked()),
        )

    def start_camera(self) -> None:
        self.stop_worker()
        cam_idx = int(self.cam_combo.currentData())
        self.worker.configure(
            source_type="camera",
            camera_index=cam_idx,
            **self._common_worker_params(),
        )
        self._set_source_buttons_enabled(False)
        self.worker.start()

    def open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Виберіть відео", "",
            "Відео (*.mp4 *.avi *.mov *.mkv *.webm *.m4v *.gif);;Усі файли (*)"
        )
        if not path:
            return
        self.stop_worker()
        self.worker.configure(
            source_type="video",
            video_path=path,
            **self._common_worker_params(),
        )
        self._set_source_buttons_enabled(False)
        self.worker.start()

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Виберіть фото", "",
            "Зображення (*.jpg *.jpeg *.png *.bmp *.webp *.gif);;Усі файли (*)"
        )
        if not path:
            return
        self.stop_worker()

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.warning(self, "Не вдалось відкрити", f"Не можу прочитати зображення:\n{path}")
            return

        params = self._common_worker_params()
        annotated = process_frame(
            frame_bgr=img,
            models=self.models,
            detector=self.worker.detector,
            do_infer=params["do_infer"],
            max_faces=params["max_faces"],
            min_face_size=params["min_face_size"],
            target_width=params["target_width"],
        )
        self.on_frame(bgr_to_qimage(annotated))
        self.fps_lbl.setText("FPS: — (фото)")

    def stop_worker(self) -> None:
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        self._set_source_buttons_enabled(True)

    def closeEvent(self, event) -> None:
        self.stop_worker()
        super().closeEvent(event)

    def on_frame(self, qimg: QImage) -> None:
        pix = QPixmap.fromImage(qimg)
        self.video.setPixmap(pix.scaled(self.video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def on_fps(self, fps: float) -> None:
        self.fps_lbl.setText(f"FPS: {fps:.1f}")

    def on_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Error", msg)
        self.stop_worker()

    def on_worker_finished(self) -> None:
        # Викликається, коли QThread.run завершився (камера спинена / кінець відео / помилка)
        self._set_source_buttons_enabled(True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--emotion-weights", dest="emotion_weights", type=str, default="")
    p.add_argument("--age-weights", dest="age_weights", type=str, default="")
    p.add_argument("--gender-weights", dest="gender_weights", type=str, default="")
    p.add_argument("--emotion-num-classes", dest="emotion_num_classes", type=int, default=0,
                   help="Optional override. If 0, inferred from checkpoint head.")
    p.add_argument("--age-num-classes", dest="age_num_classes", type=int, default=0,
                   help="Optional override. If 0, inferred from checkpoint head.")
    p.add_argument("--gender-num-classes", dest="gender_num_classes", type=int, default=0,
                   help="Optional override. If 0, inferred from checkpoint head.")
    p.add_argument("--backbone", type=str, default="google/vit-base-patch16-224-in21k")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=960, help="Resize frame to this width for speed (keeping aspect).")
    p.add_argument("--min-face", dest="min_face", type=int, default=60)
    p.add_argument("--max-faces", dest="max_faces", type=int, default=3)
    args = p.parse_args()

    # normalize empty strings
    args.emotion_weights = args.emotion_weights.strip() or ""
    args.age_weights = args.age_weights.strip() or ""
    args.gender_weights = args.gender_weights.strip() or ""
    return args


def main() -> None:
    args = parse_args()
    app = QApplication([])
    w = MainWindow(args)
    w.resize(1280, 720)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()