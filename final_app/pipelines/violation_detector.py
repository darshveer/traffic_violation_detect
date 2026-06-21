"""Unified CCTV traffic-violation inference pipeline.

:class:`ViolationDetector` chains a YOLO11n base detector with helmet/seatbelt
classifiers, self-contained logic for triple-riding / red-light / wrong-side
driving, and PaddleOCR plate reading. Each specialized detector can be replaced
by an external plug-in model when its weights are configured; otherwise the
built-in logic on top of the base detector is used.

The per-frame result schema (``infer_frame``)::

    {
      "frame_id": int,
      "timestamp": str,                 # ISO-like or seconds
      "summary": {                      # spec-shaped flat dict
          "helmet_absent": float, "helmet_present": float,
          "seatbelt_absent": float, "seatbelt_present": float,
          "triple_rider": float, "red_light_violation": float,
          "wrong_side_driving": float, "license_plate": str
      },
      "violations": [ {type, confidence, box, track_id, plate, class}, ... ],
      "detections": [ {box, label, confidence, is_violation, ...}, ... ],
      "counts": { violation_type: int, ... }
    }
"""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .common import load_yaml, resolve_path, select_device, setup_logging
from .inference_utils import (
    COCO_CLASSES,
    ID_TO_NAME,
    VEHICLE_CLASSES,
    batched,
    box_center,
    crop,
    iter_video_frames,
    load_checkpoint,
    overlap_ratio,
    point_in_box,
    save_checkpoint,
    video_metadata,
)
from .model_loader import ModelLoader
from .tracking import Tracker

logger = setup_logging()

# Ordered violation keys reported in every summary.
SUMMARY_KEYS = (
    "helmet_absent",
    "helmet_present",
    "seatbelt_absent",
    "seatbelt_present",
    "triple_rider",
    "red_light_violation",
    "wrong_side_driving",
)


class ViolationDetector:
    """Run all violation detectors on frames, images and videos.

    Parameters
    ----------
    config_path : str
        Path to ``configs/pipeline.yaml``.
    device : str, optional
        Device override (``auto``/``cuda``/``cpu``). Falls back to config value.
    model_loader : ModelLoader, optional
        Pre-built loader (mainly for testing / sharing models).
    ocr_handler : object, optional
        Pre-built OCR handler; if ``None`` one is created from the config.
    """

    def __init__(
        self,
        config_path: str = "configs/pipeline.yaml",
        device: Optional[str] = None,
        model_loader: Optional[ModelLoader] = None,
        ocr_handler: Optional[Any] = None,
    ) -> None:
        self.config: Dict[str, Any] = load_yaml(config_path)
        req_device = device or self.config.get("device", "auto")
        self.device = select_device(req_device)
        self.half = bool(self.config.get("half", False)) and self.device != "cpu"
        self.imgsz = int(self.config.get("imgsz", 640))

        th = self.config.get("thresholds", {})
        self.base_conf = float(th.get("base_conf", 0.35))
        self.iou = float(th.get("iou", 0.5))
        self.violation_conf = float(th.get("violation_conf", 0.6))
        self.helmet_conf = float(th.get("helmet_conf", 0.5))
        self.seatbelt_conf = float(th.get("seatbelt_conf", 0.5))

        # Models.
        self.loader = model_loader or ModelLoader(config_path=config_path, device=self.device)
        if not self.loader.models:
            self.loader.load_all()

        # Tracking (used by wrong-side logic across video frames).
        trk = self.config.get("tracking", {})
        self.tracker = Tracker(
            max_age=int(trk.get("max_age", 30)),
            min_hits=int(trk.get("min_hits", 3)),
            iou_threshold=float(trk.get("iou_threshold", 0.3)),
        )

        # OCR.
        ocr_cfg = self.config.get("ocr", {})
        self.ocr_enabled = bool(ocr_cfg.get("enabled", True))
        self.ocr_run_on = set(ocr_cfg.get("run_on", list(VEHICLE_CLASSES)))
        if ocr_handler is not None:
            self.ocr = ocr_handler
        elif self.ocr_enabled:
            from evidence.ocr_handler import OCRHandler

            self.ocr = OCRHandler(
                lang=ocr_cfg.get("lang", "en"),
                use_gpu=self.device != "cpu",
                min_conf=float(ocr_cfg.get("min_plate_conf", 0.5)),
            )
        else:
            self.ocr = None

        logger.info(
            "ViolationDetector ready (device=%s, half=%s, ocr=%s).",
            self.device,
            self.half,
            getattr(self.ocr, "backend", "off"),
        )

    # ====================================================================== #
    # Low-level model runners
    # ====================================================================== #
    def _predict(self, model: Any, source: Any, conf: float) -> List[List[Dict]]:
        """Run an Ultralytics model and return per-image detection dicts.

        Returns
        -------
        list[list[dict]]
            For each input image, a list of ``{box, cls_id, conf}`` dicts.
        """
        try:
            results = model.predict(
                source=source,
                conf=conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                half=self.half,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully on a frame
            logger.warning("Prediction failed: %s", exc)
            return [[]]

        out: List[List[Dict]] = []
        for res in results:
            dets: List[Dict] = []
            boxes = getattr(res, "boxes", None)
            if boxes is not None and boxes.shape[0] > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                for b, c, cf in zip(xyxy, cls, confs):
                    dets.append({"box": b.tolist(), "cls_id": int(c), "conf": float(cf)})
            out.append(dets)
        return out

    def batch_infer(self, crops: Sequence[np.ndarray], model_name: str) -> List[List[Dict]]:
        """Batch-run a named model over a list of crops for speed.

        Parameters
        ----------
        crops : sequence of np.ndarray
            Image crops to classify/detect on.
        model_name : str
            Logical model name (e.g. ``"helmet"``, ``"seatbelt"``).

        Returns
        -------
        list[list[dict]]
            Per-crop detection dicts; empty list per crop if model unavailable.
        """
        lm = self.loader.get(model_name)
        if not lm.available or len(crops) == 0:
            return [[] for _ in crops]

        conf = self.helmet_conf if model_name == "helmet" else self.seatbelt_conf
        valid = [(i, c) for i, c in enumerate(crops) if c is not None and c.size > 0]
        results: List[List[Dict]] = [[] for _ in crops]
        max_batch = 16
        for chunk in batched(valid, max_batch):
            imgs = [c for _, c in chunk]
            preds = self._predict(lm.model, imgs, conf)
            for (idx, _), pred in zip(chunk, preds):
                results[idx] = pred
        return results

    # ====================================================================== #
    # Per-frame inference
    # ====================================================================== #
    def infer_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp: Optional[str] = None,
        use_tracking: bool = False,
    ) -> Dict[str, Any]:
        """Run every violation detector on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image.
        frame_id : int
            Frame index (for reports).
        timestamp : str, optional
            Timestamp string; defaults to the current time.
        use_tracking : bool
            Enable wrong-side tracking (only meaningful for sequential video).

        Returns
        -------
        dict
            The per-frame result schema documented at module level.
        """
        if timestamp is None:
            timestamp = _dt.datetime.now().isoformat(timespec="seconds")

        base = self.loader.get("base")
        base_dets = self._predict(base.model, frame, self.base_conf)[0] if base.available else []

        # Split base detections by category for convenience.
        persons = [d for d in base_dets if d["cls_id"] == COCO_CLASSES["person"]]
        motorcycles = [d for d in base_dets if d["cls_id"] == COCO_CLASSES["motorcycle"]]
        vehicles = [d for d in base_dets if ID_TO_NAME.get(d["cls_id"]) in VEHICLE_CLASSES]
        lights = [d for d in base_dets if d["cls_id"] == COCO_CLASSES["traffic_light"]]

        violations: List[Dict] = []
        detections: List[Dict] = []

        # Base objects drawn as non-violations by default (may be upgraded).
        det_index: Dict[int, Dict] = {}
        for d in base_dets:
            rec = {
                "box": d["box"],
                "label": ID_TO_NAME.get(d["cls_id"], str(d["cls_id"])),
                "confidence": d["conf"],
                "is_violation": False,
            }
            det_index[id(d)] = rec
            detections.append(rec)

        # --- individual detectors ---------------------------------------
        violations += self._detect_triple_riding(frame, motorcycles, persons, det_index)
        violations += self._detect_red_light(frame, lights, vehicles, det_index)
        if use_tracking:
            violations += self._detect_wrong_side(frame, vehicles, det_index)
        violations += self._detect_helmet(frame, motorcycles, persons)
        violations += self._detect_seatbelt(frame, vehicles)

        # --- license plates (attach to vehicle detections) --------------
        plate_text = self._read_plates(frame, vehicles, det_index)

        summary = self._build_summary(violations, plate_text)
        counts: Dict[str, int] = {}
        for v in violations:
            counts[v["type"]] = counts.get(v["type"], 0) + 1

        return {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "summary": summary,
            "violations": violations,
            "detections": detections,
            "counts": counts,
        }

    # ------------------------------------------------------------------ detectors
    def _detect_triple_riding(
        self, frame: np.ndarray, motorcycles: List[Dict], persons: List[Dict], det_index: Dict
    ) -> List[Dict]:
        """Flag motorcycles carrying >= ``min_riders`` persons.

        Plug-in override: if a ``triple_rider`` model is loaded, its detections
        are used directly instead of the person-counting heuristic.
        """
        cfg = self.config.get("triple_rider", {})
        min_riders = int(cfg.get("min_riders", 3))
        min_overlap = float(cfg.get("overlap_ratio", 0.15))
        events: List[Dict] = []

        # Dedicated-model confidence (separate from the generic violation_conf):
        # the trained triple-rider detector is high-precision but its scores run
        # lower than COCO models, so a per-plugin threshold keeps recall usable.
        tr_conf = float(cfg.get("conf", self.violation_conf))
        plugin = self.loader.get("triple_rider")
        if plugin.available:
            names = self._class_names("triple_rider")
            is_coco = ("car" in names.values() or len(names) == 80)
            if not is_coco:
                for d in self._predict(plugin.model, frame, tr_conf)[0]:
                    if d["conf"] >= tr_conf:
                        events.append(self._event("triple_rider", d["conf"], d["box"]))
                return events

        for moto in motorcycles:
            mbox = moto["box"]
            m_h = mbox[3] - mbox[1]
            riders = []
            for p in persons:
                pbox = p["box"]
                # 1. Significant bounding box overlap
                if overlap_ratio(pbox, mbox) >= min_overlap:
                    riders.append(p)
                    continue
                # 2. Heuristic: Person is horizontally aligned and their bottom half (legs/feet) is near the motorcycle
                px_c = (pbox[0] + pbox[2]) / 2.0
                if mbox[0] <= px_c <= mbox[2]:
                    # Check if bottom Y of person is within the motorcycle box, or slightly above it
                    if mbox[1] - 0.2 * m_h <= pbox[3] <= mbox[3]:
                        riders.append(p)
                        
            if len(riders) >= min_riders:
                conf = min(0.99, moto["conf"] * (len(riders) / min_riders))
                if conf >= self.violation_conf:
                    rec = det_index.get(id(moto))
                    if rec:
                        rec["is_violation"] = True
                        rec["label"] = f"triple_rider({len(riders)})"
                    events.append(self._event("triple_rider", conf, mbox, cls="motorcycle"))
        return events

    def _detect_red_light(
        self, frame: np.ndarray, lights: List[Dict], vehicles: List[Dict], det_index: Dict
    ) -> List[Dict]:
        """Flag vehicles crossing the stop-line ROI while a light shows red.

        Plug-in override: a ``red_light`` model's detections are used directly.
        """
        events: List[Dict] = []
        plugin = self.loader.get("red_light")
        if plugin.available:
            names = self._class_names("red_light")
            is_coco = ("car" in names.values() or len(names) == 80)
            if not is_coco:
                for d in self._predict(plugin.model, frame, self.violation_conf)[0]:
                    if d["conf"] >= self.violation_conf:
                        events.append(self._event("red_light_violation", d["conf"], d["box"]))
                return events


        if not self._any_red_light(frame, lights):
            return events

        h, w = frame.shape[:2]
        cfg = self.config.get("red_light", {})
        sx1, sy1, sx2, sy2 = cfg.get("stop_line", [0.0, 0.55, 1.0, 0.65])
        band = [sx1 * w, sy1 * h, sx2 * w, sy2 * h]
        for veh in vehicles:
            cx, cy = box_center(veh["box"])
            if band[0] <= cx <= band[2] and band[1] <= cy <= band[3]:
                conf = min(0.99, veh["conf"])
                if conf >= self.violation_conf:
                    rec = det_index.get(id(veh))
                    if rec:
                        rec["is_violation"] = True
                        rec["label"] = "red_light_violation"
                    events.append(
                        self._event("red_light_violation", conf, veh["box"],
                                    cls=ID_TO_NAME.get(veh["cls_id"]))
                    )
        return events

    def _detect_wrong_side(self, frame: np.ndarray, vehicles: List[Dict], det_index: Dict) -> List[Dict]:
        """Flag tracked vehicles moving opposite the allowed direction of travel.

        Wrong-side driving is *camera-geometry specific*: the legal direction of
        travel depends on the road/lane layout in the camera view, so the allowed
        direction must be calibrated per camera (``wrong_side.allowed_direction``;
        see ``scripts/calibrate_wrong_side.py`` to estimate it from a clip).

        To suppress false positives this requires, over a sliding window of a
        track's centroids: (1) sufficient net displacement, (2) the *net* heading
        opposing the allowed direction, and (3) the *per-step* motion consistently
        opposing it (so a wiggling/parked vehicle is not flagged). By default only
        four-wheelers are considered (motorcycles excluded), configurable.

        Plug-in override: a ``wrong_side`` model's detections are used directly.
        """
        events: List[Dict] = []
        plugin = self.loader.get("wrong_side")
        if plugin.available:
            names = self._class_names("wrong_side")
            is_coco = ("car" in names.values() or len(names) == 80)
            if not is_coco:
                for d in self._predict(plugin.model, frame, self.violation_conf)[0]:
                    if d["conf"] >= self.violation_conf:
                        events.append(self._event("wrong_side_driving", d["conf"], d["box"]))
                return events

        cfg = self.config.get("wrong_side", {})
        allowed = np.array(cfg.get("allowed_direction", [0.0, 1.0]), dtype=np.float32)
        allowed = allowed / (np.linalg.norm(allowed) + 1e-9)
        min_frames = int(cfg.get("min_track_frames", 8))
        min_disp = float(cfg.get("min_displacement", 40))
        align_thr = float(cfg.get("alignment_threshold", 0.5))   # net heading vs allowed
        step_consistency = float(cfg.get("step_consistency", 0.7))  # fraction of opposing steps
        ignore_classes = set(cfg.get("ignore_classes", ["motorcycle", "bicycle"]))
        roi = cfg.get("roi")  # optional [x1,y1,x2,y2] normalised band

        h, w = frame.shape[:2]
        roi_px = [roi[0] * w, roi[1] * h, roi[2] * w, roi[3] * h] if roi else None

        det_array = np.array(
            [[*v["box"], v["cls_id"]] for v in vehicles], dtype=np.float32
        ) if vehicles else np.empty((0, 5), dtype=np.float32)
        tracks = self.tracker.update(det_array)

        for trk in tracks:
            if ID_TO_NAME.get(trk.cls_id) in ignore_classes:
                continue
            if len(trk.centroids) < min_frames:
                continue
            # ROI gate: vehicle centre must be inside the configured road band.
            if roi_px is not None:
                cx, cy = box_center(trk.box.tolist())
                if not (roi_px[0] <= cx <= roi_px[2] and roi_px[1] <= cy <= roi_px[3]):
                    continue

            recent = trk.centroids[-min_frames:]
            dx = recent[-1][0] - recent[0][0]
            dy = recent[-1][1] - recent[0][1]
            disp = float(np.hypot(dx, dy))
            if disp < min_disp:
                continue
            net_align = float(np.dot(np.array([dx, dy]) / (disp + 1e-9), allowed))
            if net_align >= -align_thr:
                continue
            # Per-step consistency: fraction of steps moving against allowed.
            opposing = 0
            steps = 0
            for (x0, y0), (x1, y1) in zip(recent[:-1], recent[1:]):
                sdx, sdy = x1 - x0, y1 - y0
                sd = float(np.hypot(sdx, sdy))
                if sd < 1e-3:
                    continue
                steps += 1
                if float(np.dot(np.array([sdx, sdy]) / sd, allowed)) < 0:
                    opposing += 1
            if steps == 0 or (opposing / steps) < step_consistency:
                continue

            conf = min(0.99, 0.6 + 0.39 * min(1.0, -net_align))
            if conf >= self.violation_conf:
                rec_box = trk.box.tolist()
                events.append(
                    self._event("wrong_side_driving", conf, rec_box,
                                track_id=trk.id, cls=ID_TO_NAME.get(trk.cls_id))
                )
        return events

    def _detect_helmet(
        self, frame: np.ndarray, motorcycles: List[Dict], persons: List[Dict]
    ) -> List[Dict]:
        """Run the helmet model on rider crops; flag ``helmet_absent``."""
        lm = self.loader.get("helmet")
        if not lm.available:
            return []

        # Rider crops: persons overlapping a motorcycle, else all motorcycles.
        rider_boxes: List[List[float]] = []
        for p in persons:
            if any(overlap_ratio(p["box"], m["box"]) > 0.1 for m in motorcycles):
                rider_boxes.append(p["box"])
        if not rider_boxes:
            rider_boxes = [m["box"] for m in motorcycles]
        if not rider_boxes:
            return []

        crops = [crop(frame, b, pad=0.1) for b in rider_boxes]
        preds = self.batch_infer(crops, "helmet")
        names = self._class_names("helmet")
        return self._classify_events(rider_boxes, preds, names, "helmet")

    def _detect_seatbelt(self, frame: np.ndarray, vehicles: List[Dict]) -> List[Dict]:
        """Run the seatbelt model on car/cab crops; flag ``seatbelt_absent``."""
        lm = self.loader.get("seatbelt")
        if not lm.available:
            return []
        car_classes = {"car", "truck", "bus"}
        car_boxes = [v["box"] for v in vehicles if ID_TO_NAME.get(v["cls_id"]) in car_classes]
        if not car_boxes:
            return []
        crops = [crop(frame, b, pad=0.05) for b in car_boxes]
        preds = self.batch_infer(crops, "seatbelt")
        names = self._class_names("seatbelt")
        return self._classify_events(car_boxes, preds, names, "seatbelt")

    # ------------------------------------------------------------------ helpers
    def _classify_events(
        self,
        boxes: List[List[float]],
        preds: List[List[Dict]],
        names: Dict[int, str],
        kind: str,
    ) -> List[Dict]:
        """Convert helmet/seatbelt crop predictions into violation events.

        Reports both the ``*_absent`` (violation) and ``*_present`` classes so the
        summary carries present-confidence too. Only ``*_absent`` is a violation.
        """
        events: List[Dict] = []
        absent_key = f"{kind}_absent"
        for box, crop_preds in zip(boxes, preds):
            if not crop_preds:
                continue
            best = max(crop_preds, key=lambda d: d["conf"])
            cls_name = names.get(best["cls_id"], "")
            conf = best["conf"]
            is_absent = self._is_absence_class(cls_name)
            vtype = absent_key if is_absent else f"{kind}_present"
            events.append(self._event(vtype, conf, box, cls=kind, is_violation=is_absent))
        return events

    @staticmethod
    def _is_absence_class(cls_name: str) -> bool:
        """Whether a helmet/seatbelt class name denotes the *violation* (absence).

        Trained datasets label the violation many ways: ``no-helmet``,
        ``no_seatbelt``, ``without helmet``, ``helmet_absent``, ``not wearing``.
        Presence classes (``helmet``, ``moto-helmet``, ``seatbelt``) are not
        violations. Normalising punctuation lets one check cover all of them.
        """
        n = cls_name.lower().replace("_", " ").replace("-", " ").strip()
        absence_tokens = ("absent", "without", "no helmet", "nohelmet",
                          "no seatbelt", "no seat belt", "not wearing", "missing")
        if any(t in n for t in absence_tokens):
            return True
        # Leading "no " (e.g. "no helmet", "no seatbelt") => absence/violation.
        return n.startswith("no ")

    def _read_plates(self, frame: np.ndarray, vehicles: List[Dict], det_index: Dict) -> str:
        """OCR vehicle plates and attach text; return the first plate found.

        If a ``plate_detector`` model is loaded, plates are first localised within
        each vehicle crop and OCR runs on the tight plate region (far more
        accurate than OCR-ing the whole vehicle). Otherwise the whole vehicle
        crop is passed to OCR as a fallback.
        """
        if self.ocr is None or not getattr(self.ocr, "available", False):
            return ""
        plate_det = self.loader.get("plate_detector")
        first_plate = ""
        for veh in vehicles:
            if ID_TO_NAME.get(veh["cls_id"]) not in self.ocr_run_on:
                continue
            vcrop = crop(frame, veh["box"], pad=0.05)
            if vcrop.size == 0:
                continue
            text = ""
            if plate_det.available:
                # Localise plate(s) inside the vehicle crop; OCR highest-conf one.
                preds = self._predict(plate_det.model, vcrop, self.base_conf)[0]
                for d in sorted(preds, key=lambda x: -x["conf"]):
                    t = self.ocr.read_plate(crop(vcrop, d["box"], pad=0.12))
                    if t:
                        text = t
                        break
            else:
                text = self.ocr.read_plate(vcrop)
            if text:
                rec = det_index.get(id(veh))
                if rec:
                    rec["plate"] = text
                if not first_plate:
                    first_plate = text
        return first_plate

    def _any_red_light(self, frame: np.ndarray, lights: List[Dict]) -> bool:
        """Whether any traffic-light crop is predominantly red (HSV heuristic)."""
        if not lights:
            return False
        import cv2

        cfg = self.config.get("red_light", {})
        ratio_thr = float(cfg.get("red_hsv_ratio", 0.18))
        for lt in lights:
            c = crop(frame, lt["box"])
            if c.size == 0:
                continue
            hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
            lower1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
            lower2 = cv2.inRange(hsv, (160, 70, 50), (180, 255, 255))
            red = (lower1 | lower2) > 0
            if red.mean() >= ratio_thr:
                return True
        return False

    def _class_names(self, model_name: str) -> Dict[int, str]:
        """Return the class-id -> name map of a loaded model."""
        lm = self.loader.get(model_name)
        if lm.available and hasattr(lm.model, "names"):
            return dict(lm.model.names)
        return {}

    @staticmethod
    def _event(
        vtype: str,
        confidence: float,
        box: Sequence[float],
        track_id: Optional[int] = None,
        plate: Optional[str] = None,
        cls: Optional[str] = None,
        is_violation: bool = True,
    ) -> Dict:
        """Build a normalized violation/observation event dict."""
        return {
            "type": vtype,
            "confidence": round(float(confidence), 4),
            "box": [float(v) for v in box[:4]],
            "track_id": track_id,
            "plate": plate,
            "class": cls,
            "is_violation": is_violation,
        }

    def _build_summary(self, violations: List[Dict], plate_text: str) -> Dict[str, Any]:
        """Collapse events into the spec-shaped flat summary dict."""
        summary: Dict[str, Any] = {k: 0.0 for k in SUMMARY_KEYS}
        for v in violations:
            t = v["type"]
            if t in summary:
                summary[t] = max(summary[t], v["confidence"])
        summary["license_plate"] = plate_text
        return summary

    # ====================================================================== #
    # Image / video entry points
    # ====================================================================== #
    def infer_image(self, image_path: str) -> Dict[str, Any]:
        """Run inference on a single image file.

        Parameters
        ----------
        image_path : str
            Path to the image.

        Returns
        -------
        dict
            The per-frame result (see :meth:`infer_frame`).

        Raises
        ------
        FileNotFoundError
            If the image cannot be read.
        """
        import cv2

        path = str(resolve_path(image_path) or image_path)
        frame = cv2.imread(path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        ts = _dt.datetime.now().isoformat(timespec="seconds")
        result = self.infer_frame(frame, frame_id=0, timestamp=ts, use_tracking=False)
        result["image_path"] = path
        result["frame"] = frame
        return result

    def infer_video(
        self,
        video_path: str,
        output_path: str,
        skip_frames: Optional[int] = None,
        annotate: bool = True,
        resume: bool = True,
        show: bool = False,
    ) -> Dict[str, Any]:
        """Process a video with frame-skipping, annotation, live window and checkpointing.

        Parameters
        ----------
        video_path : str
            Source video.
        output_path : str
            Output directory for annotated video, frames and metadata.
        skip_frames : int, optional
            Process every Nth frame (defaults to config ``video.skip_frames``).
        annotate : bool
            Write an annotated video / frames.
        resume : bool
            Resume from a checkpoint if one exists in ``output_path``.
        show : bool
            Show a pop up live window during inference.

        Returns
        -------
        dict
            ``{"records": [...per kept frame...], "video": metadata, "output": dir}``.
        """
        import cv2

        from evidence.annotate import annotate_frame

        vid_cfg = self.config.get("video", {})
        skip = int(skip_frames if skip_frames is not None else vid_cfg.get("skip_frames", 5))
        ckpt_every = int(vid_cfg.get("checkpoint_every", 200))
        save_frames = bool(vid_cfg.get("save_frames", True))

        out_dir = Path(resolve_path(output_path) or output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = out_dir / "annotated_frames"
        if save_frames and annotate:
            frames_dir.mkdir(exist_ok=True)
        ckpt_path = out_dir / "checkpoint.json"
        records_path = out_dir / "_records.jsonl"

        meta = video_metadata(str(resolve_path(video_path) or video_path))
        start_frame = 0
        records: List[Dict] = []
        if resume:
            ck = load_checkpoint(str(ckpt_path))
            if ck:
                start_frame = int(ck.get("last_frame", 0)) + 1
                records = self._load_records(records_path)
                logger.info("Resuming video from frame %d (%d records).", start_frame, len(records))

        writer = None
        if annotate and vid_cfg.get("write_annotated", True):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_fps = max(1.0, meta["fps"] / skip)
            writer = cv2.VideoWriter(
                str(out_dir / "annotated_video.mp4"),
                fourcc,
                out_fps,
                (meta["width"], meta["height"]),
            )

        try:
            from tqdm import tqdm

            iterator = iter_video_frames(
                str(resolve_path(video_path) or video_path), skip, start_frame
            )
            pbar = tqdm(iterator, desc="infer_video", unit="frame")
            for idx, frame, ts_sec in pbar:
                ts = self._fmt_ts(ts_sec)
                result = self.infer_frame(frame, frame_id=idx, timestamp=ts, use_tracking=True)
                # Drop the heavy frame array from the stored record.
                slim = {k: result[k] for k in ("frame_id", "timestamp", "summary", "violations", "counts")}
                records.append(slim)
                self._append_record(records_path, slim)

                if annotate:
                    ann = annotate_frame(frame, result["detections"], ts, result["violations"])
                    if writer is not None:
                        writer.write(ann)
                    if save_frames and result["violations"]:
                        cv2.imwrite(str(frames_dir / f"frame_{idx:06d}.jpg"), ann)
                else:
                    ann = frame.copy()
                    if show and result["violations"]:
                        from evidence.annotate import draw_violation_overlay
                        active_violations = [v for v in result["violations"] if v.get("is_violation", True)]
                        if active_violations:
                            counts = {}
                            for v in active_violations:
                                vtype = v["type"]
                                counts[vtype] = counts.get(vtype, 0) + 1
                            draw_violation_overlay(ann, counts, cv2)

                if show:
                    cv2.imshow("CCTV Traffic Violation Detection", ann)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Live display stopped by user.")
                        break

                if idx % ckpt_every == 0:
                    save_checkpoint(str(ckpt_path), {"last_frame": idx})
        finally:
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()


        save_checkpoint(str(ckpt_path), {"last_frame": -1, "done": True})
        logger.info("Video done: %d frames processed.", len(records))
        return {"records": records, "video": meta, "output": str(out_dir)}

    # ------------------------------------------------------------------ io utils
    @staticmethod
    def _fmt_ts(seconds: float) -> str:
        """Format elapsed video seconds as ``HH:MM:SS``."""
        return str(_dt.timedelta(seconds=int(seconds)))

    @staticmethod
    def _append_record(path: Path, record: Dict) -> None:
        """Append one JSON record per line (used for crash-safe resume)."""
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    @staticmethod
    def _load_records(path: Path) -> List[Dict]:
        """Load previously written JSONL records (for resume)."""
        import json

        if not path.exists():
            return []
        out: List[Dict] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return out
