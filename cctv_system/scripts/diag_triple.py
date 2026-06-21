#!/usr/bin/env python3
"""Instrument the triple-rider rider-counting to see why it (mis)fires."""
from __future__ import annotations
import sys
from pathlib import Path
from collections import Counter

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))
import cv2  # noqa: E402


def main(video, max_frames=60):
    from ultralytics import YOLO
    from pipelines.common import select_device
    from pipelines.inference_utils import COCO_CLASSES, overlap_ratio
    dev = select_device("auto")
    m = YOLO(str(PKG / "models" / "yolo11n.pt"))

    cap = cv2.VideoCapture(video)
    n = 0
    rider_hist = Counter()       # how many riders associated per motorcycle
    moto_conf_hist = []
    persons_per_frame = []
    motos_per_frame = []
    idx = 0
    while n < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        idx += 1
        if idx % 5:
            continue
        n += 1
        r = m.predict(fr, device=dev, conf=0.35, verbose=False)[0]
        xy = r.boxes.xyxy.cpu().numpy(); cl = r.boxes.cls.cpu().numpy().astype(int); cf = r.boxes.conf.cpu().numpy()
        persons = [(xy[i], cf[i]) for i in range(len(cl)) if cl[i] == COCO_CLASSES["person"]]
        motos = [(xy[i], cf[i]) for i in range(len(cl)) if cl[i] == COCO_CLASSES["motorcycle"]]
        persons_per_frame.append(len(persons)); motos_per_frame.append(len(motos))
        for mbox, mcf in motos:
            moto_conf_hist.append(round(float(mcf), 2))
            m_h = mbox[3] - mbox[1]
            riders = 0
            for pbox, pcf in persons:
                if overlap_ratio(pbox, mbox) >= 0.15:
                    riders += 1; continue
                px_c = (pbox[0] + pbox[2]) / 2.0
                if mbox[0] <= px_c <= mbox[2] and (mbox[1] - 0.2 * m_h) <= pbox[3] <= mbox[3]:
                    riders += 1
            rider_hist[riders] += 1
    cap.release()
    print(f"frames analysed: {n}")
    print(f"avg persons/frame: {sum(persons_per_frame)/max(1,len(persons_per_frame)):.1f}, "
          f"avg motos/frame: {sum(motos_per_frame)/max(1,len(motos_per_frame)):.1f}")
    print(f"rider-count distribution per motorcycle (riders:count): {dict(sorted(rider_hist.items()))}")
    mc = Counter(moto_conf_hist)
    print(f"motorcycle conf distribution: {dict(sorted(mc.items()))}")
    over = sum(c for k, c in rider_hist.items() if k >= 3)
    two = sum(c for k, c in rider_hist.items() if k == 2)
    print(f"motorcycles with >=3 riders: {over}; with exactly 2 riders: {two}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60)
