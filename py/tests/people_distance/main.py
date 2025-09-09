#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np
from pathlib import Path


def colorizeDepth(frameDepth: np.ndarray) -> np.ndarray:
    invalidMask = frameDepth == 0
    try:
        # percentile‐based log colorization
        minD = np.percentile(frameDepth[frameDepth != 0], 3)
        maxD = np.percentile(frameDepth[frameDepth != 0], 95)
        logD = np.log(frameDepth, where=frameDepth != 0)
        lmin, lmax = np.log(minD), np.log(maxD)
        np.nan_to_num(logD, copy=False, nan=lmin)
        logD = np.clip(logD, lmin, lmax)
        cd = np.interp(logD, (lmin, lmax), (0, 255)).astype(np.uint8)
        cd = cv2.applyColorMap(cd, cv2.COLORMAP_JET)
        cd[invalidMask] = 0
    except Exception:
        h, w = frameDepth.shape
        cd = np.zeros((h, w, 3), dtype=np.uint8)
    return cd


# 1) Load your superblob
blob_path = Path(__file__).parent / "scrfd_person.superblob.blob"
if not blob_path.exists():
    raise FileNotFoundError(f"Superblob not found at {blob_path}")
with open(blob_path, "rb") as f:
    blob_bytes = f.read()

# 2) Open the device with that blob
with dai.Device(blob=blob_bytes) as device:
    # 3) Grab the three output queues
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="det", maxSize=4, blocking=False)

    cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)

    while True:
        inRgb = qRgb.tryGet()
        inDepth = qDepth.tryGet()
        inDet = qDet.tryGet()

        if inRgb:
            frameRgb = inRgb.getCvFrame()
        else:
            # if you really need synchronized frames, you can block .get() instead
            continue

        if inDepth:
            # might be RAW16, so pull raw then colorize
            raw = inDepth.getFrame()  # numpy uint16
            frameDepth = colorizeDepth(raw)
        else:
            frameDepth = None

        if inDet:
            # draw all person detections on RGB
            h, w = frameRgb.shape[:2]
            for d in inDet.detections:
                x1 = int(d.xmin * w)
                y1 = int(d.ymin * h)
                x2 = int(d.xmax * w)
                y2 = int(d.ymax * h)
                cv2.rectangle(frameRgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frameRgb, f"person {d.confidence:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )

        # 4) show
        cv2.imshow("RGB", frameRgb)
        if frameDepth is not None:
            cv2.imshow("Depth", frameDepth)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
