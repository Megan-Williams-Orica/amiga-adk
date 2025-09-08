
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.core.stamp import get_stamp_by_semantics_and_clock_type, StampSemantics


async def main(service_config_path: Path) -> None:
    cfg: EventServiceConfig = proto_from_json_file(
        service_config_path, EventServiceConfig())
    client = EventClient(cfg)

    last_img = None

    # Subscribe to both topics from the Brain
    topics = ["/oak/depth_preview", "/oak/spatial"]

    async for event, message in client.subscribe(topics, decode=False):
        # Optional debug timestamp
        stamp = (
            get_stamp_by_semantics_and_clock_type(
                event, StampSemantics.DRIVER_RECEIVE, "monotonic")
            or event.timestamps[0].stamp
        )
        # print(f"Stamp: {stamp} | Path: {event.uri.path}")

        if event.uri.path.endswith("/oak/depth_preview"):
            # PNG heatmap
            img = cv2.imdecode(np.frombuffer(
                message, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            last_img = img
            cv2.namedWindow("OAK Depth", cv2.WINDOW_NORMAL)
            cv2.imshow("OAK Depth", img)
            cv2.waitKey(1)

        elif event.uri.path.endswith("/oak/spatial"):
            if last_img is None:
                continue
            overlay = last_img.copy()

            try:
                items = json.loads(message.decode("utf-8"))
            except Exception:
                items = []

            H, W = overlay.shape[:2]
            for it in items:
                x1, y1, x2, y2 = it["roi_norm"]
                xmin, ymin = int(x1 * W), int(y1 * H)
                xmax, ymax = int(x2 * W), int(y2 * H)
                X, Y, Z = it["xyz_mm"]

                cv2.rectangle(overlay, (xmin, ymin),
                              (xmax, ymax), (255, 255, 255), 2)
                text = f"X:{int(X)} Y:{int(Y)} Z:{int(Z)} mm"
                cv2.putText(overlay, text, (xmin + 6, ymin + 18),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.55, (255, 255, 255), 1)

            cv2.namedWindow("OAK Depth + Spatial", cv2.WINDOW_NORMAL)
            cv2.imshow("OAK Depth + Spatial", overlay)
            cv2.waitKey(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Laptop overlay viewer for OAK Spatial (ADK client)")
    parser.add_argument("--service-config", type=Path,
                        required=True, help="Client config JSON")
    args = parser.parse_args()
    asyncio.run(main(args.service_config))
