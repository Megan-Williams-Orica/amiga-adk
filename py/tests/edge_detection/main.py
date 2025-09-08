#!/usr/bin/env python3
from __future__ import annotations
import argparse
import asyncio
from pathlib import Path

import cv2
import numpy as np
from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfigList
from farm_ng.core.events_file_reader import proto_from_json_file
from turbojpeg import TurboJPEG

# Initialize TurboJPEG decoder.
jpeg_decoder = TurboJPEG()

# Global state
current_feed = "oak1"   # "oak1" or "oak0"
current_frame: np.ndarray | None = None
subscription_task: asyncio.Task | None = None
use_edge = False        # toggle for edge display

async def subscribe_feed(feed: str, client: EventClient, config) -> None:
    """Continuously grab JPEG frames from the Amiga EventService."""
    global current_frame
    async for event, message in client.subscribe(config.subscriptions[0], decode=True):
        try:
            frame = jpeg_decoder.decode(message.image_data)
            current_frame = frame
        except Exception as e:
            print(f"[{feed}] decode error: {e}")

async def main(service_config_path: Path) -> None:
    global current_feed, subscription_task, current_frame, use_edge

    # Load the JSON → protobuf config list
    config_list: EventServiceConfigList = proto_from_json_file(
        service_config_path, EventServiceConfigList()
    )

    # Pick out the two Oak configs
    oak1_cfg = next((c for c in config_list.configs if c.name == "oak1"), None)
    oak0_cfg = next((c for c in config_list.configs if c.name == "oak0"), None)
    if not oak1_cfg or not oak0_cfg:
        print("Need both oak0 and oak1 in the service config")
        return

    # Create clients
    clients = {
        "oak1": EventClient(oak1_cfg),
        "oak0": EventClient(oak0_cfg),
    }
    configs = {"oak1": oak1_cfg, "oak0": oak0_cfg}

    # Start the first subscription
    subscription_task = asyncio.create_task(
        subscribe_feed(current_feed, clients[current_feed], configs[current_feed])
    )

    # Prepare window    
    cv2.namedWindow("Camera Stream", cv2.WINDOW_NORMAL)
    print("Keys:  t = toggle feed, e = toggle edge, q = quit")

    while True:
        if current_frame is not None:
            frame = current_frame.copy()
            if use_edge:
                # Convert to grayscale + Canny
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 150, 150)  # adjust thresholds as needed
                display = edges
            else:
                display = frame

            cv2.imshow("Camera Stream", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            # swap oak0/oak1 feed
            if subscription_task:
                subscription_task.cancel()
            current_feed = "oak0" if current_feed == "oak1" else "oak1"
            print(f"→ Switched to {current_feed}")
            current_frame = None
            subscription_task = asyncio.create_task(
                subscribe_feed(current_feed, clients[current_feed], configs[current_feed])
            )
        elif key == ord('e'):
            use_edge = not use_edge
            mode = "EDGE" if use_edge else "RAW"
            print(f"→ Display mode: {mode}")

        await asyncio.sleep(0.067)  # ~15 FPS

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Amiga camera stream with edge-detection toggle"
    )
    parser.add_argument(
        "--service-config",
        type=Path,
        required=True,
        help="EventServiceConfigList JSON for oak0/oak1"
    )
    args = parser.parse_args()
    asyncio.run(main(args.service_config))
