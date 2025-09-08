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
from farm_ng.canbus.canbus_pb2 import Twist2d
from numpy import clip

# Constants for twist commands.
MAX_LINEAR_VELOCITY_MPS = 0.5
MAX_ANGULAR_VELOCITY_RPS = 0.5
VELOCITY_INCREMENT = 0.05

def update_twist_with_key_press(twist: Twist2d, key: int) -> Twist2d:
    """Update the twist command based on the key pressed."""
    # Stop
    if key == ord(" "):
        twist.linear_velocity_x = 0.0
        twist.linear_velocity_y = 0.0
        twist.angular_velocity = 0.0
    # Forward / reverse.
    if key == ord("w"):
        twist.linear_velocity_x += VELOCITY_INCREMENT
    elif key == ord("s"):
        twist.linear_velocity_x -= VELOCITY_INCREMENT
    # Left / right turn.
    if key == ord("a"):
        twist.angular_velocity += VELOCITY_INCREMENT
    elif key == ord("d"):
        twist.angular_velocity -= VELOCITY_INCREMENT

    twist.linear_velocity_x = clip(twist.linear_velocity_x, -MAX_LINEAR_VELOCITY_MPS, MAX_LINEAR_VELOCITY_MPS)
    twist.angular_velocity = clip(twist.angular_velocity, -MAX_ANGULAR_VELOCITY_RPS, MAX_ANGULAR_VELOCITY_RPS)
    return twist

# Initialize TurboJPEG decoder with the library path.
jpeg_decoder = TurboJPEG(lib_path='/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0')

# Global variables for camera stream.
current_feed = "oak1"   # either "oak1" or "oak0"
current_frame = None    # most recent frame from the active camera feed
subscription_task = None  # active subscription task for the camera feed

async def subscribe_feed(feed: str, client: EventClient, config) -> None:
    """Subscribe to the feed specified by the given config and client, updating current_frame."""
    global current_frame
    async for event, message in client.subscribe(config.subscriptions[0], decode=True):
        try:
            frame = jpeg_decoder.decode(message.image_data)
            current_frame = frame
        except Exception as e:
            print(f"Error decoding {feed} frame: {e}")

async def main(service_config_path: Path) -> None:
    global current_feed, subscription_task, current_frame

    # Load the service configuration as an EventServiceConfigList.
    config_list: EventServiceConfigList = proto_from_json_file(
        service_config_path, EventServiceConfigList()
    )

    # Extract configurations for the two cameras and for canbus (twist control).
    oak1_config = None
    oak0_config = None
    canbus_config = None
    for config in config_list.configs:
        if config.name == "oak1":
            oak1_config = config
        elif config.name == "oak0":
            oak0_config = config
        elif config.name == "canbus":
            canbus_config = config

    if oak1_config is None or oak0_config is None or canbus_config is None:
        print("Configs for oak1, oak0, and canbus must be provided.")
        return

    # Create separate EventClients for each camera and for twist commands.
    oak1_client = EventClient(oak1_config)
    oak0_client = EventClient(oak0_config)
    canbus_client = EventClient(canbus_config)

    # Organize camera clients and configs into dictionaries.
    camera_clients = {"oak1": oak1_client, "oak0": oak0_client}
    camera_configs = {"oak1": oak1_config, "oak0": oak0_config}

    # Start the subscription for the current camera feed.
    subscription_task = asyncio.create_task(
        subscribe_feed(current_feed, camera_clients[current_feed], camera_configs[current_feed])
    )

    # Initialize twist command.
    twist = Twist2d()

    # Create a single, resizable window for the camera feed.
    cv2.namedWindow("Camera Stream", cv2.WINDOW_NORMAL)

    # Main loop: display the current camera feed and process key presses.
    while True:
        if current_frame is not None:
            cv2.imshow("Camera Stream", current_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('t'):
            # Toggle camera feed.
            if subscription_task is not None:
                subscription_task.cancel()
            current_feed = "oak0" if current_feed == "oak1" else "oak1"
            print(f"Switching feed to {current_feed}")
            current_frame = None  # optionally clear the frame
            subscription_task = asyncio.create_task(
                subscribe_feed(current_feed, camera_clients[current_feed], camera_configs[current_feed])
            )
        elif key in [ord('w'), ord('a'), ord('s'), ord('d'), ord(' ')]:
            # Update twist command based on key press.
            twist = update_twist_with_key_press(twist, key)
            print(f"Sending twist: linear_velocity_x = {twist.linear_velocity_x:.3f}, angular_velocity = {twist.angular_velocity:.3f}")
            try:
                await canbus_client.request_reply("/twist", twist)
            except Exception as e:
                print(f"Error sending twist command: {e}")

        # Maintain a display frame rate of approximately 15 FPS.
        await asyncio.sleep(0.067)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Remote Pilot: Display camera stream with integrated keyboard twist control. "
                    "(Press 't' to toggle camera feed, 'q' to quit, and use WASD/space for movement.)"
    )
    parser.add_argument(
        "--service-config",
        type=Path,
        required=True,
        help="The service config file (EventServiceConfigList format) including configs for oak1, oak0, and canbus."
    )
    args = parser.parse_args()

    asyncio.run(main(args.service_config))
