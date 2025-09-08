import asyncio
import math
from pathlib import Path

from pyproj import Transformer

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.canbus.canbus_pb2 import Twist2d
from farm_ng.filter.filter_pb2 import FilterState

STOP_RADIUS = 0.5  # meters
MAX_LINEAR = 0.5   # m/s
MAX_ANGULAR = 0.5  # rad/s


def compute_control(current_x, current_y, heading_rad, target_x, target_y):
    dx = target_x - current_x
    dy = target_y - current_y
    distance = math.hypot(dx, dy)

    if distance < STOP_RADIUS:
        return 0.0, 0.0, True

    target_heading = math.atan2(dy, dx)
    heading_error = (target_heading - heading_rad + math.pi) % (2 * math.pi) - math.pi

    linear = min(MAX_LINEAR, distance)
    angular = max(min(heading_error, MAX_ANGULAR), -MAX_ANGULAR)

    return linear, angular, False


def latlon_to_relative_ned(base_lat, base_lon, target_lat, target_lon):
    """
    1. Convert base and target (lat, lon) from WGS84 → GDA94/UTM Zone 56 (EPSG:28356).
    2. Compute ENU deltas: Δe = target_e − base_e, Δn = target_n − base_n.
    3. Remap ENU → NED:
       rel_ned_x = Δn   (north)
       rel_ned_y = Δe   (east)
    """
    transformer = Transformer.from_crs("epsg:4326", "epsg:28356", always_xy=True)

    base_e, base_n = transformer.transform(base_lon, base_lat)
    target_e, target_n = transformer.transform(target_lon, target_lat)

    delta_e = target_e - base_e
    delta_n = target_n - base_n

    rel_ned_x = delta_n  # north → x
    rel_ned_y = delta_e  # east  → y
    return rel_ned_x, rel_ned_y


async def main(
    filter_config_path: Path,
    canbus_config_path: Path,
    base_lat: float,
    base_lon: float,
    goal_lat: float,
    goal_lon: float,
):
    # — Load service configs
    filter_config: EventServiceConfig = proto_from_json_file(
        filter_config_path, EventServiceConfig()
    )
    canbus_config: EventServiceConfig = proto_from_json_file(
        canbus_config_path, EventServiceConfig()
    )

    # — Compute target’s (x, y) in NED relative to RTK base station
    target_x, target_y = latlon_to_relative_ned(
        base_lat, base_lon, goal_lat, goal_lon
    )
    print(f"[Target in NED frame] x = {target_x:.3f} m, y = {target_y:.3f} m")

    # — Create EventClients for filter and canbus services
    filter_client = EventClient(filter_config)
    canbus_client = EventClient(canbus_config)

    print("Starting waypoint navigation...")

    first_pose_printed = False

    # — Subscribe to FilterState on the filter service
    async for _, state_msg in filter_client.subscribe(
        filter_config.subscriptions[0], decode=True
    ):
        state: FilterState = state_msg

        # Print raw pose once for debugging
        if not first_pose_printed:
            print("Raw state.pose fields:", state.pose)
            first_pose_printed = True

        # Extract (north, east) robustly from whatever field exists
        if hasattr(state.pose, "translation"):
            x = state.pose.translation.x  # north (m)
            y = state.pose.translation.y  # east  (m)
        elif hasattr(state.pose, "position"):
            x = state.pose.position.x
            y = state.pose.position.y
        else:
            # Fall back to direct x/y if they exist
            x = getattr(state.pose, "x", 0.0)
            y = getattr(state.pose, "y", 0.0)

        heading = state.heading  # rad, 0 = facing north

        linear, angular, done = compute_control(
            x, y, heading, target_x, target_y
        )

        print(f"[Pose] x={x:.3f}, y={y:.3f}, heading={heading:.3f} rad")
        print(f"[Cmd] linear={linear:.3f}, angular={angular:.3f}")

        twist = Twist2d()
        twist.linear_velocity_x = linear
        twist.angular_velocity = angular
        await canbus_client.request_reply("/twist", twist)

        if done:
            print("✅ Goal reached. Stopping robot.")
            break

        await asyncio.sleep(0.05)

    # — Once done (or loop broken), send zero‐velocity to ensure a full stop
    stop_msg = Twist2d()
    await canbus_client.request_reply("/twist", stop_msg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Navigate Amiga to a GPS waypoint via CAN twist commands."
    )
    parser.add_argument(
        "--filter-config",
        type=Path,
        required=True,
        help="Path to filter service config JSON (host=fox-furrow, port=20001).",
    )
    parser.add_argument(
        "--canbus-config",
        type=Path,
        required=True,
        help="Path to canbus service config JSON (host=fox-furrow, port=6001).",
    )
    parser.add_argument(
        "--base-lat",
        type=float,
        required=True,
        help="Latitude of RTK base station.",
    )
    parser.add_argument(
        "--base-lon",
        type=float,
        required=True,
        help="Longitude of RTK base station.",
    )
    parser.add_argument(
        "--goal-lat",
        type=float,
        required=True,
        help="Latitude of goal waypoint.",
    )
    parser.add_argument(
        "--goal-lon",
        type=float,
        required=True,
        help="Longitude of goal waypoint.",
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            args.filter_config,
            args.canbus_config,
            args.base_lat,
            args.base_lon,
            args.goal_lat,
            args.goal_lon,
        )
    )
