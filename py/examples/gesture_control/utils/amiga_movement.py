import asyncio
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from farm_ng_core_pybind import Pose3F64
from google.protobuf.empty_pb2 import Empty

from farm_ng.filter.filter_pb2 import FilterState
from util.track_planner import TrackBuilder

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig

from farm_ng.core.events_file_reader import proto_from_json_file

from farm_ng.track.track_pb2 import Track
from farm_ng.track.track_pb2 import TrackFollowRequest
from farm_ng_core_pybind import Isometry3F64
from farm_ng_core_pybind import Rotation3F64


def calculateOperatorAngle(x, y, frame_width=480):
    """Calculates the angle of the operator.

    Args:
        x, y: Points as (x, y) tuples,
        frame_width: Frame width of the OAK-D camera feed

    Returns:
        Angle in radians
    """
    # Determine the (horizontal) centre point of a human
    centre_point = (x[0] + y[0]) / 2

    offset = (centre_point - (frame_width / 2)) / (frame_width / 2)  # Normalised offset from centre (-1 to 1)

    camera_FOV = np.radians(63)  # OAK-D horizontal FOV in radians
    angle = offset * (camera_FOV / 2)  # Calculate angle based on offset and FOV

    return -angle


async def create_initial_pose(client: Optional[EventClient] = None, timeout: float = 0.5) -> Pose3F64:
    zero_tangent = np.zeros((6, 1), dtype=np.float64)
    start: Pose3F64 = Pose3F64(
        a_from_b=Isometry3F64(), frame_a="world", frame_b="robot", tangent_of_b_in_a=zero_tangent
    )
    if client is not None:
        try:
            state: FilterState = await asyncio.wait_for(
                client.request_reply("/get_state", Empty(), decode=True), timeout=timeout
            )
            start = Pose3F64.from_proto(state.pose)
        except asyncio.TimeoutError:
            print("Timeout while getting filter state")
        except Exception as e:
            print(f"Error getting filter state {e}")
    return start


async def create_initial_pose_updated(client: Optional[EventClient] = None, timeout: float = 0.5) -> Tuple[Pose3F64, float]:
    zero_tangent = np.zeros((6, 1), dtype=np.float64)
    start: Pose3F64 = Pose3F64(
        a_from_b=Isometry3F64(), frame_a="world", frame_b="robot", tangent_of_b_in_a=zero_tangent
    )
    new_start = start
    orientation = 0.0
    if client is not None:
        try:
            state: FilterState = await asyncio.wait_for(
                client.request_reply("/get_state", Empty(), decode=True), timeout=timeout
            )
            start = Pose3F64.from_proto(state.pose)
            orientation = state.heading

            fixed_isometry = Isometry3F64(
                rotation=Rotation3F64.Rz(orientation),
                translation=start.translation
            )
            new_start = Pose3F64(
                a_from_b=fixed_isometry,
                frame_a=start.frame_a,
                frame_b=start.frame_b,
                tangent_of_b_in_a=start.tangent_of_b_in_a
            )
        except asyncio.TimeoutError:
            print("Timeout while getting filter state")
        except Exception as e:
            print(f"Error getting filter state {e}")
    return new_start, orientation


async def pose_to_newpose(x_collar, y_collar, client: Optional[EventClient] = None, timeout: float = 0.5) -> Tuple[Pose3F64, float]:
    initial_pose, orientation = await create_initial_pose_updated(client)
    print(f"Initial pose: x = {initial_pose.translation[0]}, y = {initial_pose.translation[1]}, heading = {orientation}")

    final_pose = np.array([initial_pose.translation[0] + x_collar, initial_pose.translation[1] + y_collar, 0.0])
    final_dir = final_pose - np.array(initial_pose.translation)

    dir = final_dir / np.linalg.norm(final_dir)
    desired_heading = np.arctan2(dir[1], dir[0])

    new_isometry = Isometry3F64(
        rotation=Rotation3F64.Rz(desired_heading),
        translation=final_pose
    )
    new_pose = Pose3F64(
        a_from_b=new_isometry,
        frame_a="world",
        frame_b="robot",
        tangent_of_b_in_a=initial_pose.tangent_of_b_in_a
    )
    print(f"Final pose: x = {new_pose.translation[0]}, y = {new_pose.translation[1]}, heading = {desired_heading}")
    return initial_pose, orientation, new_pose, desired_heading


async def track_forwards(z_coordinate, client: Optional[EventClient] = None, save_track: Optional[Path] = None) -> Track:
    start = await create_initial_pose(client)
    print(f"Initial pose: x = {start.translation[0]}, y = {start.translation[1]}")

    trackbuilder = TrackBuilder(start=start)

    trackbuilder.create_straight_segment(next_frame_b="forwards", distance=z_coordinate, spacing=0.05)

    if save_track is not None:
        trackbuilder.save_track(save_track)

    return trackbuilder.track


async def track_to_operator(z_coordinate, left_hip, right_hip, client: Optional[EventClient] = None, save_track: Optional[Path] = None) -> Track:
    initial_pose, orientation = await create_initial_pose_updated(client)
    print(f"Initial pose: x = {initial_pose.translation[0]}, y = {initial_pose.translation[1]}, heading = {orientation}")

    human_angle = calculateOperatorAngle(left_hip, right_hip, frame_width=480)

    print(f"human angle: {human_angle}")

    desired_heading = np.arctan2(np.sin(human_angle), np.cos(human_angle))

    trackbuilder = TrackBuilder(start=initial_pose)

    print("Created track builder")

    trackbuilder.create_turn_segment(next_frame_b="rotate to operator", angle=desired_heading, spacing=0.05)

    trackbuilder.create_straight_segment(next_frame_b="move to operator", distance=z_coordinate, spacing=0.05)

    if save_track is not None:
        trackbuilder.save_track(save_track)

    return trackbuilder.track


async def track_to_dipbob(dipbob_distance, x_collar, y_collar, client: Optional[EventClient] = None, save_track: Optional[Path] = None) -> Track:
    initial_pose, orientation, final_pose, desired_heading = await pose_to_newpose(x_collar, y_collar, client)

    trackbuilder = TrackBuilder(start=initial_pose)
    trackbuilder.create_turn_segment(next_frame_b="move to dipbob", angle=desired_heading, spacing=0.05)
    trackbuilder.create_straight_segment(next_frame_b="orient over dipbob", distance=dipbob_distance, spacing=0.05)

    if save_track is not None:
        trackbuilder.save_track(save_track)

    return trackbuilder.track


async def set_track(service_config: EventServiceConfig, track: Track):
    print("Setting the track")
    await EventClient(service_config).request_reply("/set_track", TrackFollowRequest(track=track))


async def start(service_config: EventServiceConfig) -> None:
    print("Start moving towards the last known location of the operator")
    await EventClient(service_config).request_reply("/start", Empty())


async def run_path(service_config_path: Path, track_path: Path) -> None:
    service_config: EventServiceConfig = proto_from_json_file(service_config_path, EventServiceConfig())

    track: Track = proto_from_json_file(track_path, Track())

    await set_track(service_config, track)

    await start(service_config)


async def coord_move_forwards(
    movement_config: EventServiceConfig,
    filter_client: EventClient,
    z_coordinate: float,
    save_track: Optional[Path] = None,
    movement_client: Optional[EventClient] = None,
) -> None:
    """Move the robot forwards based on detected z-coordinate (forward distance in mm).

    Args:
        config: EventServiceConfig for the track_follower service
        client: EventClient to communicate with services
        z_coordinate: Forward distance in millimetres (from camera depth)
        save_track: Optional Path to save the generated track for debugging
        movement_client: Optional EventClient to use for setting and starting the track
    """
    # Convert z-coordinate distance from mm to metres and apply a scaling factor
    # (replace 1.2 with actual camera height and pitch calibration once determined)
    distance_m = (z_coordinate / 1000.0) / 1.2

    # Create track by generating waypoints from current pose to target
    track = await track_forwards(distance_m, filter_client, save_track)

    try:
        # Use provided movement client if available, otherwise use provided functions to set and start the track
        if movement_client is not None:
            await movement_client.request_reply("/set_track", TrackFollowRequest(track=track))
            await movement_client.request_reply("/start", Empty())
        else:
            # Set the track on the track_follower service
            await set_track(movement_config, track)

            # Start the robot moving along the track
            await start(movement_config)
    except Exception as e:
        print(f"Error during track execution: {e}")

    print(f"Robot is now moving forwards for {distance_m:.3f}m")


async def coord_move_to_operator(
        movement_config: EventServiceConfig,
        filter_client: EventClient,
        z_coordinate: float,
        left_hip: Tuple,
        right_hip: Tuple,
        save_track: Optional[Path] = None,
        movement_client: Optional[EventClient] = None,
) -> None:
    """Orient the robot towards the operator and move the robot forwards based on detected
    z-coordinate (forward distance in mm).

    Args:
        config: EventServiceConfig for the track_follower service
        client: EventClient to communicate with services
        z_coordinate: Forward distance in millimetres (from camera depth)
        left_hip: (x, y) coordinates of the operator's left hip keypoint
        right_hip: (x, y) coordinates of the operator's right hip keypoint
        save_track: Optional Path to save the generated track for debugging
        movement_client: Optional EventClient to use for setting and starting the track
    """
    # Convert z-coordinate distance from mm to metres and apply a scaling factor
    distance_m = (z_coordinate / 1000.0) / 1.8

    # Create track by generating waypoints from current pose to target
    track = await track_to_operator(distance_m, left_hip, right_hip, filter_client, save_track)

    try:
        # Use provided movement client if available, otherwise use provided functions to set and start the track
        if movement_client is not None:
            await movement_client.request_reply("/set_track", TrackFollowRequest(track=track))
            await movement_client.request_reply("/start", Empty())
        else:
            # Set the track on the track_follower service
            await set_track(movement_config, track)

            # Start the robot moving along the track
            await start(movement_config)
            print("Robot should have followed the track")
    except Exception as e:
        print(f"Error during track execution: {e}")

    print(f"Robot is now moving forwards towards the operator for {distance_m:.3f}m")


async def robot_dipbob(
    movement_config: EventServiceConfig,
    filter_client: EventClient,
    dipbob_distance: float,
    x_collar: float,
    y_collar: float,
    save_track: Optional[Path] = None,
    movement_client: Optional[EventClient] = None,
) -> None:
    """Orient the robot's dipbob over the collar according to the collar coordinates and dipbob distance,
    and dip the collar.

    Args:
    config: EventServiceConfig for the track follower service
    client: EventClient to communicate with services
    dipbob_distance: Forward distance in metres from the dipbob to the camera
    x_collar: x-coordinate of the collar in the camera frame
    y_collar: y-coordinate of the collar in the camera frame
    save_track: Optional Path to save the generated track for debugging
    movement_client: Optional EventClient to use for setting and starting the track
    """
    x_collar = x_collar / 2.4
    y_collar = y_collar / 3.0
    distance = dipbob_distance + x_collar

    track = await track_to_dipbob(distance, x_collar, y_collar, filter_client, save_track)

    try:
        if movement_client is not None:
            await movement_client.request_reply("/set_track", TrackFollowRequest(track=track))
            await movement_client.request_reply("/start", Empty())
        else:
            await set_track(movement_config, track)

            await start(movement_config)
    except Exception as e:
        print(f"Error during track execution: {e}")