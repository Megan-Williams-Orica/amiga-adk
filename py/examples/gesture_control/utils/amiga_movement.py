import asyncio
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from farm_ng_core_pybind import Pose3F64
from google.protobuf.empty_pb2 import Empty

from farm_ng.filter.filter_pb2 import FilterState
from utils.track_planner import TrackBuilder

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig

from farm_ng.core.events_file_reader import proto_from_json_file

from farm_ng.track.track_pb2 import Track
from farm_ng.track.track_pb2 import TrackFollowRequest
from farm_ng_core_pybind import Isometry3F64
from farm_ng_core_pybind import Rotation3F64

from utils.pose_recognition import poseKeypoints

# Initialise pose keypoints classifier to access angle calculations
keypoints_angle = poseKeypoints(confidence_threshold=0.3)


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


async def create_initial_pose_updated(client: Optional[EventClient] = None, timeout: float = 0.5) -> Pose3F64:
    # Beware: There is a bug in this function which seems to cause the robot to rotate during
    #         execution of movement forwards. A fix has been applied, but is yet to be verified
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
            orientation = state.heading

            # To fix bug: preserve pose by creating a new pose with the same information and fix translation
            preserved_pose = Pose3F64(
                a_from_b=Isometry3F64(),
                frame_a=start.frame_a,
                frame_b=start.frame_b,
                tangent_of_b_in_a=start.tangent_of_b_in_a
            )
            preserved_pose.translation = start.translation
            new_start = preserved_pose
        except asyncio.TimeoutError:
            print("Timeout while getting filter state")
        except Exception as e:
            print(f"Error getting filter state {e}")
    return new_start, orientation


async def track_forwards(z_coordinate, client: Optional[EventClient] = None, save_track: Optional[Path] = None) -> Track:
    start: Pose3F64 = await create_initial_pose(client)
    print(f"Initial pose: x = {start.translation[0]}, y = {start.translation[1]}")

    trackbuilder = TrackBuilder(start=start)

    trackbuilder.create_straight_segment(next_frame_b="forwards", distance=z_coordinate, spacing=0.05)

    if save_track is not None:
        trackbuilder.save_track(save_track)

    return trackbuilder.track

# TODO: Create a async function to create segments between the Amiga and the operator, where the Amiga
#       tracks the operator's position and orientation by creating the initial pose.
#       Then, a track is built by creating the appropriate segments. These segments should allow the Amiga
#       to line up with the operator's orientation and then move forwards towards them.

# Potentially, you could observe the heading of the robot, and take the difference between the heading and
# the operator's orientation to determine how much to rotate, and in
#  which direction. Then, you could create a
# track with a rotation segment followed by a straight segment.

# This TODO has since been implemented below, but is yet to be verified in functionality


async def track_to_operator(z_coordinate, left_hip, right_hip, client: Optional[EventClient] = None, save_track: Optional[Path] = None) -> Track:
    initial_pose: Pose3F64 = await create_initial_pose_updated(client)
    print(f"Initial pose: x = {initial_pose.translation[0]}, y = {initial_pose.translation[1]}, heading = {initial_pose.orientation}")

    human_angle = keypoints_angle.calculateHorizontalAngle(left_hip, right_hip)
    desired_heading = human_angle - initial_pose.orientation
    desired_heading_wrapped = np.arctan2(np.sin(desired_heading), np.cos(desired_heading))

    trackbuilder = TrackBuilder(start=initial_pose)

    trackbuilder.create_ab_segment(next_frame_b="rotate to operator", final_pose=Pose3F64(
        a_from_b=Isometry3F64(rotation=Rotation3F64.Rz(desired_heading_wrapped),
                              translation=np.zeros(3)), frame_a=initial_pose.frame_b, frame_b="rotated pose", tangent_of_b_in_a=np.zeros((6, 1), dtype=np.float64)), spacing=0.05)

    trackbuilder.create_straight_segment(next_frame_b="move to operator", distance=z_coordinate, spacing=0.05)

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
        left_hip: Tuple,                                 # Might have to change this
        right_hip: Tuple,                                # Might have to change this
        save_track: Optional[Path] = None,
        movement_client: Optional[EventClient] = None,
) -> None:
    # Convert z-coordinate distance from mm to metres and apply a scaling factor
    # (replace 1.2 with actual camera height and pitch calibration once determined)
    distance_m = (z_coordinate / 1000.0) / 1.2
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
    except Exception as e:
        print(f"Error during track execution: {e}")

    print(f"Robot is now moving forwards towards the operator for {distance_m:.3f}m")
