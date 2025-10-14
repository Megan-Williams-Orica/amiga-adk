# Copyright (c) farm-ng, inc.
#
# Licensed under the Amiga Development Kit License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/farm-ng/amiga-dev-kit/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import math
import socket
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig, EventServiceConfigList
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.track.track_pb2 import (
    RobotStatus,
    Track,
    TrackFollowerState,
    TrackFollowRequest,
    TrackStatusEnum,
)
from utils.actuator import BaseActuator, NullActuator, CanHBridgeActuator
from farm_ng_core_pybind import Pose3F64
from google.protobuf.empty_pb2 import Empty
from motion_planner import MotionPlanner
from utils.navigation_manager import NavigationManager
from utils.multiclient import MultiClientSubscriber as multi
from utils.pose_cache import set_latest_pose

logger = logging.getLogger("Navigation Manager")

def setup_signal_handlers(nav_manager: NavigationManager) -> None:
    """Setup signal handlers for graceful shutdown."""

    def signal_handler(signum, frame):
        logger.info(f"\nReceived signal {signum}, initiating shutdown...")
        nav_manager.shutdown_requested = True

        if nav_manager.main_task and not nav_manager.main_task.done():
            nav_manager.main_task.cancel()

        if hasattr(signal_handler, "call_count"):
            signal_handler.call_count += 1
            if signal_handler.call_count > 1:
                logger.info("Second signal received, forcing exit")
                sys.exit(1)
        else:
            signal_handler.call_count = 1

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def _update_pose_from_filter(filter_msg) -> None:
    """
    Call this from your existing filter/localization callback.
    Replace field accessors below to match your message.
    """
    # Example mappings (adjust to your types):
    # If you have a Pose3F64: pose.a_from_b.translation = [y, x, z] or [x,y,z] depending on your convention.
    x_m = float(filter_msg.pose.translation[0])  # map/world X (meters)
    y_m = float(filter_msg.pose.translation[1])  # map/world Y (meters)

    # Heading (yaw, radians). If you have a rotation SO3, the z-yaw is log()[-1]
    # yaw_rad = float(filter_msg.pose.rotation.log()[-1])
    # or if heading is provided directly:
    yaw_rad = float(getattr(filter_msg, "pose_yaw_rad", 0.0))

    set_latest_pose(x_m, y_m, yaw_rad)
        
# Vision configuration
VISION_SEARCH_RADIUS_M = 2.0  # Search radius around CSV waypoints for cone detection
VISION_WAIT_TIMEOUT_S = 10.0  # How long to wait for cone detection before skipping waypoint

def is_vision_enabled() -> bool:
    """Check if detectionPlot.py is running to determine if vision mode is active."""
    import subprocess
    try:
        result = subprocess.run(
            ["pgrep", "-f", "detectionPlot.py"],
            capture_output=True,
            text=True,
            timeout=1.0
        )
        return result.returncode == 0  # Returns 0 if process found
    except Exception:
        return False

# TODO: Refactor into a separate module
async def vision_goal_listener(motion_planner, controller_client, nav_manager, proximity_m=2.0):
    """
    Listens for UDP 'cone_goal' messages and overrides the follower by:
      /cancel -> /set_track (short vision track) -> /start
    Now also /pause if auto mode is disabled (robot not controllable).

    When vision is enabled (detectionPlot.py is sending messages):
    - CSV waypoints act as 'search zones' with radius VISION_SEARCH_RADIUS_M
    - Only cones detected within the search radius of the current CSV waypoint are accepted
    - Robot will wait/skip waypoint if no cone is detected in the zone
    """

    # --- helpers ------------------------------------------------------------
    async def get_state() -> TrackFollowerState:
        return await controller_client.request_reply("/get_state", Empty(), decode=True)

    # True if the follower is in any terminal state (COMPLETE / ABORTED / FAILED)
    def _is_terminal(st: TrackFollowerState) -> bool:
        return st.status.track_status in (
            TrackStatusEnum.TRACK_COMPLETE,
            TrackStatusEnum.TRACK_ABORTED,
            TrackStatusEnum.TRACK_FAILED,
        )
       
    # Pulls robot_status.controllable from the state (i.e., AUTO mode enabled and no safety fault).
    def controllable_from_state(st) -> bool:
        rs = getattr(st, "robot_status", None)
        return bool(getattr(rs, "controllable", False))
 
    async def wait_until(pred, timeout: float, poll_s: float = 0.05):
        """
        Poll follower state until pred(state) is True, or timeout elapses.
        Assumes get_state() returns a decoded TrackFollowerState (decode=True).
        """
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout

        # initial sample
        st = await get_state()  # TrackFollowerState
        while not pred(st):
            if loop.time() >= deadline:
                raise TimeoutError("wait condition not met")

            await asyncio.sleep(poll_s)

            try:
                st = await get_state()  # keep sampling
                if st is None:
                    # extremely defensive: treat as transient failure
                    continue
            except Exception as e:
                # transient RPC/wire hiccup; keep trying until deadline
                # (optionally log: print(f"[VISION] get_state failed: {e}"))
                continue

        return st


    # One mutex to rule all controller RPCs (prevents races with nav loop)
    if not hasattr(nav_manager, "_controller_lock"):
        nav_manager._controller_lock = asyncio.Lock()
    ctl_lock: asyncio.Lock = nav_manager._controller_lock

    # Latch state on the nav_manager object so it's visible across loops
    if not hasattr(nav_manager, "vision_latched"):
        nav_manager.vision_latched = False
    if not hasattr(nav_manager, "vision_latch_deadline"):
        nav_manager.vision_latch_deadline = 0.0

    # Track when cone was last detected in search zone for current waypoint
    if not hasattr(nav_manager, "cone_detected_for_current_wp"):
        nav_manager.cone_detected_for_current_wp = False
    if not hasattr(nav_manager, "current_waypoint_start_time"):
        nav_manager.current_waypoint_start_time = None

    # Track which waypoint index we last successfully triggered vision for
    if not hasattr(nav_manager, "vision_completed_waypoints"):
        nav_manager.vision_completed_waypoints = set()

    LATCH_MAX_S = 20.0  # safety timeout for a vision retarget (tune as you like)

    # --- socket setup -------------------------------------------------------
    loop = asyncio.get_running_loop()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try: sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except Exception: pass
    sock.bind(("127.0.0.1", 41234))
    sock.settimeout(1.0)

    last_goal = None
    last_sent_t = 0.0
    MIN_DIST_DELTA = 0.35
    MIN_PERIOD_S   = 0.8

    print("Started vision goal listener")

    try:
        while True:
            # --- while latched, just watch state and drop messages ---
            if nav_manager.vision_latched:
                # auto-unlatch on terminal state or timeout
                try:
                    st = await get_state()
                    if _is_terminal(st) or (asyncio.get_event_loop().time() >= nav_manager.vision_latch_deadline):
                        print("[VISION] unlatching (terminal state or timeout)")
                        nav_manager.vision_latched = False
                        # small grace to avoid immediate re-trigger on the same frame
                        await asyncio.sleep(0.2)
                except Exception:
                    # if state read hiccups, keep latch until deadline
                    pass

                # drain/ignore incoming UDP quickly, then continue loop
                try:
                    _ = await asyncio.wait_for(
                        asyncio.get_running_loop().run_in_executor(None, sock.recvfrom, 4096), timeout=0.01
                    )
                except Exception:
                    await asyncio.sleep(0.03)
                continue

            # ---- receive next message (non-latched path) ----
            try:
                data, _ = await loop.run_in_executor(None, sock.recvfrom, 4096)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[VISION] recv error: {e}")
                continue

            # ---- parse ----
            try:
                msg = json.loads(data.decode())
                if msg.get("class_id") != int(6):
                    continue
                name = str(msg["class_name"])
                x = float(msg["x_fwd_m"])
                y = float(msg["y_left_m"])
                
                conf = float(msg.get("confidence", 1.0))
            except Exception as e:
                print(f"[VISION] bad msg: {e}")
                continue

            # ---- pose check ----
            pose = await motion_planner._get_current_pose()
            if pose is None:
                print("[VISION] skip: pose None (filter down)")
                continue
            
            yaw = pose.a_from_b.rotation.log()[-1]
            xr, yr = pose.a_from_b.translation[0], pose.a_from_b.translation[1]
            c, s = math.cos(yaw), math.sin(yaw)

            # robot frame (x = forward, y = left)  -> world/map (X, Y)
            
            X_obj = xr + x * c - y * s
            Y_obj = yr + x * s + y * c
            print(f"[VISION] WP Location: Y={X_obj:.2f} m, X={-Y_obj:.2f} m  | robot @ ({xr:.2f}, {-yr:.2f}), ψ={math.degrees(yaw):.2f} deg")
                    
            # ---- distance from detected cone to ORIGINAL CSV waypoint (search zone center) ----
            def cone_distance_from_csv_waypoint():
                """
                Calculate distance from detected cone to the ORIGINAL CSV waypoint.
                This is used to validate if the cone is within the search zone.
                The search zone is centered at current_waypoint_index (the target we're executing).
                """
                # Get the original CSV waypoint for the CURRENT target waypoint
                # After _create_ab_segment_to_next_waypoint() increments the index,
                # current_waypoint_index already points to the target waypoint
                csv_waypoint = None
                idx = max(1, motion_planner.current_waypoint_index)

                if hasattr(motion_planner, "original_csv_waypoints"):
                    csv_waypoint = motion_planner.original_csv_waypoints.get(idx)

                # Fallback to regular waypoints if original_csv_waypoints not available
                if csv_waypoint is None and hasattr(motion_planner, "waypoints"):
                    if hasattr(motion_planner.waypoints, "get"):
                        csv_waypoint = motion_planner.waypoints.get(idx)
                    elif isinstance(motion_planner.waypoints, (list, tuple)) and idx < len(motion_planner.waypoints):
                        csv_waypoint = motion_planner.waypoints[idx]

                # Calculate distance from detected cone to CSV waypoint
                if csv_waypoint is not None:
                    csv_x = float(csv_waypoint.a_from_b.translation[0])
                    csv_y = float(csv_waypoint.a_from_b.translation[1])
                    distance = math.hypot(X_obj - csv_x, Y_obj - csv_y)
                    return distance, idx
                return None, None

            dist_goal_wp, target_wp_idx = cone_distance_from_csv_waypoint()

            # ---- cone validation gates ----
            r = math.hypot(x, y)

            # Debug output
            if dist_goal_wp is not None:
                # Get waypoint coordinates for display (target_wp_idx is the NEXT waypoint)
                csv_waypoint = motion_planner.original_csv_waypoints.get(target_wp_idx)
                if csv_waypoint is not None:
                    wp_x = float(csv_waypoint.a_from_b.translation[0])
                    wp_y = float(csv_waypoint.a_from_b.translation[1])
                    print(f"[VISION] Cone @ ({X_obj:.2f}, {Y_obj:.2f}) is {dist_goal_wp:.2f}m from CSV waypoint {target_wp_idx} @ ({wp_x:.2f}, {wp_y:.2f}) (search radius: {VISION_SEARCH_RADIUS_M:.2f}m)")
                else:
                    print(f"[VISION] Cone @ ({X_obj:.2f}, {Y_obj:.2f}) is {dist_goal_wp:.2f}m from CSV waypoint {target_wp_idx} (search radius: {VISION_SEARCH_RADIUS_M:.2f}m)")
            else:
                print(f"[VISION] Warning: Could not determine CSV waypoint location (current_idx={motion_planner.current_waypoint_index}, target_idx={max(1, motion_planner.current_waypoint_index)})")
            print(f"[VISION] Cone rf: x={x:.2f} y={y:.2f} r={r:.2f} conf={conf:.2f}")

            # NEW BEHAVIOR: Cone must be within VISION_SEARCH_RADIUS_M of the CSV waypoint (search zone)
            IN_SEARCH_ZONE = (dist_goal_wp is not None and dist_goal_wp <= VISION_SEARCH_RADIUS_M)

            # Cone quality checks (robot frame): reasonable distance, not too far left/right, good confidence
            CONE_OK = (0.3 <= r <= 6.0 and abs(y) <= 3.0 and x >= 0.3 and conf >= 0.5)

            # Combined gate: cone must be in search zone AND pass quality checks
            if not (IN_SEARCH_ZONE and CONE_OK):
                if not IN_SEARCH_ZONE:
                    if dist_goal_wp is not None:
                        print(f"[VISION] skip: cone outside search zone (dist={dist_goal_wp:.2f}m > {VISION_SEARCH_RADIUS_M:.2f}m)")
                    else:
                        print(f"[VISION] skip: no CSV waypoint found for search zone validation")
                elif not CONE_OK:
                    print(f"[VISION] skip: cone quality check failed (r={r:.2f}, y={y:.2f}, conf={conf:.2f})")
                continue

            # ---- Check if actuator is deploying ----
            if getattr(nav_manager, "actuator_deploying", False):
                print("[VISION] skip: actuator is deploying")
                continue

            # ---- Check if we've already processed this waypoint ----
            current_wp_idx = motion_planner.current_waypoint_index
            if current_wp_idx in nav_manager.vision_completed_waypoints:
                print(f"[VISION] skip: waypoint {current_wp_idx} already processed by vision")
                continue

            # ---- debounce ----
            now = asyncio.get_event_loop().time()
            if last_goal is not None:
                moved = math.hypot(x - last_goal[0], y - last_goal[1])
                if moved < MIN_DIST_DELTA and (now - last_sent_t) < MIN_PERIOD_S:
                    print(f"[VISION] skip: debounce (moved={moved:.2f}, dt={now-last_sent_t:.2f})")
                    continue

            # ---- don't stack overrides ----
            if getattr(nav_manager, "vision_active", False):
                print("[VISION] skip: vision_active already true")
                continue

            # --- Build track directly to detected cone (robot-relative coordinates) ---
            try:
                # Mark that we detected a valid cone for the current waypoint
                nav_manager.cone_detected_for_current_wp = True

                # IMPORTANT: Set vision_active FIRST before vision_executed_segment
                # This ensures the navigation manager will properly wait for vision to complete
                # If we set vision_executed_segment before vision_active, the navigation manager
                # will see vision_executed_segment=True but vision_active=False, causing it to
                # continue to the next segment before vision actually starts executing
                if hasattr(nav_manager, "vision_active"):
                    nav_manager.vision_active = True

                # Mark that vision will execute this segment (so main loop should skip execution)
                nav_manager.vision_executed_segment = True
                # Mark this waypoint as completed by vision to prevent re-triggering
                nav_manager.vision_completed_waypoints.add(current_wp_idx)
                logger.info(f"[VISION] Valid cone detected in search zone - building track to cone at ({x:.2f}m fwd, {y:.2f}m left)")

                # Build track using robot-relative coordinates (CSV waypoints unchanged!)
                track_to_cone, goal_pose = await motion_planner.build_track_to_robot_relative_goal(
                    x_fwd_m=x, y_left_m=y, standoff_m=0.75, spacing=0.1
                )
                print(f"[VISION] Built track to cone with {len(track_to_cone.waypoints)} waypoints")

            except Exception as e:
                print(f"[VISION] Failed to build track to cone: {e}")
                # Reset vision_active if we failed to build track
                if hasattr(nav_manager, "vision_active"):
                    nav_manager.vision_active = False
                continue

            print("[VISION] OK → cancel + set_track + start (driving to cone)")

            # Activate latch to lock onto cone
            nav_manager.vision_latched = True
            nav_manager.vision_latch_deadline = asyncio.get_event_loop().time() + LATCH_MAX_S
            print(f"[VISION] latched on target for up to {LATCH_MAX_S:.1f} s")

            try:
                async with ctl_lock:
                    # Get current state
                    st = await get_state()

                    # If currently following a track, pause it
                    if st.status.track_status == TrackStatusEnum.TRACK_FOLLOWING:
                        try:
                            await controller_client.request_reply("/pause", Empty())
                            print("[VISION] paused current track")
                        except Exception as e:
                            print(f"[VISION] pause failed: {e}")

                    # C) Cancel current track
                    try:
                        await controller_client.request_reply("/cancel", Empty())
                    except Exception:
                        pass

                    # D) Wait until not FOLLOWING
                    try:
                        await wait_until(lambda s: s.status.track_status != TrackStatusEnum.TRACK_FOLLOWING, timeout=3.0)
                    except TimeoutError:
                        print("[VISION] warn: still FOLLOWING after cancel; proceeding anyway")

                    # E) Set track to cone and start
                    req = TrackFollowRequest(track=track_to_cone)
                    await controller_client.request_reply("/set_track", req)
                    await wait_until(lambda s: s.status.track_status == TrackStatusEnum.TRACK_LOADED, timeout=2.0)
                    await controller_client.request_reply("/start", Empty())

                    print("[VISION] Track started - driving to cone")

                    # F) Wait for track to cone to complete
                    st = await wait_until(lambda s: _is_terminal(s), timeout=30.0)

                    if st.status.track_status == TrackStatusEnum.TRACK_COMPLETE:
                        print("[VISION] Successfully reached cone")

                        # G) Execute full deployment sequence
                        if nav_manager.actuator_enabled:
                            nav_manager.actuator_deploying = True
                            try:
                                from utils.canbus import trigger_dipbob

                                # Wait briefly before deployment
                                await asyncio.sleep(2.0)

                                # Deploy dipbob
                                await trigger_dipbob("can0")
                                logger.info("[VISION] Deploying dipbob")
                                await asyncio.sleep(3.0)

                                # Move forward (tool_to_origin) so robot center is over hole
                                origin_track = await motion_planner.create_tool_to_origin_segment()
                                req = TrackFollowRequest(track=origin_track)
                                await controller_client.request_reply("/set_track", req)
                                await wait_until(lambda s: s.status.track_status == TrackStatusEnum.TRACK_LOADED, timeout=2.0)
                                await controller_client.request_reply("/start", Empty())
                                await wait_until(lambda s: _is_terminal(s), timeout=15.0)

                                logger.info("[VISION] Tool-to-origin move complete")

                                # Open and close chute
                                await nav_manager.actuator.pulse_sequence(
                                    open_seconds=nav_manager.actuator_open_seconds,
                                    close_seconds=nav_manager.actuator_close_seconds,
                                    rate_hz=nav_manager.actuator_rate_hz,
                                    settle_before=3.0,
                                    settle_between=0.0,
                                    wait_for_enter_between=False,
                                    enter_prompt="Hole measured. Press ENTER to close the chute...",
                                    enter_timeout=30.0,
                                )
                                logger.info("[VISION] Deployment sequence complete")

                                # NOTE: Do NOT increment waypoint index here!
                                # The motion planner already increments when creating the next track segment.
                                # Double-incrementing would cause waypoints to be skipped.

                            finally:
                                nav_manager.actuator_deploying = False
                    else:
                        print(f"[VISION] Track to cone failed with status {st.status.track_status}")

            except TimeoutError as e:
                print(f"[VISION] Timeout during cone approach: {e}")
            except Exception as e:
                print(f"[VISION] Error during cone approach/deployment: {e}")
                import traceback
                traceback.print_exc()
            finally:
                if hasattr(nav_manager, "vision_active"):
                    nav_manager.vision_active = False
                await asyncio.sleep(0.1)

    except asyncio.CancelledError:
        pass
    finally:
        try: sock.close()
        except Exception: pass

async def main(args) -> None:
    """Main function to orchestrate waypoint navigation."""
    nav_manager = None
    actuator: BaseActuator = NullActuator()
    
    service_config_list = proto_from_json_file(args.config, EventServiceConfigList())
    mc = multi(service_config_list)

    filter_client = mc.clients["filter"]
    controller_client = mc.clients["track_follower"]
    canbus_client = mc.clients.get("canbus")

    try:
        # Initialize motion planner
        logger.info("Initializing motion planner...")
        motion_planner = MotionPlanner(
            client=filter_client,
            tool_config_path=args.tool_config_path, # offset of centre of robot to dipper
            # camera_offset_path=args.camera_offset_path,
            waypoints_path=args.waypoints_path,
            last_row_waypoint_index=args.last_row_waypoint_index,
            turn_direction=args.turn_direction,
            row_spacing=args.row_spacing,
            headland_buffer=args.headland_buffer,
        )

        actuator: BaseActuator = (
            CanHBridgeActuator(client=canbus_client,
                               actuator_id=args.actuator_id)
            if args.actuator_enabled and canbus_client is not None else
            NullActuator()
        )

        # Visual waypoint follower
        # asyncio.create_task(vision_goal_listener(motion_planner, controller_client, nav_manager))


        # Create nav_manager and inject actuator
        nav_manager = NavigationManager(
            filter_client=filter_client,
            controller_client=controller_client,
            motion_planner=motion_planner,
            no_stop=args.no_stop,
            canbus_client=canbus_client,
            actuator=actuator,
        )

        asyncio.create_task(
        vision_goal_listener(motion_planner, controller_client, nav_manager, proximity_m=2.0)
        )
        
        setup_signal_handlers(nav_manager=nav_manager)
        nav_manager.main_task = asyncio.current_task()

        # Run navigation
        await nav_manager.run_navigation()

    except asyncio.CancelledError:
        logger.info("Main task cancelled")
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt in main")
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}")

    finally:
        # Save navigation progress to JSON file
        progress_path = Path("./visualization/navigation_progress.json")
        try:
            serializable_progress = {}
            for segment_name, track in (nav_manager.navigation_progress if nav_manager else {}).items():
                x: List[float] = []
                y: List[float] = []
                heading: List[float] = []

                track_waypoints = [Pose3F64.from_proto(
                    pose) for pose in track.waypoints]
                for pose in track_waypoints:
                    x.append(pose.a_from_b.translation[0])
                    y.append(pose.a_from_b.translation[1])
                    heading.append(pose.a_from_b.rotation.log()[-1])

                serializable_progress[segment_name] = {
                    "waypoints_count": len(track.waypoints),
                    "x": x,
                    "y": y,
                    "heading": heading,
                }

            progress_path.parent.mkdir(parents=True, exist_ok=True)
            with open(progress_path, "w") as f:
                json.dump(serializable_progress, f, indent=2)
            logger.info(f"Navigation progress saved to {progress_path}")
        except Exception as e:
            logger.error(f"FAILED to save navigation progress: {e}")

        positions_path = Path("./visualization/robot_positions.json")
        try:
            positions_path.parent.mkdir(parents=True, exist_ok=True)
            with open(positions_path, "w") as f:
                json.dump(
                    getattr(nav_manager, "robot_positions", []), f, indent=2)
            logger.info(f"Robot positions saved to {positions_path}")
        except Exception as e:
            logger.error(f"FAILED to save robot positions: {e}")

        if nav_manager and not nav_manager.shutdown_requested:
            await nav_manager._cleanup()
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python main.py", description="Waypoint navigation using MotionPlanner and track_follower service"
    )
    # Required
    parser.add_argument("--waypoints-path", type=Path, required=True,
                        help="Path to waypoints JSON or CSV file (Track format)")
    parser.add_argument("--tool-config-path", type=Path,
                        required=True, help="Path to tool configuration JSON file")
    parser.add_argument(
        "--actuator-enabled",
        action="store_true",
        help="If set, pulse H-bridge OPEN briefly when the robot reaches each waypoint.",
    )
    parser.add_argument("--actuator-id", type=int, default=0,
                        help="H-bridge actuator ID (default: 0)")
    parser.add_argument(
        "--actuator-open-seconds",
        type=float,
        default=6,
        help="Duration to drive actuator in OPEN after reaching a waypoint (seconds).",
    )
    parser.add_argument(
        "--actuator-close-seconds",
        type=float,
        default=6,
        help="Duration to drive actuator in CLOSE after reaching a waypoint (seconds).",
    )
    parser.add_argument(
        "--actuator-rate-hz",
        type=float,
        default=10.0,
        help="Command publish rate to CAN bus while reversing (Hz).",
    )

    # MotionPlanner configuration
    parser.add_argument(
        "--last-row-waypoint-index",
        type=int,
        default=6,
        help="Index of the last waypoint in the current row (default: 6)",
    )
    parser.add_argument(
        "--turn-direction",
        choices=["left", "right"],
        default="left",
        help="Direction to turn at row ends (default: left)",
    )
    parser.add_argument(
        "--row-spacing",
        type=float,
        default=6.0,
        help="Spacing between rows in meters (default: 6.0)",
    )
    parser.add_argument(
        "--headland-buffer",
        type=float,
        default=2.0,
        help="Buffer distance for headland maneuvers in meters (default: 2.0)",
    )
    parser.add_argument("--no-stop", action="store_true",
                        help="Disable stopping at each waypoint"
    )
    parser.add_argument("--config", 
                        type=Path, 
                        required=True, 
                        help="The system config."
    )
    args = parser.parse_args()
    
    # xfm = Transforms(args.camera_offset_path)
    # os.environ["CAMERA_OFFSET_CONFIG"] = str(args.camera_offset_path)
    
    # Run the main function
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("\nFinal keyboard interrupt")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
    finally:
        logger.info("Script terminated")
        sys.exit(0)
