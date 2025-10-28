#!/usr/bin/env python3
"""
Shared navigation state for inter-process communication between
main.py navigation system and Flask GUI.

This module provides thread-safe access to navigation status for
display in the web interface.

Uses a JSON file for inter-process communication.
"""

import threading
import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

# State file location
STATE_FILE = Path("/tmp/amiga_navigation_state.json")

# Global state lock
_state_lock = threading.Lock()


@dataclass
class NavigationStateData:
    """Navigation state data structure"""
    # Waypoint tracking
    current_waypoint_index: int = 0
    total_waypoints: int = 0
    completed_waypoints: List[int] = None  # List of completed waypoint indices

    # Track follower status
    track_status: str = "IDLE"  # IDLE, FOLLOWING, TRACK_COMPLETE, TRACK_ABORTED, etc.

    # GPS/Filter state
    gps_quality: str = "UNKNOWN"  # UNKNOWN, NO_FIX, FIX, RTK_FLOAT, RTK_FIX

    # Vision system
    vision_active: bool = False
    vision_override_active: bool = False

    # System state
    navigation_running: bool = False

    def __post_init__(self):
        if self.completed_waypoints is None:
            self.completed_waypoints = []


# Global state instance (in-memory cache)
_nav_state = NavigationStateData()


def _load_state_from_file() -> NavigationStateData:
    """Load state from file, return default if file doesn't exist."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)
                return NavigationStateData(**data)
        except Exception:
            pass
    return NavigationStateData()


def _save_state_to_file(state: NavigationStateData) -> None:
    """Save state to file atomically."""
    try:
        # Write to temp file first, then rename (atomic operation)
        temp_file = STATE_FILE.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(asdict(state), f)
        temp_file.replace(STATE_FILE)
        # Debug logging
        print(f"[NAV_STATE] Saved: WP {state.current_waypoint_index}/{state.total_waypoints}, status={state.track_status}")
    except Exception as e:
        print(f"Warning: Failed to save navigation state: {e}")


def set_navigation_state(
    current_waypoint_index: Optional[int] = None,
    total_waypoints: Optional[int] = None,
    track_status: Optional[str] = None,
    gps_quality: Optional[str] = None,
    vision_active: Optional[bool] = None,
    vision_override_active: Optional[bool] = None,
    navigation_running: Optional[bool] = None,
) -> None:
    """
    Update navigation state (thread-safe, persists to file).
    Only updates fields that are not None.
    """
    global _nav_state

    with _state_lock:
        # Load current state from file
        _nav_state = _load_state_from_file()

        # Update fields
        if current_waypoint_index is not None:
            _nav_state.current_waypoint_index = current_waypoint_index
        if total_waypoints is not None:
            _nav_state.total_waypoints = total_waypoints
        if track_status is not None:
            _nav_state.track_status = track_status
        if gps_quality is not None:
            _nav_state.gps_quality = gps_quality
        if vision_active is not None:
            _nav_state.vision_active = vision_active
        if vision_override_active is not None:
            _nav_state.vision_override_active = vision_override_active
        if navigation_running is not None:
            _nav_state.navigation_running = navigation_running

        # Save to file
        _save_state_to_file(_nav_state)


def mark_waypoint_complete(waypoint_index: int) -> None:
    """Mark a waypoint as completed (thread-safe, persists to file)."""
    global _nav_state

    with _state_lock:
        _nav_state = _load_state_from_file()
        if waypoint_index not in _nav_state.completed_waypoints:
            _nav_state.completed_waypoints.append(waypoint_index)
        _save_state_to_file(_nav_state)


def get_navigation_state() -> Dict[str, Any]:
    """
    Get complete navigation state (thread-safe, reads from file).

    Returns:
        Dictionary with all navigation state fields
    """
    with _state_lock:
        state = _load_state_from_file()
        return asdict(state)


def clear_navigation_state() -> None:
    """Reset all navigation state (useful for new navigation session)."""
    with _state_lock:
        _save_state_to_file(NavigationStateData())


def get_waypoint_status(waypoint_index: int) -> str:
    """
    Get status of a specific waypoint.

    Returns:
        "completed", "current", or "pending"
    """
    with _state_lock:
        state = _load_state_from_file()
        if waypoint_index in state.completed_waypoints:
            return "completed"
        elif waypoint_index == state.current_waypoint_index:
            return "current"
        else:
            return "pending"
