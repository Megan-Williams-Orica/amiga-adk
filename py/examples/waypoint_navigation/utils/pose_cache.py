# pose_cache.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class Pose2D:
    x: float     # meters, map/world frame
    y: float     # meters, map/world frame
    yaw: float   # radians, robot heading in world frame (CCW +, 0 along +X)
    converged: bool = False  # filter convergence status

_latest: Optional[Pose2D] = None

def set_latest_pose(x: float, y: float, yaw: float, converged: bool = False) -> None:
    global _latest
    _latest = Pose2D(x, y, yaw, converged)

def get_latest_pose() -> Optional[Pose2D]:
    return _latest
