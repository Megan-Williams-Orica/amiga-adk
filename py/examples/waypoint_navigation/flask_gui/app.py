#!/usr/bin/env python3
"""
Flask GUI for Waypoint Navigation System
Provides web-based monitoring and control interface for the Amiga robot.
"""

import sys
import os
from pathlib import Path
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import json
import threading
import subprocess
import signal
import time
import asyncio
from typing import Optional

# Add parent directory to path to import from navigation system
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.pose_cache import get_latest_pose, set_latest_pose
from utils.navigation_state import get_navigation_state, get_waypoint_status
from utils.camera_frame_cache import get_latest_frame_bytes

# Import filter client dependencies
from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng_core_pybind import Pose3F64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
class NavigationState:
    """Shared state for navigation system"""
    def __init__(self):
        self.navigation_process: Optional[subprocess.Popen] = None
        self.camera_frame = None
        self.detections = []
        self.waypoints = []
        self.robot_pose = None
        self.track_status = "IDLE"
        self.current_waypoint_index = 0
        self.total_waypoints = 0
        self.gps_quality = "UNKNOWN"
        self.vision_active = False

    def is_navigation_running(self) -> bool:
        """Check if navigation process is running"""
        if self.navigation_process is None:
            return False
        return self.navigation_process.poll() is None

state = NavigationState()

# ==================== Routes ====================

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    """
    Stream camera feed with detections overlay.
    Reads the latest frame from detectionPlot via shared camera frame file.
    """
    def generate():
        while True:
            frame_bytes = get_latest_frame_bytes()
            if frame_bytes is not None:
                try:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    print(f"Error streaming frame: {e}")
            time.sleep(1/15)  # 15 FPS max (matches detectionPlot FPS)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/plot_data')
def plot_data():
    """
    Get waypoint plot data for D3.js visualization.
    Returns waypoints, robot position, and planned path.
    """
    # Try to read waypoint data from CSV with status
    waypoint_data = load_waypoint_data()

    # Update waypoint statuses from navigation state
    for wp in waypoint_data:
        wp['status'] = get_waypoint_status(wp['index'])

    # Get current robot pose
    robot_data = None
    pose = get_latest_pose()
    if pose is not None:
        robot_data = {
            'x': pose.x,
            'y': pose.y,
            'heading': pose.yaw
        }

    # Get navigation state
    nav_state = get_navigation_state()

    return jsonify({
        'waypoints': waypoint_data,
        'robot': robot_data,
        'current_index': nav_state['current_waypoint_index'],
        'total': nav_state['total_waypoints']
    })

@app.route('/detection_data')
def detection_data():
    """Get detection scatter plot data from detectionPlot.py"""
    try:
        with open('/tmp/amiga_detections.json', 'r') as f:
            detections = json.load(f)
        return jsonify({'detections': detections})
    except Exception:
        return jsonify({'detections': []})

@app.route('/robot_status')
def robot_status():
    """Get current robot status for display"""
    pose = get_latest_pose()
    nav_state = get_navigation_state()

    # Check if navigation subprocess is actually running
    subprocess_running = state.is_navigation_running()

    # Determine navigation status: subprocess must be running OR state file says running
    # But if subprocess is NOT running and state says running, that's stale - clear it
    if not subprocess_running and nav_state['navigation_running']:
        # Stale state detected - clear it
        from utils.navigation_state import clear_navigation_state
        clear_navigation_state()
        nav_state = get_navigation_state()  # Reload cleared state

    # Final determination: navigation is running if subprocess exists
    nav_running = subprocess_running

    status = {
        'navigation_running': nav_running,
        'track_status': nav_state['track_status'],
        'current_waypoint': nav_state['current_waypoint_index'],
        'total_waypoints': nav_state['total_waypoints'],
        'filter_converged': False,  # Default if no pose available
        'vision_active': nav_state['vision_active'],
        'pose': None
    }

    if pose is not None:
        import math
        status['filter_converged'] = pose.converged
        status['pose'] = {
            'x': pose.x,
            'y': pose.y,
            'heading_deg': math.degrees(pose.yaw)
        }

    return jsonify(status)

# ==================== Socket.IO Events ====================

@socketio.on('connect')
def handle_connect():
    """Client connected"""
    print(f"Client connected: {request.sid}")
    emit('status', {'message': 'Connected to navigation GUI'})

@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected"""
    print(f"Client disconnected: {request.sid}")

@socketio.on('start_navigation')
def handle_start_navigation():
    """Start the navigation system by running run.sh"""
    if state.is_navigation_running():
        emit('error', {'message': 'Navigation already running'})
        return

    try:
        # Run run.sh in the parent directory
        run_script = Path(__file__).resolve().parents[1] / 'run.sh'

        if not run_script.exists():
            emit('error', {'message': f'run.sh not found at {run_script}'})
            return

        # Start navigation process with unbuffered output
        state.navigation_process = subprocess.Popen(
            ['bash', str(run_script)],
            cwd=run_script.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            bufsize=1,  # Line buffered
            universal_newlines=True,
            preexec_fn=os.setsid  # Create new process group for clean shutdown
        )

        # Start output reader thread
        import threading
        def read_output():
            """Read navigation output and emit to clients"""
            try:
                for line in iter(state.navigation_process.stdout.readline, ''):
                    if line:
                        socketio.emit('nav_log', {'message': line.rstrip()})
            except Exception as e:
                socketio.emit('nav_log', {'message': f'Error reading output: {e}'})

        output_thread = threading.Thread(target=read_output, daemon=True)
        output_thread.start()

        emit('success', {'message': 'Navigation started'})
        print(f"Started navigation process (PID: {state.navigation_process.pid})")

    except Exception as e:
        emit('error', {'message': f'Failed to start navigation: {str(e)}'})
        print(f"Error starting navigation: {e}")

@socketio.on('stop_navigation')
def handle_stop_navigation():
    """Stop the navigation system"""
    if not state.is_navigation_running():
        emit('error', {'message': 'Navigation not running'})
        return

    try:
        # Send SIGTERM to process group
        os.killpg(os.getpgid(state.navigation_process.pid), signal.SIGTERM)
        state.navigation_process.wait(timeout=5)
        state.navigation_process = None

        # Clear navigation state file
        from utils.navigation_state import clear_navigation_state
        clear_navigation_state()

        emit('success', {'message': 'Navigation stopped'})
        print("Navigation process stopped")

    except Exception as e:
        # Force kill if graceful shutdown fails
        try:
            os.killpg(os.getpgid(state.navigation_process.pid), signal.SIGKILL)
            state.navigation_process = None
            from utils.navigation_state import clear_navigation_state
            clear_navigation_state()
            emit('warning', {'message': 'Navigation force killed'})
        except:
            emit('error', {'message': f'Failed to stop navigation: {str(e)}'})

@socketio.on('emergency_stop')
def handle_emergency_stop():
    """Emergency stop - immediately kill navigation"""
    if state.is_navigation_running():
        try:
            os.killpg(os.getpgid(state.navigation_process.pid), signal.SIGKILL)
            state.navigation_process = None
        except:
            pass

    # Clear navigation state file
    from utils.navigation_state import clear_navigation_state
    clear_navigation_state()

    emit('success', {'message': 'EMERGENCY STOP ACTIVATED'})
    print("EMERGENCY STOP")

# ==================== Helper Functions ====================

def load_waypoint_data():
    """Load waypoint data from CSV for plotting"""
    waypoints = []

    # Try to read from surveyed waypoints
    waypoint_file = Path(__file__).resolve().parents[1] / 'surveyed-waypoints' / 'physicsLabBack2Lanes.csv'

    if waypoint_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(waypoint_file)

            # Assume CSV has dx, dy columns (ENU coordinates)
            # Note: CSV has dx=X (east), dy=Y (north) in world frame
            for idx, row in df.iterrows():
                waypoints.append({
                    'x': float(row.get('dx', 0)),  # East
                    'y': float(row.get('dy', 0)),  # North
                    'index': int(idx),
                    'status': 'pending'  # Will be updated by navigation system
                })
            print(f"Loaded {len(waypoints)} waypoints from CSV")
        except Exception as e:
            print(f"Error loading waypoints: {e}")
    else:
        print(f"Waypoint file not found: {waypoint_file}")

    return waypoints

async def filter_pose_updater():
    """Subscribe to filter state and continuously update pose cache"""
    try:
        # Load filter service config from parent directory
        filter_config_path = Path(__file__).resolve().parents[1] / 'configs' / 'config.json'

        # Create filter client
        from farm_ng.core.event_service_pb2 import EventServiceConfigList, SubscribeRequest
        config_list = proto_from_json_file(filter_config_path, EventServiceConfigList())

        # Find filter service config
        filter_config = None
        for config in config_list.configs:
            if config.name == "filter":
                filter_config = config
                break

        if filter_config is None:
            print("⚠️  Filter service not found in config, pose updates disabled")
            return

        # Create subscription request manually (since it's not in config)
        from farm_ng.core.uri_pb2 import Uri
        subscription = SubscribeRequest(
            uri=Uri(path="/state", query="service_name=filter"),
            every_n=1
        )

        # Subscribe to filter state
        client = EventClient(filter_config)
        print(f"✓ Subscribed to filter service at {filter_config.host}:{filter_config.port}")

        async for event, message in client.subscribe(subscription, decode=True):
            # Update pose cache with filter state
            pose = Pose3F64.from_proto(message.pose)
            x = float(pose.a_from_b.translation[0])
            y = float(pose.a_from_b.translation[1])
            yaw = float(pose.a_from_b.rotation.log()[-1])
            converged = bool(getattr(message, "has_converged", False))
            set_latest_pose(x, y, yaw, converged)

    except Exception as e:
        print(f"⚠️  Filter pose updater error: {e}")
        import traceback
        traceback.print_exc()

def background_status_updater():
    """Background thread to emit status updates to all clients"""
    while True:
        try:
            nav_state = get_navigation_state()

            # Check if navigation subprocess is actually running
            subprocess_running = state.is_navigation_running()

            # Detect stale state: if state file says running but no subprocess exists
            if not subprocess_running and nav_state['navigation_running']:
                # Clear stale state
                from utils.navigation_state import clear_navigation_state
                clear_navigation_state()
                nav_state = get_navigation_state()  # Reload cleared state

            # Final determination: navigation is running if subprocess exists
            nav_running = subprocess_running

            status = {
                'navigation_running': nav_running,
                'track_status': nav_state['track_status'],
                'current_waypoint': nav_state['current_waypoint_index'],
                'total_waypoints': nav_state['total_waypoints'],
                'filter_converged': False,  # Default if no pose available
                'vision_active': nav_state['vision_active']
            }

            # Get robot pose
            pose = get_latest_pose()
            if pose is not None:
                import math
                status['filter_converged'] = pose.converged
                status['pose'] = {
                    'x': pose.x,
                    'y': pose.y,
                    'heading_deg': math.degrees(pose.yaw)
                }

            socketio.emit('status_update', status)

        except Exception as e:
            print(f"Error in status updater: {e}")

        time.sleep(0.5)  # Update twice per second

# ==================== Main ====================

def run_async_filter_updater():
    """Run the async filter updater in its own event loop"""
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(filter_pose_updater())

if __name__ == '__main__':
    # Start background filter pose updater
    filter_thread = threading.Thread(target=run_async_filter_updater, daemon=True)
    filter_thread.start()

    # Start background status updater
    status_thread = threading.Thread(target=background_status_updater, daemon=True)
    status_thread.start()

    print("\n" + "="*70)
    print("AMIGA WAYPOINT NAVIGATION - WEB GUI")
    print("="*70)
    print(f"Starting Flask server on http://0.0.0.0:5000")
    print("Open in browser to monitor and control navigation")
    print("="*70 + "\n")

    # Run Flask with SocketIO
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
