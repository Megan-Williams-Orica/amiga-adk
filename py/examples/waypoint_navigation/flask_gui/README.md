# Waypoint Navigation Flask GUI

Web-based monitoring and control interface for the Amiga waypoint navigation system.

## Features

- **Live Camera Feed:** Real-time camera stream with YOLO detection overlays
- **Waypoint Visualization:** Interactive D3.js plot showing waypoints, robot position, and path
- **Robot Status:** Real-time telemetry (position, heading, GPS quality, track status)
- **Navigation Control:** Start/stop navigation via web browser
- **Emergency Stop:** One-click emergency shutdown

## Installation

```bash
cd flask_gui
pip install -r requirements.txt
```

## Usage

### 1. Start the Flask GUI server

```bash
python app.py
```

The server will start on `http://0.0.0.0:5000`

### 2. Open in browser

Navigate to `http://<robot-ip>:5000` in your browser

### 3. Control navigation

- Click **"Start Navigation"** to launch the waypoint navigation system
- Monitor robot status and waypoint progress in real-time
- Click **"Stop Navigation"** to gracefully stop
- Use **"Emergency Stop"** for immediate shutdown

## Architecture

### Separate Processes

The Flask GUI runs as a **separate process** from the navigation system:

- **Process 1:** `python app.py` - Flask web server
- **Process 2:** `bash run.sh` - Navigation system (started by Flask)
- **Process 3:** `python detection/detectionPlot.py` - Camera/YOLO (separate)

### Data Flow

```
Navigation System → pose_cache.py → Flask → Socket.IO → Browser
                                      ↓
                                   HTTP/MJPEG
                                      ↑
Camera Detector → latest_frame → Flask
```

### Communication

- **Robot Pose:** Read from `utils/pose_cache.py` (shared memory)
- **Camera Feed:** Flask reads `state.camera_frame` (updated by camera thread)
- **Waypoints:** Loaded from CSV file in `surveyed-waypoints/`
- **Controls:** Browser → Socket.IO → subprocess management

## API Endpoints

### HTTP Routes

- `GET /` - Main dashboard page
- `GET /video_feed` - MJPEG camera stream
- `GET /plot_data` - JSON waypoint data for D3.js
- `GET /robot_status` - JSON robot status

### Socket.IO Events

**Client → Server:**
- `start_navigation` - Start navigation by running run.sh
- `stop_navigation` - Stop navigation (SIGTERM)
- `emergency_stop` - Force kill navigation (SIGKILL)

**Server → Client:**
- `status_update` - Real-time robot status (500ms interval)
- `success` / `error` / `warning` - Command feedback
- `status` - General status messages

## Configuration

Edit `app.py` to configure:

- Port: `socketio.run(app, port=5000)`
- Update rate: `time.sleep(0.5)` in `background_status_updater()`
- Waypoint file: `waypoint_file = Path(...) / 'physicsLabBack2Lanes.csv'`

## Troubleshooting

### Camera feed shows "Waiting for camera..."

The camera feed requires `detectionPlot.py` to be running separately and updating `state.camera_frame`. This integration is TODO.

### Navigation won't start

- Check that `run.sh` exists in parent directory
- Verify run.sh has execute permissions: `chmod +x run.sh`
- Check Flask logs for error messages

### Plot not showing waypoints

- Verify waypoint CSV file exists at configured path
- Check browser console for JavaScript errors
- Ensure CSV has `dx, dy` columns

## Next Steps

- [ ] Integrate camera feed from detectionPlot.py
- [ ] Add track status updates from navigation_manager
- [ ] Add vision detection overlay on camera
- [ ] Implement pause/resume functionality
- [ ] Add waypoint jump-to feature
