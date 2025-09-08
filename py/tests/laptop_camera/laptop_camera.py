import cv2
import depthai as dai
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

# Load YOLOv8n model
model = YOLO("yolov8n.pt")

# Create pipeline
pipeline = dai.Pipeline()

# RGB camera (CAM_A)
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(30)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)

# Mono cameras (CAM_B and CAM_C)
mono_left = pipeline.create(dai.node.MonoCamera)
mono_right = pipeline.create(dai.node.MonoCamera)
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

# Stereo depth
stereo = pipeline.create(dai.node.StereoDepth)
stereo.initialConfig.setConfidenceThreshold(200)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

# Output streams
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

def create_depth_filter():
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[0.0], [0.0]])     # [depth, depth_rate]
    kf.F = np.array([[1, 1],
                     [0, 1]])           # State transition
    kf.H = np.array([[1, 0]])           # Measurement function
    kf.P *= 10                          # Covariance
    kf.R = 0.01                         # Measurement noise
    kf.Q = np.eye(2) * 1e-4             # Process noise
    return kf

# Start device
with dai.Device(pipeline) as device:
    print("Connected to OAK-D")

    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    # Create Kalman filter for person depth
    kf = create_depth_filter()

    while True:
        in_rgb = q_rgb.get()
        in_depth = q_depth.get()

        frame = in_rgb.getCvFrame()
        depth_frame = in_depth.getFrame()

        depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_JET)

        results = model(frame, verbose=False)[0]

        for det in results.boxes:
            cls = int(det.cls)
            conf = float(det.conf)
            if cls == 0 and conf > 0.5:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Median depth in 3x3 region
                if 1 <= cx < depth_frame.shape[1] - 1 and 1 <= cy < depth_frame.shape[0] - 1:
                    roi = depth_frame[cy - 1:cy + 2, cx - 1:cx + 2]
                    valid_depths = roi[roi > 0]

                    if valid_depths.size > 0:
                        raw_depth_m = np.median(valid_depths) / 1000.0
                        kf.predict()
                        kf.update([[raw_depth_m]])
                        filtered_depth = kf.x[0, 0]
                        label = f"Person {conf:.2f} | {filtered_depth:.2f} m"
                        box_color = (0, 0, 255) if filtered_depth < 2.5 else (0, 255, 0)
                    else:
                        kf.predict()
                        filtered_depth = kf.x[0, 0]
                        label = f"Person {conf:.2f} | {filtered_depth:.2f} m (est)"
                        box_color = (0, 255, 0)
                else:
                    kf.predict()
                    filtered_depth = kf.x[0, 0]
                    label = f"Person {conf:.2f} | {filtered_depth:.2f} m (est)"
                    box_color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # Show outputs
        cv2.imshow("OAK-D RGB + Depth", frame)
        cv2.imshow("Depth Map", depth_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
