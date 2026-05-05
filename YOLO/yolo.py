"""
ZED + YOLOv8 Tennis Ball 3D Tracker
====================================
Detects a tennis ball using YOLOv8, gets 3D position from the ZED
point cloud, and tracks with a Kalman filter for velocity estimation
and trajectory prediction.

Requirements:
    pip install ultralytics opencv-python numpy

Usage:
    export DISPLAY=:1
    python3 yolo.py
    python3 yolo.py --weights yolov8s.pt --conf 0.15 --exposure 40
"""

import argparse
import time
import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────────────
# Kalman Filter
# ──────────────────────────────────────────────────────────────────────

class Kalman:
    """
    Constant-velocity Kalman filter with depth-dependent noise,
    Mahalanobis gating, and ballistic trajectory prediction.

    State: [x, y, z, vx, vy, vz]^T  (meters / m/s, camera frame)
    """

    def __init__(self, q_accel_var=4.0, mahal_gate_sq=16.0,
                 max_rejections=10, spd_ema_alpha=0.2):
        self.q_accel_var = q_accel_var
        self.mahal_gate_sq = mahal_gate_sq
        self.max_rejections = max_rejections
        self.spd_ema_alpha = spd_ema_alpha

        # Observation matrix (we measure position only)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1; self.H[1, 1] = 1; self.H[2, 2] = 1

        self.x = None          # state vector (6x1)
        self.P = None          # covariance (6x6)
        self.last_t = None
        self.consecutive_rejections = 0
        self.spd_smooth = 0.0
        self.initialized = False

    def _make_Q(self, dt):
        """Discrete-time process noise coupling position and velocity."""
        q = self.q_accel_var
        Q_block = np.array([[dt**4 / 4, dt**3 / 2],
                            [dt**3 / 2, dt**2    ]]) * q
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i,     i    ] = Q_block[0, 0]
            Q[i,     i + 3] = Q_block[0, 1]
            Q[i + 3, i    ] = Q_block[1, 0]
            Q[i + 3, i + 3] = Q_block[1, 1]
        return Q

    def _make_R(self, z_depth):
        """Depth-dependent measurement noise (ZED stereo error model)."""
        sigma_xy = 0.003 + 0.001 * z_depth
        sigma_z  = 0.002 + 0.01  * z_depth ** 2
        return np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_z ** 2])

    def _seed(self, x0, y0, z0, t_now):
        """Initialize or re-initialize the filter at a measurement."""
        self.x = np.array([[x0], [y0], [z0], [0.0], [0.0], [0.0]])
        self.P = np.diag([0.01, 0.01, 0.01, 10.0, 10.0, 10.0])
        self.last_t = t_now
        self.consecutive_rejections = 0
        self.initialized = True

    def predict(self, dt):
        """Run prediction step (public for coasting through lost frames)."""
        F = np.eye(6)
        F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._make_Q(dt)

    def _update(self, z):
        """Gated Kalman update. Returns True if measurement was accepted."""
        R = self._make_R(float(z[2, 0]))
        S = self.H @ self.P @ self.H.T + R
        innovation = z - self.H @ self.x

        try:
            mahal_sq = float(innovation.T @ np.linalg.solve(S, innovation))
        except np.linalg.LinAlgError:
            return False
        if mahal_sq > self.mahal_gate_sq:
            return False

        K = self.P @ self.H.T @ np.linalg.solve(S, np.eye(3))
        self.x = self.x + K @ innovation

        # Joseph-form covariance update
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return True

    def estimate(self, X, Y, Z, t_now):
        """
        Feed a new 3D measurement. Returns (velocity, speed).
        Handles initialization, prediction, gating, and reseeding.
        """
        if not self.initialized:
            self._seed(X, Y, Z, t_now)
            return np.array([0.0, 0.0, 0.0]), 0.0

        dt = max(t_now - self.last_t, 1e-6)
        self.last_t = t_now

        self.predict(dt)
        accepted = self._update(np.array([[X], [Y], [Z]]))

        if accepted:
            self.consecutive_rejections = 0
        else:
            self.consecutive_rejections += 1
            if self.consecutive_rejections >= self.max_rejections:
                self._seed(X, Y, Z, t_now)
                print("[Kalman] Too many outliers, reseeding filter.")
                return np.array([0.0, 0.0, 0.0]), 0.0

        vel = self.x[3:6].flatten()
        spd = float(np.linalg.norm(vel))
        self.spd_smooth = ((1.0 - self.spd_ema_alpha) * self.spd_smooth
                           + self.spd_ema_alpha * spd)
        return vel, spd

    def predict_trajectory(self, steps=30, dt=1/60):
        """Predict future positions without modifying filter state.
        Uses ballistic model (gravity on Y axis, downward)."""
        if not self.initialized:
            return []
        g = 9.81
        x_pred = self.x.copy()
        trajectory = []
        for _ in range(steps):
            F = np.eye(6)
            F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
            x_pred = F @ x_pred
            # Gravity: Y is up in ZED default, so gravity pulls -Y
            x_pred[1, 0] -= 0.5 * g * dt ** 2
            x_pred[4, 0] -= g * dt
            trajectory.append(x_pred[:3, 0].copy())
        return trajectory

    def get_predicted_position(self):
        """Return current predicted 3D position, or None."""
        if not self.initialized:
            return None
        return self.x[:3, 0].copy()

    def get_velocity(self):
        """Return current velocity vector, or None."""
        if not self.initialized:
            return None
        return self.x[3:6].flatten()

    def reset(self):
        """Reset filter state."""
        self.x = None
        self.P = None
        self.last_t = None
        self.consecutive_rejections = 0
        self.spd_smooth = 0.0
        self.initialized = False


# ──────────────────────────────────────────────────────────────────────
# YOLO Detection
# ──────────────────────────────────────────────────────────────────────

BALL_CLASS_IDS = {32, 29}  # 32 = sports ball, 29 = frisbee?

def detect_ball(model, frame_bgr, conf_threshold=0.15):
    """Run YOLO, return best ball-like detection or None."""
    results = model.predict(frame_bgr, conf=conf_threshold, verbose=False)[0]

    best = None
    best_conf = 0

    for det in results.boxes:
        cls_id = int(det.cls[0])
        conf = float(det.conf[0])

        if cls_id not in BALL_CLASS_IDS:
            continue
        if conf > best_conf:
            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
            best = (x1, y1, x2, y2, conf)
            best_conf = conf

    return best


def refine_centroid_hsv(frame_bgr, x1, y1, x2, y2):
    """Refine centroid within bbox using HSV for tennis ball yellow-green."""
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 80, 80])
    upper = np.array([65, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    moments = cv2.moments(mask)
    if moments["m00"] < 10:
        return None

    cx = x1 + moments["m10"] / moments["m00"]
    cy = y1 + moments["m01"] / moments["m00"]
    return (cx, cy)


# ──────────────────────────────────────────────────────────────────────
# 3D Position from Point Cloud
# ──────────────────────────────────────────────────────────────────────

def get_3d_from_bbox(cx, cy, point_cloud_np, sample_radius=5):
    """Get 3D position from point cloud at centroid, median over small region."""
    h, w = point_cloud_np.shape[:2]
    icx, icy = int(cx), int(cy)

    y0 = max(0, icy - sample_radius)
    y1 = min(h, icy + sample_radius + 1)
    x0 = max(0, icx - sample_radius)
    x1 = min(w, icx + sample_radius + 1)

    region = point_cloud_np[y0:y1, x0:x1, :3]
    valid = np.isfinite(region[:, :, 2]) & (region[:, :, 2] > 0)

    if np.any(valid):
        X, Y, Z = np.median(region[valid], axis=0)
        return X, Y, Z

    # Fallback: single pixel
    if 0 <= icy < h and 0 <= icx < w:
        pt = point_cloud_np[icy, icx, :3]
        if np.isfinite(pt[2]) and pt[2] > 0:
            return pt[0], pt[1], pt[2]

    return None


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8s.pt",
                        help="YOLO model weights (default: yolov8s.pt)")
    parser.add_argument("--conf", type=float, default=0.15,
                        help="Detection confidence threshold (default: 0.15)")
    parser.add_argument("--fps", type=int, default=60,
                        help="ZED camera FPS (default: 60)")
    parser.add_argument("--exposure", type=int, default=30,
                        help="Camera exposure 1-100 (default: 30)")
    parser.add_argument("--gain", type=int, default=80,
                        help="Camera gain 1-100 (default: 80)")
    parser.add_argument("--max-lost", type=int, default=10,
                        help="Max frames to coast when detection is lost (default: 10)")
    args = parser.parse_args()

    # ── YOLO ──
    print("Loading YOLO model...")
    model = YOLO(args.weights)

    # ── ZED ──
    print("Opening ZED camera...")
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = args.fps
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_minimum_distance = 0.15

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Cannot open ZED: {repr(status)}")

    zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, args.exposure)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, args.gain)

    # Camera intrinsics for projecting predicted trajectory to 2D
    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters
    fx = calib.left_cam.fx
    fy = calib.left_cam.fy
    cx_cam = calib.left_cam.cx
    cy_cam = calib.left_cam.cy
    actual_fps = cam_info.camera_configuration.fps
    print(f"ZED running at {actual_fps} FPS, exposure={args.exposure}, gain={args.gain}")

    # ── Kalman tracker ──
    kf = Kalman()
    lost_count = 0

    image_zed = sl.Mat()
    point_cloud_zed = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    cv2.namedWindow("ZED + YOLO Tennis Tracker")
    print("Tracking... Press 'r' to reset, 'q' to quit.")

    try:
        while True:
            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                continue

            # ── Get image ──
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            frame_bgra = image_zed.get_data()
            frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

            # ── Get point cloud ──
            zed.retrieve_measure(point_cloud_zed, sl.MEASURE.XYZRGBA)
            pc = point_cloud_zed.get_data()

            # ── YOLO detection ──
            detection = detect_ball(model, frame, args.conf)

            if detection is not None:
                # ── Ball detected ──
                lost_count = 0
                x1, y1, x2, y2, conf = detection

                # Compute centroid (try HSV refinement first)
                refined = refine_centroid_hsv(frame, x1, y1, x2, y2)
                if refined is not None:
                    cx, cy = refined
                else:
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                # Draw bounding box and centroid
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"Ball {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 3D lookup from point cloud
                result = get_3d_from_bbox(cx, cy, pc, sample_radius=5)

                if result is not None:
                    X, Y, Z = result
                    vel, spd = kf.estimate(X, Y, Z, time.perf_counter())

                    cv2.putText(frame, f"X:{X:.3f} Y:{Y:.3f} Z:{Z:.3f}m",
                                (x1, y1 - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, f"Vx:{vel[0]:.2f} Vy:{vel[1]:.2f} Vz:{vel[2]:.2f} m/s",
                                (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    cv2.putText(frame, f"|V|: {kf.spd_smooth:.2f} m/s  ({kf.spd_smooth * 2.237:.1f} mph)",
                                (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                    # Draw predicted trajectory
                    if kf.initialized:
                        traj = kf.predict_trajectory(steps=40, dt=1.0/actual_fps)
                        for i, pt3d in enumerate(traj):
                            if pt3d[2] > 0.1:
                                u = int(fx * pt3d[0] / pt3d[2] + cx_cam)
                                v = int(fy * pt3d[1] / pt3d[2] + cy_cam)
                                alpha = max(0.2, 1.0 - i / len(traj))
                                color = (0, int(100 + 155 * alpha), int(255 * alpha))
                                cv2.circle(frame, (u, v), 3, color, -1)
                else:
                    cv2.putText(frame, "Depth: N/A", (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            elif kf.initialized:
                # ── Ball lost — coast using Kalman prediction ──
                lost_count += 1

                if lost_count <= args.max_lost:
                    kf.predict(1.0 / actual_fps)
                    kf.last_t = time.perf_counter()

                    pred_3d = kf.get_predicted_position()
                    vel = kf.get_velocity()

                    if pred_3d is not None and pred_3d[2] > 0.1:
                        u = int(fx * pred_3d[0] / pred_3d[2] + cx_cam)
                        v = int(fy * pred_3d[1] / pred_3d[2] + cy_cam)

                        # Orange circle for predicted position
                        cv2.circle(frame, (u, v), 8, (0, 165, 255), 2)
                        cv2.putText(frame, f"Predicted ({lost_count}/{args.max_lost})",
                                    (u + 12, v - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                        cv2.putText(frame, f"|V|: {kf.spd_smooth:.1f} m/s  ({kf.spd_smooth * 2.237:.1f} mph)",
                                    (u + 12, v + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                        # Draw predicted trajectory while coasting
                        traj = kf.predict_trajectory(steps=40, dt=1.0/actual_fps)
                        for i, pt3d in enumerate(traj):
                            if pt3d[2] > 0.1:
                                tu = int(fx * pt3d[0] / pt3d[2] + cx_cam)
                                tv = int(fy * pt3d[1] / pt3d[2] + cy_cam)
                                alpha = max(0.2, 1.0 - i / len(traj))
                                color = (0, int(80 + 85 * alpha), int(200 * alpha))
                                cv2.circle(frame, (tu, tv), 3, color, -1)
                else:
                    # Lost for too long — reset
                    kf.reset()
                    lost_count = 0

            cv2.imshow("ZED + YOLO Tennis Tracker", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                kf.reset()
                lost_count = 0
                print("Reset.")

    finally:
        zed.close()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()