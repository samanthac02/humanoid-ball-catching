"""
ZED + HSV Tennis Ball 3D Tracker
====================================
Detects a tennis ball using HSV color filtering, gets 3D position
from the ZED point cloud, and tracks with a Kalman filter for
velocity estimation and trajectory prediction.

No ML model needed — runs at full camera FPS with sub-millisecond
detection latency.

Requirements:
    pip install opencv-python numpy

Usage:
    export DISPLAY=:1
    python3 hsv_tracker.py
    python3 hsv_tracker.py --exposure 30 --min-radius 5 --max-radius 200
"""

import argparse
import time
import cv2
import numpy as np
import pyzed.sl as sl


# ──────────────────────────────────────────────────────────────────────
# Kalman Filter
# ──────────────────────────────────────────────────────────────────────

class KalmanTracker:
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

        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1; self.H[1, 1] = 1; self.H[2, 2] = 1

        self.x = None
        self.P = None
        self.last_t = None
        self.consecutive_rejections = 0
        self.spd_smooth = 0.0
        self.initialized = False

    def _make_Q(self, dt):
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
        sigma_xy = 0.003 + 0.001 * z_depth
        sigma_z  = 0.002 + 0.01  * z_depth ** 2
        return np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_z ** 2])

    def _seed(self, x0, y0, z0, t_now):
        self.x = np.array([[x0], [y0], [z0], [0.0], [0.0], [0.0]])
        self.P = np.diag([0.01, 0.01, 0.01, 10.0, 10.0, 10.0])
        self.last_t = t_now
        self.consecutive_rejections = 0
        self.initialized = True

    def predict(self, dt):
        F = np.eye(6)
        F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._make_Q(dt)

    def _update(self, z):
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

        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return True

    def estimate(self, X, Y, Z, t_now):
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
        if not self.initialized:
            return []
        g = 9.81
        x_pred = self.x.copy()
        trajectory = []
        for _ in range(steps):
            F = np.eye(6)
            F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
            x_pred = F @ x_pred
            x_pred[1, 0] -= 0.5 * g * dt ** 2
            x_pred[4, 0] -= g * dt
            trajectory.append(x_pred[:3, 0].copy())
        return trajectory

    def get_predicted_position(self):
        if not self.initialized:
            return None
        return self.x[:3, 0].copy()

    def get_velocity(self):
        if not self.initialized:
            return None
        return self.x[3:6].flatten()

    def reset(self):
        self.x = None
        self.P = None
        self.last_t = None
        self.consecutive_rejections = 0
        self.spd_smooth = 0.0
        self.initialized = False


# ──────────────────────────────────────────────────────────────────────
# HSV Tennis Ball Detection
# ──────────────────────────────────────────────────────────────────────

class HSVDetector:
    """
    Detects a tennis ball using HSV color filtering with contour
    analysis for size and circularity validation.
    """

    def __init__(self, h_low=25, h_high=65, s_low=80, s_high=255,
                 v_low=80, v_high=255, min_radius=5, max_radius=200,
                 min_circularity=0.5):
        self.lower = np.array([h_low, s_low, v_low])
        self.upper = np.array([h_high, s_high, v_high])
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_circularity = min_circularity

        # Morphological kernel for cleaning up the mask
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def detect(self, frame_bgr):
        """
        Detect tennis ball in frame.
        Returns (cx, cy, radius, mask) or None if not found.
        cx, cy are subpixel float centroid coordinates.
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)

        # Morphological close then open to remove noise and fill gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = None
        best_radius = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # too small
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(contour)

            # Size filter
            if radius < self.min_radius or radius > self.max_radius:
                continue

            # Circularity filter: how well does the contour fill its circle?
            circle_area = np.pi * radius ** 2
            circularity = area / circle_area if circle_area > 0 else 0
            if circularity < self.min_circularity:
                continue

            # Pick the largest valid contour
            if radius > best_radius:
                # Subpixel centroid via image moments
                moments = cv2.moments(contour)
                if moments["m00"] > 0:
                    mcx = moments["m10"] / moments["m00"]
                    mcy = moments["m01"] / moments["m00"]
                else:
                    mcx, mcy = cx, cy

                best = (mcx, mcy, radius, mask)
                best_radius = radius

        return best


# ──────────────────────────────────────────────────────────────────────
# 3D Position from Point Cloud
# ──────────────────────────────────────────────────────────────────────

def get_3d_from_point(cx, cy, point_cloud_np, sample_radius=5):
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
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--exposure", type=int, default=30)
    parser.add_argument("--gain", type=int, default=80)
    parser.add_argument("--max-lost", type=int, default=10,
                        help="Max frames to coast when detection is lost")
    # HSV tuning
    parser.add_argument("--h-low", type=int, default=35)
    parser.add_argument("--h-high", type=int, default=55)
    parser.add_argument("--s-low", type=int, default=100)
    parser.add_argument("--v-low", type=int, default=100)
    parser.add_argument("--min-radius", type=int, default=5)
    parser.add_argument("--max-radius", type=int, default=200)
    parser.add_argument("--show-mask", action="store_true",
                        help="Show HSV mask in a second window for tuning")
    args = parser.parse_args()

    # ── HSV Detector ──
    detector = HSVDetector(
        h_low=args.h_low, h_high=args.h_high,
        s_low=args.s_low, v_low=args.v_low,
        min_radius=args.min_radius, max_radius=args.max_radius,
    )

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

    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters
    fx = calib.left_cam.fx
    fy = calib.left_cam.fy
    cx_cam = calib.left_cam.cx
    cy_cam = calib.left_cam.cy
    actual_fps = cam_info.camera_configuration.fps
    print(f"ZED running at {actual_fps} FPS, exposure={args.exposure}, gain={args.gain}")

    # ── Kalman tracker ──
    kf = KalmanTracker()
    lost_count = 0

    image_zed = sl.Mat()
    point_cloud_zed = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    cv2.namedWindow("ZED + HSV Tennis Tracker")
    if args.show_mask:
        cv2.namedWindow("HSV Mask")
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

            # ── HSV detection ──
            detection = detector.detect(frame)

            if detection is not None:
                # ── Ball detected ──
                lost_count = 0
                cx, cy, radius, mask = detection

                # Draw circle and centroid
                cv2.circle(frame, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)

                # Green mask overlay on detected ball pixels
                ball_mask = mask > 0
                frame[ball_mask] = (frame[ball_mask] * 0.5
                                    + np.array([0, 255, 0]) * 0.5).astype(np.uint8)

                # Show mask window if requested
                if args.show_mask:
                    cv2.imshow("HSV Mask", mask)

                # 3D lookup
                result = get_3d_from_point(cx, cy, pc, sample_radius=max(3, int(radius * 0.3)))

                if result is not None:
                    X, Y, Z = result
                    vel, spd = kf.estimate(X, Y, Z, time.perf_counter())

                    cv2.putText(frame, f"X:{X:.3f} Y:{Y:.3f} Z:{Z:.3f}m",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, f"Vx:{vel[0]:.2f} Vy:{vel[1]:.2f} Vz:{vel[2]:.2f} m/s",
                                (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    cv2.putText(frame, f"|V|: {kf.spd_smooth:.2f} m/s  ({kf.spd_smooth * 2.237:.1f} mph)",
                                (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    cv2.putText(frame, f"R: {radius:.0f}px",
                                (10, 105),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

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
                    cv2.putText(frame, "Depth: N/A", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            elif kf.initialized:
                # ── Ball lost — coast using Kalman prediction ──
                lost_count += 1

                if lost_count <= args.max_lost:
                    kf.predict(1.0 / actual_fps)
                    kf.last_t = time.perf_counter()

                    pred_3d = kf.get_predicted_position()

                    if pred_3d is not None and pred_3d[2] > 0.1:
                        u = int(fx * pred_3d[0] / pred_3d[2] + cx_cam)
                        v = int(fy * pred_3d[1] / pred_3d[2] + cy_cam)

                        cv2.circle(frame, (u, v), 8, (0, 165, 255), 2)
                        cv2.putText(frame, f"Predicted ({lost_count}/{args.max_lost})",
                                    (u + 12, v - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                        cv2.putText(frame, f"|V|: {kf.spd_smooth:.1f} m/s  ({kf.spd_smooth * 2.237:.1f} mph)",
                                    (u + 12, v + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                        traj = kf.predict_trajectory(steps=40, dt=1.0/actual_fps)
                        for i, pt3d in enumerate(traj):
                            if pt3d[2] > 0.1:
                                tu = int(fx * pt3d[0] / pt3d[2] + cx_cam)
                                tv = int(fy * pt3d[1] / pt3d[2] + cy_cam)
                                alpha = max(0.2, 1.0 - i / len(traj))
                                color = (0, int(80 + 85 * alpha), int(200 * alpha))
                                cv2.circle(frame, (tu, tv), 3, color, -1)
                else:
                    kf.reset()
                    lost_count = 0

            cv2.imshow("ZED + HSV Tennis Tracker", frame)
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