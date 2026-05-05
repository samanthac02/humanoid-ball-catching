# RUN with
# export DISPLAY=:1
# python3 thrower_prior.py
#
# Approach:
#   1. YOLO detects the person (thrower) continuously
#   2. YOLO watches for the tennis ball to appear near the thrower
#   3. Once detected, seeds Kalman filter with position + velocity
#      prior pointing away from the thrower
#   4. Tracks ball with YOLO + Kalman coasting through lost frames
#
# Requirements:
#   pip install ultralytics opencv-python numpy

import argparse
import numpy as np
import cv2
import time
import pyzed.sl as sl
from ultralytics import YOLO


# ====== Kalman Filter ======

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

    def seed(self, x0, y0, z0, t_now, vx=0.0, vy=0.0, vz=0.0,
             vel_uncertainty=5.0):
        """Initialize filter with optional velocity prior."""
        self.x = np.array([[x0], [y0], [z0], [vx], [vy], [vz]])
        self.P = np.diag([0.01, 0.01, 0.01,
                          vel_uncertainty, vel_uncertainty, vel_uncertainty])
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
            self.seed(X, Y, Z, t_now)
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
                self.seed(X, Y, Z, t_now)
                print("[Kalman] Too many outliers, reseeding.")
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


# ====== YOLO Detection ======

PERSON_CLASS_ID = 0
BALL_CLASS_IDS = {32, 29}  # sports ball + frisbee


def detect_all(model, frame_bgr, conf=0.15):
    """
    Run YOLO once and return both person and ball detections.
    Returns (person_bbox, ball_bbox) where each is (x1, y1, x2, y2, conf) or None.
    """
    results = model.predict(frame_bgr, conf=conf, verbose=False)[0]

    best_person = None
    best_person_area = 0
    best_ball = None
    best_ball_conf = 0

    for det in results.boxes:
        cls_id = int(det.cls[0])
        c = float(det.conf[0])
        x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)

        if cls_id == PERSON_CLASS_ID and c > 0.5:
            area = (x2 - x1) * (y2 - y1)
            if area > best_person_area:
                best_person = (x1, y1, x2, y2, c)
                best_person_area = area

        elif cls_id in BALL_CLASS_IDS:
            if c > best_ball_conf:
                best_ball = (x1, y1, x2, y2, c)
                best_ball_conf = c

    return best_person, best_ball


def bbox_center(bbox):
    """Get center of a bounding box (x1, y1, x2, y2, ...)."""
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_near_person(ball_bbox, person_bbox, margin=1.5):
    """Check if the ball bbox is near the person (within expanded person region)."""
    if person_bbox is None or ball_bbox is None:
        return False

    px1, py1, px2, py2 = person_bbox[:4]
    pw = px2 - px1
    ph = py2 - py1

    # Expanded region around person
    ex = int(pw * margin)
    ey = int(ph * margin * 0.5)

    rx1 = px1 - ex
    ry1 = py1 - ey
    rx2 = px2 + ex
    ry2 = py2 + ey

    bcx, bcy = bbox_center(ball_bbox)
    return rx1 <= bcx <= rx2 and ry1 <= bcy <= ry2


# ====== 3D from Point Cloud ======

def get_3d_from_point(cx, cy, point_cloud_np, sample_radius=5):
    h, w = point_cloud_np.shape[:2]
    icx, icy = int(cx), int(cy)

    y0 = max(0, icy - sample_radius)
    y1 = min(h, icy + sample_radius + 1)
    x0 = max(0, icx - sample_radius)
    x1 = min(w, icx + sample_radius + 1)

    region = point_cloud_np[y0:y1, x0:x1, :3]
    valid = np.isfinite(region[:, :, 2]) & (region[:, :, 2] > 0)

    if np.any(valid):
        return np.median(region[valid], axis=0)

    if 0 <= icy < h and 0 <= icx < w:
        pt = point_cloud_np[icy, icx, :3]
        if np.isfinite(pt[2]) and pt[2] > 0:
            return pt

    return None


def get_person_center_3d(person_bbox, point_cloud_np):
    """Get approximate 3D position of the thrower's center mass."""
    px1, py1, px2, py2 = person_bbox[:4]
    cx = (px1 + px2) // 2
    cy = py1 + (py2 - py1) // 3  # upper third (chest area)
    return get_3d_from_point(cx, cy, point_cloud_np, sample_radius=10)


# ====== Tracking States ======

STATE_WAITING = 0      # Waiting — looking for person
STATE_WATCHING = 1     # Person found — watching for ball near them
STATE_TRACKING = 2     # Ball detected — actively tracking

STATE_NAMES = {
    STATE_WAITING: "WAITING for person",
    STATE_WATCHING: "WATCHING for ball",
    STATE_TRACKING: "TRACKING ball",
}


# ====== Main ======

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8s.pt")
    parser.add_argument("--conf", type=float, default=0.15,
                        help="Ball detection confidence threshold")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--exposure", type=int, default=30)
    parser.add_argument("--gain", type=int, default=80)
    parser.add_argument("--max-lost", type=int, default=15)
    args = parser.parse_args()

    # ── YOLO (single model for both person + ball) ──
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

    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters
    fx = calib.left_cam.fx
    fy = calib.left_cam.fy
    cx_cam = calib.left_cam.cx
    cy_cam = calib.left_cam.cy
    actual_fps = cam_info.camera_configuration.fps
    print(f"ZED running at {actual_fps} FPS")

    image_zed = sl.Mat()
    point_cloud_zed = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    # ── State ──
    kf = KalmanTracker()
    state = STATE_WAITING
    lost_count = 0
    person_bbox = None
    person_3d = None

    cv2.namedWindow("ZED + Thrower Prior Tracker")
    print("Tracking... Press 'r' to reset, 'q' to quit.")

    try:
        while True:
            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            frame_bgra = image_zed.get_data()
            frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

            zed.retrieve_measure(point_cloud_zed, sl.MEASURE.XYZRGBA)
            pc = point_cloud_zed.get_data()

            # ── Run YOLO once per frame for both person and ball ──
            person_det, ball_det = detect_all(model, frame, conf=args.conf)

            # Update person tracking (always, regardless of state)
            if person_det is not None:
                person_bbox = person_det
                person_3d = get_person_center_3d(person_bbox, pc)

                if state == STATE_WAITING:
                    state = STATE_WATCHING
                    print("[State] Person detected — watching for ball.")

            # ── State machine ──

            if state == STATE_WAITING:
                cv2.putText(frame, "Waiting for person...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)

            elif state == STATE_WATCHING:
                # Draw person
                if person_bbox is not None:
                    px1, py1, px2, py2 = person_bbox[:4]
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 200, 0), 2)
                    cv2.putText(frame, "Thrower", (px1, py1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

                # Check if ball appeared near the thrower
                if ball_det is not None and bbox_near_person(ball_det, person_bbox):
                    bx1, by1, bx2, by2, bconf = ball_det
                    bcx, bcy = bbox_center(ball_det)

                    ball_3d = get_3d_from_point(bcx, bcy, pc, sample_radius=5)

                    if ball_3d is not None:
                        X, Y, Z = ball_3d

                        # Compute velocity prior: ball moves away from thrower
                        vx0, vy0, vz0 = 0.0, 0.0, 0.0
                        if person_3d is not None:
                            ball_pos = np.array([X, Y, Z])
                            direction = ball_pos - person_3d
                            dist = np.linalg.norm(direction)
                            if dist > 0.01:
                                direction /= dist
                                throw_speed_prior = 5.0  # m/s
                                vx0 = direction[0] * throw_speed_prior
                                vy0 = direction[1] * throw_speed_prior
                                vz0 = direction[2] * throw_speed_prior

                        kf.seed(X, Y, Z, time.perf_counter(),
                                vx=vx0, vy=vy0, vz=vz0,
                                vel_uncertainty=3.0)
                        state = STATE_TRACKING
                        lost_count = 0
                        print(f"[State] Ball detected! v=({vx0:.1f}, {vy0:.1f}, {vz0:.1f})")

                cv2.putText(frame, "Watching for ball near thrower...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            elif state == STATE_TRACKING:
                if ball_det is not None:
                    # ── Ball detected ──
                    lost_count = 0
                    bx1, by1, bx2, by2, bconf = ball_det
                    bcx, bcy = bbox_center(ball_det)

                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    cv2.circle(frame, (int(bcx), int(bcy)), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"Ball {bconf:.2f}", (bx1, by1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    ball_3d = get_3d_from_point(bcx, bcy, pc, sample_radius=5)

                    if ball_3d is not None:
                        X, Y, Z = ball_3d
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
                    # ── Ball lost — coast with Kalman ──
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
                            cv2.putText(frame, f"|V|: {kf.spd_smooth:.1f} m/s",
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
                        print("[State] Ball lost — returning to watch mode.")
                        kf.reset()
                        lost_count = 0
                        state = STATE_WATCHING if person_bbox is not None else STATE_WAITING

                # Draw person bbox while tracking (dimmed)
                if person_bbox is not None:
                    px1, py1, px2, py2 = person_bbox[:4]
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (100, 80, 0), 1)

            # ── Status bar ──
            cv2.putText(frame, STATE_NAMES[state], (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("ZED + Thrower Prior Tracker", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                kf.reset()
                state = STATE_WAITING
                lost_count = 0
                person_bbox = None
                person_3d = None
                print("Reset.")

    finally:
        zed.close()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()