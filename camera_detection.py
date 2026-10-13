# Python 3.12.10
# Create by Paul Mendoza


import customtkinter as ctk
import cv2
import mediapipe as mp
import PIL.Image
import numpy as np
import threading
import time
from datetime import datetime
from collections import deque

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

PROCESS_WIDTH = 640          # downscale frames before detection (huge FPS win)
DISPLAY_INTERVAL = 1 / 60    # cap UI update to 60 Hz
DETECTION_EVERY_N = 1        # run MediaPipe every Nth frame
CAMERA_BUFFER = 1            # minimizes latency


class CameraDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Advanced Camera Detector")
        self.geometry("1200x750")
        self.minsize(1000, 650)

        # State
        self.cap = None
        self.running = False
        self.thread = None
        self.frame_count = 0
        self.fps = 0.0
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()
        self.last_display_time = 0.0
        self.last_detection_frame = 0
        self.detection_count = {"faces": 0, "eyes": 0, "hands": 0}
        self._display_size = (900, 600)   # cached, updated only on resize
        self._resize_job = None

        self.detect_face = ctk.BooleanVar(value=True)
        self.detect_eyes = ctk.BooleanVar(value=True)
        self.detect_hands = ctk.BooleanVar(value=False)
        self.detect_motion = ctk.BooleanVar(value=False)
        self.mirror = ctk.BooleanVar(value=True)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=3,
            refine_landmarks=False,        # faster (we only need eye contours)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            model_complexity=0,             # fastest
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

        self.prev_gray = None


        self._latest_frame = None

        self._hand_lm_spec = self.mp_draw.DrawingSpec(color=(255, 0, 128), thickness=2)
        self._hand_conn_spec = self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2)

        self._eye_sets = [
            np.array([33, 133, 160, 159, 158, 144, 153, 145], np.int32),
            np.array([362, 263, 387, 386, 385, 373, 380, 374], np.int32),
        ]

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        video_frame = ctk.CTkFrame(self, corner_radius=10)
        video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        video_frame.grid_rowconfigure(0, weight=1)
        video_frame.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(
            video_frame, text="Camera Offline\n\nPress  ▶  Start Camera",
            font=("Segoe UI", 20), fg_color="#111111", corner_radius=8
        )
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.video_label.bind("<Configure>", self._on_label_resize)

        self.status = ctk.CTkLabel(
            self, text="Ready.", anchor="w",
            font=("Segoe UI", 12), text_color="#9aa0a6"
        )
        self.status.grid(row=1, column=0, columnspan=2, padx=15, pady=(0, 8), sticky="ew")

        panel = ctk.CTkFrame(self, width=280, corner_radius=10)
        panel.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="ns")
        panel.grid_propagate(False)

        ctk.CTkLabel(panel, text="⚙  Controls",
                     font=("Segoe UI", 20, "bold")).pack(pady=(15, 10))

        btn_row = ctk.CTkFrame(panel, fg_color="transparent")
        btn_row.pack(pady=5, padx=15, fill="x")
        btn_row.grid_columnconfigure((0, 1), weight=1)

        self.start_btn = ctk.CTkButton(
            btn_row, text="▶ Start", command=self.start_camera,
            fg_color="#2ecc71", hover_color="#27ae60", height=40
        )
        self.start_btn.grid(row=0, column=0, padx=3, sticky="ew")

        self.stop_btn = ctk.CTkButton(
            btn_row, text="■ Stop", command=self.stop_camera,
            fg_color="#e74c3c", hover_color="#c0392b", height=40,
            state="disabled"
        )
        self.stop_btn.grid(row=0, column=1, padx=3, sticky="ew")

        self.snap_btn = ctk.CTkButton(
            panel, text="📷 Take Snapshot", command=self.take_snapshot,
            height=38, state="disabled"
        )
        self.snap_btn.pack(pady=8, padx=15, fill="x")

        self._section(panel, "Detection Modules")
        self._toggle(panel, "Face Detection", self.detect_face)
        self._toggle(panel, "Eye / Landmarks", self.detect_eyes)
        self._toggle(panel, "Hand Tracking", self.detect_hands)
        self._toggle(panel, "Motion Detection", self.detect_motion)
        self._toggle(panel, "Mirror View", self.mirror)

        self._section(panel, "Statistics")
        self.stats_label = ctk.CTkLabel(
            panel, text="FPS: 0\nFaces: 0\nEyes: 0\nHands: 0",
            justify="left", anchor="w", font=("Consolas", 13)
        )
        self.stats_label.pack(pady=5, padx=20, fill="x")

        self._section(panel, "Camera Source")
        self.cam_index = ctk.CTkOptionMenu(
            panel, values=["0", "1", "2", "3"], width=200
        )
        self.cam_index.pack(pady=5, padx=15)

        ctk.CTkLabel(panel, text="").pack(expand=True, fill="both")
        ctk.CTkLabel(panel, text="Advanced Camera Detector v1.1",
                     font=("Segoe UI", 10), text_color="#666").pack(pady=10)

    def _section(self, parent, title):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", padx=15, pady=(15, 5))
        ctk.CTkLabel(f, text=title.upper(),
                     font=("Segoe UI", 11, "bold"),
                     text_color="#5dade2", anchor="w").pack(fill="x")
        ctk.CTkFrame(parent, height=1, fg_color="#333").pack(
            fill="x", padx=15, pady=(0, 8))

    def _toggle(self, parent, text, var):
        ctk.CTkSwitch(parent, text=text, variable=var,
                      font=("Segoe UI", 13)).pack(anchor="w", padx=25, pady=4)

    def _on_label_resize(self, event):
        """Debounce resize — only recompute display size 150 ms after last event."""
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(150, lambda: self._commit_resize(event.width, event.height))

    def _commit_resize(self, w, h):
        self._resize_job = None
        if w > 50 and h > 50:
            self._display_size = (w, h)

    def start_camera(self):
        if self.running:
            return
        idx = int(self.cam_index.get())
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(idx)  # fallback
        if not self.cap.isOpened():
            self.status.configure(text=f"❌ Cannot open camera {idx}",
                                  text_color="#e74c3c")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER)

        self.running = True
        self.last_time = time.time()
        self.last_display_time = 0.0
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.snap_btn.configure(state="normal")
        self.status.configure(text=f"✅ Camera {idx} running.",
                              text_color="#2ecc71")

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop_camera(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.snap_btn.configure(state="disabled")
        self.video_label.configure(image=None,
                                   text="Camera Offline\n\nPress  ▶  Start Camera")
        self.status.configure(text="Camera stopped.", text_color="#9aa0a6")

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                self._post_status("⚠ Frame grab failed.", "#f39c12")
                break

            if self.mirror.get():
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            if w > PROCESS_WIDTH:
                scale = PROCESS_WIDTH / w
                proc = cv2.resize(frame, (PROCESS_WIDTH, int(h * scale)),
                                  interpolation=cv2.INTER_LINEAR)
            else:
                proc = frame

            self.last_detection_frame += 1
            if self.last_detection_frame >= DETECTION_EVERY_N:
                self.last_detection_frame = 0
                proc = self._process(proc)

            self._latest_frame = proc

            now = time.time()
            if now - self.last_display_time >= DISPLAY_INTERVAL:
                self.last_display_time = now
                self._display(proc)

            self._update_fps()

    def _process(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False   # micro-optimization for MediaPipe

        faces = eyes = hands_count = 0

        need_mesh = self.detect_face.get() or self.detect_eyes.get()
        if need_mesh:
            mesh_res = self.face_mesh.process(rgb)
            if mesh_res.multi_face_landmarks:
                faces = len(mesh_res.multi_face_landmarks)
                draw_eyes = self.detect_eyes.get()
                for lm in mesh_res.multi_face_landmarks:
                    xs = np.fromiter((p.x for p in lm.landmark), np.float32, 468)
                    ys = np.fromiter((p.y for p in lm.landmark), np.float32, 468)
                    x1, y1 = int(xs.min() * w), int(ys.min() * h)
                    x2, y2 = int(xs.max() * w), int(ys.max() * h)
                    if self.detect_face.get():
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "Face", (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    if draw_eyes:
                        for idx_set in self._eye_sets:
                            pts = np.column_stack((
                                (xs[idx_set] * w).astype(np.int32),
                                (ys[idx_set] * h).astype(np.int32),
                            ))
                            cv2.polylines(frame, [pts], False, (0, 200, 255), 1)
                        eyes += 2

        if self.detect_hands.get():
            hand_res = self.hands.process(rgb)
            if hand_res.multi_hand_landmarks:
                hands_count = len(hand_res.multi_hand_landmarks)
                for hand_lm in hand_res.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_lm, self.mp_hands.HAND_CONNECTIONS,
                        self._hand_lm_spec, self._hand_conn_spec,
                    )

        if self.detect_motion.get():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
                diff = cv2.absdiff(self.prev_gray, gray)
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                thresh = cv2.dilate(thresh, None, iterations=2)
                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if cv2.contourArea(c) < 1000:
                        continue
                    x, y, cw, ch = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + cw, y + ch), (255, 100, 0), 2)
            self.prev_gray = gray

        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, datetime.now().strftime("%H:%M:%S"),
                    (w - 100, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 2)

        self.detection_count["faces"] = faces
        self.detection_count["eyes"] = eyes
        self.detection_count["hands"] = hands_count
        return frame

    def _display(self, frame):
        max_w, max_h = self._display_size
        h, w = frame.shape[:2]
        scale = min(max_w / w, max_h / h)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))

        if new_size != (w, h):
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(rgb)
        imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=new_size)

        self.video_label.after(0, self._set_image, imgtk)

    def _set_image(self, imgtk):
        self.video_label.configure(image=imgtk, text="")
        self.video_label._image_ref = imgtk  # prevent GC

    def _update_fps(self):
        self.frame_count += 1
        now = time.time()
        dt = now - self.last_time
        if dt >= 0.5:
            instant = self.frame_count / dt
            self.fps_history.append(instant)
            self.fps = sum(self.fps_history) / len(self.fps_history)
            self.frame_count = 0
            self.last_time = now
            self._post_stats()

    def _post_stats(self):
        txt = (f"FPS: {self.fps:.1f}\n"
               f"Faces: {self.detection_count['faces']}\n"
               f"Eyes: {self.detection_count['eyes']}\n"
               f"Hands: {self.detection_count['hands']}")
        self.stats_label.after(0, lambda: self.stats_label.configure(text=txt))

    def _post_status(self, text, color):
        self.status.after(0, lambda: self.status.configure(text=text, text_color=color))

    def take_snapshot(self):
        if self._latest_frame is None:
            return
        filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, self._latest_frame)
        self.status.configure(text=f"📷 Saved {filename}", text_color="#3498db")

    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = CameraDetectorApp()
    app.mainloop()