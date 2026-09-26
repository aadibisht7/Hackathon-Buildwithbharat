import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
from deepface import DeepFace

import pickle
import os
import time
import threading

from collections import defaultdict, deque

# Voice registration
from voice_registration import register_from_voice

# Conversation recording / summarization / storage
from speech_recorder import SpeechRecorder
from summarizer import summarize_text
from conversation_manager import save_conversation_files


# ============================================================
# CONFIGURATION
# ============================================================

DB_FILE = "family_deepface_data.pkl"

MODEL_NAME = "ArcFace"

DETECTOR_BACKEND = "yolov12s"  # retinaface / mtcnn / opencv also work

DISTANCE_METRIC = "cosine"

MATCH_THRESHOLD = 0.62

FACE_DETECTION_CONFIDENCE = 0.80

MIN_FACE_SIZE = 80

AI_FRAME_SKIP = 3

REQUIRED_CONFIRMATIONS = 3

MAX_EMBEDDINGS_PER_PERSON = 8

# How long an old face track is kept (in seconds)
TRACK_TIMEOUT = 1.5


# ============================================================
# MAIN CLASS
# ============================================================

class DementiaAssistantPro:

    def __init__(self, root):

        self.root = root
        self.root.title("ForgetMeNot")
        self.root.geometry("1550x820")

        # ----------------------------------------------------
        # Database
        # ----------------------------------------------------

        self.people = self.load_database()

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open camera.")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # ----------------------------------------------------
        # AI state
        # ----------------------------------------------------

        self.ai_running = True
        self.processing = False
        self.frame_count = 0
        self.last_results = []

        self.identity_history = defaultdict(
            lambda: deque(maxlen=REQUIRED_CONFIRMATIONS)
        )

        # Lightweight centroid-based face tracker: gives each face a
        # stable track_id across frames so the UI card can smoothly
        # follow one specific head instead of jumping between detections.
        # track_id -> {"pos": (cx, cy), "last_seen": timestamp}
        self.tracked_faces = {}
        self.next_track_id = 0

        # Smooth (EMA) on-screen position of each identity card, keyed
        # by track_id. Higher ui_smoothing = snappier, lower = smoother.
        self.ui_positions = {}
        self.ui_smoothing = 0.25

        self.current_person = None
        self.current_people = []
        self.previously_recognized = set()

        # ----------------------------------------------------
        # Conversation recording state
        # ----------------------------------------------------

        self.speech_recorder = None
        self.recording = False
        self.recording_target_person = None

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        self.create_ui()

        # ----------------------------------------------------
        # Warm up AI model
        # ----------------------------------------------------

        self.status_label.config(text="Loading AI model...")

        threading.Thread(target=self.warmup_model, daemon=True).start()

        # ----------------------------------------------------
        # Start camera loop
        # ----------------------------------------------------

        self.update_video()

        # ----------------------------------------------------
        # Closing
        # ----------------------------------------------------

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ========================================================
    # DATABASE
    # ========================================================

    def load_database(self):

        if not os.path.exists(DB_FILE):
            print("No database found.")
            return {}

        try:
            with open(DB_FILE, "rb") as f:
                data = pickle.load(f)

            print(f"Loaded {len(data)} registered people.")
            return data

        except Exception as e:
            print("Database loading error:", e)
            return {}

    def save_database(self):

        try:
            with open(DB_FILE, "wb") as f:
                pickle.dump(self.people, f)

            print("Database saved.")

        except Exception as e:
            print("Database saving error:", e)

    # ========================================================
    # UI
    # ========================================================

    def create_ui(self):

        # ----------------------------------------------------
        # Two-column layout: video/controls on the left,
        # conversation panel on the right.
        # ----------------------------------------------------

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True)

        right_frame = tk.Frame(main_frame, width=360, bg="#f2f2f2")
        right_frame.pack(side="right", fill="y")
        right_frame.pack_propagate(False)

        # ----------------------------------------------------
        # Video
        # ----------------------------------------------------

        self.video_label = tk.Label(left_frame, bg="black")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        # ----------------------------------------------------
        # Bottom registration panel
        # ----------------------------------------------------

        panel = tk.Frame(left_frame)
        panel.pack(fill="x", padx=10, pady=10)

        tk.Label(panel, text="Name:").grid(row=0, column=0, padx=5)

        self.name_entry = tk.Entry(panel, width=18)
        self.name_entry.grid(row=0, column=1, padx=5)

        tk.Label(panel, text="Relationship:").grid(row=0, column=2, padx=5)

        self.relationship_entry = tk.Entry(panel, width=18)
        self.relationship_entry.grid(row=0, column=3, padx=5)

        self.register_btn = tk.Button(
            panel,
            text="📷 Register Person",
            font=("Arial", 11, "bold"),
            command=self.register_person
        )
        self.register_btn.grid(row=0, column=4, padx=10)

        self.register_voice_btn = tk.Button(
            panel,
            text="🎤 Voice Register",
            font=("Arial", 11, "bold"),
            command=self.voice_register_person
        )
        self.register_voice_btn.grid(row=0, column=5, padx=10)

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        self.status_label = tk.Label(
            left_frame, text="Starting...", font=("Arial", 12, "bold")
        )
        self.status_label.pack(pady=5)

        # ----------------------------------------------------
        # Right panel: conversation recording + retrieval
        # ----------------------------------------------------

        tk.Label(
            right_frame,
            text="Conversation",
            font=("Arial", 13, "bold"),
            bg="#f2f2f2"
        ).pack(pady=(15, 5))

        self.record_btn = tk.Button(
            right_frame,
            text="🎙️ Start Conversation",
            font=("Arial", 11, "bold"),
            command=self.toggle_conversation_recording
        )
        self.record_btn.pack(pady=5, padx=15, fill="x")

        # "Last conversation with X" retrieval area
        retrieval_frame = tk.Frame(right_frame, bg="#f2f2f2")
        retrieval_frame.pack(pady=(15, 5), padx=15, fill="x")

        self.last_conv_label = tk.Label(
            retrieval_frame,
            text="",
            font=("Arial", 10),
            bg="#f2f2f2",
            wraplength=320,
            justify="left"
        )
        self.last_conv_label.pack(anchor="w")

        self.view_conv_btn = tk.Button(
            retrieval_frame,
            text="📖 Retrieve Last Conversation",
            state="disabled",
            command=lambda: None
        )
        self.view_conv_btn.pack(pady=5, fill="x")

        # Scrollable transcript / summary text area
        text_frame = tk.Frame(right_frame)
        text_frame.pack(padx=15, pady=10, fill="both", expand=True)

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")

        self.conversation_text = tk.Text(
            text_frame,
            wrap="word",
            font=("Arial", 10),
            yscrollcommand=scrollbar.set,
            state="disabled"
        )
        self.conversation_text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.conversation_text.yview)

        self._set_conversation_text("No conversation loaded yet.")

    def _set_conversation_text(self, text):
        self.conversation_text.config(state="normal")
        self.conversation_text.delete("1.0", tk.END)
        self.conversation_text.insert(tk.END, text)
        self.conversation_text.config(state="disabled")

    def _append_conversation_text(self, text):
        self.conversation_text.config(state="normal")
        self.conversation_text.insert(tk.END, text)
        self.conversation_text.see(tk.END)
        self.conversation_text.config(state="disabled")

    # ========================================================
    # AI MODEL WARMUP
    # ========================================================

    def warmup_model(self):

        try:
            print("Loading DeepFace model...")

            dummy = np.zeros((224, 224, 3), dtype=np.uint8)

            DeepFace.represent(
                img_path=dummy,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False
            )

            print("AI model ready.")

            self.root.after(
                0,
                lambda: self.status_label.config(
                    text="AI Ready — Looking for faces..."
                )
            )

        except Exception as e:
            print("Model warmup error:", e)
            self.root.after(
                0,
                lambda: self.status_label.config(text=f"AI Error: {e}")
            )

    # ========================================================
    # EMBEDDING NORMALIZATION
    # ========================================================

    def normalize_embedding(self, embedding):
        embedding = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)

        if norm == 0:
            return embedding

        return embedding / norm

    # ========================================================
    # COSINE DISTANCE
    # ========================================================

    def cosine_distance(self, embedding1, embedding2):
        embedding1 = self.normalize_embedding(embedding1)
        embedding2 = self.normalize_embedding(embedding2)

        similarity = np.dot(embedding1, embedding2)
        similarity = np.clip(similarity, -1, 1)

        return 1 - similarity

    # ========================================================
    # FIND BEST MATCH
    # ========================================================

    def find_best_match(self, embedding):

        if not self.people:
            return None, None

        best_name = None
        best_distance = float("inf")

        for name, data in self.people.items():

            embeddings = data.get("embeddings", [])

            for stored_embedding in embeddings:

                distance = self.cosine_distance(embedding, stored_embedding)

                if distance < best_distance:
                    best_distance = distance
                    best_name = name

        if best_name is not None and best_distance <= MATCH_THRESHOLD:
            return best_name, best_distance

        return None, best_distance

    # ========================================================
    # FACE QUALITY CHECK
    # ========================================================

    def good_face(self, face):

        if face is None:
            return False

        h, w = face.shape[:2]

        if w < MIN_FACE_SIZE:
            return False

        if h < MIN_FACE_SIZE:
            return False

        return True

    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(self, frame):

        results = []
        now = time.time()
        used_track_ids = set()

        # Remove stale tracks that exceeded TRACK_TIMEOUT
        stale_ids = [
            tid for tid, info in self.tracked_faces.items()
            if now - info["last_seen"] > TRACK_TIMEOUT
        ]
        for tid in stale_ids:
            del self.tracked_faces[tid]
            if tid in self.identity_history:
                del self.identity_history[tid]
            if tid in self.ui_positions:
                del self.ui_positions[tid]

        try:
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                align=True
            )

            for face_data in faces:

                face_image = face_data.get("face")
                facial_area = face_data.get("facial_area", {})
                confidence = face_data.get("confidence", 0)

                if face_image is None:
                    continue

                if face_image.max() <= 1.0:
                    face_image = (face_image * 255).astype(np.uint8)
                else:
                    face_image = face_image.astype(np.uint8)

                if confidence < FACE_DETECTION_CONFIDENCE:
                    continue

                if not self.good_face(face_image):
                    continue

                representation = DeepFace.represent(
                    img_path=face_image,
                    model_name=MODEL_NAME,
                    detector_backend="skip",
                    enforce_detection=False
                )

                if not representation:
                    continue

                embedding = representation[0]["embedding"]

                name, distance = self.find_best_match(embedding)

                x = facial_area.get("x", 0)
                y = facial_area.get("y", 0)
                w = facial_area.get("w", 0)
                h = facial_area.get("h", 0)

                # --------------------------------------------
                # Assign a stable track_id based on face position,
                # so the same head keeps the same id across frames.
                # --------------------------------------------

                cx = x + w // 2
                cy = y + h // 2

                track_id = self._match_track_id(cx, cy, used_track_ids)
                used_track_ids.add(track_id)
                self.tracked_faces[track_id] = {
                    "pos": (cx, cy),
                    "last_seen": now
                }

                stable_name = self.stabilize_identity(track_id, name)

                results.append({
                    "name": name,
                    "stable_name": stable_name,
                    "distance": distance,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "track_id": track_id
                })

        except Exception as e:
            print("AI processing error:", e)

        return results

    def _match_track_id(self, cx, cy, used_track_ids, max_distance=120):
        """
        Finds the closest existing track (within TRACK_TIMEOUT) to
        the point (cx, cy) that hasn't already been claimed this frame.
        Falls back to handing out a brand new track_id.
        """

        best_id = None
        best_dist = max_distance

        for track_id, info in self.tracked_faces.items():

            if track_id in used_track_ids:
                continue

            px, py = info["pos"]
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5

            if dist < best_dist:
                best_dist = dist
                best_id = track_id

        if best_id is None:
            best_id = self.next_track_id
            self.next_track_id += 1

        return best_id

    # ========================================================
    # IDENTITY STABILIZATION
    # ========================================================

    def stabilize_identity(self, track_id, name):

        if name is None:
            return None

        history = self.identity_history[track_id]
        history.append(name)

        if len(history) < REQUIRED_CONFIRMATIONS:
            return None

        counts = {}

        for item in history:
            counts[item] = counts.get(item, 0) + 1

        best_name = max(counts, key=counts.get)

        if counts[best_name] >= REQUIRED_CONFIRMATIONS:
            return best_name

        return None

    # ========================================================
    # DRAW RESULTS
    # ========================================================

    def draw_results(self, frame, results):

        recognized_people = []

        for result in results:

            x = result["x"]
            y = result["y"]
            w = result["w"]
            h = result["h"]

            track_id = result.get("track_id")
            name = result["name"]
            stable_name = result.get("stable_name")

            # ------------------------------------------------
            # Determine displayed information
            # ------------------------------------------------

            if stable_name:

                display_name = stable_name

                relationship = self.people[stable_name].get(
                    "relationship", ""
                )

                if stable_name not in recognized_people:
                    recognized_people.append(stable_name)

                status = "Recognized"

            elif name:

                display_name = "Checking..."
                relationship = ""
                status = "Identifying"

            else:

                display_name = "Unknown"
                relationship = ""
                status = "Unknown"

            # ------------------------------------------------
            # Face center
            # ------------------------------------------------

            target_x = x + w // 2

            # Put card above head
            target_y = y - 20

            # ------------------------------------------------
            # Smooth movement
            # ------------------------------------------------

            if track_id not in self.ui_positions:
                self.ui_positions[track_id] = (target_x, target_y)

            else:
                old_x, old_y = self.ui_positions[track_id]

                new_x = old_x + (target_x - old_x) * self.ui_smoothing
                new_y = old_y + (target_y - old_y) * self.ui_smoothing

                self.ui_positions[track_id] = (new_x, new_y)

            card_x, card_y = self.ui_positions[track_id]
            card_x = int(card_x)
            card_y = int(card_y)

            # ------------------------------------------------
            # Card dimensions
            # ------------------------------------------------

            if relationship:
                card_width = 230
                card_height = 68
            else:
                card_width = 180
                card_height = 58

            # ------------------------------------------------
            # Keep card inside screen
            # ------------------------------------------------

            card_x = max(
                card_width // 2 + 10,
                min(card_x, frame.shape[1] - card_width // 2 - 10)
            )

            card_y = max(card_height + 10, card_y)

            left = int(card_x - card_width // 2)
            top = int(card_y - card_height)
            right = int(card_x + card_width // 2)
            bottom = int(card_y)

            # ------------------------------------------------
            # Create translucent overlay
            # ------------------------------------------------

            overlay = frame.copy()

            # ------------------------------------------------
            # Rounded effect
            # ------------------------------------------------

            radius = 18

            cv2.rectangle(
                overlay, (left + radius, top), (right - radius, bottom),
                (25, 25, 25), -1
            )
            cv2.rectangle(
                overlay, (left, top + radius), (right, bottom - radius),
                (25, 25, 25), -1
            )

            cv2.circle(overlay, (left + radius, top + radius), radius, (25, 25, 25), -1)
            cv2.circle(overlay, (right - radius, top + radius), radius, (25, 25, 25), -1)
            cv2.circle(overlay, (left + radius, bottom - radius), radius, (25, 25, 25), -1)
            cv2.circle(overlay, (right - radius, bottom - radius), radius, (25, 25, 25), -1)

            # ------------------------------------------------
            # Transparency
            # ------------------------------------------------

            cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

            # ------------------------------------------------
            # Small status indicator
            # ------------------------------------------------

            dot_x = left + 22
            dot_y = top + 23

            if status == "Recognized":
                dot_color = (80, 220, 120)
            elif status == "Identifying":
                dot_color = (230, 190, 70)
            else:
                dot_color = (150, 150, 150)

            cv2.circle(frame, (dot_x, dot_y), 5, dot_color, -1)

            # ------------------------------------------------
            # Name
            # ------------------------------------------------

            text_x = left + 38

            cv2.putText(
                frame, display_name, (text_x, top + 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA
            )

            # ------------------------------------------------
            # Relationship
            # ------------------------------------------------

            if relationship:
                cv2.putText(
                    frame, relationship, (text_x, top + 51),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (175, 175, 175), 1, cv2.LINE_AA
                )

            # ------------------------------------------------
            # Small connector line
            # ------------------------------------------------

            line_start = (card_x, bottom)
            line_end = (x + w // 2, y)

            cv2.line(frame, line_start, line_end, (100, 100, 100), 1, cv2.LINE_AA)

        # ----------------------------------------------------
        # Save recognized people (used by conversation tagging /
        # the "retrieve last conversation" hook)
        # ----------------------------------------------------

        self.current_people = recognized_people

        if recognized_people:
            self.current_person = recognized_people[-1]

        # ----------------------------------------------------
        # Remove old UI tracks that expired from tracked_faces
        # ----------------------------------------------------

        for track_id in list(self.ui_positions.keys()):
            if track_id not in self.tracked_faces:
                del self.ui_positions[track_id]

        return frame

    # ========================================================
    # PERSON RECOGNIZED HOOK
    # ========================================================

    def on_person_recognized(self, name):
        """Called once per new stable detection of `name`."""
        self.refresh_last_conversation_button(name)

    # ========================================================
    # CAMERA LOOP
    # ========================================================

    def update_video(self):

        if not self.ai_running:
            return

        ret, frame = self.cap.read()

        if not ret:
            self.root.after(30, self.update_video)
            return

        frame = cv2.flip(frame, 1)
        self.frame_count += 1

        if self.frame_count % AI_FRAME_SKIP == 0 and not self.processing:
            self.processing = True
            ai_frame = frame.copy()

            threading.Thread(
                target=self.run_ai, args=(ai_frame,), daemon=True
            ).start()

        frame = self.draw_results(frame, self.last_results)

        # Fire the "person recognized" hook once per new appearance
        currently_recognized = set(self.current_people)

        for name in currently_recognized - self.previously_recognized:
            self.on_person_recognized(name)

        self.previously_recognized = currently_recognized

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image = image.resize((950, 620))
        photo = ImageTk.PhotoImage(image=image)

        self.video_label.configure(image=photo)
        self.video_label.image = photo

        self.root.after(15, self.update_video)

    # ========================================================
    # BACKGROUND AI
    # ========================================================

    def run_ai(self, frame):

        try:
            results = self.process_frame(frame)
            self.last_results = results

            recognized = None

            for result in results:
                if result["name"]:
                    recognized = result["name"]
                    break

            if recognized:
                relationship = self.people[recognized].get("relationship", "")

                self.root.after(
                    0,
                    lambda n=recognized, r=relationship:
                    self.status_label.config(text=f"Detected: {n} ({r})")
                )

            elif results:
                self.root.after(
                    0,
                    lambda: self.status_label.config(
                        text="Face detected — identifying..."
                    )
                )

            else:
                self.root.after(
                    0,
                    lambda: self.status_label.config(
                        text="AI Ready — Looking for faces..."
                    )
                )

        except Exception as e:
            print("Background AI error:", e)

        finally:
            self.processing = False

    # ========================================================
    # NORMAL REGISTRATION
    # ========================================================

    def register_person(self):

        name = self.name_entry.get().strip()
        relationship = self.relationship_entry.get().strip()

        if not name:
            messagebox.showwarning("Missing Name", "Please enter the person's name.")
            return

        if not relationship:
            messagebox.showwarning("Missing Relationship", "Please enter the relationship.")
            return

        self.status_label.config(text=f"Look at the camera to register {name}...")
        self.register_btn.config(state="disabled")
        self.register_voice_btn.config(state="disabled")

        threading.Thread(
            target=self.capture_registration, args=(name, relationship), daemon=True
        ).start()

    # ========================================================
    # CAPTURE REGISTRATION
    # ========================================================

    def capture_registration(self, name, relationship):

        embeddings = []

        print(f"Starting registration for {name}")

        start_time = time.time()

        while len(embeddings) < MAX_EMBEDDINGS_PER_PERSON:

            if time.time() - start_time > 30:
                print("Registration timeout.")
                break

            ret, frame = self.cap.read()

            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            try:
                faces = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False,
                    align=True
                )

                if len(faces) == 0:
                    self.root.after(
                        0,
                        lambda: self.status_label.config(text="No face detected...")
                    )
                    time.sleep(0.2)
                    continue

                face_data = faces[0]
                face_image = face_data.get("face")
                confidence = face_data.get("confidence", 0)

                if face_image is None:
                    continue

                if confidence < FACE_DETECTION_CONFIDENCE:
                    continue

                if face_image.max() <= 1.0:
                    face_image = (face_image * 255).astype(np.uint8)
                else:
                    face_image = face_image.astype(np.uint8)

                if not self.good_face(face_image):
                    continue

                representation = DeepFace.represent(
                    img_path=face_image,
                    model_name=MODEL_NAME,
                    detector_backend="skip",
                    enforce_detection=False
                )

                if not representation:
                    continue

                embedding = representation[0]["embedding"]
                embedding = self.normalize_embedding(embedding)

                duplicate = False

                for old_embedding in embeddings:
                    distance = self.cosine_distance(embedding, old_embedding)
                    if distance < 0.05:
                        duplicate = True
                        break

                if duplicate:
                    continue

                embeddings.append(embedding)
                count = len(embeddings)

                self.root.after(
                    0,
                    lambda c=count: self.status_label.config(
                        text=f"Capturing face {c}/{MAX_EMBEDDINGS_PER_PERSON}..."
                    )
                )

                print(f"Captured embedding {count}/{MAX_EMBEDDINGS_PER_PERSON}")
                time.sleep(0.5)

            except Exception as e:
                print("Registration AI error:", e)

        if len(embeddings) < REQUIRED_CONFIRMATIONS:
            self.root.after(
                0,
                lambda: messagebox.showwarning(
                    "Registration Failed",
                    "Could not capture enough good face samples."
                )
            )
            self.root.after(
                0, lambda: self.status_label.config(text="Registration failed.")
            )

        else:
            self.people[name] = {
                "relationship": relationship,
                "embeddings": embeddings,
                "created": time.time(),
                "conversations": []
            }

            self.save_database()

            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Registration Successful",
                    f"{name} has been registered successfully."
                )
            )
            self.root.after(
                0,
                lambda: self.status_label.config(
                    text=f"Registered: {name} ({relationship})"
                )
            )

        self.root.after(0, lambda: self.register_btn.config(state="normal"))
        self.root.after(0, lambda: self.register_voice_btn.config(state="normal"))

    # ========================================================
    # VOICE REGISTRATION BUTTON
    # ========================================================

    def voice_register_person(self):

        self.register_voice_btn.config(state="disabled")
        self.register_btn.config(state="disabled")

        self.status_label.config(
            text="🎤 Listening... Say: Hello, I am Rahul, your son"
        )

        threading.Thread(target=self.run_voice_registration, daemon=True).start()

    def run_voice_registration(self):

        try:
            person = register_from_voice()

            if person is None:
                self.root.after(
                    0,
                    lambda: self.status_label.config(
                        text="Could not understand registration."
                    )
                )
                return

            name = person["name"]
            relationship = person["relationship"]

            print("Voice registration:")
            print("Name:", name)
            print("Relationship:", relationship)

            self.root.after(
                0,
                lambda: self.status_label.config(
                    text=f"Voice recognized: {name} ({relationship})"
                )
            )

            time.sleep(1)

            self.root.after(
                0,
                lambda: self.status_label.config(
                    text=f"Look at the camera to register {name}..."
                )
            )

            self.capture_registration(name, relationship)

        except Exception as e:
            print("Voice registration error:", e)
            self.root.after(
                0,
                lambda err=e: self.status_label.config(
                    text=f"Voice registration error: {err}"
                )
            )

        finally:
            self.root.after(0, lambda: self.register_voice_btn.config(state="normal"))
            self.root.after(0, lambda: self.register_btn.config(state="normal"))

    # ========================================================
    # CONVERSATION RECORDING
    # ========================================================

    def toggle_conversation_recording(self):
        if not self.recording:
            self.start_conversation_recording()
        else:
            self.stop_conversation_recording()

    def start_conversation_recording(self):

        target = self.current_person

        if target is None:
            target = self.prompt_for_person_name()
            if target is None:
                return

        self.recording_target_person = target

        self._set_conversation_text(f"🎙️ Recording conversation with {target}...\n\n")

        self.speech_recorder = SpeechRecorder(
            on_partial_text=self.on_partial_transcript
        )

        try:
            self.speech_recorder.start()
        except Exception as e:
            messagebox.showerror(
                "Microphone Error", f"Could not start recording:\n{e}"
            )
            return

        self.recording = True
        self.record_btn.config(text="⏹ Stop & Save Conversation")

    def prompt_for_person_name(self):

        if not self.people:
            messagebox.showwarning(
                "No People Registered", "Please register a person first."
            )
            return None

        name = simpledialog.askstring(
            "Who is this conversation with?",
            "No one is currently recognized.\nEnter the registered person's name:"
        )

        if name and name in self.people:
            return name

        if name:
            messagebox.showwarning(
                "Unknown Person", f"'{name}' is not a registered person."
            )

        return None

    def on_partial_transcript(self, text):
        self.root.after(0, lambda: self._append_conversation_text(text + " "))

    def stop_conversation_recording(self):

        if not self.speech_recorder:
            print("ERROR: No speech recorder exists.")
            return

        print("\n========================================")
        print("STOPPING CONVERSATION RECORDING")
        print("========================================")

        try:
            transcript = self.speech_recorder.stop()

            print("Speech recorder stopped.")
            print("Transcript received:")
            print(repr(transcript))
            print("Transcript length:", len(transcript) if transcript else 0)

        except Exception as e:
            print("ERROR while stopping recorder:", repr(e))

            self.recording = False
            self.record_btn.config(
                text="🎙️ Start Conversation",
                state="normal"
            )

            messagebox.showerror(
                "Recording Error",
                f"Could not stop recording:\n\n{e}"
            )
            return

        self.recording = False

        self.record_btn.config(
            text="🎙️ Start Conversation",
            state="disabled"
        )

        self._set_conversation_text(
            "🤖 Generating AI summary...\n\nPlease wait..."
        )

        target = self.recording_target_person

        print("Conversation target:", target)
        print("Starting processing thread...")

        threading.Thread(
            target=self._process_and_save_conversation,
            args=(target, transcript),
            daemon=True
        ).start()

    def _process_and_save_conversation(self, name, transcript):

        try:

            print("\n========================================")
            print("CONVERSATION PROCESSING")
            print("========================================")

            print("Person:", name)
            print("Transcript:", repr(transcript))
            print("Transcript length:", len(transcript) if transcript else 0)

            # -----------------------------------------
            # CHECK TRANSCRIPT
            # -----------------------------------------

            if not transcript or not transcript.strip():

                print("ERROR: Transcript is empty!")

                raise ValueError(
                    "The speech recorder returned an empty transcript."
                )

            # -----------------------------------------
            # GENERATE SUMMARY
            # -----------------------------------------

            print("\nSTEP 1 -> Calling summarize_text()...")

            summary = summarize_text(
                transcript,
                name=name
            )

            print("STEP 2 -> Summary returned:")
            print(repr(summary))

            if not summary or not summary.strip():

                raise ValueError(
                    "summarize_text() returned an empty summary."
                )

            # -----------------------------------------
            # SAVE FILES
            # -----------------------------------------

            print("\nSTEP 3 -> Saving conversation files...")

            record = save_conversation_files(
                name,
                transcript,
                summary
            )

            print("STEP 4 -> Conversation files saved.")

            print("Transcript file:")
            print(record["transcript_file"])

            print("Summary file:")
            print(record["summary_file"])

            # -----------------------------------------
            # DATABASE
            # -----------------------------------------

            print("\nSTEP 5 -> Updating database...")

            self.people.setdefault(
                name,
                {
                    "relationship": "",
                    "embeddings": [],
                    "conversations": []
                }
            )

            self.people[name].setdefault(
                "conversations",
                []
            )

            self.people[name]["conversations"].append(record)

            self.save_database()

            print("STEP 6 -> Database saved.")

            print("\n========================================")
            print("CONVERSATION PROCESSING COMPLETE")
            print("========================================\n")

            # -----------------------------------------
            # UPDATE UI
            # -----------------------------------------

            self.root.after(
                0,
                lambda n=name, r=record:
                self._show_summary(n, r)
            )

        except Exception as e:

            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("CONVERSATION PROCESSING ERROR")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(repr(e))
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

            error_message = str(e)

            self.root.after(
                0,
                lambda err=error_message:
                messagebox.showerror(
                    "Conversation Error",
                    f"Could not process conversation:\n\n{err}"
                )
            )

        finally:

            self.root.after(
                0,
                lambda: self.record_btn.config(
                    state="normal"
                )
            )

    def _show_summary(self, name, record):
        self._set_conversation_text(
            f"Conversation with {name}\n{record['display_date']}\n\n{record['summary']}"
        )
        self.refresh_last_conversation_button(name)

    # ========================================================
    # CONVERSATION RETRIEVAL
    # ========================================================

    def refresh_last_conversation_button(self, name):
        """
        Called whenever a person is (re)recognized, or a new conversation
        is saved for them. Updates the "last conversation" label/button.
        """
        conversations = self.people.get(name, {}).get("conversations", [])

        if conversations:
            last = conversations[-1]
            self.last_conv_label.config(
                text=f"Retrieve last conversation with {name}\n({last['display_date']})"
            )
            self.view_conv_btn.config(
                state="normal",
                command=lambda: self.load_last_conversation(name)
            )
        else:
            self.last_conv_label.config(text=f"No past conversations with {name} yet.")
            self.view_conv_btn.config(state="disabled", command=lambda: None)

    def load_last_conversation(self, name):
        conversations = self.people.get(name, {}).get("conversations", [])

        if not conversations:
            return

        last = conversations[-1]

        self._set_conversation_text(
            f"Last conversation with {name}\n{last['display_date']}\n\n{last['summary']}"
        )

    # ========================================================
    # CLOSE APPLICATION
    # ========================================================

    def on_closing(self):

        print("Closing application...")

        self.ai_running = False

        if self.recording and self.speech_recorder:
            try:
                self.speech_recorder.stop()
            except Exception:
                pass

        time.sleep(0.2)

        if self.cap is not None:
            self.cap.release()

        self.root.destroy()


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()
    app = DementiaAssistantPro(root)
    root.mainloop()