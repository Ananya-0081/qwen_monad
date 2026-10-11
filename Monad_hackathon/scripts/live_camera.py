"""
DaanDristi live webcam inference demo.

Flow:

    webcam
        -> fixed ROI
        -> empty-background calibration
        -> note-presence gate
        -> persistent presence confirmation
        -> MobileNetV3 classification
        -> if confidence >= 0.80: ACCEPT
        -> write JSONL + SQLite
        -> create blockchain proof
        -> register proof on Monad through API
        -> exit

    if confidence < 0.80:
        -> show LOW CONFIDENCE
        -> keep camera open
        -> retry classification

Run:

    python scripts/live_camera.py
    python scripts/live_camera.py --camera 1
    python scripts/live_camera.py --checkpoint models/mobilenet_v3_clean.pt
    python scripts/live_camera.py --log_path logs/inference.jsonl

Press Q at any time before acceptance to quit.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ============================================================
# ML IMPORTS
# ============================================================

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ============================================================
# BLOCKCHAIN
# ============================================================

from src.blockchain import register_blockchain_proof

# ============================================================
# SQLITE
# ============================================================

try:
    from sqlite.importer import import_single_record
    from sqlite.config import DATABASE_FILE as _SQLITE_DB_FILE

    _SQLITE_AVAILABLE = True

except Exception as _sqlite_import_error:

    _SQLITE_AVAILABLE = False

    print(
        f"Warning: SQLite module unavailable "
        f"({_sqlite_import_error}). "
        "Donations will only be written to the JSONL log."
    )

# ============================================================
# WINDOW SETTINGS
# ============================================================

WINDOW_NAME = "DaanDristi - Live Demo"

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

WINDOW_POS_X = 50
WINDOW_POS_Y = 50

# ============================================================
# MODEL / PREPROCESSING
# ============================================================

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

DENOM_LABELS = {
    "10": "Rs.10",
    "20": "Rs.20",
    "50": "Rs.50",
    "100": "Rs.100",
    "200": "Rs.200",
    "500": "Rs.500",
}

MODEL_VERSION = "mobilenet_v3_clean_v1"

# ============================================================
# PRESENCE GATE / TIMING
# ============================================================

WARMUP_FRAMES = 20

PRESENCE_PIXEL_THRESHOLD = 30

PRESENCE_RATIO_THRESHOLD = 0.08

MIN_CONTOUR_RATIO = 0.04

MAX_CONTOUR_RATIO = 0.85

REQUIRED_PRESENCE_FRAMES = 4

ACCEPT_CONFIDENCE = 0.80

CLASSIFICATION_COOLDOWN = 0.30

# ============================================================
# LOAD MODEL
# ============================================================


def load_model(checkpoint_path, device):
    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
    )

    class_names = ckpt["class_names"]

    model = models.mobilenet_v3_large(
        weights=None
    )

    in_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        in_features,
        len(class_names),
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.to(device)

    model.eval()

    print(f"Device       : {device}")
    print(f"Model path   : {checkpoint_path}")
    print(f"Ckpt epoch   : {ckpt.get('epoch', '?')}")
    print(f"Ckpt val_acc : {ckpt.get('val_acc', '?')}")
    print(f"Classes      : {class_names}")

    return model, class_names


# ============================================================
# CLASSIFICATION
# ============================================================


@torch.inference_mode()
def classify_roi(model, class_names, roi_bgr, device):

    roi_rgb = cv2.cvtColor(
        roi_bgr,
        cv2.COLOR_BGR2RGB,
    )

    pil_img = Image.fromarray(
        roi_rgb
    )

    tensor = EVAL_TRANSFORM(
        pil_img
    ).unsqueeze(0).to(device)

    logits = model(tensor)

    probs = torch.softmax(
        logits,
        dim=1,
    )[0]

    confidence, index = torch.max(
        probs,
        dim=0,
    )

    label = class_names[
        index.item()
    ]

    return label, float(
        confidence.item()
    )


# ============================================================
# ROI
# ============================================================


def compute_roi(frame_h, frame_w):

    roi_w = int(
        frame_w * 0.55
    )

    roi_h = int(
        roi_w * 0.45
    )

    roi_h = min(
        roi_h,
        int(frame_h * 0.65),
    )

    x1 = (frame_w - roi_w) // 2
    y1 = (frame_h - roi_h) // 2

    x2 = x1 + roi_w
    y2 = y1 + roi_h

    return x1, y1, x2, y2


# ============================================================
# IMAGE PREPROCESSING
# ============================================================


def preprocess_gray(roi_bgr):

    gray = cv2.cvtColor(
        roi_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (21, 21),
        0,
    )

    return gray


# ============================================================
# BACKGROUND
# ============================================================


def build_background(background_frames):

    background = np.median(
        np.stack(
            background_frames,
            axis=0,
        ),
        axis=0,
    )

    return background.astype(
        np.uint8
    )


# ============================================================
# NOTE PRESENCE DETECTION
# ============================================================


def detect_note_presence(
    background_gray,
    current_gray,
):

    diff = cv2.absdiff(
        background_gray,
        current_gray,
    )

    _, mask = cv2.threshold(
        diff,
        PRESENCE_PIXEL_THRESHOLD,
        255,
        cv2.THRESH_BINARY,
    )

    kernel = np.ones(
        (5, 5),
        np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    roi_area = (
        mask.shape[0]
        * mask.shape[1]
    )

    foreground_pixels = cv2.countNonZero(
        mask
    )

    foreground_ratio = (
        foreground_pixels
        / float(roi_area)
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    largest_area = 0.0

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area > largest_area:
            largest_area = area

    contour_ratio = (
        largest_area
        / float(roi_area)
    )

    presence_detected = (
        foreground_ratio
        >= PRESENCE_RATIO_THRESHOLD
        and
        MIN_CONTOUR_RATIO
        <= contour_ratio
        <= MAX_CONTOUR_RATIO
    )

    return (
        presence_detected,
        foreground_ratio,
        contour_ratio,
    )


# ============================================================
# UI
# ============================================================


def draw_overlay(
    frame,
    roi_coords,
    label,
    confidence,
    fps,
    status,
    presence_ratio,
):

    x1, y1, x2, y2 = roi_coords

    h, w = frame.shape[:2]

    if status.startswith("WAITING"):
        color = (
            180,
            180,
            180,
        )

    elif status.startswith("CHECKING"):
        color = (
            0,
            180,
            255,
        )

    elif status.startswith("LOW CONFIDENCE"):
        color = (
            0,
            120,
            255,
        )

    elif (
        status.startswith("DETECTED")
        or
        status.startswith("ACCEPT")
    ):
        color = (
            0,
            220,
            0,
        )

    else:
        color = (
            0,
            180,
            255,
        )

    # ROI rectangle
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    # Corner brackets
    bracket = 20
    thickness = 3

    corners = [
        (x1, y1, 1, 1),
        (x2, y1, -1, 1),
        (x1, y2, 1, -1),
        (x2, y2, -1, -1),
    ]

    for cx, cy, dx, dy in corners:

        cv2.line(
            frame,
            (cx, cy),
            (
                cx + dx * bracket,
                cy,
            ),
            color,
            thickness,
        )

        cv2.line(
            frame,
            (cx, cy),
            (
                cx,
                cy + dy * bracket,
            ),
            color,
            thickness,
        )

    # Instruction
    cv2.putText(
        frame,
        "Place note inside the box",
        (
            x1,
            max(25, y1 - 12),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    if label is None:

        cv2.putText(
            frame,
            status,
            (
                x1,
                y1 + (y2 - y1) // 2,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    else:

        display_label = DENOM_LABELS.get(
            label,
            label,
        )

        confidence_text = (
            f"{confidence * 100:.1f}%"
        )

        panel_y = min(
            h - 65,
            y2 + 35,
        )

        cv2.putText(
            frame,
            f"{display_label} {confidence_text}",
            (
                x1,
                panel_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            color,
            3,
            cv2.LINE_AA,
        )

        status_y = min(
            h - 45,
            panel_y + 25,
        )

        cv2.putText(
            frame,
            status,
            (
                x1,
                status_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

        # Confidence bar
        bar_x = x1

        bar_y = min(
            h - 25,
            status_y + 12,
        )

        bar_w = x2 - x1
        bar_h = 8

        cv2.rectangle(
            frame,
            (
                bar_x,
                bar_y,
            ),
            (
                bar_x + bar_w,
                bar_y + bar_h,
            ),
            (80, 80, 80),
            -1,
        )

        fill_width = int(
            bar_w
            * min(confidence, 1.0)
        )

        cv2.rectangle(
            frame,
            (
                bar_x,
                bar_y,
            ),
            (
                bar_x + fill_width,
                bar_y + bar_h,
            ),
            color,
            -1,
        )

    # FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (
            max(10, w - 140),
            30,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    # Title
    cv2.putText(
        frame,
        "DaanDristi",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Presence
    cv2.putText(
        frame,
        f"Presence: {presence_ratio * 100:.1f}%",
        (
            15,
            h - 35,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    # Quit
    cv2.putText(
        frame,
        "Q = quit",
        (
            15,
            h - 15,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )


# ============================================================
# LOGGING
# ============================================================


def append_log(
    log_path,
    camera_index,
    label,
    confidence,
):

    """
    Called exactly once per successful script execution,
    only when confidence >= ACCEPT_CONFIDENCE.

    Always writes decision = ACCEPT.
    """

    record = {
        "pocket_id": f"webcam-{camera_index}",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "item_status": "NOTE_DETECTED",
        "predicted_denomination": label,
        "classifier_confidence": confidence,
        "confidence": confidence,
        "source": "webcam",
        "decision": "ACCEPT",
        "target_bin": f"BIN_{label}",
        "model_version": MODEL_VERSION,
    }

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # JSONL
    with log_path.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                record,
                separators=(",", ":"),
            )
            + "\n"
        )

    # SQLite
    if _SQLITE_AVAILABLE:

        try:

            result = import_single_record(
                record=record,
                database_file=_SQLITE_DB_FILE,
            )

            print(
                f"SQLite: "
                f"{result.inserted} record(s) "
                f"inserted into "
                f"{_SQLITE_DB_FILE}"
            )

        except Exception as _db_err:

            print(
                "Warning: SQLite write failed "
                f"({_db_err}). "
                "JSONL record is still safe."
            )

    return record


# ============================================================
# MAIN
# ============================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "DaanDristi live camera "
            "note detection + classification"
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            _PROJECT_ROOT
            / "models"
            / "mobilenet_v3_clean.pt"
        ),
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--log_path",
        type=str,
        default=str(
            _PROJECT_ROOT
            / "logs"
            / "inference.jsonl"
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # DEVICE + MODEL
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, class_names = load_model(
        args.checkpoint,
        device,
    )

    # ========================================================
    # CAMERA
    # ========================================================

    cap = cv2.VideoCapture(
        args.camera
    )

    if not cap.isOpened():

        print(
            f"ERROR: Could not open camera "
            f"index {args.camera}. "
            f"Try --camera 1."
        )

        return

    log_path = Path(
        args.log_path
    )

    # ========================================================
    # WINDOW
    # ========================================================

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
    )

    cv2.moveWindow(
        WINDOW_NAME,
        WINDOW_POS_X,
        WINDOW_POS_Y,
    )

    ret, initial_frame = cap.read()

    if ret:

        cv2.imshow(
            WINDOW_NAME,
            initial_frame,
        )

        cv2.waitKey(1)

        try:

            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_TOPMOST,
                1,
            )

        except Exception:
            pass

    print()
    print(
        f"Camera opened (index {args.camera})."
    )

    print(
        "KEEP THE ROI EMPTY during startup calibration."
    )

    print(
        "Waiting for a real note..."
    )

    print()

    # ========================================================
    # STATE
    # ========================================================

    background_frames = []

    background_gray = None

    presence_streak = 0

    note_confirmed = False

    note_confirmed_announced = False

    last_label = None

    last_confidence = 0.0

    last_classification_time = 0.0

    frame_counter = 0

    prev_time = time.perf_counter()

    fps = 0.0

    try:

        while True:

            ret, frame = cap.read()

            if not ret:

                print(
                    "Failed to read frame."
                )

                break

            # =================================================
            # ROI
            # =================================================

            h, w = frame.shape[:2]

            roi_coords = compute_roi(
                h,
                w,
            )

            x1, y1, x2, y2 = roi_coords

            roi_crop = frame[
                y1:y2,
                x1:x2,
            ]

            roi_gray = preprocess_gray(
                roi_crop
            )

            # =================================================
            # FPS
            # =================================================

            now = time.perf_counter()

            dt = now - prev_time

            if dt > 0:
                fps = 1.0 / dt

            prev_time = now

            # =================================================
            # BACKGROUND CALIBRATION
            # =================================================

            if background_gray is None:

                background_frames.append(
                    roi_gray.copy()
                )

                frame_counter += 1

                status = (
                    "CALIBRATING BACKGROUND "
                    f"{frame_counter}/{WARMUP_FRAMES}"
                )

                if frame_counter >= WARMUP_FRAMES:

                    background_gray = (
                        build_background(
                            background_frames
                        )
                    )

                    print(
                        "Background calibration complete."
                    )

                    status = (
                        "WAITING FOR NOTE"
                    )

                draw_overlay(
                    frame,
                    roi_coords,
                    None,
                    0.0,
                    fps,
                    status,
                    0.0,
                )

                cv2.imshow(
                    WINDOW_NAME,
                    frame,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key == ord("q"):
                    break

                continue

            # =================================================
            # PRESENCE DETECTION
            # =================================================

            (
                presence_detected,
                presence_ratio,
                contour_ratio,
            ) = detect_note_presence(
                background_gray,
                roi_gray,
            )

            if presence_detected:

                presence_streak += 1

            else:

                presence_streak = 0

                if note_confirmed:

                    note_confirmed = False
                    note_confirmed_announced = False
                    last_label = None
                    last_confidence = 0.0

            # =================================================
            # CONFIRM PRESENCE
            # =================================================

            if (
                not note_confirmed
                and
                presence_streak
                >= REQUIRED_PRESENCE_FRAMES
            ):

                note_confirmed = True

                last_classification_time = 0.0

            # =================================================
            # CLASSIFY
            # =================================================

            if note_confirmed:

                if not note_confirmed_announced:

                    print(
                        "Physical object detected in ROI."
                    )

                    note_confirmed_announced = True

                time_since_last = (
                    now
                    - last_classification_time
                )

                if (
                    time_since_last
                    >= CLASSIFICATION_COOLDOWN
                ):

                    (
                        last_label,
                        last_confidence,
                    ) = classify_roi(
                        model,
                        class_names,
                        roi_crop,
                        device,
                    )

                    last_classification_time = now

                    print(
                        f"Prediction: "
                        f"{DENOM_LABELS.get(last_label, last_label)} "
                        f"Confidence: "
                        f"{last_confidence * 100:.2f}%"
                    )

                # =================================================
                # ACCEPT
                # =================================================

                if (
                    last_confidence
                    >= ACCEPT_CONFIDENCE
                ):

                    target_bin = (
                        f"BIN_{last_label}"
                    )

                    draw_overlay(
                        frame,
                        roi_coords,
                        last_label,
                        last_confidence,
                        fps,
                        "ACCEPTED",
                        presence_ratio,
                    )

                    cv2.imshow(
                        WINDOW_NAME,
                        frame,
                    )

                    cv2.waitKey(2000)

                    # =================================================
                    # SAVE ML RECORD
                    # =================================================

                    record = append_log(
                        log_path,
                        args.camera,
                        last_label,
                        last_confidence,
                    )

                    # =================================================
                    # BLOCKCHAIN REGISTRATION
                    # =================================================

                    blockchain_result = (
                        register_blockchain_proof(
                            record
                        )
                    )

                    if blockchain_result:

                        print()
                        print(
                            "=== BLOCKCHAIN PROOF ==="
                        )

                        print(
                            "Proof hash: "
                            f"{blockchain_result['proof_hash']}"
                        )

                        print(
                            "Transaction: "
                            f"{blockchain_result['transaction_hash']}"
                        )

                        print(
                            "Block: "
                            f"{blockchain_result['block_number']}"
                        )

                    else:

                        print()
                        print(
                            "Blockchain proof was not registered."
                        )

                        print(
                            "The ML record is still safely "
                            "stored in JSONL/SQLite."
                        )

                    # =================================================
                    # FINAL RECORD
                    # =================================================

                    print()
                    print(
                        "=== FINAL RECORD ==="
                    )

                    print(
                        json.dumps(
                            record,
                            indent=2,
                        )
                    )

                    print()

                    print(
                        f"One record appended to: "
                        f"{log_path}"
                    )

                    print(
                        f"Target bin: "
                        f"{target_bin}"
                    )

                    # Exactly one detection per execution.
                    break

                # =================================================
                # LOW CONFIDENCE
                # =================================================

                status = (
                    "LOW CONFIDENCE - "
                    "HOLD NOTE STEADY"
                )

                draw_overlay(
                    frame,
                    roi_coords,
                    last_label,
                    last_confidence,
                    fps,
                    status,
                    presence_ratio,
                )

                cv2.imshow(
                    WINDOW_NAME,
                    frame,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key == ord("q"):
                    break

                continue

            # =================================================
            # WAITING / CHECKING
            # =================================================

            if presence_detected:

                status = (
                    "CHECKING... "
                    f"{presence_streak}/"
                    f"{REQUIRED_PRESENCE_FRAMES}"
                )

            else:

                status = (
                    "WAITING FOR NOTE"
                )

            draw_overlay(
                frame,
                roi_coords,
                None,
                0.0,
                fps,
                status,
                presence_ratio,
            )

            cv2.imshow(
                WINDOW_NAME,
                frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

    finally:

        cap.release()

        cv2.destroyAllWindows()

        print(
            "Camera released. Goodbye."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()