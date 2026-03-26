import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import pandas as pd

# Default ROI tuned for video1.mp4 (x, y, w, h)
DEFAULT_ROI = (1360, 120, 620, 420)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Детекция состояния столика по видео")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", default="output.mp4", help="Path to output video")
    parser.add_argument(
        "--events-csv",
        default="events.csv",
        help="Path to save events DataFrame as CSV",
    )
    parser.add_argument(
        "--summary-txt", default="summary.txt", help="Path to save text summary"
    )
    parser.add_argument(
        "--table-roi",
        default=None,
        help="Manual ROI in format x,y,w,h. If omitted, built-in default is used.",
    )
    parser.add_argument(
        "--select-roi",
        action="store_true",
        help="Pick ROI interactively via cv2.selectROI",
    )
    parser.add_argument("--start-sec", type=float, default=0.0, help="Start second")
    parser.add_argument(
        "--end-sec",
        type=float,
        default=None,
        help="End second (inclusive). If omitted, process full video.",
    )

    # Detection parameters
    parser.add_argument(
        "--detection-scale",
        type=float,
        default=0.5,
        help="Scale factor for detection pipeline (0..1], output video stays full-size",
    )
    parser.add_argument(
        "--detect-every-n",
        type=int,
        default=1,
        help="Run detector every Nth frame for speed",
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=0.008,
        help="Foreground pixel ratio in ROI to treat frame as motion",
    )
    parser.add_argument(
        "--occupied-hold-sec",
        type=float,
        default=12.0,
        help="Keep state OCCUPIED this many seconds after last motion",
    )
    parser.add_argument(
        "--min-state-change-sec",
        type=float,
        default=3.0,
        help="Minimal gap between state transitions to avoid flicker",
    )
    parser.add_argument(
        "--warmup-sec",
        type=float,
        default=5.0,
        help="Warm-up time for background subtractor before logging events",
    )
    parser.add_argument(
        "--bg-history",
        type=int,
        default=500,
        help="MOG2 history parameter",
    )
    parser.add_argument(
        "--bg-var-threshold",
        type=float,
        default=32.0,
        help="MOG2 varThreshold parameter",
    )
    parser.add_argument(
        "--shadow-threshold",
        type=int,
        default=200,
        help="Mask threshold to remove shadows (0..255)",
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Show processing window (ESC to stop)",
    )

    return parser.parse_args()


def parse_roi(roi_text: str) -> Tuple[int, int, int, int]:
    parts = [int(p.strip()) for p in roi_text.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must have 4 integers: x,y,w,h")
    x, y, w, h = parts
    if w <= 0 or h <= 0:
        raise ValueError("ROI width and height must be > 0")
    return x, y, w, h


def clamp_roi(
    roi: Tuple[int, int, int, int], frame_w: int, frame_h: int
) -> Tuple[int, int, int, int]:
    x, y, w, h = roi
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return x, y, w, h


def seconds_to_hms(seconds: float) -> str:
    seconds = max(seconds, 0)
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def add_event(
    events: List[Dict[str, object]],
    timestamp_sec: float,
    frame_idx: int,
    event_type: str,
    state: str,
    motion_ratio: float,
    delay_from_empty_sec: Optional[float] = None,
) -> None:
    events.append(
        {
            "timestamp_sec": round(timestamp_sec, 3),
            "timestamp_hms": seconds_to_hms(timestamp_sec),
            "frame_idx": frame_idx,
            "event_type": event_type,
            "state": state,
            "motion_ratio": round(motion_ratio, 6),
            "delay_from_empty_sec": (
                None if delay_from_empty_sec is None else round(delay_from_empty_sec, 3)
            ),
        }
    )


def select_roi_interactive(frame) -> Tuple[int, int, int, int]:
    selected = cv2.selectROI(
        "Select table ROI", frame, fromCenter=False, showCrosshair=True
    )
    cv2.destroyWindow("Select table ROI")
    x, y, w, h = [int(v) for v in selected]
    if w <= 0 or h <= 0:
        raise ValueError("Empty ROI selected")
    return x, y, w, h


def run(args: argparse.Namespace) -> None:
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    start_sec = max(0.0, args.start_sec)
    end_sec = duration_sec if args.end_sec is None else min(duration_sec, args.end_sec)
    if end_sec <= start_sec:
        raise ValueError("--end-sec must be greater than --start-sec")

    start_frame = int(start_sec * fps)
    end_frame = min(total_frames - 1, int(end_sec * fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Cannot read first frame at chosen start time")

    if args.select_roi:
        roi = select_roi_interactive(first_frame)
    elif args.table_roi:
        roi = parse_roi(args.table_roi)
    else:
        roi = DEFAULT_ROI

    roi = clamp_roi(roi, frame_w, frame_h)
    x, y, w, h = roi

    cap.release()
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open VideoWriter for: {args.output}")

    detection_scale = float(args.detection_scale)
    if detection_scale <= 0 or detection_scale > 1.0:
        cap.release()
        writer.release()
        raise ValueError("--detection-scale must be within (0, 1]")

    if args.detect_every_n <= 0:
        cap.release()
        writer.release()
        raise ValueError("--detect-every-n must be >= 1")

    sx = int(x * detection_scale)
    sy = int(y * detection_scale)
    sw = max(1, int(w * detection_scale))
    sh = max(1, int(h * detection_scale))

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=args.bg_history,
        varThreshold=args.bg_var_threshold,
        detectShadows=True,
    )

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    events: List[Dict[str, object]] = []
    approach_delays: List[float] = []

    state = "EMPTY"
    state_initialized = False
    last_motion_sec = -10_000.0
    last_state_change_sec = -10_000.0
    last_empty_sec: Optional[float] = None

    warmup_end_sec = start_sec + args.warmup_sec

    frame_idx = start_frame - 1
    processed_frames = 0
    motion_ratio = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        if frame_idx > end_frame:
            break

        timestamp_sec = frame_idx / fps

        if processed_frames % args.detect_every_n == 0:
            if detection_scale != 1.0:
                det_frame = cv2.resize(
                    frame,
                    (0, 0),
                    fx=detection_scale,
                    fy=detection_scale,
                    interpolation=cv2.INTER_AREA,
                )
            else:
                det_frame = frame

            fg_mask = bg_subtractor.apply(det_frame)
            roi_mask = fg_mask[sy : sy + sh, sx : sx + sw]

            _, roi_bin = cv2.threshold(
                roi_mask, args.shadow_threshold, 255, cv2.THRESH_BINARY
            )
            roi_bin = cv2.morphologyEx(roi_bin, cv2.MORPH_OPEN, open_kernel)
            roi_bin = cv2.morphologyEx(roi_bin, cv2.MORPH_CLOSE, close_kernel)

            motion_ratio = float(cv2.countNonZero(roi_bin)) / float(sw * sh)
            if motion_ratio >= args.motion_threshold:
                last_motion_sec = timestamp_sec

        occupied_candidate = (timestamp_sec - last_motion_sec) <= args.occupied_hold_sec

        if timestamp_sec >= warmup_end_sec:
            if not state_initialized:
                state = "OCCUPIED" if occupied_candidate else "EMPTY"
                state_initialized = True
                last_state_change_sec = timestamp_sec
                add_event(
                    events,
                    timestamp_sec,
                    frame_idx,
                    event_type="INIT_STATE",
                    state=state,
                    motion_ratio=motion_ratio,
                )
            else:
                next_state = "OCCUPIED" if occupied_candidate else "EMPTY"
                can_switch = (
                    timestamp_sec - last_state_change_sec
                ) >= args.min_state_change_sec

                if next_state != state and can_switch:
                    state = next_state
                    last_state_change_sec = timestamp_sec

                    if state == "EMPTY":
                        add_event(
                            events,
                            timestamp_sec,
                            frame_idx,
                            event_type="TABLE_EMPTY",
                            state=state,
                            motion_ratio=motion_ratio,
                        )
                        last_empty_sec = timestamp_sec
                    else:
                        add_event(
                            events,
                            timestamp_sec,
                            frame_idx,
                            event_type="TABLE_OCCUPIED",
                            state=state,
                            motion_ratio=motion_ratio,
                        )
                        if last_empty_sec is not None:
                            delay = timestamp_sec - last_empty_sec
                            approach_delays.append(delay)
                            add_event(
                                events,
                                timestamp_sec,
                                frame_idx,
                                event_type="APPROACH",
                                state=state,
                                motion_ratio=motion_ratio,
                                delay_from_empty_sec=delay,
                            )
                            last_empty_sec = None

        if state_initialized:
            box_color = (0, 0, 255) if state == "OCCUPIED" else (0, 180, 0)
            state_label = state
        else:
            box_color = (0, 255, 255)
            state_label = "WARMUP"

        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 5)
        cv2.putText(
            frame,
            f"State: {state_label}",
            (x, max(30, y - 40)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            box_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Motion ratio: {motion_ratio:.4f}",
            (x, max(60, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Time: {seconds_to_hms(timestamp_sec)}",
            (40, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

        if args.display:
            preview = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
            cv2.imshow("Table state", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

        processed_frames += 1

    cap.release()
    writer.release()
    if args.display:
        cv2.destroyAllWindows()

    events_df = pd.DataFrame(events)
    if not events_df.empty:
        events_df = events_df.sort_values("timestamp_sec").reset_index(drop=True)
    events_df.to_csv(args.events_csv, index=False)

    approach_series = pd.Series(approach_delays, dtype="float64")
    mean_delay = (
        float(approach_series.mean()) if not approach_series.empty else float("nan")
    )

    lines = [
        f"video={video_path}",
        f"output={args.output}",
        f"events_csv={args.events_csv}",
        f"roi={x},{y},{w},{h}",
        f"processed_seconds={start_sec:.2f}..{end_sec:.2f}",
        f"events_total={len(events_df)}",
        f"approach_events={len(approach_delays)}",
    ]
    if approach_delays:
        lines.append(f"mean_delay_sec={mean_delay:.3f}")
    else:
        lines.append("mean_delay_sec=NA")

    Path(args.summary_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[DONE] Processing complete")
    print(f"Video: {video_path}")
    print(f"ROI: {x},{y},{w},{h}")
    print(f"Output video: {args.output}")
    print(f"Events CSV: {args.events_csv}")
    print(f"Summary: {args.summary_txt}")
    print(f"Events total: {len(events_df)}")
    if approach_delays:
        print(f"Approach events: {len(approach_delays)}")
        print(f"Mean delay (sec): {mean_delay:.3f}")
    else:
        print("Approach events: 0")
        print("Mean delay (sec): NA (not enough EMPTY->OCCUPIED pairs)")


if __name__ == "__main__":
    run(parse_args())
