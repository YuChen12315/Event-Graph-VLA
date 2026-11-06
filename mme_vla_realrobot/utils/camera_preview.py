import threading
import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
import cv2

class CameraPreview:
    """Latest-frame mailbox; OpenCV GUI stays on the main event-loop thread."""

    def __init__(self, output_dir: str = "/home/main3/yu/record/deco_task7", fps: float = 60.0, label: str = "deco_task7_stage1"):
        self._lock = threading.Lock()
        self._frame = None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.label = label
        self.writer = None
        self.record_path = None

    def update(self, frame):
        with self._lock:
            self._frame = frame

    def _start_recording(self, frame):
        if self.writer is not None:
            return
        height, width = frame.shape[:2]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = self.output_dir / f"{self.label}_{timestamp}.mp4"
        index = 1
        while candidate.exists():
            candidate = self.output_dir / f"{self.label}_{timestamp}_{index}.mp4"
            index += 1
        self.writer = cv2.VideoWriter(
            str(candidate),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height),
        )
        if not self.writer.isOpened():
            print(f"[policy_server] failed to open video writer: {candidate}", flush=True)
            self.writer = None
            return
        self.record_path = candidate
        print(f"[policy_server] recording preview to {candidate}", flush=True)

    def _stop_recording(self):
        if self.writer is not None:
            try:
                if self.writer.isOpened():
                    self.writer.release()
            except cv2.error:
                pass
            self.writer = None
            self.record_path = None

    async def run(self):
        title = "Pi05 server - received stereo (Q/Esc closes preview)"
        created = False
        headless = False
        try:
            if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                print("[policy_server] no desktop display; recording continues without preview window", flush=True)
                headless = True
            if not headless:
                cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                created = True
                cv2.resizeWindow(title, 1280, 360)
            while True:
                with self._lock:
                    frame, self._frame = self._frame, None
                if frame is not None:
                    if self.writer is None:
                        self._start_recording(frame)
                    if self.writer is not None and self.writer.isOpened():
                        self.writer.write(frame)
                    if not headless:
                        height, width = frame.shape[:2]
                        if width > 1280:
                            display = cv2.resize(frame, (1280, max(1, round(height * 1280 / width))))
                        else:
                            display = frame
                        cv2.imshow(title, display)
                if headless:
                    await asyncio.sleep(1 / 30)
                    continue
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                await asyncio.sleep(1 / 30)
        except cv2.error as exc:
            print(f"[policy_server] preview disabled: {exc}", flush=True)
        finally:
            self._stop_recording()
            if created:
                try:
                    cv2.destroyWindow(title)
                except cv2.error:
                    pass
