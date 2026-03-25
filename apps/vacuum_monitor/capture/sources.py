# apps/vacuum_monitor/capture/sources.py
import time
import cv2
import numpy as np
from dataclasses import dataclass, field
import mss  # 화면 캡처용

@dataclass
class SourceBase:
    def read(self) -> np.ndarray:
        raise NotImplementedError
    def release(self):
        pass

@dataclass
class FileSource(SourceBase):
    path: str
    loop: bool = True
    cap: cv2.VideoCapture = field(init=False, default=None)
    def __post_init__(self):
        self.cap = cv2.VideoCapture(self.path)
    def read(self):
        ret, frame = self.cap.read()
        if not ret and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return frame if ret else None
    def release(self):
        if self.cap: self.cap.release()

@dataclass
class UvcSource(SourceBase):
    index: int
    cap: cv2.VideoCapture = field(init=False, default=None)
    def __post_init__(self):
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            # UVC 실패 시 그냥 빈 화면이라도 내보내서 프로그램 죽는 것 방지
            print(f"[Warn] Camera index {self.index} failed. Check connections.")
    def read(self):
        if not self.cap or not self.cap.isOpened(): return None
        ret, frame = self.cap.read()
        return frame if ret else None
    def release(self):
        if self.cap: self.cap.release()

# [핵심] 화면 캡처 소스
@dataclass
class ScreenSource(SourceBase):
    monitor_idx: int = 1
    sct: mss.mss = field(init=False, default=None)
    monitor: dict = field(init=False, default=None)
    def __post_init__(self):
        self.sct = mss.mss()
        # 모니터 인덱스 안전하게 가져오기
        idx = self.monitor_idx if self.monitor_idx < len(self.sct.monitors) else 1
        self.monitor = self.sct.monitors[idx]
    def read(self):
        img = self.sct.grab(self.monitor)
        frame = np.array(img)
        # 색상 변환 (BGRA -> BGR)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    def release(self):
        if self.sct: self.sct.close()

@dataclass
class HikrobotMvsSource(SourceBase):
    config: dict
    def __post_init__(self): pass
    def read(self): return np.zeros((480, 640, 3), dtype=np.uint8)