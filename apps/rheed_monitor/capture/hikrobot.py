# apps/rheed_monitor/capture/hikrobot.py
"""
HIKROBOT GigE 카메라 소스 (MVS SDK ctypes 래퍼)

사전 조건:
  1. HIKROBOT MVS 소프트웨어 설치
     https://www.hikrobotics.com → 다운로드 → Machine Vision → MVS
     (설치 시 GigE 필터 드라이버 + Python SDK 자동 포함)
  2. 카메라 연결 NIC에 Jumbo Frame 활성화 (MTU=9000 권장)
  3. 카메라와 PC가 같은 서브넷이거나, MVS IP 할당 완료 상태

MVS Python SDK 경로 (설치 후 자동 탐색):
  C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import cast, POINTER, memmove
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── MVS SDK 경로 자동 탐색 ──────────────────────────────────────────────────
_MVS_CANDIDATE_PATHS = [
    r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport",
    r"C:\Program Files\MVS\Development\Samples\Python\MvImport",
    r"C:\Program Files\HIKRobot\MVS\Development\Samples\Python\MvImport",
    r"D:\MVS\Development\Samples\Python\MvImport",
]

_mvs_available = False
for _p in _MVS_CANDIDATE_PATHS:
    if Path(_p).exists():
        if _p not in sys.path:
            sys.path.insert(0, _p)
        _mvs_available = True
        break

_MVS_IMPORT_ERROR: Optional[str] = None
if _mvs_available:
    try:
        from MvCameraControl_class import (  # type: ignore
            MvCamera,
            MV_CC_DEVICE_INFO_LIST,
            MV_CC_DEVICE_INFO,
            MV_FRAME_OUT,
            MV_GIGE_DEVICE,
            MV_ACCESS_Exclusive,
        )
        # 픽셀 타입 상수 (모노 판단용)
        _MONO8 = 0x01080001
    except Exception as e:
        _mvs_available = False
        _MVS_IMPORT_ERROR = str(e)
else:
    _MVS_IMPORT_ERROR = (
        "MVS SDK not found. Install HIKROBOT MVS and restart.\n"
        f"Expected: {_MVS_CANDIDATE_PATHS[0]}"
    )


def mvs_available() -> bool:
    return _mvs_available


def list_devices() -> list[str]:
    """검색된 GigE 카메라 목록 반환 (UI 표시용)."""
    if not _mvs_available:
        return [f"[MVS 없음] {_MVS_IMPORT_ERROR}"]
    device_list = MV_CC_DEVICE_INFO_LIST()
    MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, device_list)
    infos: list[str] = []
    for i in range(device_list.nDeviceNum):
        try:
            st = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            gi = st.SpecialInfo.stGigEInfo
            ip = gi.nCurrentIp
            ip_str = f"{(ip>>24)&0xFF}.{(ip>>16)&0xFF}.{(ip>>8)&0xFF}.{ip&0xFF}"
            sn = bytes(gi.chSerialNumber).rstrip(b"\x00").decode(errors="replace")
            model = bytes(gi.chModelName).rstrip(b"\x00").decode(errors="replace")
        except Exception:
            ip_str, sn, model = "?", "?", "?"
        infos.append(f"[{i}] {model}  IP={ip_str}  SN={sn}")
    return infos if infos else ["카메라를 찾을 수 없습니다. 연결/드라이버 확인 필요."]


class HikrobotCamera:
    """
    HIKROBOT GigE 카메라 래퍼.

    Parameters
    ----------
    device_index : 카메라 인덱스 (0 = 첫 번째 검색된 카메라)
    exposure_us  : 노출 시간 (마이크로초)
    gain_db      : 게인 (dB)
    """

    def __init__(
        self,
        device_index: int = 0,
        exposure_us: float = 10_000.0,
        gain_db: float = 0.0,
    ):
        if not _mvs_available:
            raise RuntimeError(_MVS_IMPORT_ERROR)

        self._cam: Optional[MvCamera] = None
        self._device_index = device_index
        self._exposure_us = exposure_us
        self._gain_db = gain_db
        self._open()

    # ── 내부 초기화 ─────────────────────────────────────────────────────────
    def _open(self) -> None:
        device_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, device_list)
        self._check(ret, "MV_CC_EnumDevices")

        n = device_list.nDeviceNum
        if n == 0:
            raise RuntimeError("GigE 카메라를 찾을 수 없습니다. 케이블/드라이버 확인.")
        if self._device_index >= n:
            raise RuntimeError(f"device_index={self._device_index} 초과 (발견={n})")

        cam = MvCamera()
        st = cast(device_list.pDeviceInfo[self._device_index],
                  POINTER(MV_CC_DEVICE_INFO)).contents
        self._check(cam.MV_CC_CreateHandle(st), "MV_CC_CreateHandle")
        self._check(cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0), "MV_CC_OpenDevice")

        cam.MV_CC_SetEnumValue("TriggerMode", 0)            # 연속 취득
        cam.MV_CC_SetFloatValue("ExposureTime", self._exposure_us)
        cam.MV_CC_SetFloatValue("Gain", self._gain_db)
        cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", False)  # 최대 FPS

        self._check(cam.MV_CC_StartGrabbing(), "MV_CC_StartGrabbing")
        self._cam = cam

    @staticmethod
    def _check(ret: int, name: str) -> None:
        if ret != 0:
            raise RuntimeError(f"{name} 실패: 0x{ret:08X}")

    # ── 프레임 읽기 ─────────────────────────────────────────────────────────
    def read(self) -> Optional[np.ndarray]:
        if self._cam is None:
            return None

        st_frame = MV_FRAME_OUT()
        ret = self._cam.MV_CC_GetImageBuffer(st_frame, 1000)  # timeout 1s
        if ret != 0:
            return None

        try:
            w = st_frame.stFrameInfo.nWidth
            h = st_frame.stFrameInfo.nHeight
            n_bytes = st_frame.stFrameInfo.nFrameLen
            pixel_type = st_frame.stFrameInfo.enPixelType

            buf = (ctypes.c_ubyte * n_bytes)()
            memmove(buf, st_frame.pBufAddr, n_bytes)
            data = np.frombuffer(buf, dtype=np.uint8)

            if pixel_type == _MONO8:
                # 모노 → 3채널 BGR
                gray = data[: w * h].reshape(h, w)
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            elif n_bytes >= w * h * 3:
                # RGB8 packed
                frame = data[: w * h * 3].reshape(h, w, 3)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                # Bayer (BayerBG8 기본 가정 — 카메라 설정에 따라 조정)
                bayer = data[: w * h].reshape(h, w)
                frame = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2BGR)

            return frame.copy()
        finally:
            self._cam.MV_CC_FreeImageBuffer(st_frame)

    # ── 설정 변경 ────────────────────────────────────────────────────────────
    def set_exposure(self, us: float) -> None:
        if self._cam:
            self._cam.MV_CC_SetFloatValue("ExposureTime", us)

    def set_gain(self, db: float) -> None:
        if self._cam:
            self._cam.MV_CC_SetFloatValue("Gain", db)

    # ── 해제 ─────────────────────────────────────────────────────────────────
    def release(self) -> None:
        if self._cam:
            self._cam.MV_CC_StopGrabbing()
            self._cam.MV_CC_CloseDevice()
            self._cam.MV_CC_DestroyHandle()
            self._cam = None

    def __del__(self) -> None:
        self.release()
