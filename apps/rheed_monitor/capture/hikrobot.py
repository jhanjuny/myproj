# apps/rheed_monitor/capture/hikrobot.py
"""
HIKROBOT GigE 카메라 소스 (MVS SDK ctypes 래퍼)

MVS SDK 탐색 우선순위:
  1. 환경변수  HIKROBOT_MVS_PATH
  2. 설정 파일 rheed_config.yaml의 mvs_sdk_path 항목
  3. Windows 레지스트리 (HIKROBOT 설치 정보)
  4. 표준 설치 경로 후보 목록 (여러 드라이브/경로)

수동 경로 지정:
  set_mvs_path("C:\\YourPath\\MvImport") 를 호출한 뒤 import
  또는 HIKROBOT_MVS_PATH 환경변수 설정 후 재시작
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import cast, POINTER, memmove
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np


# ── MVS SDK 경로 탐색 ──────────────────────────────────────────────────────

_MVS_CANDIDATE_PATHS: List[str] = [
    # Program Files (x86) — 32-bit 설치
    r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport",
    r"C:\Program Files (x86)\HIKRobot\MVS\Development\Samples\Python\MvImport",
    r"C:\Program Files (x86)\Hikrobot\MVS\Development\Samples\Python\MvImport",
    # Program Files — 64-bit 설치
    r"C:\Program Files\MVS\Development\Samples\Python\MvImport",
    r"C:\Program Files\HIKRobot\MVS\Development\Samples\Python\MvImport",
    r"C:\Program Files\Hikrobot\MVS\Development\Samples\Python\MvImport",
    # D 드라이브
    r"D:\MVS\Development\Samples\Python\MvImport",
    r"D:\HIKRobot\MVS\Development\Samples\Python\MvImport",
    r"D:\Program Files\MVS\Development\Samples\Python\MvImport",
    r"D:\Program Files (x86)\MVS\Development\Samples\Python\MvImport",
    # E 드라이브
    r"E:\MVS\Development\Samples\Python\MvImport",
    r"E:\HIKRobot\MVS\Development\Samples\Python\MvImport",
]


def _find_via_registry() -> Optional[str]:
    """Windows 레지스트리에서 MVS 설치 경로를 탐색."""
    try:
        import winreg
        key_paths = [
            r"SOFTWARE\Hikrobot\MVS",
            r"SOFTWARE\HIKRobot\MVS",
            r"SOFTWARE\WOW6432Node\Hikrobot\MVS",
            r"SOFTWARE\WOW6432Node\HIKRobot\MVS",
            r"SOFTWARE\HIKROBOT\MVS",
            r"SOFTWARE\WOW6432Node\HIKROBOT\MVS",
        ]
        value_names = ["InstallPath", "installpath", "Path", "Install"]
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            for key_path in key_paths:
                try:
                    with winreg.OpenKey(hive, key_path) as key:
                        for vname in value_names:
                            try:
                                install_dir, _ = winreg.QueryValueEx(key, vname)
                                candidate = Path(install_dir) / "Development/Samples/Python/MvImport"
                                if candidate.exists():
                                    return str(candidate)
                                # MvCameraControl_class.py 직접 탐색
                                for p in Path(install_dir).rglob("MvCameraControl_class.py"):
                                    return str(p.parent)
                            except (FileNotFoundError, OSError):
                                pass
                except (FileNotFoundError, OSError):
                    pass
    except (ImportError, Exception):
        pass
    return None


def _find_via_env() -> Optional[str]:
    """환경변수 HIKROBOT_MVS_PATH에서 경로 탐색."""
    p = os.environ.get("HIKROBOT_MVS_PATH", "").strip()
    if p and Path(p).exists():
        return p
    return None


def _find_via_config() -> Optional[str]:
    """rheed_config.yaml 또는 config.yaml에서 mvs_sdk_path 탐색."""
    try:
        import yaml
        cfg_candidates = []
        if getattr(sys, "frozen", False):
            cfg_candidates.append(Path(sys.executable).parent / "rheed_config.yaml")
        else:
            cfg_candidates.append(Path(__file__).parents[3] / "apps/rheed_monitor/config.yaml")
            cfg_candidates.append(Path(__file__).parents[3] / "rheed_config.yaml")

        for cfg_path in cfg_candidates:
            if cfg_path.exists():
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                p = cfg.get("mvs_sdk_path", "").strip()
                if p and Path(p).exists():
                    return p
    except Exception:
        pass
    return None


def _find_all_drives_glob() -> Optional[str]:
    """모든 드라이브에서 MvCameraControl_class.py 위치 탐색 (느리므로 마지막 수단)."""
    import subprocess
    try:
        # wmic logical disk로 드라이브 목록 가져오기
        result = subprocess.run(
            ["wmic", "logicaldisk", "get", "caption"],
            capture_output=True, text=True, timeout=5
        )
        drives = [line.strip() for line in result.stdout.splitlines()
                  if line.strip() and ":" in line and line.strip() != "Caption"]
        for drive in drives:
            # Hikrobot 디렉토리만 탐색 (전체 탐색은 너무 느림)
            for top_dir in ["Hikrobot", "HIKRobot", "HIKROBOT", "MVS"]:
                candidate_root = Path(drive) / "Program Files" / top_dir
                if candidate_root.exists():
                    for p in candidate_root.rglob("MvCameraControl_class.py"):
                        return str(p.parent)
                candidate_root2 = Path(drive) / "Program Files (x86)" / top_dir
                if candidate_root2.exists():
                    for p in candidate_root2.rglob("MvCameraControl_class.py"):
                        return str(p.parent)
    except Exception:
        pass
    return None


# ── 경로 탐색 실행 ──────────────────────────────────────────────────────────

_mvs_available = False
_mvs_path_found: Optional[str] = None
_MVS_IMPORT_ERROR: Optional[str] = None
_all_searched: List[str] = []


def _add_mvs_runtime_to_path(mvs_import_path: str) -> None:
    """
    MvImport 경로에서 MVS 루트를 역산해 Runtime DLL 디렉토리를 PATH에 추가.

    MVS 설치 구조:
        [MVS_ROOT]\Development\Samples\Python\MvImport\  ← mvs_import_path
        [MVS_ROOT]\Runtime\Win64_x64\MvCameraControl.dll ← 여기가 필요

    MvCameraControl_class.py는 ctypes로 MvCameraControl.dll을 로드하므로
    Runtime 디렉토리가 PATH 또는 add_dll_directory에 없으면 ImportError 발생.
    """
    p = Path(mvs_import_path)
    # MvImport → Python → Samples → Development → [MVS_ROOT]
    candidates = [
        p.parents[3],   # MvImport/Python/Samples/Development → 4단계 위 = MVS_ROOT (표준)
        p.parents[4],   # 혹시 한 단계 더 위
    ]
    runtime_subdirs = [
        "Runtime/Win64_x64",
        "Runtime/Win32_i86",
        "Runtime",
        "bin",
        "lib",
    ]
    added = []
    for root in candidates:
        for sub in runtime_subdirs:
            dll_dir = root / sub
            if dll_dir.exists():
                s = str(dll_dir)
                if s not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = s + os.pathsep + os.environ.get("PATH", "")
                try:
                    os.add_dll_directory(s)
                except (AttributeError, OSError):
                    pass
                added.append(s)

    # MvImport 디렉토리 자체도 추가 (일부 버전은 여기에 DLL 포함)
    s = str(p)
    if s not in os.environ.get("PATH", ""):
        os.environ["PATH"] = s + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(s)
    except (AttributeError, OSError):
        pass


def _preload_mvs_dll(mvs_import_path: str) -> Optional[str]:
    """
    MvCameraControl.dll을 ctypes로 직접 로드합니다 (전체 경로 사용).

    MvCameraControl_class.py는 모듈 로드 시 ctypes.CDLL('MvCameraControl.dll')을
    이름만으로 호출합니다. PyInstaller 동결 EXE에서는 PATH 수정만으로는 부족할 수
    있으므로, 미리 전체 경로로 DLL을 로드해 프로세스 DLL 캐시에 등록합니다.

    Returns: 로드 성공한 DLL 경로 또는 None
    """
    p = Path(mvs_import_path)
    # MVS_ROOT = MvImport의 4단계 위 (Development/Samples/Python/MvImport)
    dll_names = ["MvCameraControl.dll", "MvCameraControl_d.dll"]
    search_roots = []
    if len(p.parents) > 3:
        search_roots.append(p.parents[3])
    if len(p.parents) > 4:
        search_roots.append(p.parents[4])
    search_roots.append(p)  # MvImport 자체에 DLL이 있는 경우도 처리

    runtime_subdirs = [
        "Runtime/Win64_x64",
        "Runtime/Win32_i86",
        "Runtime/Win64",
        "Runtime",
        "bin",
        "lib",
        "",  # root 자체
    ]
    for root in search_roots:
        for sub in runtime_subdirs:
            check_dir = root / sub if sub else root
            for dll_name in dll_names:
                dll_path = check_dir / dll_name
                if dll_path.exists():
                    try:
                        ctypes.WinDLL(str(dll_path))
                        return str(dll_path)
                    except OSError:
                        pass
    return None


def _try_load_mvs(path: str) -> bool:
    global _mvs_available, _mvs_path_found, _MVS_IMPORT_ERROR

    # 1. Runtime DLL 디렉토리를 PATH / add_dll_directory에 추가
    _add_mvs_runtime_to_path(path)

    # 2. MvCameraControl.dll을 전체 경로로 미리 로드
    #    (PyInstaller 동결 EXE에서 ctypes.CDLL('MvCameraControl.dll')이 실패하는 문제 방지)
    preloaded = _preload_mvs_dll(path)

    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        from MvCameraControl_class import (  # type: ignore  # noqa: F401
            MvCamera, MV_CC_DEVICE_INFO_LIST, MV_CC_DEVICE_INFO,
            MV_FRAME_OUT, MV_GIGE_DEVICE, MV_ACCESS_Exclusive,
        )
        _mvs_available = True
        _mvs_path_found = path
        return True
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        preload_info = f"\n[DLL 사전 로드: {preloaded or '실패'}]"
        _MVS_IMPORT_ERROR = f"{e}{preload_info}\n\n[상세]\n{tb}"
        return False


def _init_mvs() -> None:
    global _MVS_IMPORT_ERROR, _all_searched

    # 1. 환경변수
    p = _find_via_env()
    if p:
        _all_searched.append(f"[ENV] {p}")
        if _try_load_mvs(p):
            return

    # 2. 설정 파일
    p = _find_via_config()
    if p:
        _all_searched.append(f"[CFG] {p}")
        if _try_load_mvs(p):
            return

    # 3. 레지스트리
    p = _find_via_registry()
    if p:
        _all_searched.append(f"[REG] {p}")
        if _try_load_mvs(p):
            return

    # 4. 표준 경로 후보
    for cp in _MVS_CANDIDATE_PATHS:
        _all_searched.append(cp)
        if Path(cp).exists():
            if _try_load_mvs(cp):
                return

    # 5. 드라이브 전체 탐색 (마지막 수단)
    p = _find_all_drives_glob()
    if p:
        _all_searched.append(f"[GLOB] {p}")
        if _try_load_mvs(p):
            return

    _MVS_IMPORT_ERROR = (
        "MVS SDK(MvCameraControl_class.py)를 찾을 수 없습니다.\n"
        "해결 방법:\n"
        "  1. HIKROBOT MVS 소프트웨어 설치 후 재시작\n"
        "     https://www.hikrobotics.com → 머신비전 → MVS\n"
        "  2. 또는 RheedSetup.exe → 'MVS SDK 경로' 항목에\n"
        "     MvCameraControl_class.py 위치를 직접 입력\n"
        "  3. 또는 환경변수 HIKROBOT_MVS_PATH 설정 후 재시작\n"
        f"\n탐색한 경로 ({len(_all_searched)}개):\n" +
        "\n".join(f"  - {s}" for s in _all_searched[:15])
    )


_init_mvs()


def set_mvs_path(path: str) -> bool:
    """
    MVS SDK 경로를 런타임에 지정합니다. (setup_wizard 또는 main.py에서 호출)
    Returns True if MVS module loaded successfully.
    """
    global _mvs_available, _mvs_path_found
    if _mvs_available:
        return True
    return _try_load_mvs(path)


def mvs_available() -> bool:
    return _mvs_available


def mvs_error_message() -> str:
    return _MVS_IMPORT_ERROR or ""


def list_devices() -> list[str]:
    """검색된 GigE 카메라 목록 반환 (UI 표시용)."""
    if not _mvs_available:
        return [f"[MVS 없음] {_MVS_IMPORT_ERROR}"]
    from MvCameraControl_class import MvCamera, MV_CC_DEVICE_INFO_LIST, MV_CC_DEVICE_INFO, MV_GIGE_DEVICE  # type: ignore
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

        from MvCameraControl_class import (  # type: ignore
            MvCamera, MV_CC_DEVICE_INFO_LIST, MV_CC_DEVICE_INFO,
            MV_GIGE_DEVICE, MV_ACCESS_Exclusive,
        )
        self._MvCamera = MvCamera
        self._MV_CC_DEVICE_INFO_LIST = MV_CC_DEVICE_INFO_LIST
        self._MV_CC_DEVICE_INFO = MV_CC_DEVICE_INFO
        self._MV_GIGE_DEVICE = MV_GIGE_DEVICE
        self._MV_ACCESS_Exclusive = MV_ACCESS_Exclusive

        self._cam: Optional[MvCamera] = None
        self._device_index = device_index
        self._exposure_us = exposure_us
        self._gain_db = gain_db
        self._MONO8 = 0x01080001
        self._open()

    def _open(self) -> None:
        from MvCameraControl_class import MV_FRAME_OUT  # type: ignore
        self._MV_FRAME_OUT = MV_FRAME_OUT

        device_list = self._MV_CC_DEVICE_INFO_LIST()
        ret = self._MvCamera.MV_CC_EnumDevices(self._MV_GIGE_DEVICE, device_list)
        self._check(ret, "MV_CC_EnumDevices")

        n = device_list.nDeviceNum
        if n == 0:
            raise RuntimeError("GigE 카메라를 찾을 수 없습니다. 케이블/드라이버 확인.")
        if self._device_index >= n:
            raise RuntimeError(f"device_index={self._device_index} 초과 (발견={n})")

        cam = self._MvCamera()
        st = cast(device_list.pDeviceInfo[self._device_index],
                  POINTER(self._MV_CC_DEVICE_INFO)).contents
        self._check(cam.MV_CC_CreateHandle(st), "MV_CC_CreateHandle")
        self._check(cam.MV_CC_OpenDevice(self._MV_ACCESS_Exclusive, 0), "MV_CC_OpenDevice")

        cam.MV_CC_SetEnumValue("TriggerMode", 0)
        cam.MV_CC_SetFloatValue("ExposureTime", self._exposure_us)
        cam.MV_CC_SetFloatValue("Gain", self._gain_db)
        cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", False)
        self._check(cam.MV_CC_StartGrabbing(), "MV_CC_StartGrabbing")
        self._cam = cam

    @staticmethod
    def _check(ret: int, name: str) -> None:
        if ret != 0:
            raise RuntimeError(f"{name} 실패: 0x{ret:08X}")

    def read(self) -> Optional[np.ndarray]:
        if self._cam is None:
            return None
        st_frame = self._MV_FRAME_OUT()
        ret = self._cam.MV_CC_GetImageBuffer(st_frame, 1000)
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

            if pixel_type == self._MONO8:
                gray = data[: w * h].reshape(h, w)
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            elif n_bytes >= w * h * 3:
                frame = data[: w * h * 3].reshape(h, w, 3)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                bayer = data[: w * h].reshape(h, w)
                frame = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2BGR)

            return frame.copy()
        finally:
            self._cam.MV_CC_FreeImageBuffer(st_frame)

    def set_exposure(self, us: float) -> None:
        if self._cam:
            self._cam.MV_CC_SetFloatValue("ExposureTime", us)

    def set_gain(self, db: float) -> None:
        if self._cam:
            self._cam.MV_CC_SetFloatValue("Gain", db)

    def release(self) -> None:
        if self._cam:
            self._cam.MV_CC_StopGrabbing()
            self._cam.MV_CC_CloseDevice()
            self._cam.MV_CC_DestroyHandle()
            self._cam = None

    def __del__(self) -> None:
        self.release()
