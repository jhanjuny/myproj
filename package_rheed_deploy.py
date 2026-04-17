
"""
배포 폴더(rheed_deploy/)를 생성합니다.
build_rheed.bat 실행 후 이 스크립트를 실행하세요.
"""
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).parent
DEPLOY = ROOT / "rheed_deploy"

if DEPLOY.exists():
    shutil.rmtree(DEPLOY)
DEPLOY.mkdir()

files = {
    ROOT / "dist/RheedMonitor.exe": DEPLOY / "RheedMonitor.exe",
    ROOT / "dist/RheedSetup.exe":   DEPLOY / "RheedSetup.exe",
    ROOT / "RHEED_Monitor_Manual.pdf": DEPLOY / "RHEED_Monitor_Manual.pdf",
}

missing = []
for src, dst in files.items():
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  copied: {dst.name}")
    else:
        print(f"  MISSING: {src}")
        missing.append(src)

# README
(DEPLOY / "README.txt").write_text(
    "RHEED Monitor 배포 패키지\n"
    "=" * 40 + "\n\n"
    "사전 준비:\n"
    "  1. HIKROBOT MVS 설치 (hikrobotics.com)\n"
    "  2. GigE 카메라 연결 및 IP 할당 확인\n\n"
    "실행 순서:\n"
    "  1. RheedSetup.exe  -- 카메라/검출 설정 저장\n"
    "  2. RheedMonitor.exe -- 메인 모니터링 프로그램\n\n"
    "출력 파일:\n"
    "  이 폴더 안 outputs/rheed/ 에 저장됩니다.\n"
    "  세션 종료 시 ZIP으로 자동 압축됩니다.\n\n"
    "자세한 사용법은 RHEED_Monitor_Manual.pdf 참고.\n",
    encoding="utf-8"
)
print("  created: README.txt")

if missing:
    print(f"\n경고: {len(missing)}개 파일 없음. build_rheed.bat을 먼저 실행하세요.")
    sys.exit(1)
else:
    print(f"\n배포 폴더 생성 완료: {DEPLOY}")
    size = sum(f.stat().st_size for f in DEPLOY.rglob("*") if f.is_file())
    print(f"총 크기: {size/1024/1024:.1f} MB")
