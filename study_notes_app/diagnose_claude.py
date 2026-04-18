"""
Claude CLI 진단 스크립트
앱 오류가 계속 날 때 이 스크립트를 실행해서 결과를 알려주세요.
실행: python diagnose_claude.py
"""
import os, sys, shutil, subprocess, re

SEP = "=" * 60

def section(title):
    print(f"\n{SEP}\n{title}\n{SEP}")

# ── 1. 환경 변수 ───────────────────────────────────────────────────────────────
section("1. 환경 변수")
for k in ("USERNAME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "HOME"):
    print(f"  {k:15s} = {os.environ.get(k, '(없음)')}")
print(f"  expanduser('~')  = {os.path.expanduser('~')}")

# ── 2. shutil.which ───────────────────────────────────────────────────────────
section("2. shutil.which('claude')")
which_result = shutil.which("claude")
print(f"  결과: {which_result}")

# ── 3. APPDATA 기반 네이티브 EXE 탐색 ─────────────────────────────────────────
section("3. APPDATA 기반 claude.exe 탐색")
appdata = os.environ.get("APPDATA", "")
cc_dir = os.path.join(appdata, "Claude", "claude-code")
print(f"  탐색 경로: {cc_dir}")
print(f"  존재 여부: {os.path.isdir(cc_dir)}")
if os.path.isdir(cc_dir):
    for ver in sorted(os.listdir(cc_dir), reverse=True):
        for ext in ("claude.exe", "claude.EXE"):
            p = os.path.join(cc_dir, ver, ext)
            exists = os.path.isfile(p)
            size = os.path.getsize(p) // 1024 // 1024 if exists else 0
            print(f"  [{ver}] {ext}: exists={exists}" + (f" ({size}MB)" if exists else ""))

# ── 4. 알려진 CMD 파일 ─────────────────────────────────────────────────────────
section("4. 알려진 CMD 파일")
known_locations = [
    r"C:\Users\Hanjun Sim\claude.CMD",
    os.path.join(os.environ.get("APPDATA",""), "npm", "claude.cmd"),
    os.path.join(os.environ.get("APPDATA",""), "npm", "claude.CMD"),
]
for loc in known_locations:
    exists = os.path.isfile(loc)
    print(f"  {loc}: {exists}")
    if exists:
        with open(loc, encoding="utf-8", errors="replace") as f:
            content = f.read()
        print(f"    내용:\n      " + content.replace("\n", "\n      "))
        m = re.search(r'"([^"]+\.exe)"', content, re.IGNORECASE)
        if m:
            exe = m.group(1)
            print(f"    → 추출된 EXE: {exe}")
            print(f"    → EXE 존재: {os.path.isfile(exe)}")

# ── 5. --version 테스트 ────────────────────────────────────────────────────────
section("5. claude --version 테스트")
candidates = []
if which_result:
    candidates.append(which_result)
# Direct EXE candidates
_appdata = os.environ.get("APPDATA","")
_cc = os.path.join(_appdata, "Claude", "claude-code")
if os.path.isdir(_cc):
    for _v in sorted(os.listdir(_cc), reverse=True):
        for _e in ("claude.exe", "claude.EXE"):
            _p = os.path.join(_cc, _v, _e)
            if os.path.isfile(_p):
                candidates.append(_p)

for exe in dict.fromkeys(candidates):  # deduplicate, preserve order
    print(f"\n  테스트: {exe}")
    try:
        r = subprocess.run(
            [exe, "--version"],
            capture_output=True, timeout=15,
            cwd=os.path.expanduser("~"),
        )
        print(f"    RC={r.returncode}")
        print(f"    stdout={r.stdout[:300]}")
        print(f"    stderr={r.stderr[:300]}")
    except FileNotFoundError as e:
        print(f"    ❌ FileNotFoundError: {e}")
    except subprocess.TimeoutExpired:
        print(f"    ⏱ Timeout (15s)")
    except Exception as e:
        print(f"    ❌ {type(e).__name__}: {e}")

# ── 6. 짧은 프롬프트로 실제 호출 테스트 ──────────────────────────────────────
section("6. 짧은 프롬프트로 실제 호출 (최대 90초 대기)")
test_exe = None
# Prefer native EXE
_appdata = os.environ.get("APPDATA","")
_cc = os.path.join(_appdata, "Claude", "claude-code")
if os.path.isdir(_cc):
    for _v in sorted(os.listdir(_cc), reverse=True):
        for _e in ("claude.exe", "claude.EXE"):
            _p = os.path.join(_cc, _v, _e)
            if os.path.isfile(_p):
                test_exe = _p
                break
        if test_exe:
            break
if not test_exe and which_result:
    test_exe = which_result

if test_exe:
    print(f"  사용 EXE: {test_exe}")
    print(f"  cwd: {os.path.expanduser('~')}")
    print(f"  명령: [exe, '-p', '--dangerously-skip-permissions', 'say: HELLO']")
    print(f"  실행 중 (최대 90초)...")
    try:
        r = subprocess.run(
            [test_exe, "-p", "--dangerously-skip-permissions", "say: HELLO"],
            capture_output=True, timeout=90,
            cwd=os.path.expanduser("~"),
        )
        print(f"  RC={r.returncode}")
        print(f"  stdout={r.stdout[:500]}")
        print(f"  stderr={r.stderr[:500]}")
    except subprocess.TimeoutExpired:
        print(f"  ⏱ Timeout (90초) - 응답 없음")
    except Exception as e:
        print(f"  ❌ {type(e).__name__}: {e}")
else:
    print("  ❌ 테스트할 EXE를 찾지 못했습니다.")

print(f"\n{SEP}\n진단 완료\n{SEP}")
