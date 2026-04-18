"""
Claude AI client — supports two backends:

  1. "api"  → Anthropic SDK (API key required)
  2. "cli"  → Claude Code CLI subprocess (team / pro 계정, API key 불필요)
              `claude` 명령이 PATH에 있어야 함 (Claude Code 설치 전제)
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import os
from typing import Iterator

from src.config import CLAUDE_MAX_TOKENS, CLAUDE_MODEL
from src.ai.prompts import SYSTEM_PROMPT, SUBJECT_INFERENCE_PROMPT, build_note_user_message
from src.processors.base import ProcessedContent, ImageBlock


# ── Backend: Anthropic SDK (API key) ─────────────────────────────────────────

class ApiKeyClient:
    """Calls Anthropic API directly with an API key."""

    def __init__(self, api_key: str):
        import anthropic as _anthropic
        self._client = _anthropic.Anthropic(api_key=api_key)

    def generate_note_stream(self, content: ProcessedContent) -> Iterator[str]:
        import anthropic as _anthropic
        messages = _build_messages(content)
        with self._client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=messages,
        ) as stream:
            for chunk in stream.text_stream:
                yield chunk

    def infer_subject_and_chapter(self, note_markdown: str) -> dict:
        preview = note_markdown[:1500]
        prompt = SUBJECT_INFERENCE_PROMPT.format(note_preview=preview)
        try:
            resp = self._client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            m = re.search(r'\{.*?\}', raw, re.DOTALL)
            if m:
                return json.loads(m.group())
        except Exception:
            pass
        return _fallback_subject(note_markdown)


# ── Backend: Claude Code CLI (team / pro 계정) ────────────────────────────────

class CliClient:
    """
    Calls `claude -p <prompt>` via subprocess.
    Authentication is handled by Claude Code (team/pro 계정).
    Images are saved to temp files and referenced as file paths.
    Streaming is simulated: the full response is yielded in one chunk
    (the CLI doesn't expose per-token streaming externally).
    """

    def __init__(self):
        self._claude = _find_claude_cli()
        if not self._claude:
            raise RuntimeError(
                "claude CLI를 찾을 수 없습니다.\n"
                "Claude Code가 설치되어 있는지 확인하세요.\n"
                "설치 후 터미널에서 'claude' 명령이 실행되는지 확인하세요."
            )

    def generate_note_stream(self, content: ProcessedContent) -> Iterator[str]:
        prompt = _sanitize(_build_cli_prompt(content))

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        try:
            output = _run_claude(self._claude, prompt, env, timeout=300)
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "Claude CLI 응답 시간 초과 (5분).\n"
                "파일 내용이 너무 길거나 네트워크 문제일 수 있습니다."
            )

        if not output:
            raise RuntimeError("Claude CLI에서 빈 응답을 받았습니다.")

        yield output

    def infer_subject_and_chapter(self, note_markdown: str) -> dict:
        preview = note_markdown[:1500]
        prompt = _sanitize(SUBJECT_INFERENCE_PROMPT.format(note_preview=preview))
        try:
            raw = _run_claude(self._claude, prompt,
                              os.environ.copy(), timeout=60)
            m = re.search(r'\{.*?\}', raw, re.DOTALL)
            if m:
                return json.loads(m.group())
        except Exception:
            pass
        return _fallback_subject(note_markdown)


# ── Unified facade ────────────────────────────────────────────────────────────

class ClaudeClient:
    """Public interface — wraps either ApiKeyClient or CliClient."""

    def __init__(self, api_key: str | None = None, use_cli: bool = False):
        if use_cli:
            self._backend = CliClient()
        elif api_key:
            self._backend = ApiKeyClient(api_key)
        else:
            raise ValueError("API 키 또는 CLI 모드 중 하나를 선택하세요.")

    def generate_note_stream(self, content: ProcessedContent) -> Iterator[str]:
        return self._backend.generate_note_stream(content)

    def generate_note(self, content: ProcessedContent) -> str:
        return "".join(self.generate_note_stream(content))

    def infer_subject_and_chapter(self, note_markdown: str) -> dict:
        return self._backend.infer_subject_and_chapter(note_markdown)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_messages(content: ProcessedContent) -> list[dict]:
    file_info = _build_file_info(content)
    text_block = {
        "type": "text",
        "text": build_note_user_message(content.text, file_info),
    }
    image_blocks = []
    for img in content.images[:20]:
        try:
            image_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.media_type,
                    "data": img.to_base64(),
                },
            })
        except Exception:
            continue

    if image_blocks:
        parts: list[dict] = [text_block]
        for img_block, img_data in zip(image_blocks, content.images[:20]):
            parts.append(img_block)
            if img_data.caption:
                parts.append({"type": "text", "text": f"[이미지: {img_data.caption}]"})
        return [{"role": "user", "content": parts}]

    return [{"role": "user", "content": [text_block]}]


def _build_cli_prompt(content: ProcessedContent) -> str:
    file_info = _build_file_info(content)
    user_msg = build_note_user_message(content.text, file_info)
    return f"{SYSTEM_PROMPT}\n\n---\n\n{user_msg}"


def _build_file_info(content: ProcessedContent) -> str:
    parts = []
    if content.source_path:
        parts.append(f"파일명: {content.source_path.name}")
    parts.append(f"형식: {content.file_type}")
    for k, v in content.metadata.items():
        parts.append(f"{k}: {v}")
    if content.images:
        parts.append(f"포함 이미지: {len(content.images)}개")
    return "\n".join(parts)


def _run_claude(claude_exe: str, prompt: str, env: dict, timeout: int = 300) -> str:
    """
    Run claude CLI and return text output.

    Root problem on Windows:
      • npm-installed claude is a .CMD wrapper → goes through cmd.exe → 8 192-char limit
      • `claude -p -` (stdin sentinel) is NOT reliably supported by Claude Code CLI

    Strategy order (file-based approaches are the only ones that bypass arg limits):
      1. Native .EXE only — direct stdin PIPE to `claude --print`
      2. cmd.exe file redirect:  claude --print < prompt.txt
      3. cmd.exe type-pipe:      type prompt.txt | claude --print
      4. PowerShell pipe:        Get-Content … | claude --print
      5. Last resort: truncate prompt to cmd-safe length and pass as -p arg
    """
    prompt_bytes = prompt.encode("utf-8")
    is_cmd_script = claude_exe.lower().endswith((".cmd", ".bat"))

    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_path = os.path.join(tmpdir, "prompt.txt")
        with open(prompt_path, "wb") as f:
            f.write(prompt_bytes)

        # ── Strategy 1: stdin PIPE (native .EXE only) ────────────────────────
        if not is_cmd_script:
            for print_flag in (["--print"], ["-p", "--"]):
                try:
                    proc = subprocess.Popen(
                        [claude_exe] + print_flag,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        env=env,
                    )
                    stdout_b, _ = proc.communicate(input=prompt_bytes, timeout=timeout)
                    out = _decode_output(stdout_b)
                    if out:
                        return out
                except (OSError, subprocess.SubprocessError, ValueError):
                    continue

        # ── Strategy 2: cmd.exe file redirect  < prompt.txt ──────────────────
        q = claude_exe.replace('"', '""')
        p = prompt_path.replace('"', '""')

        for print_arg in ("--print", "-p -"):
            cmd = f'chcp 65001 > nul 2>&1 && "{q}" {print_arg} < "{p}"'
            try:
                r = subprocess.run(
                    ["cmd.exe", "/c", cmd],
                    capture_output=True, env=env, timeout=timeout,
                )
                out = _decode_output(r.stdout)
                if out:
                    return out
            except (OSError, subprocess.SubprocessError):
                continue

        # ── Strategy 3: cmd.exe type-pipe  type prompt.txt | claude --print ──
        for print_arg in ("--print", "-p -"):
            cmd = f'chcp 65001 > nul 2>&1 && type "{p}" | "{q}" {print_arg}'
            try:
                r = subprocess.run(
                    ["cmd.exe", "/c", cmd],
                    capture_output=True, env=env, timeout=timeout,
                )
                out = _decode_output(r.stdout)
                if out:
                    return out
            except (OSError, subprocess.SubprocessError):
                continue

        # ── Strategy 4: PowerShell pipe ───────────────────────────────────────
        for print_arg in ("--print", "-p -"):
            ps = (
                f'$env:PYTHONIOENCODING="utf-8"; '
                f'Get-Content -Raw -Encoding UTF8 "{p}" | & "{q}" {print_arg}'
            )
            try:
                r = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
                    capture_output=True, env=env, timeout=timeout,
                )
                out = _decode_output(r.stdout)
                if out:
                    return out
            except (OSError, subprocess.SubprocessError):
                continue

    # ── Strategy 5: last resort — truncate to cmd-safe size (~5 000 chars) ───
    # cmd.exe limit ≈ 8 192; subtract exe path + flags overhead
    truncated = prompt[:5000]
    try:
        if is_cmd_script:
            safe = truncated.replace('"', ' ')   # drop quotes to avoid quoting hell
            r = subprocess.run(
                ["cmd.exe", "/c", f'chcp 65001 > nul 2>&1 && "{claude_exe}" -p "{safe}"'],
                capture_output=True, env=env, timeout=timeout,
            )
        else:
            r = subprocess.run(
                [claude_exe, "-p", truncated],
                capture_output=True, env=env, timeout=timeout,
            )
        out = _decode_output(r.stdout)
        if out:
            return out
        err = _decode_output(r.stderr) or "(오류 정보 없음)"
        raise RuntimeError(f"Claude CLI 오류:\n{err}")
    except subprocess.TimeoutExpired:
        raise
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Claude CLI 실행 실패: {exc}") from exc


def _sanitize(text: str) -> str:
    """Remove characters that break subprocess pipes (null bytes, surrogates)."""
    return text.replace("\x00", "").encode("utf-8", errors="replace").decode("utf-8")


def _decode_output(raw: bytes) -> str:
    """Decode subprocess bytes output — tries UTF-8 first, then CP949 (Korean Windows)."""
    if not raw:
        return ""
    for enc in ("utf-8", "cp949", "euc-kr", "latin-1"):
        try:
            return raw.decode(enc).strip()
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace").strip()


def _fallback_subject(note_markdown: str) -> dict:
    for line in note_markdown.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return {"subject": "미분류", "chapter": line[2:].strip()}
    return {"subject": "미분류", "chapter": "단원 1"}


def _find_claude_cli() -> str | None:
    """
    Find the claude executable.
    Prefer the native .EXE (no cmd.exe wrapper → no 8192-char arg limit).
    """
    appdata = os.environ.get("APPDATA", "")
    localappdata = os.environ.get("LOCALAPPDATA", "")
    home = os.path.expanduser("~")

    # ── 1. Native .EXE builds (Claude Desktop / Claude Code installer) ────────
    native_candidates = []
    # Versioned claude-code directory (Claude Desktop)
    cc_dir = os.path.join(appdata, "Claude", "claude-code")
    if os.path.isdir(cc_dir):
        for ver in sorted(os.listdir(cc_dir), reverse=True):
            native_candidates.append(os.path.join(cc_dir, ver, "claude.exe"))
            native_candidates.append(os.path.join(cc_dir, ver, "claude.EXE"))
    native_candidates += [
        os.path.join(localappdata, "Programs", "claude", "claude.exe"),
        os.path.join(appdata, "Claude", "claude.exe"),
    ]
    for c in native_candidates:
        if os.path.isfile(c):
            return c

    # ── 2. PATH lookup (may return .CMD wrapper — acceptable fallback) ─────────
    found = shutil.which("claude")
    if found:
        return found

    # ── 3. Common npm / manual install locations ──────────────────────────────
    for c in [
        os.path.join(appdata, "npm", "claude.cmd"),
        os.path.join(home, "AppData", "Roaming", "npm", "claude.cmd"),
    ]:
        if os.path.isfile(c):
            return c

    return None


# ── Settings helpers ──────────────────────────────────────────────────────────

def load_settings() -> dict:
    from src.config import SETTINGS_FILE
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_settings(data: dict):
    from src.config import SETTINGS_FILE
    existing = load_settings()
    existing.update(data)
    SETTINGS_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def load_api_key() -> str | None:
    return load_settings().get("api_key") or None


def load_use_cli() -> bool:
    return load_settings().get("use_cli", False)


def save_api_key(key: str):
    save_settings({"api_key": key})


def save_use_cli(value: bool):
    save_settings({"use_cli": value})


def validate_api_key(key: str) -> bool:
    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=key)
        client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=8,
            messages=[{"role": "user", "content": "hi"}],
        )
        return True
    except Exception:
        return False


def validate_cli() -> bool:
    """Return True if the claude CLI is reachable and responds to --version."""
    path = _find_claude_cli()
    if not path:
        return False
    is_cmd = path.lower().endswith((".cmd", ".bat"))
    try:
        if is_cmd:
            # .CMD wrappers must go through cmd.exe
            result = subprocess.run(
                ["cmd.exe", "/c", f'"{path}" --version'],
                capture_output=True, timeout=15,
            )
        else:
            result = subprocess.run(
                [path, "--version"],
                capture_output=True, timeout=15,
            )
        return result.returncode == 0
    except Exception:
        return False
