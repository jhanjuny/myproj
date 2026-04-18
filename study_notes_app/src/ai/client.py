"""
Claude AI client — supports three backends:

  1. "api"    → Anthropic SDK (API key required)
  2. "cli"    → Claude Code CLI subprocess (team / pro 계정, API key 불필요)
  3. "ollama" → Local Ollama server (http://localhost:11434, 완전 무료·오프라인)
               `ollama` 설치 후 원하는 모델을 pull 해두면 사용 가능
               https://ollama.com
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


# ── Backend: Ollama local server ──────────────────────────────────────────────

_OLLAMA_DEFAULT_URL   = "http://localhost:11434"
_OLLAMA_DEFAULT_MODEL = "gemma3:4b"


class OllamaClient:
    """
    Calls a local Ollama server via its HTTP API.
    Streaming is token-by-token using the /api/chat endpoint (NDJSON).
    Images are passed as base64 blobs (multimodal models only; ignored otherwise).
    """

    def __init__(
        self,
        model: str = _OLLAMA_DEFAULT_MODEL,
        base_url: str = _OLLAMA_DEFAULT_URL,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")

    def generate_note_stream(self, content: ProcessedContent) -> Iterator[str]:
        import requests

        messages = self._build_ollama_messages(content)
        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json={"model": self._model, "messages": messages, "stream": True},
                stream=True,
                timeout=300,
            )
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Ollama 서버 연결 실패: {e}\n"
                "Ollama가 실행 중인지 확인하세요 (ollama serve)."
            ) from e

        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            token = data.get("message", {}).get("content", "")
            if token:
                yield token
            if data.get("done"):
                break

    def infer_subject_and_chapter(self, note_markdown: str) -> dict:
        import requests

        preview = note_markdown[:1500]
        prompt  = SUBJECT_INFERENCE_PROMPT.format(note_preview=preview)
        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=60,
            )
            resp.raise_for_status()
            raw = resp.json()["message"]["content"].strip()
            m = re.search(r'\{.*?\}', raw, re.DOTALL)
            if m:
                return json.loads(m.group())
        except Exception:
            pass
        return _fallback_subject(note_markdown)

    # ── private ───────────────────────────────────────────────────────────────

    def _build_ollama_messages(self, content: ProcessedContent) -> list[dict]:
        file_info = _build_file_info(content)
        user_text = (
            f"{SYSTEM_PROMPT}\n\n---\n\n"
            f"{build_note_user_message(content.text, file_info)}"
        )
        msg: dict = {"role": "user", "content": user_text}

        # Attach images for multimodal models (llava, gemma3, etc.)
        if content.images:
            msg["images"] = [
                img.to_base64()
                for img in content.images[:10]   # conservative limit
                if img.data
            ]
        return [msg]


# ── Unified facade ────────────────────────────────────────────────────────────

class ClaudeClient:
    """Public interface — wraps ApiKeyClient, CliClient, or OllamaClient."""

    def __init__(
        self,
        api_key: str | None = None,
        use_cli: bool = False,
        use_ollama: bool = False,
        ollama_model: str = _OLLAMA_DEFAULT_MODEL,
        ollama_base_url: str = _OLLAMA_DEFAULT_URL,
    ):
        if use_ollama:
            self._backend = OllamaClient(model=ollama_model, base_url=ollama_base_url)
        elif use_cli:
            self._backend = CliClient()
        elif api_key:
            self._backend = ApiKeyClient(api_key)
        else:
            raise ValueError("API 키, CLI 모드, 또는 Ollama 모드 중 하나를 선택하세요.")

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


def load_use_ollama() -> bool:
    return load_settings().get("use_ollama", False)


def save_use_ollama(value: bool):
    save_settings({"use_ollama": value})


def save_api_key(key: str):
    save_settings({"api_key": key})


def save_use_cli(value: bool):
    save_settings({"use_cli": value})


def load_ollama_settings() -> tuple[str, str]:
    """Return (model, base_url) from persisted settings."""
    s = load_settings()
    return (
        s.get("ollama_model",    _OLLAMA_DEFAULT_MODEL),
        s.get("ollama_base_url", _OLLAMA_DEFAULT_URL),
    )


def save_ollama_settings(model: str, base_url: str):
    save_settings({"ollama_model": model, "ollama_base_url": base_url})


def list_ollama_models(base_url: str = _OLLAMA_DEFAULT_URL) -> list[str]:
    """Return names of all locally available Ollama models."""
    try:
        import requests
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        if resp.status_code == 200:
            return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        pass
    return []


def validate_ollama(
    base_url: str = _OLLAMA_DEFAULT_URL,
    model: str = _OLLAMA_DEFAULT_MODEL,
) -> bool:
    """Return True if Ollama is reachable and the given model is available."""
    try:
        import requests
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        available = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        # Accept exact match or base-name match (e.g. "gemma3" matches "gemma3:4b")
        target = model.split(":")[0]
        return target in available or model in [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return False


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
