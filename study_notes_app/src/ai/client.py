"""
Claude API client with Vision support.
Uses the Anthropic SDK with prompt caching for long documents.
"""
from __future__ import annotations

import json
import re
from typing import Iterator

import anthropic

from src.config import CLAUDE_MAX_TOKENS, CLAUDE_MODEL
from src.ai.prompts import SYSTEM_PROMPT, SUBJECT_INFERENCE_PROMPT, build_note_user_message
from src.processors.base import ProcessedContent, ImageBlock


class ClaudeClient:
    def __init__(self, api_key: str):
        self._client = anthropic.Anthropic(api_key=api_key)

    # ── Note generation (streaming) ───────────────────────────────────────────

    def generate_note_stream(
        self,
        content: ProcessedContent,
    ) -> Iterator[str]:
        """
        Stream-generate a markdown note from ProcessedContent.
        Yields text chunks as they arrive.
        """
        messages = self._build_messages(content)

        with self._client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},  # cache the long system prompt
                }
            ],
            messages=messages,
        ) as stream:
            for chunk in stream.text_stream:
                yield chunk

    def generate_note(self, content: ProcessedContent) -> str:
        """Non-streaming version — returns complete markdown."""
        return "".join(self.generate_note_stream(content))

    # ── Subject / chapter inference ───────────────────────────────────────────

    def infer_subject_and_chapter(self, note_markdown: str) -> dict:
        """
        Extract subject name and chapter title from the generated note.
        Returns {"subject": "...", "chapter": "..."}.
        Falls back to defaults if parsing fails.
        """
        preview = note_markdown[:1500]
        prompt = SUBJECT_INFERENCE_PROMPT.format(note_preview=preview)

        try:
            resp = self._client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            # Extract JSON even if wrapped in markdown fences
            m = re.search(r'\{.*?\}', raw, re.DOTALL)
            if m:
                return json.loads(m.group())
        except Exception:
            pass

        # Fallback: try to read the first # heading
        for line in note_markdown.splitlines():
            line = line.strip()
            if line.startswith("# "):
                return {"subject": "미분류", "chapter": line[2:].strip()}

        return {"subject": "미분류", "chapter": "단원 1"}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_messages(self, content: ProcessedContent) -> list[dict]:
        """Build the messages list with text + vision blocks."""
        file_info = self._build_file_info(content)
        text_block = {
            "type": "text",
            "text": build_note_user_message(content.text, file_info),
        }

        # Add image blocks (Claude Vision)
        image_blocks = []
        for img in content.images[:20]:  # API limit safety
            try:
                image_blocks.append(self._image_block(img))
            except Exception:
                continue

        if image_blocks:
            # Interleave: text prompt, then each image with its caption
            parts: list[dict] = [text_block]
            for i, (img_block, img_data) in enumerate(
                zip(image_blocks, content.images[:20])
            ):
                parts.append(img_block)
                if img_data.caption:
                    parts.append({"type": "text", "text": f"[이미지 설명: {img_data.caption}]"})
            return [{"role": "user", "content": parts}]

        return [{"role": "user", "content": [text_block]}]

    @staticmethod
    def _image_block(img: ImageBlock) -> dict:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img.media_type,
                "data": img.to_base64(),
            },
        }

    @staticmethod
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


# ── Settings helpers ──────────────────────────────────────────────────────────

def load_api_key() -> str | None:
    """Load the Claude API key from settings file."""
    import json
    from src.config import SETTINGS_FILE

    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            return data.get("api_key") or None
        except Exception:
            pass
    return None


def save_api_key(key: str):
    """Persist the API key to the settings file."""
    import json
    from src.config import SETTINGS_FILE

    data = {}
    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    data["api_key"] = key
    SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def validate_api_key(key: str) -> bool:
    """Quick smoke-test to verify the key works."""
    try:
        client = anthropic.Anthropic(api_key=key)
        client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=8,
            messages=[{"role": "user", "content": "hi"}],
        )
        return True
    except Exception:
        return False
