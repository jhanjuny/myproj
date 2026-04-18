"""
Base processor interface.
Every processor returns ProcessedContent so the AI layer receives
a uniform structure regardless of the source file type.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ImageBlock:
    """One image extracted from the source material."""
    data: bytes                  # raw bytes (JPEG/PNG)
    media_type: str = "image/png"
    caption: str = ""            # slide title, page note, etc.
    page_or_slide: int = 0

    def to_base64(self) -> str:
        return base64.standard_b64encode(self.data).decode()


@dataclass
class ProcessedContent:
    """Unified extraction result handed to the AI pipeline."""
    text: str = ""
    images: list[ImageBlock] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)   # title, author, slide_count…
    source_path: Optional[Path] = None
    file_type: str = ""

    def has_text(self) -> bool:
        return bool(self.text.strip())

    def has_images(self) -> bool:
        return bool(self.images)

    def is_empty(self) -> bool:
        return not self.has_text() and not self.has_images()


class ProcessorError(Exception):
    """Raised when a file cannot be processed (e.g. OCR failure)."""


class BaseProcessor:
    """Abstract base – subclasses must implement `process`."""

    supported_extensions: tuple[str, ...] = ()

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_extensions

    def process(self, path: Path) -> ProcessedContent:
        raise NotImplementedError
