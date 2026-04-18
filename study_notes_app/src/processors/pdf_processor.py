"""
PDF processor using PyMuPDF (fitz).
Extracts:
  • All text (preserving layout where possible)
  • All embedded images (resized if > MAX_IMAGE_BYTES)
  • Rendered page snapshots for pages that are purely image/scan based
"""
from __future__ import annotations

import io
from pathlib import Path

from src.config import MAX_IMAGE_BYTES, MAX_IMAGES_PER_FILE
from src.processors.base import BaseProcessor, ImageBlock, ProcessedContent, ProcessorError


class PdfProcessor(BaseProcessor):
    supported_extensions = (".pdf",)

    def process(self, path: Path) -> ProcessedContent:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ProcessorError("PyMuPDF is not installed (pip install PyMuPDF)")

        try:
            doc = fitz.open(str(path))
        except Exception as e:
            raise ProcessorError(f"Cannot open PDF: {e}")

        text_parts: list[str] = []
        images: list[ImageBlock] = []
        image_count = 0

        for page_num, page in enumerate(doc, start=1):
            # ── Text ──────────────────────────────────────────────────────────
            page_text = page.get_text("text").strip()
            if page_text:
                text_parts.append(f"[Page {page_num}]\n{page_text}")

            # ── Embedded images ───────────────────────────────────────────────
            if image_count < MAX_IMAGES_PER_FILE:
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    try:
                        base_img = doc.extract_image(xref)
                        raw = base_img["image"]
                        media = f"image/{base_img['ext']}" if base_img["ext"] != "jpg" else "image/jpeg"

                        if len(raw) > MAX_IMAGE_BYTES:
                            raw = _downscale_image(raw, MAX_IMAGE_BYTES)
                            media = "image/jpeg"

                        images.append(ImageBlock(
                            data=raw,
                            media_type=media,
                            caption=f"Image from page {page_num}",
                            page_or_slide=page_num,
                        ))
                        image_count += 1
                        if image_count >= MAX_IMAGES_PER_FILE:
                            break
                    except Exception:
                        continue

            # ── Fallback: render page as image if no text extracted ───────────
            if not page_text and image_count < MAX_IMAGES_PER_FILE:
                try:
                    mat = fitz.Matrix(1.5, 1.5)  # 1.5x zoom → decent quality
                    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                    raw = pix.tobytes("jpeg", jpg_quality=85)
                    if len(raw) <= MAX_IMAGE_BYTES:
                        images.append(ImageBlock(
                            data=raw,
                            media_type="image/jpeg",
                            caption=f"Scanned page {page_num}",
                            page_or_slide=page_num,
                        ))
                        image_count += 1
                except Exception:
                    pass

        doc.close()

        metadata = {
            "title": path.stem,
            "page_count": len(doc),
        }

        content = ProcessedContent(
            text="\n\n".join(text_parts),
            images=images,
            metadata=metadata,
            source_path=path,
            file_type="pdf",
        )

        if content.is_empty():
            raise ProcessorError(f"Could not extract any content from {path.name}")

        return content


def _downscale_image(raw: bytes, max_bytes: int) -> bytes:
    """Reduce JPEG quality until under max_bytes."""
    from PIL import Image

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    for quality in (80, 60, 40, 25):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return data
    # Last resort: resize
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=50)
    return buf.getvalue()
