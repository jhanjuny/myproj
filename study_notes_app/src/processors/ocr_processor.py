"""
OCR processor for handwritten notes and scanned images.
Uses EasyOCR.  If confidence is below OCR_CONFIDENCE_THRESHOLD for the
majority of text blocks, raises ProcessorError with a clear message
so the GUI can show "인식 불가" (unrecognizable).
"""
from __future__ import annotations

import io
from pathlib import Path

from src.config import MAX_IMAGE_BYTES, OCR_CONFIDENCE_THRESHOLD
from src.processors.base import BaseProcessor, ImageBlock, ProcessedContent, ProcessorError


class OcrProcessor(BaseProcessor):
    supported_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")

    _reader = None  # lazy singleton

    @classmethod
    def _get_reader(cls):
        if cls._reader is None:
            try:
                import easyocr
                # Korean + English; downloads model on first use (~500 MB)
                cls._reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
            except ImportError:
                raise ProcessorError(
                    "easyocr가 설치되지 않았습니다.\n"
                    "pip install easyocr 를 실행한 후 재시도하세요."
                )
        return cls._reader

    def process(self, path: Path) -> ProcessedContent:
        from PIL import Image
        import numpy as np

        # ── Load image ────────────────────────────────────────────────────────
        try:
            img = Image.open(str(path))
        except Exception as e:
            raise ProcessorError(f"이미지를 열 수 없습니다: {e}")

        # Convert to RGB numpy array for EasyOCR
        img_rgb = img.convert("RGB")
        img_array = np.array(img_rgb)

        # ── Run OCR ───────────────────────────────────────────────────────────
        try:
            reader = self._get_reader()
            results = reader.readtext(img_array, detail=1)
        except ProcessorError:
            raise
        except Exception as e:
            raise ProcessorError(
                f"OCR 처리 중 오류가 발생했습니다: {e}\n"
                "파일 형식이나 이미지 품질을 확인해주세요."
            )

        if not results:
            raise ProcessorError(
                f"'{path.name}' 파일에서 텍스트를 인식하지 못했습니다.\n"
                "이미지가 너무 흐리거나 필기가 인식 불가능한 상태입니다."
            )

        # ── Filter by confidence ──────────────────────────────────────────────
        confident = [r for r in results if r[2] >= OCR_CONFIDENCE_THRESHOLD]
        low_conf_ratio = 1 - len(confident) / len(results)

        if low_conf_ratio > 0.6:
            avg_conf = sum(r[2] for r in results) / len(results)
            raise ProcessorError(
                f"'{path.name}' 파일의 OCR 인식률이 너무 낮습니다 "
                f"(평균 신뢰도: {avg_conf:.0%}).\n"
                "다음을 확인해주세요:\n"
                "  • 이미지 해상도가 충분한가요?\n"
                "  • 필기가 명확하게 찍혔나요?\n"
                "  • 조명이 균일한가요?"
            )

        # ── Reconstruct text preserving reading order (top-to-bottom) ─────────
        # Sort by vertical position of bounding box centre
        def y_centre(r):
            bbox = r[0]
            return (bbox[0][1] + bbox[2][1]) / 2

        sorted_results = sorted(confident, key=y_centre)

        lines: list[str] = []
        prev_y = None
        line_gap_threshold = _estimate_line_gap(sorted_results)
        current_line: list[tuple[float, str]] = []

        for r in sorted_results:
            bbox, text, conf = r
            cy = y_centre(r)
            cx = (bbox[0][0] + bbox[2][0]) / 2

            if prev_y is not None and abs(cy - prev_y) > line_gap_threshold:
                # Flush current line sorted by x
                current_line.sort(key=lambda t: t[0])
                lines.append(" ".join(t[1] for t in current_line))
                current_line = []

            current_line.append((cx, text))
            prev_y = cy

        if current_line:
            current_line.sort(key=lambda t: t[0])
            lines.append(" ".join(t[1] for t in current_line))

        ocr_text = "\n".join(lines)

        # ── Package image for Claude Vision too ───────────────────────────────
        buf = io.BytesIO()
        img_rgb.save(buf, format="PNG")
        raw = buf.getvalue()
        if len(raw) > MAX_IMAGE_BYTES:
            # Downscale
            img_rgb.thumbnail((1600, 1200))
            buf = io.BytesIO()
            img_rgb.save(buf, format="JPEG", quality=80)
            raw = buf.getvalue()
            media = "image/jpeg"
        else:
            media = "image/png"

        image_block = ImageBlock(
            data=raw,
            media_type=media,
            caption=f"손필기 이미지: {path.name}",
            page_or_slide=1,
        )

        metadata = {
            "title": path.stem,
            "ocr_blocks": len(confident),
            "avg_confidence": sum(r[2] for r in confident) / len(confident),
        }

        return ProcessedContent(
            text=f"[OCR 인식 결과 — {path.name}]\n{ocr_text}",
            images=[image_block],
            metadata=metadata,
            source_path=path,
            file_type="image",
        )


def _estimate_line_gap(results: list) -> float:
    """Estimate typical character height to use as line-break threshold."""
    if not results:
        return 20.0
    heights = []
    for r in results:
        bbox = r[0]
        h = abs(bbox[2][1] - bbox[0][1])
        heights.append(h)
    avg_h = sum(heights) / len(heights)
    return avg_h * 0.8  # gap > 80% of avg char height → new line
