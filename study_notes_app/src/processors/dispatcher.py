"""
File → processor routing.
Also handles multiple files: merges them into one ProcessedContent
so the AI layer receives a single unified analysis request.
"""
from __future__ import annotations

from pathlib import Path

from src.config import SUPPORTED_EXTENSIONS
from src.processors.base import ProcessedContent, ProcessorError
from src.processors.pdf_processor import PdfProcessor
from src.processors.docx_processor import DocxProcessor
from src.processors.pptx_processor import PptxProcessor
from src.processors.video_processor import VideoProcessor
from src.processors.ocr_processor import OcrProcessor


_PROCESSORS = [
    PdfProcessor(),
    DocxProcessor(),
    PptxProcessor(),
    VideoProcessor(),
    OcrProcessor(),
]


def dispatch(path: Path) -> ProcessedContent:
    """Route a single file to the appropriate processor."""
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ProcessorError(
            f"지원하지 않는 파일 형식입니다: '{ext}'\n"
            f"지원 형식: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    for proc in _PROCESSORS:
        if proc.can_handle(path):
            return proc.process(path)

    raise ProcessorError(f"'{path.name}' 처리기를 찾을 수 없습니다.")


def dispatch_multiple(
    paths: list[Path],
    progress_callback=None,  # callable(current: int, total: int, name: str)
) -> tuple[list[ProcessedContent], list[tuple[Path, str]]]:
    """
    Process multiple files.
    Returns (successes, failures) where failures = [(path, error_message)].
    """
    successes: list[ProcessedContent] = []
    failures: list[tuple[Path, str]] = []
    total = len(paths)

    for i, path in enumerate(paths):
        if progress_callback:
            progress_callback(i, total, path.name)
        try:
            content = dispatch(path)
            successes.append(content)
        except ProcessorError as e:
            failures.append((path, str(e)))
        except Exception as e:
            failures.append((path, f"예기치 않은 오류: {e}"))

    if progress_callback:
        progress_callback(total, total, "완료")

    return successes, failures


def merge_contents(contents: list[ProcessedContent]) -> ProcessedContent:
    """
    Merge multiple ProcessedContent objects into one.
    Text and images are concatenated with clear separators.
    """
    merged_text_parts = []
    merged_images = []
    merged_sources = []

    for content in contents:
        fname = content.source_path.name if content.source_path else "unknown"
        merged_sources.append(fname)

        if content.has_text():
            merged_text_parts.append(
                f"{'='*60}\n파일: {fname}\n{'='*60}\n{content.text}"
            )

        merged_images.extend(content.images)

    return ProcessedContent(
        text="\n\n".join(merged_text_parts),
        images=merged_images[:20],  # cap total images to avoid token overrun
        metadata={"sources": merged_sources, "file_count": len(contents)},
        file_type="merged",
    )
