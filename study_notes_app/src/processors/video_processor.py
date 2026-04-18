"""
Video processor.
Transcribes audio with OpenAI Whisper (runs locally, no API key needed).
Also extracts key-frame thumbnails at regular intervals for Claude Vision.
"""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

from src.config import MAX_IMAGE_BYTES, MAX_IMAGES_PER_FILE, WHISPER_MODEL
from src.processors.base import BaseProcessor, ImageBlock, ProcessedContent, ProcessorError


class VideoProcessor(BaseProcessor):
    supported_extensions = (".mp4", ".mov", ".avi", ".mkv", ".m4v")

    _whisper_model = None

    @classmethod
    def _get_model(cls):
        if cls._whisper_model is None:
            try:
                import whisper
                cls._whisper_model = whisper.load_model(WHISPER_MODEL)
            except ImportError:
                raise ProcessorError(
                    "openai-whisper가 설치되지 않았습니다.\n"
                    "pip install openai-whisper ffmpeg-python 을 실행하세요.\n"
                    "ffmpeg도 시스템에 설치되어 있어야 합니다."
                )
        return cls._whisper_model

    def process(self, path: Path) -> ProcessedContent:
        # ── Extract audio to temp WAV ─────────────────────────────────────────
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "audio.wav")

            try:
                import ffmpeg
                (
                    ffmpeg
                    .input(str(path))
                    .output(wav_path, ac=1, ar="16000", vn=None)
                    .overwrite_output()
                    .run(quiet=True)
                )
            except Exception as e:
                raise ProcessorError(
                    f"동영상에서 오디오를 추출할 수 없습니다: {e}\n"
                    "ffmpeg가 설치되어 있는지 확인하세요."
                )

            # ── Transcribe ────────────────────────────────────────────────────
            try:
                model = self._get_model()
                result = model.transcribe(
                    wav_path,
                    language=None,         # auto-detect (Korean/English)
                    task="transcribe",
                    fp16=False,
                    verbose=False,
                )
            except ProcessorError:
                raise
            except Exception as e:
                raise ProcessorError(f"음성 인식 중 오류: {e}")

        transcript = result.get("text", "").strip()
        segments = result.get("segments", [])
        detected_language = result.get("language", "unknown")

        if not transcript:
            raise ProcessorError(
                f"'{path.name}'에서 음성을 인식하지 못했습니다.\n"
                "무음이거나 인식 불가능한 언어일 수 있습니다."
            )

        # ── Build timestamped text ────────────────────────────────────────────
        text_parts = [f"[동영상 전사 — {path.name}  언어: {detected_language}]\n"]
        for seg in segments:
            start = _fmt_time(seg["start"])
            end = _fmt_time(seg["end"])
            text_parts.append(f"[{start} → {end}] {seg['text'].strip()}")

        full_text = "\n".join(text_parts)

        # ── Extract keyframe thumbnails ───────────────────────────────────────
        images = _extract_keyframes(path, max_frames=min(10, MAX_IMAGES_PER_FILE))

        duration = segments[-1]["end"] if segments else 0
        metadata = {
            "title": path.stem,
            "duration_sec": duration,
            "language": detected_language,
            "segment_count": len(segments),
        }

        return ProcessedContent(
            text=full_text,
            images=images,
            metadata=metadata,
            source_path=path,
            file_type="video",
        )


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _extract_keyframes(path: Path, max_frames: int = 10) -> list[ImageBlock]:
    """Extract evenly spaced keyframes using ffmpeg."""
    images: list[ImageBlock] = []

    try:
        import ffmpeg
        from PIL import Image

        # Get video duration
        probe = ffmpeg.probe(str(path))
        video_streams = [s for s in probe["streams"] if s["codec_type"] == "video"]
        if not video_streams:
            return images
        duration = float(probe["format"]["duration"])

        interval = max(duration / max_frames, 5.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            frame_pattern = os.path.join(tmpdir, "frame_%04d.jpg")
            (
                ffmpeg
                .input(str(path))
                .filter("fps", fps=f"1/{interval:.1f}")
                .output(frame_pattern, vframes=max_frames, **{"q:v": 5})
                .overwrite_output()
                .run(quiet=True)
            )
            for i, fname in enumerate(sorted(os.listdir(tmpdir))[:max_frames]):
                fpath = os.path.join(tmpdir, fname)
                raw = open(fpath, "rb").read()
                if len(raw) > MAX_IMAGE_BYTES:
                    img = Image.open(fpath)
                    img.thumbnail((960, 540))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=75)
                    raw = buf.getvalue()
                timestamp = _fmt_time(i * interval)
                images.append(ImageBlock(
                    data=raw,
                    media_type="image/jpeg",
                    caption=f"Frame at {timestamp}",
                    page_or_slide=i + 1,
                ))
    except Exception:
        pass  # keyframe extraction is best-effort

    return images
