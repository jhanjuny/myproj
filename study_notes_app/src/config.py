import os
import sys
from pathlib import Path

# ── Base paths ────────────────────────────────────────────────────────────────
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data"
NOTES_DIR = DATA_DIR / "notes"
UPLOADS_DIR = DATA_DIR / "uploads"
RESOURCES_DIR = BASE_DIR / "resources"

for _d in (DATA_DIR, NOTES_DIR, UPLOADS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH = DATA_DIR / "study_notes.db"

# ── Claude model ──────────────────────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-6"
CLAUDE_MAX_TOKENS = 8192

# ── Supported file types ──────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".pptx": "pptx",
    ".ppt": "pptx",
    ".mp4": "video",
    ".mov": "video",
    ".avi": "video",
    ".mkv": "video",
    ".m4v": "video",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".bmp": "image",
    ".tiff": "image",
    ".tif": "image",
    ".webp": "image",
}

# ── Settings file ─────────────────────────────────────────────────────────────
SETTINGS_FILE = DATA_DIR / "settings.json"

# ── Whisper model size ────────────────────────────────────────────────────────
# tiny / base / small / medium / large  (tradeoff: speed vs accuracy)
WHISPER_MODEL = "base"

# ── Image size limit for Claude Vision (bytes) ────────────────────────────────
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB per image

# ── Max images extracted per file ────────────────────────────────────────────
MAX_IMAGES_PER_FILE = 20

# ── OCR confidence threshold ──────────────────────────────────────────────────
OCR_CONFIDENCE_THRESHOLD = 0.3  # below this → "unrecognizable" error

# ── App metadata ──────────────────────────────────────────────────────────────
APP_NAME = "StudyNotes AI"
APP_VERSION = "1.0.0"
