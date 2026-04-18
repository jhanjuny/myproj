"""
High-level note generation orchestrator.
Ties together: dispatcher → Claude → database storage.
Designed to run in a QThread so the GUI stays responsive.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt5.QtCore import QThread, pyqtSignal

from src.processors.dispatcher import dispatch_multiple, merge_contents
from src.processors.base import ProcessorError
from src.ai.client import ClaudeClient
from src.storage import database as db


class NoteGeneratorThread(QThread):
    """
    QThread that:
      1. Processes uploaded files
      2. Streams the note from Claude
      3. Saves to the database
      4. Emits signals for UI updates
    """

    # ── Signals ───────────────────────────────────────────────────────────────
    progress = pyqtSignal(str)           # status text
    chunk_received = pyqtSignal(str)     # streaming markdown chunk
    note_complete = pyqtSignal(int)      # note_id when done
    file_error = pyqtSignal(str, str)    # (filename, error_message)
    error = pyqtSignal(str)              # fatal error

    def __init__(
        self,
        api_key: str,
        file_paths: list[Path],
        subject_id: int | None = None,
        chapter_id: int | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._api_key = api_key
        self._file_paths = file_paths
        self._subject_id = subject_id
        self._chapter_id = chapter_id
        self._note_buffer = ""

    def run(self):
        try:
            self._run_pipeline()
        except Exception as e:
            self.error.emit(f"예기치 않은 오류가 발생했습니다:\n{e}")

    def _run_pipeline(self):
        # ── Step 1: Extract content from files ────────────────────────────────
        self.progress.emit("📂 파일 분석 중...")

        def on_progress(i, total, name):
            self.progress.emit(f"📂 파일 처리 중 ({i+1}/{total}): {name}")

        successes, failures = dispatch_multiple(self._file_paths, on_progress)

        for path, err_msg in failures:
            self.file_error.emit(path.name, err_msg)

        if not successes:
            self.error.emit("처리 가능한 파일이 없습니다.")
            return

        # ── Step 2: Merge if multiple files ───────────────────────────────────
        self.progress.emit("🔗 파일 내용 병합 중...")
        if len(successes) > 1:
            merged = merge_contents(successes)
        else:
            merged = successes[0]

        # ── Step 3: Stream note from Claude ───────────────────────────────────
        self.progress.emit("🤖 AI 노트 생성 중 (스트리밍)...")

        client = ClaudeClient(self._api_key)
        self._note_buffer = ""

        try:
            for chunk in client.generate_note_stream(merged):
                self._note_buffer += chunk
                self.chunk_received.emit(chunk)
        except Exception as e:
            self.error.emit(f"AI 노트 생성 실패:\n{str(e)}")
            return

        if not self._note_buffer.strip():
            self.error.emit("AI가 빈 응답을 반환했습니다. 다시 시도해주세요.")
            return

        # ── Step 4: Infer subject / chapter from the note ─────────────────────
        self.progress.emit("📚 과목 및 단원 분류 중...")

        inferred = {"subject": "미분류", "chapter": "단원 1"}
        if self._subject_id is None or self._chapter_id is None:
            try:
                inferred = client.infer_subject_and_chapter(self._note_buffer)
            except Exception:
                pass

        subject_id = self._subject_id
        if subject_id is None:
            subject_id = db.create_subject(inferred["subject"])

        chapter_id = self._chapter_id
        if chapter_id is None:
            existing_chapters = db.get_chapters(subject_id)
            chapter_title = inferred["chapter"]
            matching = [c for c in existing_chapters if c["title"] == chapter_title]
            if matching:
                chapter_id = matching[0]["id"]
            else:
                chapter_id = db.create_chapter(
                    subject_id,
                    chapter_title,
                    order_idx=len(existing_chapters),
                )

        # ── Step 5: Save note to database ─────────────────────────────────────
        self.progress.emit("💾 노트 저장 중...")

        note_title = _extract_title(self._note_buffer)
        source_files = [str(p.name) for p in self._file_paths]
        note_id = db.create_note(
            chapter_id=chapter_id,
            title=note_title,
            markdown=self._note_buffer,
            source_files=source_files,
        )

        self.progress.emit("✅ 완료!")
        self.note_complete.emit(note_id)


def _extract_title(markdown: str) -> str:
    for line in markdown.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return "새 노트"
