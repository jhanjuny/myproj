"""
Main application window.
Layout:
  ┌────────────────────────────────────────────────┐
  │  Toolbar (highlight tools, edit, export, …)    │
  ├──────────┬─────────────────────┬───────────────┤
  │ Sidebar  │   NoteViewer        │  UploadPanel  │
  │ (tree)   │   (QWebEngineView)  │  / Editor     │
  └──────────┴─────────────────────┴───────────────┘
  │  Status bar                                    │
  └────────────────────────────────────────────────┘
"""
from __future__ import annotations

import json
from pathlib import Path

from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QToolBar, QAction, QLabel, QStatusBar,
    QInputDialog, QLineEdit, QMessageBox, QShortcut,
    QStackedWidget,
)

from src.gui.styles import DARK_QSS
from src.gui.widgets.note_viewer import NoteViewer
from src.gui.widgets.note_editor import NoteEditorPanel
from src.gui.widgets.sidebar import SidebarWidget
from src.gui.widgets.upload_panel import UploadPanel
from src.gui.widgets.processing_dialog import ProcessingDialog
from src.ai.client import ClaudeClient, load_api_key, save_api_key, validate_api_key
from src.ai.note_generator import NoteGeneratorThread
from src.storage import database as db


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("StudyNotes AI")
        self.resize(1400, 860)
        self.setMinimumSize(900, 600)

        self.setStyleSheet(DARK_QSS)

        self._current_note_id: int | None = None
        self._api_key: str | None = load_api_key()
        self._generator_thread: NoteGeneratorThread | None = None
        self._processing_dialog: ProcessingDialog | None = None

        db.init_db()
        self._build_ui()
        self._bind_shortcuts()

        if not self._api_key:
            self._prompt_api_key()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Central widget + outer layout ─────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Toolbar ───────────────────────────────────────────────────────────
        self._build_toolbar()

        # ── Main splitter: [sidebar | viewer | right panel] ───────────────────
        self._splitter = QSplitter(Qt.Horizontal)

        # Left: sidebar
        self._sidebar = SidebarWidget()
        self._sidebar.note_selected.connect(self._on_note_selected)
        self._sidebar.chapter_selected.connect(self._on_chapter_selected)
        self._splitter.addWidget(self._sidebar)

        # Centre: note viewer
        self._viewer = NoteViewer()
        self._viewer.highlights_changed.connect(self._on_highlights_changed)
        self._splitter.addWidget(self._viewer)

        # Right: stacked (upload panel / editor)
        self._right_stack = QStackedWidget()
        self._right_stack.setMinimumWidth(300)
        self._right_stack.setMaximumWidth(420)

        self._upload_panel = UploadPanel()
        self._upload_panel.generate_requested.connect(self._on_generate_requested)
        self._right_stack.addWidget(self._upload_panel)  # index 0

        self._editor_panel = NoteEditorPanel()
        self._editor_panel.save_requested.connect(self._on_note_saved)
        self._editor_panel.discard_requested.connect(self._show_upload_panel)
        self._right_stack.addWidget(self._editor_panel)  # index 1

        self._right_stack.setCurrentIndex(0)
        self._splitter.addWidget(self._right_stack)

        # Splitter proportions: 18% | 57% | 25%
        self._splitter.setStretchFactor(0, 18)
        self._splitter.setStretchFactor(1, 57)
        self._splitter.setStretchFactor(2, 25)

        outer.addWidget(self._splitter)

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_bar = self.statusBar()
        self._status_label = QLabel("준비")
        self._status_bar.addWidget(self._status_label)
        self._api_status = QLabel()
        self._update_api_status_indicator()
        self._status_bar.addPermanentWidget(self._api_status)

    def _build_toolbar(self):
        tb = QToolBar("Main Toolbar")
        tb.setMovable(False)
        self.addToolBar(tb)

        # Highlight actions
        tb.addWidget(QLabel("  하이라이트: "))
        for label, color, obj in [
            ("🟡", "yellow", "hl_yellow"),
            ("🟢", "green",  "hl_green"),
            ("🔵", "blue",   "hl_blue"),
            ("🔴", "pink",   "hl_pink"),
        ]:
            act = QAction(label, self)
            act.setToolTip(f"{color} 하이라이트")
            act.triggered.connect(lambda checked, c=color: self._viewer.apply_highlight(c))
            tb.addAction(act)

        clear_act = QAction("✖", self)
        clear_act.setToolTip("하이라이트 모두 지우기")
        clear_act.triggered.connect(self._viewer.clear_highlights)
        tb.addAction(clear_act)

        tb.addSeparator()

        # Edit note
        self._edit_act = QAction("✏️  편집", self)
        self._edit_act.setToolTip("현재 노트를 편집합니다 (Ctrl+E)")
        self._edit_act.triggered.connect(self._edit_current_note)
        self._edit_act.setEnabled(False)
        tb.addAction(self._edit_act)

        # Export
        export_act = QAction("💾  내보내기", self)
        export_act.setToolTip("마크다운으로 내보내기")
        export_act.triggered.connect(self._export_current_note)
        tb.addAction(export_act)

        tb.addSeparator()

        # Settings
        settings_act = QAction("⚙️  설정", self)
        settings_act.triggered.connect(self._open_settings)
        tb.addAction(settings_act)

        tb.addSeparator()

        # Search bar
        tb.addWidget(QLabel("  검색: "))
        self._search_toolbar = QLineEdit()
        self._search_toolbar.setPlaceholderText("노트 전체 검색...")
        self._search_toolbar.setFixedWidth(200)
        self._search_toolbar.returnPressed.connect(
            lambda: self._sidebar._on_search(self._search_toolbar.text())
        )
        tb.addWidget(self._search_toolbar)

    def _bind_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+E"), self, self._edit_current_note)
        QShortcut(QKeySequence("Ctrl+S"), self, self._save_from_shortcut)
        QShortcut(QKeySequence("Ctrl+N"), self, self._show_upload_panel)

    # ── Slot: note selected from sidebar ─────────────────────────────────────

    def _on_note_selected(self, note_id: int):
        note = db.get_note(note_id)
        if note is None:
            return
        self._current_note_id = note_id
        self._viewer.render_note(note["markdown"], note_id)
        if note["highlights"]:
            self._viewer.restore_highlights(json.dumps(note["highlights"]))
        self._edit_act.setEnabled(True)
        self._status_label.setText(f"📄  {note['title']}")

    def _on_chapter_selected(self, chapter_id: int):
        """When user clicks a chapter, prime the upload panel to use it."""
        self._upload_panel._chapter_combo.setCurrentIndex(0)

    # ── Slot: generate requested ──────────────────────────────────────────────

    def _on_generate_requested(
        self, file_paths: list, subject_id, chapter_id
    ):
        if not self._api_key:
            self._prompt_api_key()
            if not self._api_key:
                return

        if not file_paths:
            return

        # Show processing dialog
        self._processing_dialog = ProcessingDialog(self)

        # Create and start the generator thread
        self._generator_thread = NoteGeneratorThread(
            api_key=self._api_key,
            file_paths=[Path(p) for p in file_paths],
            subject_id=subject_id,
            chapter_id=chapter_id,
            parent=self,
        )
        self._generator_thread.progress.connect(self._processing_dialog.set_status)
        self._generator_thread.chunk_received.connect(self._processing_dialog.append_chunk)
        self._generator_thread.note_complete.connect(self._on_note_complete)
        self._generator_thread.file_error.connect(self._processing_dialog.add_file_error)
        self._generator_thread.error.connect(self._on_generator_error)
        self._generator_thread.start()

        self._processing_dialog.exec_()  # blocks until user closes

    def _on_note_complete(self, note_id: int):
        if self._processing_dialog:
            self._processing_dialog.mark_done()
        self._sidebar.refresh()
        self._upload_panel.refresh_subjects()
        self._on_note_selected(note_id)

    def _on_generator_error(self, message: str):
        if self._processing_dialog:
            self._processing_dialog.mark_error(message)

    # ── Slot: highlights changed ──────────────────────────────────────────────

    def _on_highlights_changed(self, json_str: str):
        if self._current_note_id is None:
            return
        try:
            highlights = json.loads(json_str)
            db.update_note_highlights(self._current_note_id, highlights)
        except Exception:
            pass

    # ── Edit mode ─────────────────────────────────────────────────────────────

    def _edit_current_note(self):
        if self._current_note_id is None:
            return
        note = db.get_note(self._current_note_id)
        if note is None:
            return
        self._editor_panel.set_markdown(note["markdown"])
        self._right_stack.setCurrentIndex(1)

    def _on_note_saved(self, new_markdown: str):
        if self._current_note_id is None:
            return
        db.update_note_markdown(self._current_note_id, new_markdown)
        self._viewer.render_note(new_markdown, self._current_note_id)
        self._show_upload_panel()
        self._status_label.setText("✅  노트 저장 완료")

    def _save_from_shortcut(self):
        if self._right_stack.currentIndex() == 1:
            self._on_note_saved(self._editor_panel.get_markdown())

    def _show_upload_panel(self):
        self._right_stack.setCurrentIndex(0)

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_current_note(self):
        if self._current_note_id is None:
            QMessageBox.information(self, "내보내기", "먼저 노트를 선택해주세요.")
            return
        note = db.get_note(self._current_note_id)
        if note is None:
            return

        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "마크다운으로 내보내기", f"{note['title']}.md",
            "Markdown (*.md);;All files (*)"
        )
        if path:
            Path(path).write_text(note["markdown"], encoding="utf-8")
            self._status_label.setText(f"📥  내보내기 완료: {path}")

    # ── Settings ──────────────────────────────────────────────────────────────

    def _open_settings(self):
        from PyQt5.QtWidgets import QDialog, QFormLayout, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("설정")
        dlg.setFixedWidth(440)
        dlg.setStyleSheet(DARK_QSS)

        form = QFormLayout(dlg)

        key_edit = QLineEdit(self._api_key or "")
        key_edit.setEchoMode(QLineEdit.Password)
        key_edit.setPlaceholderText("sk-ant-…")
        form.addRow("Claude API 키:", key_edit)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)

        if dlg.exec_():
            new_key = key_edit.text().strip()
            if new_key:
                self._status_label.setText("🔑  API 키 확인 중...")
                if validate_api_key(new_key):
                    self._api_key = new_key
                    save_api_key(new_key)
                    self._update_api_status_indicator()
                    self._status_label.setText("✅  API 키가 저장되었습니다")
                else:
                    QMessageBox.warning(self, "오류", "API 키가 유효하지 않습니다.")
                    self._status_label.setText("❌  API 키 오류")

    def _prompt_api_key(self):
        key, ok = QInputDialog.getText(
            self,
            "Claude API 키 설정",
            "Anthropic API 키를 입력하세요 (https://console.anthropic.com):",
            QLineEdit.Password,
        )
        if ok and key.strip():
            if validate_api_key(key.strip()):
                self._api_key = key.strip()
                save_api_key(self._api_key)
                self._update_api_status_indicator()
            else:
                QMessageBox.warning(self, "오류", "유효하지 않은 API 키입니다.")

    def _update_api_status_indicator(self):
        if self._api_key:
            self._api_status.setText("🟢  API 연결됨")
            self._api_status.setStyleSheet("color: #9ece6a;")
        else:
            self._api_status.setText("🔴  API 키 없음")
            self._api_status.setStyleSheet("color: #f7768e;")
