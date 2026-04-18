"""
Subject / Chapter / Note tree sidebar.
"""
from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QFont, QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QPushButton, QLabel, QLineEdit, QInputDialog, QMenu, QAction,
    QColorDialog, QMessageBox,
)

from src.storage import database as db


class SidebarWidget(QWidget):
    """Left-side panel: subject tree + search."""

    note_selected = pyqtSignal(int)         # note_id
    chapter_selected = pyqtSignal(int)      # chapter_id (for new-note target)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(220)
        self.setMaximumWidth(340)
        self._build_ui()
        self.refresh()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header
        hdr = QHBoxLayout()
        title = QLabel("📚 노트")
        title.setObjectName("title")
        hdr.addWidget(title)
        hdr.addStretch()

        add_subject_btn = QPushButton("+ 과목")
        add_subject_btn.setObjectName("secondary")
        add_subject_btn.setFixedWidth(64)
        add_subject_btn.clicked.connect(self._add_subject)
        hdr.addWidget(add_subject_btn)
        layout.addLayout(hdr)

        # Search
        self._search = QLineEdit()
        self._search.setPlaceholderText("🔍 노트 검색...")
        self._search.textChanged.connect(self._on_search)
        layout.addWidget(self._search)

        # Tree
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._context_menu)
        self._tree.itemClicked.connect(self._on_item_clicked)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self._tree)

    # ── Data loading ──────────────────────────────────────────────────────────

    def refresh(self):
        self._tree.clear()
        for subject in db.get_subjects():
            s_item = self._make_subject_item(subject)
            self._tree.addTopLevelItem(s_item)
            for chapter in db.get_chapters(subject["id"]):
                c_item = self._make_chapter_item(chapter)
                s_item.addChild(c_item)
                for note in db.get_notes(chapter["id"]):
                    n_item = self._make_note_item(note)
                    c_item.addChild(n_item)
        self._tree.expandAll()

    def _make_subject_item(self, subject: dict) -> QTreeWidgetItem:
        item = QTreeWidgetItem([f"📖  {subject['name']}"])
        item.setData(0, Qt.UserRole, ("subject", subject["id"]))
        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        item.setFont(0, font)
        return item

    def _make_chapter_item(self, chapter: dict) -> QTreeWidgetItem:
        item = QTreeWidgetItem([f"  📁  {chapter['title']}"])
        item.setData(0, Qt.UserRole, ("chapter", chapter["id"]))
        font = QFont()
        font.setPointSize(10)
        item.setFont(0, font)
        return item

    def _make_note_item(self, note: dict) -> QTreeWidgetItem:
        item = QTreeWidgetItem([f"      📄  {note['title']}"])
        item.setData(0, Qt.UserRole, ("note", note["id"]))
        return item

    # ── Signals ───────────────────────────────────────────────────────────────

    def _on_item_clicked(self, item, col):
        data = item.data(0, Qt.UserRole)
        if data is None:
            return
        kind, id_ = data
        if kind == "note":
            self.note_selected.emit(id_)
        elif kind == "chapter":
            self.chapter_selected.emit(id_)

    def _on_item_double_clicked(self, item, col):
        data = item.data(0, Qt.UserRole)
        if data is None:
            return
        kind, id_ = data
        if kind == "subject":
            self._rename_subject(id_, item)
        elif kind == "chapter":
            self._rename_chapter(id_, item)
        elif kind == "note":
            self._rename_note(id_, item)

    # ── Context menu ──────────────────────────────────────────────────────────

    def _context_menu(self, pos):
        item = self._tree.itemAt(pos)
        if item is None:
            # Right-click on empty area → add subject
            menu = QMenu(self)
            menu.addAction("+ 새 과목 추가", self._add_subject)
            menu.exec_(self._tree.mapToGlobal(pos))
            return

        data = item.data(0, Qt.UserRole)
        if data is None:
            return
        kind, id_ = data

        menu = QMenu(self)
        if kind == "subject":
            menu.addAction("✏️  이름 변경", lambda: self._rename_subject(id_, item))
            menu.addAction("📁  단원 추가", lambda: self._add_chapter(id_, item))
            menu.addSeparator()
            menu.addAction("🗑️  과목 삭제", lambda: self._delete_subject(id_))
        elif kind == "chapter":
            menu.addAction("✏️  이름 변경", lambda: self._rename_chapter(id_, item))
            menu.addSeparator()
            menu.addAction("🗑️  단원 삭제", lambda: self._delete_chapter(id_))
        elif kind == "note":
            menu.addAction("✏️  이름 변경", lambda: self._rename_note(id_, item))
            menu.addSeparator()
            menu.addAction("🗑️  노트 삭제", lambda: self._delete_note(id_))

        menu.exec_(self._tree.mapToGlobal(pos))

    # ── CRUD operations ───────────────────────────────────────────────────────

    def _add_subject(self):
        name, ok = QInputDialog.getText(self, "새 과목", "과목 이름:")
        if ok and name.strip():
            db.create_subject(name.strip())
            self.refresh()

    def _rename_subject(self, subject_id, item):
        old = item.text(0).replace("📖  ", "").strip()
        name, ok = QInputDialog.getText(self, "과목 이름 변경", "새 이름:", text=old)
        if ok and name.strip():
            db.rename_subject(subject_id, name.strip())
            self.refresh()

    def _delete_subject(self, subject_id):
        reply = QMessageBox.question(
            self, "과목 삭제", "이 과목과 모든 노트를 삭제하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            db.delete_subject(subject_id)
            self.refresh()

    def _add_chapter(self, subject_id, parent_item):
        title, ok = QInputDialog.getText(self, "새 단원", "단원 이름:")
        if ok and title.strip():
            existing = db.get_chapters(subject_id)
            db.create_chapter(subject_id, title.strip(), len(existing))
            self.refresh()

    def _rename_chapter(self, chapter_id, item):
        old = item.text(0).strip().replace("📁  ", "").strip()
        title, ok = QInputDialog.getText(self, "단원 이름 변경", "새 이름:", text=old)
        if ok and title.strip():
            db.rename_chapter(chapter_id, title.strip())
            self.refresh()

    def _delete_chapter(self, chapter_id):
        reply = QMessageBox.question(
            self, "단원 삭제", "이 단원과 모든 노트를 삭제하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            db.delete_chapter(chapter_id)
            self.refresh()

    def _rename_note(self, note_id, item):
        old = item.text(0).strip().replace("📄  ", "").strip()
        title, ok = QInputDialog.getText(self, "노트 이름 변경", "새 이름:", text=old)
        if ok and title.strip():
            db.rename_note(note_id, title.strip())
            self.refresh()

    def _delete_note(self, note_id):
        reply = QMessageBox.question(
            self, "노트 삭제", "이 노트를 삭제하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            db.delete_note(note_id)
            self.refresh()

    # ── Search ────────────────────────────────────────────────────────────────

    def _on_search(self, query: str):
        if not query.strip():
            self.refresh()
            return
        results = db.search_notes(query.strip())
        self._tree.clear()
        for r in results:
            item = QTreeWidgetItem([
                f"📄  {r['note_title']}  —  {r['subject_name']} / {r['chapter_title']}"
                if "note_title" in r else
                f"📄  {r['title']}  —  {r['subject_name']} / {r['chapter_title']}"
            ])
            item.setData(0, Qt.UserRole, ("note", r["id"]))
            self._tree.addTopLevelItem(item)
