"""Application-wide QSS stylesheet."""

DARK_QSS = """
QMainWindow, QDialog, QWidget {
    background-color: #1a1b26;
    color: #c0caf5;
    font-family: 'Segoe UI', 'Malgun Gothic', sans-serif;
    font-size: 13px;
}

QSplitter::handle {
    background-color: #2a2b3d;
    width: 2px;
}

/* ── Sidebar / Tree ─────────────────────────────────────── */
QTreeWidget {
    background-color: #16161e;
    border: none;
    color: #c0caf5;
    font-size: 13px;
    padding: 4px;
}
QTreeWidget::item {
    height: 28px;
    padding-left: 4px;
    border-radius: 4px;
}
QTreeWidget::item:selected {
    background-color: #3d59a1;
    color: #ffffff;
}
QTreeWidget::item:hover:!selected {
    background-color: #2a2b3d;
}
QTreeWidget::branch {
    background: transparent;
}

/* ── Toolbar ─────────────────────────────────────────────── */
QToolBar {
    background-color: #16161e;
    border-bottom: 1px solid #2a2b3d;
    spacing: 6px;
    padding: 4px 8px;
}
QToolButton {
    background-color: transparent;
    border: none;
    border-radius: 6px;
    padding: 6px 10px;
    color: #c0caf5;
    font-size: 13px;
}
QToolButton:hover {
    background-color: #2a2b3d;
}
QToolButton:pressed {
    background-color: #3d59a1;
}

/* ── Buttons ─────────────────────────────────────────────── */
QPushButton {
    background-color: #3d59a1;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 7px 16px;
    font-size: 13px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #4a6bbd;
}
QPushButton:pressed {
    background-color: #2d4580;
}
QPushButton:disabled {
    background-color: #2a2b3d;
    color: #565f89;
}
QPushButton#danger {
    background-color: #f7768e;
}
QPushButton#danger:hover {
    background-color: #ff8fa3;
}
QPushButton#secondary {
    background-color: #2a2b3d;
    color: #c0caf5;
}
QPushButton#secondary:hover {
    background-color: #33344d;
}

/* ── Input fields ────────────────────────────────────────── */
QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #16161e;
    border: 1px solid #2a2b3d;
    border-radius: 6px;
    padding: 6px 10px;
    color: #c0caf5;
    selection-background-color: #3d59a1;
}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border-color: #7aa2f7;
}

/* ── ComboBox ────────────────────────────────────────────── */
QComboBox {
    background-color: #16161e;
    border: 1px solid #2a2b3d;
    border-radius: 6px;
    padding: 5px 10px;
    color: #c0caf5;
    min-width: 120px;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox QAbstractItemView {
    background-color: #1a1b26;
    border: 1px solid #2a2b3d;
    selection-background-color: #3d59a1;
}

/* ── Progress bar ────────────────────────────────────────── */
QProgressBar {
    background-color: #2a2b3d;
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #7aa2f7, stop:1 #bb9af7);
    border-radius: 4px;
}

/* ── ScrollBar ───────────────────────────────────────────── */
QScrollBar:vertical {
    background: #16161e;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #3b4261;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #565f89;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

/* ── Menu ─────────────────────────────────────────────────── */
QMenu {
    background-color: #1a1b26;
    border: 1px solid #2a2b3d;
    border-radius: 8px;
    padding: 4px;
}
QMenu::item {
    padding: 6px 20px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #3d59a1;
}
QMenu::separator {
    height: 1px;
    background: #2a2b3d;
    margin: 4px 8px;
}

/* ── Status bar ───────────────────────────────────────────── */
QStatusBar {
    background-color: #16161e;
    color: #565f89;
    border-top: 1px solid #2a2b3d;
    font-size: 12px;
}

/* ── Labels ───────────────────────────────────────────────── */
QLabel#title {
    font-size: 18px;
    font-weight: 700;
    color: #7aa2f7;
}
QLabel#subtitle {
    font-size: 13px;
    color: #565f89;
}

/* ── Tab bar ──────────────────────────────────────────────── */
QTabWidget::pane {
    border: none;
    background: #1a1b26;
}
QTabBar::tab {
    background: #16161e;
    color: #565f89;
    padding: 8px 18px;
    border: none;
    border-bottom: 2px solid transparent;
}
QTabBar::tab:selected {
    color: #7aa2f7;
    border-bottom-color: #7aa2f7;
}
QTabBar::tab:hover:!selected {
    color: #c0caf5;
}

/* ── Upload drop zone ─────────────────────────────────────── */
QFrame#dropzone {
    border: 2px dashed #3d59a1;
    border-radius: 12px;
    background-color: #1e1f2e;
}
QFrame#dropzone:hover {
    border-color: #7aa2f7;
    background-color: #212234;
}

/* ── Highlight colour swatches ───────────────────────────── */
QPushButton#hl_yellow { background-color: #f6c90e; color: #000; }
QPushButton#hl_green  { background-color: #73daca; color: #000; }
QPushButton#hl_blue   { background-color: #7dcfff; color: #000; }
QPushButton#hl_pink   { background-color: #f7768e; color: #fff; }
QPushButton#hl_clear  { background-color: #2a2b3d; color: #c0caf5; }
"""
