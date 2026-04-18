"""
NoteViewer: QWebEngineView that renders markdown + LaTeX.
Uses MathJax 3 (loaded from CDN; bundled copy as fallback).
Highlights are applied via JavaScript inside the page.
"""
from __future__ import annotations

import json
import re

from PyQt5.QtCore import QUrl, pyqtSlot, pyqtSignal
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtCore import QObject


# ── The full HTML template ────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  :root {
    --bg: #1a1b26;
    --surface: #1e1f2e;
    --text: #c0caf5;
    --heading1: #7aa2f7;
    --heading2: #bb9af7;
    --heading3: #73daca;
    --accent: #3d59a1;
    --border: #2a2b3d;
    --code-bg: #16161e;
    --def-bg: #1c2a3a;
    --def-border: #7aa2f7;
    --concept-bg: #1c2e24;
    --concept-border: #73daca;
    --formula-bg: #2a1f0e;
    --formula-border: #e0af68;
    --example-bg: #231f2e;
    --example-border: #bb9af7;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 28px 48px;
    background: var(--bg);
    color: var(--text);
    font-family: 'Malgun Gothic', 'Noto Sans KR', 'Segoe UI', sans-serif;
    font-size: 15px;
    line-height: 1.85;
    max-width: 960px;
    margin-left: auto;
    margin-right: auto;
  }

  /* ── Headings ── */
  h1 { color: var(--heading1); font-size: 1.9em; border-bottom: 2px solid var(--accent); padding-bottom: 10px; margin-top: 8px; }
  h2 { color: var(--heading2); font-size: 1.4em; border-left: 4px solid var(--heading2); padding-left: 12px; margin-top: 36px; }
  h3 { color: var(--heading3); font-size: 1.15em; margin-top: 24px; }
  h4 { color: #9ece6a; margin-top: 18px; }

  /* ── Tables ── */
  table { width: 100%; border-collapse: collapse; margin: 16px 0; }
  th { background: var(--accent); color: #fff; padding: 8px 14px; text-align: left; }
  td { border: 1px solid var(--border); padding: 7px 14px; }
  tr:nth-child(even) td { background: #1e1f2e; }

  /* ── Code ── */
  code { background: var(--code-bg); padding: 2px 6px; border-radius: 4px; font-family: 'Consolas', monospace; font-size: 0.9em; color: #9ece6a; }
  pre { background: var(--code-bg); padding: 14px 18px; border-radius: 8px; overflow-x: auto; border: 1px solid var(--border); }
  pre code { background: transparent; padding: 0; color: #a9b1d6; }

  /* ── Blockquote (info box) ── */
  blockquote { border-left: 4px solid var(--accent); margin: 12px 0; padding: 8px 16px; background: #1e1f2e; border-radius: 0 8px 8px 0; color: #a9b1d6; }

  /* ── Lists ── */
  ul, ol { padding-left: 1.6em; }
  li { margin: 4px 0; }

  /* ── Horizontal rule ── */
  hr { border: none; border-top: 1px solid var(--border); margin: 28px 0; }

  /* ── Math overflow ── */
  mjx-container { overflow-x: auto; max-width: 100%; }
  .MathJax_Display { overflow-x: auto; }

  /* ── Highlighting ── */
  .hl-yellow { background: #ffe06644; border-radius: 3px; padding: 1px 2px; }
  .hl-green  { background: #73daca44; border-radius: 3px; padding: 1px 2px; }
  .hl-blue   { background: #7dcfff44; border-radius: 3px; padding: 1px 2px; }
  .hl-pink   { background: #f7768e44; border-radius: 3px; padding: 1px 2px; }

  /* ── Section cards ── */
  .def-card     { background: var(--def-bg);     border-left: 4px solid var(--def-border);     padding: 10px 16px; border-radius: 0 8px 8px 0; margin: 10px 0; }
  .concept-card { background: var(--concept-bg); border-left: 4px solid var(--concept-border); padding: 10px 16px; border-radius: 0 8px 8px 0; margin: 10px 0; }
  .formula-card { background: var(--formula-bg); border: 1px solid var(--formula-border);      padding: 14px 20px; border-radius: 8px; margin: 16px 0; }
  .example-card { background: var(--example-bg); border-left: 4px solid var(--example-border); padding: 10px 16px; border-radius: 0 8px 8px 0; margin: 10px 0; }

  /* ── Streaming cursor ── */
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
  .streaming-cursor::after { content: '▋'; animation: blink 1s infinite; color: var(--heading1); }
</style>

<!-- MathJax 3 configuration -->
<script>
MathJax = {
  tex: {
    inlineMath:  [['$','$'], ['\\(','\\)']],
    displayMath: [['$$','$$'], ['\\[','\\]']],
    processEscapes: true,
    processEnvironments: true,
    packages: {'[+]': ['ams', 'boldsymbol', 'physics', 'color']},
    tags: 'ams',
  },
  options: {
    skipHtmlTags: ['script','noscript','style','textarea','pre'],
    ignoreHtmlClass: 'no-mathjax',
  },
  startup: {
    ready() {
      MathJax.startup.defaultReady();
    }
  }
};
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

<!-- marked.js for markdown rendering -->
<script src="https://cdn.jsdelivr.net/npm/marked@9/marked.min.js"></script>
</head>
<body>

<div id="note-content"></div>

<script>
"use strict";

// ── Markdown renderer setup ────────────────────────────────────────────────
const renderer = new marked.Renderer();

// Keep LaTeX delimiters safe from marked escaping
function protectMath(src) {
  const blocks = [];
  // Display math first $$...$$
  src = src.replace(/\$\$([\s\S]*?)\$\$/g, (_, m) => {
    blocks.push('$$' + m + '$$');
    return '\x00MATH' + (blocks.length - 1) + '\x00';
  });
  // Inline math $...$
  src = src.replace(/\$([^\n$]+?)\$/g, (_, m) => {
    blocks.push('$' + m + '$');
    return '\x00MATH' + (blocks.length - 1) + '\x00';
  });
  return { src, blocks };
}

function restoreMath(html, blocks) {
  return html.replace(/\x00MATH(\d+)\x00/g, (_, i) => blocks[i]);
}

marked.setOptions({ breaks: true, gfm: true });

let currentMarkdown = '';
let isStreaming = false;
let mathQueued = false;

function renderMarkdown(md) {
  const { src, blocks } = protectMath(md);
  let html = marked.parse(src);
  html = restoreMath(html, blocks);
  return html;
}

// ── Main render function (called from Python) ─────────────────────────────
window.renderNote = function(markdown) {
  currentMarkdown = markdown;
  isStreaming = false;
  const el = document.getElementById('note-content');
  el.innerHTML = renderMarkdown(markdown);
  el.classList.remove('streaming-cursor');
  typesetMath();
};

// ── Streaming append (called from Python for each chunk) ──────────────────
window.appendChunk = function(chunk) {
  currentMarkdown += chunk;
  isStreaming = true;
  const el = document.getElementById('note-content');
  el.innerHTML = renderMarkdown(currentMarkdown);
  el.classList.add('streaming-cursor');
  // Throttle MathJax calls during streaming
  if (!mathQueued) {
    mathQueued = true;
    setTimeout(() => { typesetMath(); mathQueued = false; }, 800);
  }
};

window.finalizeStream = function() {
  isStreaming = false;
  const el = document.getElementById('note-content');
  el.classList.remove('streaming-cursor');
  typesetMath();
};

// ── MathJax typesetting ───────────────────────────────────────────────────
function typesetMath() {
  if (window.MathJax && MathJax.typesetPromise) {
    MathJax.typesetPromise([document.getElementById('note-content')])
      .catch(err => console.warn('MathJax error:', err));
  }
}

// ── Highlighting ──────────────────────────────────────────────────────────
window.applyHighlight = function(color) {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed) return null;
  const range = sel.getRangeAt(0);
  const span = document.createElement('span');
  span.className = 'hl-' + color;
  span.dataset.hl = color;
  try {
    range.surroundContents(span);
  } catch(e) {
    const frag = range.extractContents();
    span.appendChild(frag);
    range.insertNode(span);
  }
  sel.removeAllRanges();
  return collectHighlights();
};

window.removeHighlight = function(spanEl) {
  const parent = spanEl.parentNode;
  while (spanEl.firstChild) parent.insertBefore(spanEl.firstChild, spanEl);
  parent.removeChild(spanEl);
  return collectHighlights();
};

window.clearAllHighlights = function() {
  document.querySelectorAll('[data-hl]').forEach(el => {
    const p = el.parentNode;
    while (el.firstChild) p.insertBefore(el.firstChild, el);
    p.removeChild(el);
  });
  return '[]';
};

function collectHighlights() {
  const result = [];
  document.querySelectorAll('[data-hl]').forEach((el, i) => {
    result.push({ id: i, text: el.textContent.slice(0, 80), color: el.dataset.hl });
  });
  return JSON.stringify(result);
}

// Restore highlights from saved JSON
window.restoreHighlights = function(json) {
  // Highlights are stored as DOM mutations — after render we re-apply
  // This is a best-effort approach; full fidelity would require offset tracking
  const items = JSON.parse(json);
  items.forEach(item => {
    // Simple text-search re-highlight
    highlightTextInPage(item.text, item.color);
  });
};

function highlightTextInPage(searchText, color) {
  if (!searchText || searchText.length < 3) return;
  const content = document.getElementById('note-content');
  const walker = document.createTreeWalker(content, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    const idx = node.nodeValue.indexOf(searchText);
    if (idx >= 0) {
      const range = document.createRange();
      range.setStart(node, idx);
      range.setEnd(node, idx + searchText.length);
      const span = document.createElement('span');
      span.className = 'hl-' + color;
      span.dataset.hl = color;
      try { range.surroundContents(span); } catch(e) {}
      break;
    }
  }
}

// ── Context menu: right-click to remove highlight ─────────────────────────
document.addEventListener('contextmenu', (e) => {
  const hl = e.target.closest('[data-hl]');
  if (hl) { e.preventDefault(); removeHighlight(hl); }
});

// ── Scroll to top helper ──────────────────────────────────────────────────
window.scrollToTop = function() { window.scrollTo(0, 0); };
</script>
</body>
</html>
"""


class NoteViewer(QWebEngineView):
    """Renders notes as rich HTML with MathJax math typesetting."""

    highlights_changed = pyqtSignal(str)  # JSON of current highlights

    def __init__(self, parent=None):
        super().__init__(parent)
        settings = self.settings()
        settings.setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)

        self.setHtml(HTML_TEMPLATE, QUrl("about:blank"))
        self._current_note_id: int | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def render_note(self, markdown: str, note_id: int | None = None):
        self._current_note_id = note_id
        escaped = _js_string(markdown)
        self.page().runJavaScript(f"window.renderNote({escaped});")

    def append_chunk(self, chunk: str):
        escaped = _js_string(chunk)
        self.page().runJavaScript(f"window.appendChunk({escaped});")

    def finalize_stream(self):
        self.page().runJavaScript("window.finalizeStream();")

    def apply_highlight(self, color: str):
        """Highlight selected text with colour ('yellow','green','blue','pink')."""
        self.page().runJavaScript(
            f"window.applyHighlight('{color}');",
            self._on_highlight_result,
        )

    def clear_highlights(self):
        self.page().runJavaScript(
            "window.clearAllHighlights();",
            self._on_highlight_result,
        )

    def restore_highlights(self, highlights_json: str):
        escaped = _js_string(highlights_json)
        self.page().runJavaScript(f"window.restoreHighlights({escaped});")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_highlight_result(self, result):
        if result:
            self.highlights_changed.emit(result)


def _js_string(text: str) -> str:
    """Safely escape a Python string for embedding in a JS function call."""
    return json.dumps(text, ensure_ascii=False)
