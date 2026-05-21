#!/usr/bin/env node
/**
 * codex-pty.mjs
 *
 * Spawns the Codex interactive TUI inside a pseudo-terminal (ConPTY on
 * Windows) and types a prompt just like a human would, then captures the
 * response.
 *
 * Key design: instead of stripping ANSI escape codes from a concatenation of
 * all PTY frames (which mixes spinner/cursor noise into the content), we feed
 * the raw bytes through a VirtualScreen — a minimal terminal emulator that
 * tracks cursor position and text overwrites.  finish() reads the *final*
 * rendered screen state, giving clean output.
 *
 * State machine:
 *   booting → [dialog_wait] → typing → waiting → done
 *
 * Usage:
 *   node codex-pty.mjs "my prompt here"
 *   echo "my prompt" | node codex-pty.mjs --stdin
 *   node codex-pty.mjs --mode rescue --file buggy.py --error "NameError: x"
 *   node codex-pty.mjs --mode review  --file code.py
 */

import pty from 'node-pty';
import { appendFileSync, readFileSync, writeFileSync, statSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const __dir = dirname(fileURLToPath(import.meta.url));
const LOG   = join(__dir, 'sessions.log');

// ── Timing (ms) ───────────────────────────────────────────────
const TRUST_WATCH_MS = 8000;   // max wait for dialogs before typing
const AFTER_TRUST_MS = 5000;   // pause after accepting trust (sandbox setup)
const CHAR_MS        = 2;      // delay between typed characters
const POST_ENTER_MS  = 400;    // pause before pressing Enter
const IDLE_MS        = 15000;  // silence after last AI output → done (tool calls can pause >6s)
const FIRST_IDLE_MS  = 20000;  // longer first window (AI may be slow to start)
const MAX_MS         = 210_000; // hard timeout (~3.5 min)

// ── Terminal dimensions ───────────────────────────────────────
const COLS = 200;
const ROWS = 100;  // large enough for long AI responses + user prompt echo

// ── Log rotation ──────────────────────────────────────────────
// Keep sessions.log under LOG_MAX_BYTES.  When exceeded, discard the oldest
// content and retain LOG_KEEP_BYTES of recent history aligned to a session
// boundary ("--- SESSION ---") so no session is left half-truncated.
const LOG_MAX_BYTES  = 500_000;  // ~500 KB trigger
const LOG_KEEP_BYTES = 200_000;  // retain ~200 KB after rotation

function rotateLOG() {
  try {
    let size;
    try { size = statSync(LOG).size; } catch { return; } // file doesn't exist yet
    if (size <= LOG_MAX_BYTES) return;

    const content = readFileSync(LOG, 'utf8');
    const cutPos   = content.length - LOG_KEEP_BYTES;
    // Align to the next "--- SESSION ---" boundary so sessions stay intact
    const boundary = content.indexOf('\n--- SESSION ---', cutPos);
    const trimmed  = boundary >= 0 ? content.slice(boundary + 1) : content.slice(cutPos);
    writeFileSync(LOG, trimmed, 'utf8');
  } catch { /* ignore I/O errors */ }
}

// ── Log ───────────────────────────────────────────────────────
function log(tag, msg) {
  try { appendFileSync(LOG, `[${new Date().toISOString()}][${tag}] ${msg}\n`); } catch {}
}

// ── VirtualScreen ─────────────────────────────────────────────
// Minimal terminal emulator: parses raw PTY bytes (including ANSI cursor-
// movement sequences) and maintains a 2-D character buffer.  Reading
// getText() at any point gives the current rendered screen state.
class VirtualScreen {
  constructor(cols = COLS, rows = ROWS) {
    this.cols = cols;
    this.rows = rows;
    this.lines = Array.from({ length: rows }, () => '');
    this.cx = 0;   // cursor column (0-based)
    this.cy = 0;   // cursor row    (0-based)
    this.savedCx = 0;
    this.savedCy = 0;
  }

  write(data) {
    let i = 0;
    while (i < data.length) {
      const ch = data[i];
      if (ch === '\x1B') {
        i += this._esc(data, i);
      } else if (ch === '\r') {
        this.cx = 0; i++;
      } else if (ch === '\n') {
        this.cy = Math.min(this.cy + 1, this.rows - 1); i++;
      } else if (ch === '\b') {
        this.cx = Math.max(0, this.cx - 1); i++;
      } else {
        // Handle multi-codepoint characters (emoji etc.)
        const cp = data.codePointAt(i);
        const c  = String.fromCodePoint(cp);
        this._put(c);
        i += c.length;
      }
    }
  }

  _put(ch) {
    if (this.cy < 0 || this.cy >= this.rows) return;
    const line = this.lines[this.cy].padEnd(this.cx, ' ');
    this.lines[this.cy] = line.slice(0, this.cx) + ch + line.slice(this.cx + 1);
    this.cx = Math.min(this.cx + 1, this.cols - 1);
  }

  _esc(data, pos) {
    // Returns number of bytes consumed starting at pos (which is the ESC byte)
    if (pos + 1 >= data.length) return data.length - pos;
    const next = data[pos + 1];

    // ── CSI  \x1B[ ─────────────────────────────────────────
    if (next === '[') {
      let end = pos + 2;
      while (end < data.length && !/[@-~]/.test(data[end])) end++;
      if (end >= data.length) return data.length - pos;

      const final    = data[end];
      const paramStr = data.slice(pos + 2, end);
      const nums     = paramStr.split(';').map(p => parseInt(p, 10) || 0);
      const p0 = nums[0] || 0, p1 = nums[1] || 0;

      switch (final) {
        case 'H': case 'f': // cursor position (1-based)
          this.cy = Math.max(0, Math.min(this.rows - 1, (p0 || 1) - 1));
          this.cx = Math.max(0, Math.min(this.cols - 1, (p1 || 1) - 1));
          break;
        case 'A': this.cy = Math.max(0, this.cy - (p0 || 1)); break;
        case 'B': this.cy = Math.min(this.rows - 1, this.cy + (p0 || 1)); break;
        case 'C': this.cx = Math.min(this.cols - 1, this.cx + (p0 || 1)); break;
        case 'D': this.cx = Math.max(0, this.cx - (p0 || 1)); break;
        case 'E': this.cy = Math.min(this.rows - 1, this.cy + (p0 || 1)); this.cx = 0; break;
        case 'F': this.cy = Math.max(0, this.cy - (p0 || 1)); this.cx = 0; break;
        case 'G': this.cx = Math.max(0, Math.min(this.cols - 1, (p0 || 1) - 1)); break;
        case 'd': this.cy = Math.max(0, Math.min(this.rows - 1, (p0 || 1) - 1)); break;
        case 'J':
          if (p0 === 0) {
            for (let r = this.cy + 1; r < this.rows; r++) this.lines[r] = '';
            this.lines[this.cy] = this.lines[this.cy].slice(0, this.cx);
          } else if (p0 === 1) {
            for (let r = 0; r < this.cy; r++) this.lines[r] = '';
            this.lines[this.cy] = ' '.repeat(this.cx) + this.lines[this.cy].slice(this.cx);
          } else if (p0 === 2 || p0 === 3) {
            this.lines = Array.from({ length: this.rows }, () => '');
            this.cx = 0; this.cy = 0;
          }
          break;
        case 'K':
          if      (p0 === 0) this.lines[this.cy] = this.lines[this.cy].slice(0, this.cx);
          else if (p0 === 1) this.lines[this.cy] = ' '.repeat(this.cx) + this.lines[this.cy].slice(this.cx);
          else if (p0 === 2) this.lines[this.cy] = '';
          break;
        case 'S': { // scroll up
          const n = p0 || 1;
          this.lines.splice(0, n);
          while (this.lines.length < this.rows) this.lines.push('');
          break;
        }
        case 'T': { // scroll down
          const n = p0 || 1;
          for (let k = 0; k < n; k++) { this.lines.pop(); this.lines.unshift(''); }
          break;
        }
        case 's': this.savedCx = this.cx; this.savedCy = this.cy; break;
        case 'u': this.cx = this.savedCx; this.cy = this.savedCy; break;
        // 'm' colors, 'h'/'l' modes, 'r' scroll region, '?' flags — ignore
      }
      return end - pos + 1;
    }

    // ── OSC  \x1B] ─────────────────────────────────────────
    if (next === ']') {
      const bel = data.indexOf('\x07',   pos + 2);
      const st  = data.indexOf('\x1B\\', pos + 2);
      if (bel >= 0 && (st < 0 || bel < st)) return bel - pos + 1;
      if (st  >= 0)                          return st  - pos + 2;
      return data.length - pos; // unterminated — consume rest
    }

    // ── DCS  \x1BP ─────────────────────────────────────────
    if (next === 'P') {
      const st = data.indexOf('\x1B\\', pos + 2);
      return st >= 0 ? st - pos + 2 : data.length - pos;
    }

    // ── SS2/SS3  \x1BN / \x1BO — single extra char ─────────
    if (next === 'N' || next === 'O') return 3;

    // Other 2-char sequences (\x1B= \x1B> \x1B7 \x1B8 \x1B( \x1B) …)
    return 2;
  }

  /** Return the visible screen as plain text with trailing spaces stripped. */
  getText() {
    return this.lines.map(l => l.trimEnd()).join('\n');
  }
}

// ── Post-render TUI chrome filter ────────────────────────────
// Even after proper screen rendering the status bar / suggestion row remain.
const CHROME_LINE_RE = [
  /^\s*0;/,                       // terminal-title artifact that leaked through
  /^[⠀-⣿\s]+$/,                 // braille-only line (spinner leftover)
  /^›\s/,                         // suggestion item
  /gpt-[0-9.]+.*default/,         // model indicator
  /esc to int/i,                  // "esc to interrupt"
  /Working.*esc/i,
  /^C:\\Users\\/,                 // bare Windows path
  /^Tip:/,                        // "Tip: Start a fresh idea..." header
  /^[─═━]{5,}$/,                  // horizontal separator lines
];

function filterChrome(text) {
  return text
    .split('\n')
    .filter(line => {
      const t = line.trim();
      return t !== '' && !CHROME_LINE_RE.some(re => re.test(t));
    })
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

// ── AI response extractor ─────────────────────────────────────
// After Enter, Codex re-renders the full screen including the echoed user
// message.  The AI response section starts at the first line prefixed with
// '•' (Codex's thinking/planning bullet). We take from there to the last
// non-empty content line (discarding trailing separators / chrome).
function extractAIResponse(text) {
  const lines = text.split('\n');

  // Find the first line that looks like the AI's response start.
  // Codex prefixes its thinking step with a bullet '•'.
  let start = lines.findIndex(l => /^\s*[•·]/.test(l));

  // Fallback: if no bullet, find first non-indented, non-empty, non-chrome
  // line that appears after the echoed user prompt (which is always indented).
  if (start < 0) {
    start = lines.findIndex((l, i) => {
      if (i === 0) return false;        // skip very first line (title bar)
      if (l.trim() === '') return false;
      if (/^\s{2,}/.test(l)) return false; // indented → user message
      if (CHROME_LINE_RE.some(re => re.test(l.trim()))) return false;
      return true;
    });
  }

  if (start < 0) return text; // nothing identifiable — return raw

  // Trim trailing separator/empty lines from the bottom
  let end = lines.length - 1;
  while (end > start && (
    lines[end].trim() === '' ||
    /^[─═━]{5,}$/.test(lines[end].trim())
  )) end--;

  return lines.slice(start, end + 1).join('\n');
}

// ── Dialog detection patterns (whitespace-stripped lowercase) ─
const TRUST_PATTERNS  = ['sandbox', 'entertocontin', 'trustthecontents', 'doyoutrust'];
const UPDATE_PATTERNS = ['updateavailable', 'updatenow', 'skipuntilnext'];

// Simple ANSI strip for dialog detection only (not for output)
const ANSI_RE = /\x1B(?:[@-Z\\-_]|\[[0-9;?]*[ -/]*[@-~]|\][^\x07\x1B]*(?:\x07|\x1B\\))/g;
const stripAnsi = s => s.replace(ANSI_RE, '');

// ── Build prompt from CLI args ────────────────────────────────
async function buildPrompt(argv) {
  if (argv[0] === '--stdin') {
    const chunks = [];
    for await (const chunk of process.stdin) chunks.push(chunk);
    return Buffer.concat(chunks).toString('utf8').trim();
  }

  if (argv[0] === '--mode') {
    const mode = argv[1];
    const fIdx = argv.indexOf('--file');
    const eIdx = argv.indexOf('--error');
    const file  = fIdx >= 0 ? argv[fIdx + 1] : '';
    const error = eIdx >= 0 ? argv.slice(eIdx + 1).join(' ') : '';
    const code  = file ? readFileSync(file, 'utf8') : '';

    if (mode === 'rescue') {
      return [
        'RESCUE MODE — independent bug investigation',
        `File: ${file}`,
        error ? `Error: ${error}` : '',
        '',
        '```',
        code,
        '```',
        '',
        'Diagnose the root cause. Be specific about which line and why. Suggest a minimal fix.',
      ].filter(l => l !== undefined).join('\n');
    }

    return [
      'REVIEW MODE — independent code review',
      `File: ${file}`,
      '',
      '```',
      code,
      '```',
      '',
      'Check for bugs, logic errors, and critical issues. Be concise.',
    ].join('\n');
  }

  return argv.join(' ').trim();
}

// ── Main ──────────────────────────────────────────────────────
(async () => {
  const argv   = process.argv.slice(2);
  const prompt = await buildPrompt(argv);

  if (!prompt) {
    process.stderr.write([
      'Usage:',
      '  node codex-pty.mjs "your prompt"',
      '  node codex-pty.mjs --stdin',
      '  node codex-pty.mjs --mode rescue --file path.py --error "msg"',
      '  node codex-pty.mjs --mode review  --file path.py',
      '',
    ].join('\n'));
    process.exit(1);
  }

  rotateLOG();  // trim log before writing this session
  log('START', `prompt_len=${prompt.length}`);

  // ── Spawn ─────────────────────────────────────────────────
  // --dangerously-bypass-approvals-and-sandbox: skip all confirmation prompts
  // so shell commands (Get-Content, ls, etc.) run without user approval.
  // Safe for this automation context — tool calls are read-only file inspection.
  const proc = pty.spawn('cmd.exe', ['/c', 'codex', '--dangerously-bypass-approvals-and-sandbox'], {
    name: 'xterm-256color',
    cols: COLS,
    rows: ROWS,
    cwd: process.cwd(),
    env: { ...process.env, TERM: 'xterm-256color', COLORTERM: 'truecolor' },
  });

  // ── State ─────────────────────────────────────────────────
  let rawBuf        = '';            // full raw output for logging
  let screen        = new VirtualScreen(); // tracks boot + dialog phase
  let replyScreen   = null;          // created fresh when Enter is pressed
  let phase         = 'booting';
  let handledDialogs = new Set();
  let dialogCooldown = false;
  let idleTimer      = null;
  let trustWatchTimer = null;
  let maxTimer        = null;
  let finished        = false;

  // ── finish ────────────────────────────────────────────────
  function finish(reason) {
    if (finished) return;
    finished = true;
    clearTimeout(idleTimer);
    clearTimeout(trustWatchTimer);
    clearTimeout(maxTimer);

    // Use the reply-phase VirtualScreen for clean output.
    // extractAIResponse removes the echoed user prompt (Codex re-renders the
    // full screen after Enter, so the raw screen includes the user's message).
    const raw    = replyScreen ? replyScreen.getText() : screen.getText();
    const reply  = filterChrome(extractAIResponse(raw));

    log('FINISH', `reason=${reason} reply_chars=${reply.length}`);
    try { appendFileSync(LOG, `--- SESSION ---\n${raw}\n--- END ---\n\n`); } catch {}

    process.stdout.write(
      '\n╔══ CODEX RESPONSE ══════════════════════════════════╗\n' +
      reply + '\n' +
      '╚════════════════════════════════════════════════════╝\n',
      () => {
        try { execSync(`taskkill /PID ${proc.pid} /T /F`, { stdio: 'ignore' }); } catch {}
        process.exit(0);
      }
    );
  }

  // ── typePrompt ────────────────────────────────────────────
  async function typePrompt() {
    if (finished) return;
    phase = 'typing';
    log('TYPE', `typing ${prompt.length} chars`);

    for (const ch of prompt) {
      if (finished) return;
      proc.write(ch);
      await new Promise(r => setTimeout(r, CHAR_MS));
    }

    await new Promise(r => setTimeout(r, POST_ENTER_MS));
    if (finished) return;

    // Start a fresh VirtualScreen so we only capture the AI reply
    replyScreen = new VirtualScreen();
    proc.write('\r');

    phase = 'waiting';
    idleTimer = setTimeout(() => finish('idle'), FIRST_IDLE_MS);
    log('TYPE', 'Enter sent; waiting for AI response');
  }

  // ── onData ────────────────────────────────────────────────
  proc.onData(chunk => {
    rawBuf += chunk;

    // Feed to the appropriate screen
    if (replyScreen) {
      replyScreen.write(chunk);
    } else {
      screen.write(chunk);
    }

    // Dialog detection (booting / dialog_wait only)
    if (!finished && (phase === 'booting' || phase === 'dialog_wait') && !dialogCooldown) {
      const compact = stripAnsi(rawBuf).replace(/\s+/g, '').toLowerCase();

      if (!handledDialogs.has('update') && UPDATE_PATTERNS.some(p => compact.includes(p))) {
        handledDialogs.add('update');
        dialogCooldown = true;
        clearTimeout(trustWatchTimer);
        log('UPDATE', 'update dialog — skipping (2)');
        setTimeout(() => {
          if (!finished) {
            proc.write('2\r');
            phase = 'dialog_wait';
            dialogCooldown = false;
            trustWatchTimer = setTimeout(() => {
              if (!finished && phase !== 'typing' && phase !== 'waiting') typePrompt();
            }, TRUST_WATCH_MS);
          }
        }, 400);

      } else if (!handledDialogs.has('trust') && TRUST_PATTERNS.some(p => compact.includes(p))) {
        handledDialogs.add('trust');
        dialogCooldown = true;
        clearTimeout(trustWatchTimer);
        log('TRUST', 'trust dialog — accepting (Enter)');
        setTimeout(() => {
          if (!finished) {
            proc.write('\r');
            phase = 'dialog_wait';
            dialogCooldown = false;
            log('TRUST', `waiting ${AFTER_TRUST_MS}ms for sandbox`);
            setTimeout(typePrompt, AFTER_TRUST_MS);
          }
        }, 400);
      }
    }

    // Idle reset while waiting for response
    if (phase === 'waiting') {
      clearTimeout(idleTimer);
      idleTimer = setTimeout(() => finish('idle'), IDLE_MS);
    }
  });

  proc.onExit(() => { if (!finished) finish('exit'); });

  // ── Fallback: no dialogs → type directly ─────────────────
  trustWatchTimer = setTimeout(() => {
    if (!finished && phase === 'booting') {
      log('BOOT', 'no dialogs — typing directly');
      typePrompt();
    }
  }, TRUST_WATCH_MS);

  // ── Hard timeout ──────────────────────────────────────────
  maxTimer = setTimeout(() => finish('timeout'), MAX_MS);

})().catch(err => {
  process.stderr.write(`codex-pty fatal: ${err.message}\n`);
  process.exit(1);
});
