# apps/rheed_monitor/make_manual.py
"""RHEED Monitor Manual PDF (English - fpdf2 built-in font compatible)."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def make_manual(out_path: Path) -> None:
    try:
        from fpdf import FPDF
    except ImportError:
        print("fpdf2 not installed. pip install fpdf2")
        return

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def h1(t):
        pdf.set_font("Helvetica", "B", 20)
        pdf.cell(0, 12, t, new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(4)

    def h2(t):
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_fill_color(210, 225, 245)
        pdf.cell(0, 8, t, new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.ln(2)

    def body(t):
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, t)
        pdf.ln(2)

    def bullets(items):
        pdf.set_font("Helvetica", "", 11)
        for item in items:
            pdf.set_x(16)
            pdf.multi_cell(0, 6, f"  - {item}")
        pdf.ln(2)

    def code(lines):
        pdf.set_font("Courier", "", 10)
        pdf.set_fill_color(240, 240, 240)
        for line in lines:
            pdf.cell(0, 5.5, line, new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.ln(2)

    # Title
    h1("RHEED Monitor")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 7, "User Manual  v1.0", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)

    # 1. Overview
    h2("1. Overview")
    body(
        "RHEED Monitor captures and analyzes RHEED (Reflection High-Energy Electron "
        "Diffraction) patterns from an MBE system in real time using a HIKROBOT GigE camera.\n\n"
        "Features:\n"
        "  - Live camera feed with spot overlay\n"
        "  - Automatic spot detection: position (x, y) and brightness\n"
        "  - Real-time graphs (X position / Y position / Brightness vs. time)\n"
        "  - Timed screenshots (configurable interval)\n"
        "  - Full session video recording\n"
        "  - Auto-compressed archive on session close"
    )

    # 2. System Requirements
    h2("2. System Requirements")
    bullets([
        "Windows 10 / 11 (64-bit)",
        "HIKROBOT MVS software installed (includes GigE filter driver + Python SDK)",
        "Dedicated GigE NIC for camera recommended (Jumbo Frame MTU=9000)",
        "Camera and PC on the same subnet (or IP assigned via MVS)",
    ])

    # 3. First-time Setup
    h2("3. First-time Setup  --  RheedSetup.exe")
    body("Run RheedSetup.exe before first use or whenever the camera/environment changes.")
    bullets([
        "Camera Index: 0 = first detected camera (increase if multiple cameras connected)",
        "Exposure (us): Adjust for RHEED screen brightness (default 10,000 us)",
        "Gain (dB): Additional amplification if needed (default 0)",
        "Threshold Fraction: Fraction of max brightness to use as spot threshold (default 0.5)",
        "Min Brightness: Frames darker than this = NO SPOT (default 20 / 255)",
        "Broad Area Threshold: Spots larger than this area (px2) = broad pattern (default 500)",
        "Screenshot Interval: Auto-screenshot period in seconds (default 30 s)",
        "Video FPS: Recording frame rate (default 15 fps)",
    ])
    body("Click [Save] to write settings to rheed_config.yaml beside the EXE.")

    # 4. Operation
    h2("4. Operation  --  RheedMonitor.exe")
    bullets([
        "[Connect Camera]  :  Scan for GigE cameras and connect.",
        "[Start Recording] :  Begin session (video + timed screenshots + CSV data).",
        "[Screenshot Now]  :  Save a manual screenshot immediately.",
        "[End Session & Save] :  Stop recording, compress all files, show save path.",
        "Screenshot Interval spinner:  Change auto-screenshot period while running.",
    ])

    # 5. Spot Detection
    h2("5. Spot Detection Logic")
    body(
        "Each frame is processed as follows:\n"
        "  1. Convert to grayscale + Gaussian blur (noise reduction)\n"
        "  2. If max brightness < Min Brightness  ->  NO SPOT\n"
        "  3. Threshold = max brightness x Threshold Fraction\n"
        "  4. Connected-component analysis on thresholded image\n"
        "  5. Centroid of each component = reported (x, y) coordinate"
    )
    bullets([
        "Dot  [D]: small area spot  -> crosshair marker (green = primary spot)",
        "Broad [B]: large diffuse pattern -> square marker, centroid = center of intensity",
        "NO SPOT: displayed in red when no component passes the threshold",
        "Graph shows NaN gap when spot is absent (line breaks instead of zero)",
    ])

    # 6. Output Structure
    h2("6. Output File Structure")
    code([
        "outputs/rheed/",
        "  run_YYYYMMDD_HHMMSS/",
        "    video.mp4              <- full session video (original frames)",
        "    screenshots/           <- auto + manual PNGs with spot overlay",
        "      YYYYMMDD_HHMMSS_001.png",
        "    spot_data.csv          <- timestamp, spot_count, x, y, brightness, area, type",
        "    session.log            <- session events log",
        "  run_YYYYMMDD_HHMMSS.zip  <- compressed archive (video stored uncompressed)",
    ])
    body("The ZIP file is created automatically when you click [End Session & Save].\n"
         "Video (.mp4) is stored without re-compression inside the ZIP to avoid quality loss.")

    # 7. Troubleshooting
    h2("7. Troubleshooting")
    bullets([
        "'MVS SDK not found': Install HIKROBOT MVS software, then restart the program.",
        "'No cameras found': Check cable, verify camera appears in MVS software.",
        "No spot detected: Lower Threshold Fraction or Min Brightness in RheedSetup.",
        "Broad spot shown as dot: Lower Broad Area Threshold in RheedSetup.",
        "Video not saved: Make sure to click [Start Recording] before [End Session].",
        "Graph shows all gaps: Camera connected but scene too dark -- adjust exposure.",
    ])

    pdf.output(str(out_path))
    print(f"Manual saved: {out_path}")


if __name__ == "__main__":
    out = REPO_ROOT / "RHEED_Monitor_Manual.pdf"
    make_manual(out)
