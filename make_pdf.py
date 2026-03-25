# make_pdf.py
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import mm
import os

def create_manual_pdf(filename="Vacuum_Monitor_Manual.pdf"):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    # 1. 윈도우 한글 폰트(맑은고딕) 등록
    font_path = "C:/Windows/Fonts/malgun.ttf"
    if os.path.exists(font_path):
        pdfmetrics.registerFont(TTFont("Malgun", font_path))
        font_title = "Malgun"
        font_body = "Malgun"
    else:
        # 폰트가 없으면 기본 폰트 (한글 깨질 수 있음)
        font_title = "Helvetica-Bold"
        font_body = "Helvetica"

    # --- 제목 ---
    c.setFont(font_title, 24)
    c.drawString(20 * mm, height - 30 * mm, "Vacuum Monitor System v2.0 매뉴얼")
    
    c.setFont(font_body, 10)
    c.drawString(20 * mm, height - 40 * mm, "작성일: 2026.01.26 | 대상: 관리자 및 사용자")

    # --- 본문 내용 ---
    content = [
        "",
        "================================================================",
        "1. 시스템 개요",
        "================================================================",
        "본 시스템은 듀얼 카메라(Top/Bottom)를 이용해 진공 게이지 수치를",
        "자동으로 인식(OCR)하고, 이상 발생 시 이메일 알림을 보냅니다.",
        "",
        "================================================================",
        "2. 설치 및 폴더 구성 (Portable)",
        "================================================================",
        "설치가 필요 없습니다. 아래 폴더 구조를 유지한 채 복사해서 사용하세요.",
        "📂 Vacuum_v2/",
        "   ├── VacuumSetup.exe     (초기 설정 및 좌표 수정용)",
        "   ├── VacuumMonitor.exe   (실행용 - 감시 프로그램)",
        "   └── tesseract_portable/ (OCR 엔진 폴더 *삭제 금지*)",
        "",
        "================================================================",
        "3. 초기 설정 방법 (최초 1회)",
        "================================================================",
        "[Step 1] 이메일 알림 계정 등록 (CMD창)",
        "   > setx VACUUM_SMTP_USER \"본인_이메일@gmail.com\"",
        "   > setx VACUUM_SMTP_PASS \"구글_앱_비밀번호_16자리\"",
        "",
        "[Step 2] 좌표 설정 (VacuumSetup.exe 실행)",
        "   - [1]번 키: 전체 화면에서 게이지 위치 잡기 (Crop)",
        "   - [2]번 키 -> [N]키: 숫자 영역 박스 그리기 (ROI)",
        "     (박스는 초록색으로 숫자를 꽉 차게 그려주세요)",
        "   - [S]번 키: 저장 후 종료",
        "",
        "================================================================",
        "4. 모니터링 시작",
        "================================================================",
        "1. VacuumMonitor.exe를 실행합니다.",
        "2. 프로그램이 설정된 ROI를 읽어 화면에 표시합니다.",
        "3. 'records.csv' 파일에 1초 단위로 데이터가 저장됩니다.",
        "4. 종료하려면 화면 클릭 후 'Q'를 누르세요.",
        "",
        "================================================================",
        "5. 시스템 로직 (System Logic)",
        "================================================================",
        "- Sampling: 1초 간격 촬영 / 10분 간격 엑셀 저장",
        "- Anomaly Check: 최근 1분 평균 대비 10배(log10 1.0) 급변 시 경고",
        "- Missing Check: 3회 연속 인식 실패 시 경고",
        "- Auto Cleanup: 30일 지난 로그 및 사진 자동 삭제 (매시간 체크)",
        "",
        "================================================================",
        "6. 주의 사항 및 문제 해결",
        "================================================================",
        "- [주의] 'tesseract_portable' 폴더가 없으면 에러가 발생합니다.",
        "- [에러] 'No cameras available': Setup.exe를 먼저 실행해 저장하세요.",
        "- [에러] 숫자가 튐: Setup.exe에서 ROI 박스를 숫자에 딱 맞게 줄이세요.",
        "- [참고] 이메일이 안 오면 setx 명령어를 다시 확인하세요."
    ]

    # --- 텍스트 그리기 ---
    y = height - 60 * mm
    line_height = 6 * mm

    c.setFont(font_body, 11)
    
    for line in content:
        # 제목 강조
        if line.startswith("="):
            c.setFont(font_title, 10)
        elif line.startswith("[") or line.startswith("1.") or line.startswith("2."):
            c.setFont(font_title, 12)
            y -= 2 * mm
        else:
            c.setFont(font_body, 11)
            
        c.drawString(20 * mm, y, line)
        y -= line_height
        
        # 페이지 넘김 처리
        if y < 20 * mm:
            c.showPage()
            c.setFont(font_body, 11)
            y = height - 20 * mm

    c.save()
    print(f"\n[성공] PDF 파일이 생성되었습니다: {os.path.abspath(filename)}")

if __name__ == "__main__":
    create_manual_pdf()