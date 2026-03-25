# make_manual.py
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import mm
import os

def create_manual_pdf(filename="Vacuum_Monitor_Manual.pdf"):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    # 1. 한글 폰트 등록 (Windows 맑은고딕 기준)
    # 폰트가 없다면 기본 폰트로 대체되거나 에러가 날 수 있음
    font_path = "C:/Windows/Fonts/malgun.ttf"
    if os.path.exists(font_path):
        pdfmetrics.registerFont(TTFont("Malgun", font_path))
        font_name = "Malgun"
        font_bold = "Malgun" # 편의상 동일 폰트 사용
    else:
        print("한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
        font_name = "Helvetica"
        font_bold = "Helvetica-Bold"

    # 타이틀
    c.setFont(font_bold, 24)
    c.drawString(20 * mm, height - 30 * mm, "Vacuum Monitor System v2.0 매뉴얼")
    
    c.setFont(font_name, 10)
    c.drawString(20 * mm, height - 40 * mm, "작성일: 2026.01.26 | 대상: 관리자 및 사용자")
    
    # 본문 시작 위치
    y = height - 60 * mm
    line_height = 6 * mm

    content = [
        "[1. 시스템 개요]",
        "본 시스템은 듀얼 카메라를 이용해 진공 게이지 수치를 자동 인식(OCR)하고,",
        "이상 발생 시 이메일 알림을 보내는 자동화 모니터링 도구입니다.",
        "",
        "[2. 설치 및 폴더 구성]",
        "- 설치가 필요 없는 포터블(Portable) 방식입니다.",
        "- VacuumSetup.exe: 초기 설정용",
        "- VacuumMonitor.exe: 감시 실행용",
        "- tesseract_portable/: OCR 엔진 (삭제 금지)",
        "",
        "[3. 초기 설정 방법]",
        "Step 1: 윈도우 CMD창에서 이메일 계정 설정 (최초 1회)",
        "  > setx VACUUM_SMTP_USER \"이메일주소\"",
        "  > setx VACUUM_SMTP_PASS \"앱비밀번호\"",
        "",
        "Step 2: VacuumSetup.exe 실행",
        "  - [1]번 키: 전체 화면에서 게이지 위치 잡기 (Crop)",
        "  - [2]번 키 -> [N]키: 숫자 영역 박스 그리기 (ROI)",
        "  - [S]번 키: 저장 후 종료",
        "",
        "[4. 시스템 로직]",
        "- Sampling: 1초 간격으로 게이지 촬영",
        "- Anomaly Check: 최근 1분 평균 대비 10배 이상 급변 시 경고",
        "- Auto Cleanup: 30일 지난 로그 및 사진 자동 삭제",
        "",
        "[5. 주의 사항]",
        "- 실행 파일 옆에 반드시 'tesseract_portable' 폴더가 있어야 합니다.",
        "- 카메라 연결이 끊기면 프로그램을 재시작해야 합니다.",
        "- 조명 반사가 심하면 인식률이 떨어질 수 있습니다."
    ]

    c.setFont(font_name, 11)
    
    for line in content:
        if line.startswith("["):
            c.setFont(font_bold, 14)
            y -= 4 * mm
        else:
            c.setFont(font_name, 11)
        
        c.drawString(20 * mm, y, line)
        y -= line_height
        
        if y < 20 * mm: # 페이지 넘김
            c.showPage()
            c.setFont(font_name, 11)
            y = height - 20 * mm

    c.save()
    print(f"PDF 생성 완료: {filename}")

if __name__ == "__main__":
    create_manual_pdf()