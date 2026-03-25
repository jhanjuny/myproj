# apps/vacuum_monitor/alerts/email_smtp.py
from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText
from email.utils import formatdate
from typing import Any, Dict, List


def _as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(v).strip() for v in x if str(v).strip()]
    s = str(x).strip()
    return [s] if s else []


def send_email_smtp(cfg: Dict[str, Any], body: str) -> None:
    """
    cfg["alerts"]["email"] 설정을 읽어서 SMTP로 메일을 발송합니다.

    필수(권장) YAML 키:
      alerts:
        email:
          enabled: true
          to: ["recipient@example.com"]
          from: "sender@example.com"         # Gmail 사용 시 로그인 계정과 동일 권장
          subject_prefix: "[VACUUM ALERT]"
          smtp_host: "smtp.gmail.com"
          smtp_port: 587
          starttls: true
          username_env: "VACUUM_SMTP_USER"
          password_env: "VACUUM_SMTP_PASS"

    동작 원칙:
    - enabled=false면 아무것도 하지 않고 return
    - username/password가 비어 있으면 즉시 ValueError로 터뜨려 원인을 명확히 함
    - starttls=true면 EHLO -> STARTTLS -> EHLO 후 login
    - Gmail은 From 주소가 로그인 계정과 다르면 거절/스팸 위험이 커서 기본적으로 From을 username으로 강제
    """

    alerts_cfg = cfg.get("alerts", {}) if isinstance(cfg, dict) else {}
    email_cfg = (alerts_cfg.get("email", {}) or {}) if isinstance(alerts_cfg, dict) else {}

    enabled = bool(email_cfg.get("enabled", False))
    if not enabled:
        return

    to_list = _as_list(email_cfg.get("to", []))
    if not to_list:
        raise ValueError("alerts.email.to is empty (no recipients).")

    smtp_host = str(email_cfg.get("smtp_host", "")).strip()
    smtp_port = int(email_cfg.get("smtp_port", 0) or 0)
    if not smtp_host or smtp_port <= 0:
        raise ValueError(f"SMTP server not configured: smtp_host={smtp_host!r}, smtp_port={smtp_port!r}")

    starttls = bool(email_cfg.get("starttls", True))
    smtp_ssl = bool(email_cfg.get("smtp_ssl", False))  # 필요하면 465에서 사용

    user_env = str(email_cfg.get("username_env", "VACUUM_SMTP_USER"))
    pass_env = str(email_cfg.get("password_env", "VACUUM_SMTP_PASS"))

    username = os.getenv(user_env, "").strip()
    password = os.getenv(pass_env, "").strip()

    # App Password를 "fjly jrtn ..." 처럼 공백 포함으로 넣는 경우가 잦아서 제거
    password = password.replace(" ", "")

    if not username or not password:
        raise ValueError(
            "SMTP credentials missing. "
            f"Set env vars {user_env} and {pass_env}. "
            f"(username_len={len(username)}, password_len={len(password)})"
        )

    subject_prefix = str(email_cfg.get("subject_prefix", "[VACUUM ALERT]")).strip()
    subject_suffix = str(email_cfg.get("subject_suffix", "vacuum monitor")).strip()
    subject = f"{subject_prefix} {subject_suffix}".strip()

    from_addr_cfg = str(email_cfg.get("from", "")).strip()
    # Gmail은 from이 login 계정과 다르면 거절/스팸 가능성이 높아 from을 username으로 강제(가장 안전)
    force_from_login = bool(email_cfg.get("force_from_login", True))
    from_addr = username if force_from_login else (from_addr_cfg or username)

    # 메시지 작성
    msg = MIMEText(body, _charset="utf-8")
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_list)
    msg["Date"] = formatdate(localtime=True)
    msg["Subject"] = subject

    debug = bool(email_cfg.get("debug", False))

    if smtp_ssl:
        # SSL 직접 연결(주로 465)
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=20) as s:
            if debug:
                s.set_debuglevel(1)
            s.ehlo()
            s.login(username, password)
            s.sendmail(from_addr, to_list, msg.as_string())
        return

    # 일반 SMTP(주로 587) + STARTTLS
    with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as s:
        if debug:
            s.set_debuglevel(1)

        s.ehlo()
        if starttls:
            s.starttls()
            s.ehlo()

        s.login(username, password)
        s.sendmail(from_addr, to_list, msg.as_string())
