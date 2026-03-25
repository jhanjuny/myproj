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
    cfg['alerts']['email'] 설정을 읽어서 SMTP로 메일을 보냅니다.
    - 인증정보는 YAML에 직접 넣지 않고 ENV에서 읽습니다.
    - Gmail(smtp.gmail.com:587)은 STARTTLS 후 로그인해야 합니다.
    - Gmail에서 From은 보통 로그인 계정과 동일해야 안정적이어서 username으로 고정합니다.
    """
    alerts_cfg = cfg.get('alerts', {}) if isinstance(cfg, dict) else {}
    email_cfg = (alerts_cfg.get('email', {}) or {}) if isinstance(alerts_cfg, dict) else {}

    if not bool(email_cfg.get('enabled', False)):
        return

    to_list = _as_list(email_cfg.get('to', []))
    if not to_list:
        raise ValueError('alerts.email.to is empty')

    smtp_host = str(email_cfg.get('smtp_host', '')).strip()
    smtp_port = int(email_cfg.get('smtp_port', 0) or 0)
    starttls = bool(email_cfg.get('starttls', True))

    user_env = str(email_cfg.get('username_env', 'VACUUM_SMTP_USER'))
    pass_env = str(email_cfg.get('password_env', 'VACUUM_SMTP_PASS'))
    username = os.getenv(user_env, '').strip()
    password = os.getenv(pass_env, '').strip().replace(' ', '')

    if not username or not password:
        raise ValueError(
            f'SMTP credentials missing. Set env {user_env}/{pass_env}. '
            f'(username_len={len(username)}, password_len={len(password)})'
        )

    subject_prefix = str(email_cfg.get('subject_prefix', '[VACUUM ALERT]')).strip()
    subject_suffix = str(email_cfg.get('subject_suffix', 'vacuum monitor')).strip()
    subject = f'{subject_prefix} {subject_suffix}'.strip()

    # Gmail: From은 로그인 계정으로 고정(중요)
    from_addr = username

    msg = MIMEText(body, _charset='utf-8')
    msg['From'] = from_addr
    msg['To'] = ', '.join(to_list)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    debug = bool(email_cfg.get('debug', False))

    with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as s:
        if debug:
            s.set_debuglevel(1)

        s.ehlo()
        if starttls:
            s.starttls()
            s.ehlo()

        s.login(username, password)
        s.sendmail(from_addr, to_list, msg.as_string())
