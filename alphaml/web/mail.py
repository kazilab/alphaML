"""
Outgoing email over SMTP, for password-reset links. Configured only through the environment (the same
SMTP settings as the U-MIMIC workbench, with an ALPHAML_ prefix). The SMTP password is never logged or returned.
"""
import logging
import os
import smtplib
import ssl
from dataclasses import dataclass, field
from email.message import EmailMessage

logger = logging.getLogger('alphaml.web')

SECURITY_MODES = ('starttls', 'ssl', 'none')  # starttls: port 587, ssl: port 465, none: a local relay


@dataclass(frozen=True)
class MailSettings:
    host: str = ''
    port: int = 587
    security: str = 'starttls'
    username: str = ''
    password: str = field(default='', repr=False)
    sender: str = ''

    @classmethod
    def from_env(cls):
        settings = cls(host=os.environ.get('ALPHAML_SMTP_HOST', '').strip(),
                       port=int(os.environ.get('ALPHAML_SMTP_PORT') or 587),
                       security=os.environ.get('ALPHAML_SMTP_SECURITY', 'starttls').strip().lower() or 'starttls',
                       username=os.environ.get('ALPHAML_SMTP_USERNAME', '').strip(),
                       password=os.environ.get('ALPHAML_SMTP_PASSWORD', ''),
                       sender=os.environ.get('ALPHAML_SMTP_FROM', '').strip())
        if settings.security not in SECURITY_MODES:
            raise ValueError(f"ALPHAML_SMTP_SECURITY must be one of {', '.join(SECURITY_MODES)}")
        return settings

    @property
    def configured(self):
        return bool(self.host and self.sender)


def send_mail(settings, to, subject, body):
    """Send one plain-text message. Returns whether the server accepted it."""
    message = EmailMessage()
    message['From'] = settings.sender
    message['To'] = to
    message['Subject'] = subject
    message.set_content(body)
    context = ssl.create_default_context()
    try:
        if settings.security == 'ssl':
            server = smtplib.SMTP_SSL(settings.host, settings.port, context=context, timeout=20)
        else:
            server = smtplib.SMTP(settings.host, settings.port, timeout=20)
        with server:
            if settings.security == 'starttls':
                server.starttls(context=context)
            if settings.username:
                server.login(settings.username, settings.password)
            server.send_message(message)
        return True
    except (OSError, smtplib.SMTPException) as exc:
        # The exception text can echo server replies; log only its type, never the password
        logger.error(f'Could not send mail via the SMTP server {settings.host}: {type(exc).__name__}')
        return False
