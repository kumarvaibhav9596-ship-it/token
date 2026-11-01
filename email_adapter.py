"""Local-only mock outbox for demonstrating support email replies."""
from __future__ import annotations

import csv
import re
import threading
from datetime import datetime, timezone
from email.utils import parseaddr
from pathlib import Path


OUTBOX_PATH = Path(__file__).resolve().parent / "tools" / "emails_outbox.csv"
_OUTBOX_LOCK = threading.Lock()


class EmailValidationError(ValueError):
    pass


def validate_email_address(value: str) -> str:
    if not isinstance(value, str) or len(value) > 254 or "\r" in value or "\n" in value:
        raise EmailValidationError("Enter a valid email address.")
    name, address = parseaddr(value.strip())
    if name or address != value.strip() or not re.fullmatch(r"[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+", address):
        raise EmailValidationError("Enter a valid email address.")
    return address


def queue_email(recipient: str, subject: str, body: str) -> None:
    """Write an explicitly approved draft to a local CSV; never contact a mail server."""
    recipient = validate_email_address(recipient)
    if not isinstance(subject, str) or not subject.strip() or len(subject) > 200 or "\r" in subject or "\n" in subject:
        raise EmailValidationError("Email subject is invalid.")
    if not isinstance(body, str) or not body.strip() or len(body) > 8000:
        raise EmailValidationError("Email reply is invalid.")
    OUTBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _OUTBOX_LOCK:
        exists = OUTBOX_PATH.exists() and OUTBOX_PATH.stat().st_size > 0
        with OUTBOX_PATH.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("queued_at_utc", "recipient", "subject", "body", "status"))
            if not exists:
                writer.writeheader()
            writer.writerow({
                "queued_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "recipient": recipient, "subject": subject.strip(), "body": body,
                "status": "MOCK_QUEUED_NOT_SENT",
            })
