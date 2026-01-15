"""
WhatsApp Chat Parser
Supports exports from Android and iOS, multiple date formats and locales.
"""

import re
import uuid
from datetime import datetime
from typing import Optional
from app.models.schemas import ParsedMessage

# ── Regex patterns for different WhatsApp export formats ───────────────────────

# Android: [DD/MM/YYYY, HH:MM:SS] Sender: Message
# iOS:     [DD/MM/YYYY, HH:MM:SS] Sender: Message  (same)
# Some locales use 12h: [DD/MM/YYYY, HH:MM:SS AM/PM]
# Some use dashes:      DD-MM-YYYY HH:MM:SS

DATE_PATTERNS = [
    # [DD/MM/YYYY, HH:MM:SS AM/PM] Sender: Message  (iOS, 12h)
    r"^\[(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)\]\s+(.+?):\s(.+)$",
    # DD/MM/YYYY, HH:MM - Sender: Message  (Android)
    r"^(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)\s*-\s*(.+?):\s(.+)$",
    # M/D/YY, H:MM AM/PM - Sender: Message (US format Android)
    r"^(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}),\s+(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))\s*-\s*(.+?):\s(.+)$",
]

DATE_FORMATS = [
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%y %H:%M:%S",
    "%d/%m/%y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%y %H:%M:%S",
    "%m/%d/%y %H:%M",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M",
    "%d/%m/%Y %I:%M:%S %p",
    "%d/%m/%Y %I:%M %p",
    "%m/%d/%Y %I:%M %p",
    "%m/%d/%y %I:%M %p",
    "%d/%m/%y %I:%M %p",
]

SYSTEM_MESSAGE_PATTERNS = [
    r"messages and calls are end-to-end encrypted",
    r"<media omitted>",
    r"this message was deleted",
    r"you deleted this message",
    r"missed voice call",
    r"missed video call",
    r"changed the subject",
    r"added you",
    r"left",
    r"changed their phone number",
    r"changed this group",
    r"security code changed",
    r"tap to learn more",
    r"voice call",
    r"video call",
    r"joined using this group",
    r"created group",
    r"changed the group",
    r"removed",
    r"you're now an admin",
    r"disappearing messages",
]

MEDIA_PATTERNS = [
    r"<media omitted>",
    r"\.(jpg|jpeg|png|gif|mp4|mov|avi|mp3|ogg|pdf|doc|docx|xlsx|zip)\s*\(file attached\)",
    r"image omitted",
    r"video omitted",
    r"audio omitted",
    r"document omitted",
    r"sticker omitted",
    r"gif omitted",
]

DELETED_PATTERNS = [
    r"this message was deleted",
    r"you deleted this message",
    r"message deleted",
]


def _parse_datetime(date_str: str, time_str: str) -> Optional[datetime]:
    """Try multiple date formats to parse the timestamp."""
    combined = f"{date_str.strip()} {time_str.strip()}"
    # Normalize separators
    combined = combined.replace("-", "/")
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(combined, fmt)
        except ValueError:
            continue
    return None


def _extract_emojis(text: str) -> list[str]:
    """Extract emoji characters from text."""
    try:
        import emoji
        return [c for c in text if c in emoji.EMOJI_DATA]
    except ImportError:
        # Fallback: basic emoji range detection
        emojis = []
        for char in text:
            cp = ord(char)
            if (0x1F300 <= cp <= 0x1F9FF or
                    0x2600 <= cp <= 0x26FF or
                    0x2700 <= cp <= 0x27BF or
                    0xFE00 <= cp <= 0xFE0F):
                emojis.append(char)
        return emojis


def _is_system_message(text: str) -> bool:
    text_lower = text.lower().strip()
    for pattern in SYSTEM_MESSAGE_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def _is_media_message(text: str) -> bool:
    text_lower = text.lower().strip()
    for pattern in MEDIA_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def _is_deleted_message(text: str) -> bool:
    text_lower = text.lower().strip()
    for pattern in DELETED_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def parse_whatsapp_export(raw_text: str) -> list[ParsedMessage]:
    """
    Parse a WhatsApp exported chat text into structured messages.
    Handles multi-line messages and various export formats.
    """
    lines = raw_text.splitlines()
    messages: list[ParsedMessage] = []
    
    current_msg: Optional[dict] = None
    
    def finalize_message():
        if current_msg is None:
            return
        content = current_msg["content"].strip()
        if not content:
            return
        if _is_system_message(content):
            return

        is_media = _is_media_message(content)
        is_deleted = _is_deleted_message(content)
        emojis = _extract_emojis(content)
        words = content.split() if not is_media else []

        messages.append(ParsedMessage(
            id=str(uuid.uuid4()),
            timestamp=current_msg["timestamp"],
            sender=current_msg["sender"].strip(),
            content=content,
            is_media=is_media,
            is_deleted=is_deleted,
            word_count=len(words),
            char_count=len(content),
            has_emoji=len(emojis) > 0,
            emojis=emojis,
        ))

    for line in lines:
        line = line.strip()
        if not line:
            continue

        matched = False
        for pattern in DATE_PATTERNS:
            m = re.match(pattern, line, re.IGNORECASE)
            if m:
                # Save previous message
                finalize_message()

                groups = m.groups()
                date_str, time_str, sender, content = groups[0], groups[1], groups[2], groups[3]
                
                timestamp = _parse_datetime(date_str, time_str)
                if timestamp is None:
                    # Could not parse date, treat as continuation
                    if current_msg:
                        current_msg["content"] += f"\n{line}"
                    break

                current_msg = {
                    "timestamp": timestamp,
                    "sender": sender,
                    "content": content,
                }
                matched = True
                break

        if not matched and current_msg is not None:
            # Continuation of multi-line message
            current_msg["content"] += f"\n{line}"

    # Don't forget the last message
    finalize_message()
    
    return messages


def get_participants(messages: list[ParsedMessage]) -> list[str]:
    """Get unique participants sorted by message count."""
    from collections import Counter
    counter = Counter(m.sender for m in messages)
    return [name for name, _ in counter.most_common()]


def get_date_range(messages: list[ParsedMessage]) -> dict[str, str]:
    if not messages:
        return {}
    sorted_msgs = sorted(messages, key=lambda m: m.timestamp)
    return {
        "start": sorted_msgs[0].timestamp.strftime("%B %d, %Y"),
        "end": sorted_msgs[-1].timestamp.strftime("%B %d, %Y"),
        "start_iso": sorted_msgs[0].timestamp.isoformat(),
        "end_iso": sorted_msgs[-1].timestamp.isoformat(),
    }
