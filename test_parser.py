"""
Tests for WhatsApp chat parser.
Run: pytest backend/tests/ -v
"""

import pytest
from datetime import datetime
from app.services.parser import parse_whatsapp_export, get_participants, get_date_range


ANDROID_SAMPLE = """1/1/2024, 10:30 AM - Alice: Hey, how are you?
1/1/2024, 10:31 AM - Bob: I'm good! You?
1/1/2024, 10:32 AM - Alice: Great actually 😊
1/1/2024, 10:33 AM - Alice: We should catch up sometime
1/1/2024, 10:35 AM - Bob: Absolutely! This weekend?
1/1/2024, 10:36 AM - Alice: Perfect 👍
1/2/2024, 9:00 AM - Bob: Good morning!
1/2/2024, 9:05 AM - Alice: Morning ☀️
1/2/2024, 9:06 AM - Bob: Did you sleep well?
1/2/2024, 9:07 AM - Alice: Yes! Best sleep in a while"""

IOS_SAMPLE = """[01/01/2024, 10:30:00 AM] Alice: Hey there!
[01/01/2024, 10:31:00 AM] Bob: Hi!
[01/01/2024, 10:32:00 AM] Alice: What's up?"""

MULTILINE_SAMPLE = """1/1/2024, 10:00 AM - Alice: This is a message
that spans multiple
lines in the chat
1/1/2024, 10:01 AM - Bob: Got it!"""

SYSTEM_MSG_SAMPLE = """1/1/2024, 10:00 AM - Messages and calls are end-to-end encrypted.
1/1/2024, 10:01 AM - Alice: Hello!
1/1/2024, 10:02 AM - Bob: Hi there!"""


def test_android_format_parsing():
    messages = parse_whatsapp_export(ANDROID_SAMPLE)
    assert len(messages) == 10
    assert messages[0].sender == "Alice"
    assert messages[0].content == "Hey, how are you?"
    assert messages[1].sender == "Bob"


def test_ios_format_parsing():
    messages = parse_whatsapp_export(IOS_SAMPLE)
    assert len(messages) == 3
    assert messages[0].sender == "Alice"


def test_emoji_detection():
    messages = parse_whatsapp_export(ANDROID_SAMPLE)
    alice_third = next(m for m in messages if '😊' in m.content)
    assert alice_third.has_emoji
    assert '😊' in alice_third.emojis


def test_multiline_messages():
    messages = parse_whatsapp_export(MULTILINE_SAMPLE)
    assert len(messages) == 2
    assert "multiple" in messages[0].content
    assert "lines" in messages[0].content


def test_system_messages_filtered():
    messages = parse_whatsapp_export(SYSTEM_MSG_SAMPLE)
    assert all(m.sender in ("Alice", "Bob") for m in messages)
    assert len(messages) == 2


def test_get_participants():
    messages = parse_whatsapp_export(ANDROID_SAMPLE)
    participants = get_participants(messages)
    assert "Alice" in participants
    assert "Bob" in participants


def test_date_range():
    messages = parse_whatsapp_export(ANDROID_SAMPLE)
    date_range = get_date_range(messages)
    assert "start" in date_range
    assert "end" in date_range


def test_word_count():
    messages = parse_whatsapp_export(ANDROID_SAMPLE)
    assert messages[0].word_count > 0


def test_empty_input():
    messages = parse_whatsapp_export("")
    assert messages == []


def test_minimal_input():
    minimal = "1/1/2024, 10:00 AM - Alice: Hi\n1/1/2024, 10:01 AM - Bob: Hello"
    messages = parse_whatsapp_export(minimal)
    assert len(messages) == 2
