"""Tests for statistical analysis service."""

import pytest
from datetime import datetime
from app.services.parser import parse_whatsapp_export
from app.services.analyzer import compute_stats, format_stats_for_context


SAMPLE_CHAT = """1/1/2024, 10:00 AM - Alice: Good morning!
1/1/2024, 10:05 AM - Bob: Morning! How are you?
1/1/2024, 10:06 AM - Alice: Great! Excited for today
1/1/2024, 10:10 AM - Bob: Me too! What time are we meeting?
1/1/2024, 10:11 AM - Alice: 3pm works for me 👍
1/1/2024, 10:12 AM - Bob: Perfect, see you then!
2/1/2024, 9:00 AM - Alice: Hey did you make it home okay?
2/1/2024, 9:30 AM - Bob: Yes! Thanks for asking 😊
2/1/2024, 9:31 AM - Alice: Great, I had such a good time
2/1/2024, 9:35 AM - Bob: Same! We should do it again soon"""


def test_compute_stats_basic():
    messages = parse_whatsapp_export(SAMPLE_CHAT)
    stats = compute_stats(messages)
    
    assert stats.total_messages == 10
    assert "Alice" in stats.participants
    assert "Bob" in stats.participants


def test_participant_stats():
    messages = parse_whatsapp_export(SAMPLE_CHAT)
    stats = compute_stats(messages)
    
    alice = next(p for p in stats.participant_stats if p.name == "Alice")
    bob = next(p for p in stats.participant_stats if p.name == "Bob")
    
    assert alice.message_count == 5
    assert bob.message_count == 5
    assert alice.word_count > 0


def test_message_counts_sum():
    messages = parse_whatsapp_export(SAMPLE_CHAT)
    stats = compute_stats(messages)
    
    total = sum(p.message_count for p in stats.participant_stats)
    assert total == stats.total_messages


def test_monthly_breakdown():
    messages = parse_whatsapp_export(SAMPLE_CHAT)
    stats = compute_stats(messages)
    
    assert len(stats.messages_by_month) >= 1


def test_format_stats_for_context():
    messages = parse_whatsapp_export(SAMPLE_CHAT)
    stats = compute_stats(messages)
    
    formatted = format_stats_for_context(stats)
    assert "Alice" in formatted
    assert "Bob" in formatted
    assert "STATISTICS" in formatted


def test_empty_messages_raises():
    with pytest.raises(ValueError):
        compute_stats([])
