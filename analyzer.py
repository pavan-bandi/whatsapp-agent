"""
Statistical Analysis Service
Computes objective statistics over the parsed conversation.
"""

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any
import math

from app.models.schemas import ParsedMessage, ParticipantStats, ConversationStats


def _most_common_emojis(messages: list[ParsedMessage], n: int = 10) -> list[dict]:
    counter: Counter = Counter()
    for msg in messages:
        counter.update(msg.emojis)
    return [{"emoji": e, "count": c} for e, c in counter.most_common(n)]


def _response_times(messages: list[ParsedMessage], sender: str) -> list[float]:
    """Calculate average response time in minutes for a sender."""
    times = []
    for i in range(1, len(messages)):
        prev = messages[i - 1]
        curr = messages[i]
        if curr.sender == sender and prev.sender != sender:
            delta = (curr.timestamp - prev.timestamp).total_seconds() / 60
            if 0 < delta < 1440:  # ignore gaps > 24h (they're not responses)
                times.append(delta)
    return times


def _messages_by_hour(messages: list[ParsedMessage]) -> dict[str, int]:
    counter: Counter = Counter()
    for msg in messages:
        hour = msg.timestamp.strftime("%H:00")
        counter[hour] += 1
    return dict(counter)


def _messages_by_month(messages: list[ParsedMessage]) -> dict[str, int]:
    counter: Counter = Counter()
    for msg in messages:
        month = msg.timestamp.strftime("%Y-%m")
        counter[month] += 1
    return dict(sorted(counter.items()))


def _conversation_initiations(messages: list[ParsedMessage]) -> dict[str, int]:
    """Count how many times each person started a new conversation (after 6h gap)."""
    initiations: dict[str, int] = defaultdict(int)
    if not messages:
        return {}
    
    sorted_msgs = sorted(messages, key=lambda m: m.timestamp)
    # First message
    initiations[sorted_msgs[0].sender] += 1
    
    for i in range(1, len(sorted_msgs)):
        gap = (sorted_msgs[i].timestamp - sorted_msgs[i - 1].timestamp).total_seconds() / 3600
        if gap >= 6:  # New conversation after 6h silence
            initiations[sorted_msgs[i].sender] += 1
    
    return dict(initiations)


def _detect_phases(messages: list[ParsedMessage]) -> list[dict[str, Any]]:
    """
    Detect conversation phases based on message frequency.
    Groups messages by month and identifies high/low activity periods.
    """
    if not messages:
        return []
    
    monthly = _messages_by_month(messages)
    if len(monthly) < 2:
        return []
    
    months = sorted(monthly.keys())
    counts = [monthly[m] for m in months]
    
    if not counts:
        return []
    
    mean = sum(counts) / len(counts)
    std = math.sqrt(sum((c - mean) ** 2 for c in counts) / len(counts)) if len(counts) > 1 else 0
    
    phases = []
    current_phase = None
    
    for month, count in zip(months, counts):
        if std > 0:
            z = (count - mean) / std
        else:
            z = 0
        
        if z > 0.5:
            phase_type = "high_activity"
        elif z < -0.5:
            phase_type = "low_activity"
        else:
            phase_type = "normal"
        
        if current_phase is None or current_phase["type"] != phase_type:
            if current_phase:
                phases.append(current_phase)
            current_phase = {
                "type": phase_type,
                "start": month,
                "end": month,
                "avg_messages": count,
                "months": [month],
            }
        else:
            current_phase["end"] = month
            current_phase["months"].append(month)
            current_phase["avg_messages"] = sum(
                monthly[m] for m in current_phase["months"]
            ) / len(current_phase["months"])
    
    if current_phase:
        phases.append(current_phase)
    
    return phases


def _longest_silence(messages: list[ParsedMessage]) -> float:
    """Find the longest gap between messages in days."""
    if len(messages) < 2:
        return 0
    sorted_msgs = sorted(messages, key=lambda m: m.timestamp)
    max_gap = 0.0
    for i in range(1, len(sorted_msgs)):
        gap = (sorted_msgs[i].timestamp - sorted_msgs[i - 1].timestamp).total_seconds() / 86400
        max_gap = max(max_gap, gap)
    return round(max_gap, 1)


def compute_stats(messages: list[ParsedMessage]) -> ConversationStats:
    """Compute full statistical analysis of the conversation."""
    if not messages:
        raise ValueError("No messages to analyze")
    
    sorted_msgs = sorted(messages, key=lambda m: m.timestamp)
    participants = list({m.sender for m in messages})
    
    total_days = max(
        1,
        (sorted_msgs[-1].timestamp - sorted_msgs[0].timestamp).days + 1
    )
    
    monthly_counts = _messages_by_month(messages)
    hourly_counts = _messages_by_hour(messages)
    initiations = _conversation_initiations(messages)
    
    # Most active periods
    most_active_hour = max(hourly_counts, key=hourly_counts.get) if hourly_counts else "N/A"
    most_active_month = max(monthly_counts, key=monthly_counts.get) if monthly_counts else "N/A"
    
    day_counter: Counter = Counter()
    for msg in messages:
        day_counter[msg.timestamp.strftime("%A")] += 1
    most_active_day = day_counter.most_common(1)[0][0] if day_counter else "N/A"
    
    # Per-participant stats
    participant_stats = []
    for p in participants:
        p_messages = [m for m in messages if m.sender == p]
        p_words = sum(m.word_count for m in p_messages)
        p_chars = sum(m.char_count for m in p_messages)
        
        resp_times = _response_times(sorted_msgs, p)
        avg_resp = round(sum(resp_times) / len(resp_times), 1) if resp_times else None
        
        participant_stats.append(ParticipantStats(
            name=p,
            message_count=len(p_messages),
            word_count=p_words,
            avg_message_length=round(p_chars / len(p_messages), 1) if p_messages else 0,
            most_used_emojis=_most_common_emojis(p_messages, 5),
            avg_response_time_minutes=avg_resp,
            messages_by_hour=_messages_by_hour(p_messages),
            messages_by_month=_messages_by_month(p_messages),
            initiation_count=initiations.get(p, 0),
        ))
    
    media_count = sum(1 for m in messages if m.is_media)
    deleted_count = sum(1 for m in messages if m.is_deleted)
    total_words = sum(m.word_count for m in messages)
    
    date_range = {
        "start": sorted_msgs[0].timestamp.strftime("%B %d, %Y"),
        "end": sorted_msgs[-1].timestamp.strftime("%B %d, %Y"),
        "start_iso": sorted_msgs[0].timestamp.isoformat(),
        "end_iso": sorted_msgs[-1].timestamp.isoformat(),
        "days": str(total_days),
    }
    
    return ConversationStats(
        total_messages=len(messages),
        total_words=total_words,
        date_range=date_range,
        participants=participants,
        participant_stats=participant_stats,
        most_active_hour=most_active_hour,
        most_active_day=most_active_day,
        most_active_month=most_active_month,
        avg_daily_messages=round(len(messages) / total_days, 1),
        longest_silence_days=_longest_silence(messages),
        top_topics=[],  # Populated by the agent via LLM
        media_count=media_count,
        deleted_count=deleted_count,
        messages_by_month=monthly_counts,
        conversation_phases=_detect_phases(messages),
    )


def format_stats_for_context(stats: ConversationStats) -> str:
    """Format stats into a compact text summary for LLM context injection."""
    lines = [
        "=== CONVERSATION STATISTICS ===",
        f"Date range: {stats.date_range.get('start')} → {stats.date_range.get('end')} ({stats.date_range.get('days')} days)",
        f"Total messages: {stats.total_messages:,}",
        f"Total words: {stats.total_words:,}",
        f"Avg messages/day: {stats.avg_daily_messages}",
        f"Longest silence: {stats.longest_silence_days} days",
        f"Most active: {stats.most_active_day}s, {stats.most_active_hour}",
        "",
        "=== PER-PARTICIPANT STATS ===",
    ]
    
    for p in stats.participant_stats:
        msg_pct = round(p.message_count / stats.total_messages * 100, 1) if stats.total_messages else 0
        lines.append(f"\n[{p.name}]")
        lines.append(f"  Messages: {p.message_count:,} ({msg_pct}% of total)")
        lines.append(f"  Words: {p.word_count:,}")
        lines.append(f"  Avg message length: {p.avg_message_length} chars")
        if p.avg_response_time_minutes is not None:
            lines.append(f"  Avg response time: {p.avg_response_time_minutes} min")
        lines.append(f"  Conversation initiations: {p.initiation_count}")
        if p.most_used_emojis:
            emoji_str = " ".join(f"{e['emoji']}×{e['count']}" for e in p.most_used_emojis[:5])
            lines.append(f"  Top emojis: {emoji_str}")
    
    if stats.conversation_phases:
        lines.append("\n=== CONVERSATION PHASES ===")
        for ph in stats.conversation_phases:
            lines.append(f"  {ph['start']} → {ph['end']}: {ph['type'].replace('_', ' ')} "
                         f"(avg {ph['avg_messages']:.0f} msg/month)")
    
    return "\n".join(lines)
