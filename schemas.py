from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime
from enum import Enum


class ModelProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class ConfidenceLevel(str, Enum):
    FACT = "fact"
    INFERENCE = "inference"
    SPECULATION = "speculation"


# ── Message models ─────────────────────────────────────────────

class ParsedMessage(BaseModel):
    id: str
    timestamp: datetime
    sender: str
    content: str
    is_media: bool = False
    is_deleted: bool = False
    word_count: int = 0
    char_count: int = 0
    has_emoji: bool = False
    emojis: list[str] = []
    reply_to: Optional[str] = None


class ConversationChunk(BaseModel):
    chunk_id: str
    messages: list[ParsedMessage]
    start_time: datetime
    end_time: datetime
    senders: list[str]
    summary: str = ""
    text_content: str = ""


# ── Upload & session models ─────────────────────────────────────

class UploadResponse(BaseModel):
    session_id: str
    message: str
    participants: list[str]
    total_messages: int
    date_range: dict[str, str]
    stats_preview: dict[str, Any]


class SessionInfo(BaseModel):
    session_id: str
    participants: list[str]
    total_messages: int
    date_range: dict[str, str]
    created_at: datetime
    is_indexed: bool = False


# ── Query models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    session_id: str
    question: str
    model_provider: ModelProvider = ModelProvider.ANTHROPIC
    model_name: Optional[str] = None
    include_evidence: bool = True


class EvidenceItem(BaseModel):
    quote: str
    sender: str
    timestamp: str
    relevance: str


class ConfidencedClaim(BaseModel):
    claim: str
    confidence: ConfidenceLevel
    evidence: list[EvidenceItem] = []


class QueryResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    confidence_breakdown: list[ConfidencedClaim] = []
    retrieved_context_count: int = 0
    model_used: str
    processing_time_ms: int
    follow_up_suggestions: list[str] = []


# ── Analysis models ─────────────────────────────────────────────

class ParticipantStats(BaseModel):
    name: str
    message_count: int
    word_count: int
    avg_message_length: float
    most_used_emojis: list[dict[str, Any]]
    avg_response_time_minutes: Optional[float]
    messages_by_hour: dict[str, int]
    messages_by_month: dict[str, int]
    initiation_count: int  # how many conversations they started


class ConversationStats(BaseModel):
    total_messages: int
    total_words: int
    date_range: dict[str, str]
    participants: list[str]
    participant_stats: list[ParticipantStats]
    most_active_hour: str
    most_active_day: str
    most_active_month: str
    avg_daily_messages: float
    longest_silence_days: float
    top_topics: list[str]
    media_count: int
    deleted_count: int
    messages_by_month: dict[str, int]
    conversation_phases: list[dict[str, Any]]


class StatsResponse(BaseModel):
    session_id: str
    stats: ConversationStats


# ── Model info ──────────────────────────────────────────────────

class ModelInfo(BaseModel):
    provider: str
    model_id: str
    display_name: str
    description: str


class AvailableModelsResponse(BaseModel):
    providers: list[str]
    models: list[ModelInfo]
    default_provider: str
    default_model: str
