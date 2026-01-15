"""
API Routes
- POST /upload          - Upload WhatsApp export
- POST /query           - Ask a question
- GET  /stats/{session} - Get conversation statistics
- GET  /models          - Available AI models
- DELETE /session/{id}  - Delete session
"""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from app.models.schemas import (
    UploadResponse, QueryRequest, QueryResponse,
    StatsResponse, AvailableModelsResponse, ModelInfo,
)
from app.services.parser import parse_whatsapp_export, get_participants, get_date_range
from app.services.analyzer import compute_stats
from app.services.session_store import session_store
from app.services.vectorstore import index_conversation, delete_session
from app.services.agent import run_agent, get_available_models
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat-intelligence"])


# ── Upload ──────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_chat(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Upload a WhatsApp exported .txt file."""
    
    # Validate file
    if not file.filename.endswith(".txt"):
        raise HTTPException(400, "File must be a .txt WhatsApp export")
    
    content = await file.read()
    
    # Check size
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(413, f"File too large. Max {settings.max_file_size_mb}MB")
    
    # Decode
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = content.decode("latin-1")
    
    # Parse
    messages = parse_whatsapp_export(text)
    if len(messages) < 5:
        raise HTTPException(
            422,
            "Could not parse chat file. Please make sure it's a valid WhatsApp export (.txt without media)."
        )
    
    participants = get_participants(messages)
    date_range = get_date_range(messages)
    
    # Compute stats
    try:
        stats = compute_stats(messages)
    except Exception as e:
        logger.error(f"Stats computation failed: {e}")
        raise HTTPException(500, f"Failed to analyze chat: {str(e)}")
    
    # Create session
    session_id = session_store.create_session(
        messages=messages,
        stats=stats,
        participants=participants,
        date_range=date_range,
    )
    
    # Index in background (Weaviate)
    if settings.has_openai:
        background_tasks.add_task(_index_session, session_id, messages)
    else:
        logger.warning("No OpenAI key set — skipping Weaviate indexing. RAG will be unavailable.")
    
    # Stats preview
    stats_preview = {
        "total_messages": stats.total_messages,
        "total_words": stats.total_words,
        "most_active_hour": stats.most_active_hour,
        "most_active_day": stats.most_active_day,
        "avg_daily_messages": stats.avg_daily_messages,
        "participant_breakdown": {
            p.name: {
                "messages": p.message_count,
                "pct": round(p.message_count / stats.total_messages * 100, 1),
            }
            for p in stats.participant_stats
        },
    }
    
    return UploadResponse(
        session_id=session_id,
        message=f"Successfully parsed {len(messages):,} messages between {len(participants)} participants.",
        participants=participants,
        total_messages=len(messages),
        date_range=date_range,
        stats_preview=stats_preview,
    )


async def _index_session(session_id: str, messages):
    """Background task: index session in Weaviate."""
    try:
        count = index_conversation(session_id, messages)
        session_store.mark_indexed(session_id)
        logger.info(f"Session {session_id} indexed: {count} chunks")
    except Exception as e:
        logger.error(f"Indexing failed for session {session_id}: {e}")


# ── Query ────────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_chat(request: QueryRequest):
    """Ask a natural language question about the conversation."""
    
    # Validate session
    session = session_store.get_session(request.session_id)
    if not session:
        raise HTTPException(404, f"Session '{request.session_id}' not found")
    
    # Validate provider
    if request.model_provider.value not in settings.available_providers:
        available = ", ".join(settings.available_providers) or "none"
        raise HTTPException(
            400,
            f"Provider '{request.model_provider.value}' not configured. "
            f"Available: {available}"
        )
    
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    
    stats = session_store.get_stats(request.session_id)
    
    try:
        response = await run_agent(request=request, stats=stats)
        return response
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(500, f"Analysis failed: {str(e)}")


# ── Stats ────────────────────────────────────────────────────────────────────

@router.get("/stats/{session_id}", response_model=StatsResponse)
async def get_stats(session_id: str):
    """Get full statistical analysis for a session."""
    stats = session_store.get_stats(session_id)
    if not stats:
        raise HTTPException(404, f"Session '{session_id}' not found")
    
    return StatsResponse(session_id=session_id, stats=stats)


# ── Models ───────────────────────────────────────────────────────────────────

@router.get("/models", response_model=AvailableModelsResponse)
async def get_models():
    """Get list of available AI models."""
    models_data = get_available_models()
    
    return AvailableModelsResponse(
        providers=settings.available_providers,
        models=[ModelInfo(**m) for m in models_data],
        default_provider=settings.default_model_provider,
        default_model=(
            settings.default_anthropic_model
            if settings.default_model_provider == "anthropic"
            else settings.default_openai_model
        ),
    )


# ── Session management ───────────────────────────────────────────────────────

@router.delete("/session/{session_id}")
async def delete_session_endpoint(session_id: str):
    """Delete a session and its indexed data."""
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(404, f"Session '{session_id}' not found")
    
    session_store.delete_session(session_id)
    
    if settings.has_openai:
        delete_session(session_id)
    
    return {"message": "Session deleted successfully"}


@router.get("/session/{session_id}/status")
async def get_session_status(session_id: str):
    """Check session status including indexing state."""
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(404, f"Session '{session_id}' not found")
    
    return {
        "session_id": session_id,
        "is_indexed": session["is_indexed"],
        "participants": session["participants"],
        "total_messages": len(session["messages"]),
    }


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "providers": settings.available_providers,
        "weaviate_url": settings.weaviate_url,
        "rag_available": settings.has_openai,
    }
