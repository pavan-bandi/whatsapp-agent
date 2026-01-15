"""
WhatsApp AI Chat Intelligence Agent — Backend
FastAPI application entry point.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes import router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="WhatsApp AI Chat Intelligence Agent",
    description=(
        "An AI system that deeply understands exported WhatsApp conversations "
        "and acts as an intelligent analyst for relationships, friendships, "
        "communication behavior, and conversational history."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router)


@app.on_event("startup")
async def startup():
    logger.info("=" * 60)
    logger.info("WhatsApp AI Chat Intelligence Agent starting up")
    logger.info(f"Available providers: {settings.available_providers}")
    logger.info(f"RAG (Weaviate): {'enabled' if settings.has_openai else 'disabled (no OpenAI key)'}")
    logger.info(f"Weaviate URL: {settings.weaviate_url}")
    logger.info("=" * 60)
    
    if not settings.available_providers:
        logger.warning(
            "WARNING: No AI providers configured! "
            "Please set ANTHROPIC_API_KEY and/or OPENAI_API_KEY in your .env file."
        )


@app.get("/")
async def root():
    return {
        "name": "WhatsApp AI Chat Intelligence Agent",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "providers": settings.available_providers,
    }
