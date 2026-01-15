"""
Weaviate Vector Store Service
Handles embedding, indexing, and semantic retrieval of conversation chunks.
"""

import uuid
import logging
from datetime import datetime
from typing import Optional, Any

import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Configure, Property, DataType
from langchain_weaviate import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from app.core.config import settings
from app.models.schemas import ParsedMessage, ConversationChunk

logger = logging.getLogger(__name__)

COLLECTION_PREFIX = "WhatsAppChat"


def _get_collection_name(session_id: str) -> str:
    safe_id = session_id.replace("-", "_")
    return f"{COLLECTION_PREFIX}_{safe_id}"


def _get_embeddings():
    """Get the embedding model (always use OpenAI embeddings for Weaviate)."""
    if not settings.openai_api_key:
        raise ValueError(
            "OpenAI API key is required for embeddings (used with Weaviate). "
            "Please set OPENAI_API_KEY in your .env file."
        )
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


def _get_weaviate_client() -> weaviate.WeaviateClient:
    """Connect to Weaviate instance."""
    connect_kwargs: dict[str, Any] = {}
    
    if settings.weaviate_api_key:
        auth = weaviate.auth.AuthApiKey(api_key=settings.weaviate_api_key)
        connect_kwargs["auth_credentials"] = auth
    
    # Parse URL for host/port
    url = settings.weaviate_url.rstrip("/")
    if url.startswith("http://"):
        host = url[7:]
        port = 8080
        if ":" in host:
            host, p = host.rsplit(":", 1)
            port = int(p)
        client = weaviate.connect_to_local(host=host, port=port, **connect_kwargs)
    elif url.startswith("https://"):
        host = url[8:]
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url, **connect_kwargs
        )
    else:
        client = weaviate.connect_to_local(**connect_kwargs)
    
    return client


def chunk_messages(
    messages: list[ParsedMessage],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> list[ConversationChunk]:
    """Split messages into overlapping chunks for indexing."""
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    if not messages:
        return []
    
    sorted_msgs = sorted(messages, key=lambda m: m.timestamp)
    chunks: list[ConversationChunk] = []
    
    i = 0
    while i < len(sorted_msgs):
        chunk_msgs = sorted_msgs[i: i + chunk_size]
        if not chunk_msgs:
            break
        
        # Build text representation
        text_lines = []
        for msg in chunk_msgs:
            ts = msg.timestamp.strftime("%Y-%m-%d %H:%M")
            if msg.is_media:
                text_lines.append(f"[{ts}] {msg.sender}: [media]")
            elif msg.is_deleted:
                text_lines.append(f"[{ts}] {msg.sender}: [deleted message]")
            else:
                text_lines.append(f"[{ts}] {msg.sender}: {msg.content}")
        
        text_content = "\n".join(text_lines)
        senders = list({m.sender for m in chunk_msgs})
        
        chunk = ConversationChunk(
            chunk_id=str(uuid.uuid4()),
            messages=chunk_msgs,
            start_time=chunk_msgs[0].timestamp,
            end_time=chunk_msgs[-1].timestamp,
            senders=senders,
            text_content=text_content,
        )
        chunks.append(chunk)
        
        step = chunk_size - chunk_overlap
        i += max(1, step)
    
    return chunks


def index_conversation(session_id: str, messages: list[ParsedMessage]) -> int:
    """
    Chunk and index all messages into Weaviate.
    Returns number of chunks indexed.
    """
    chunks = chunk_messages(messages)
    if not chunks:
        return 0
    
    embeddings = _get_embeddings()
    collection_name = _get_collection_name(session_id)
    
    docs = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk.text_content,
            metadata={
                "chunk_id": chunk.chunk_id,
                "session_id": session_id,
                "start_time": chunk.start_time.isoformat(),
                "end_time": chunk.end_time.isoformat(),
                "senders": ", ".join(chunk.senders),
                "message_count": len(chunk.messages),
            }
        )
        docs.append(doc)
    
    with _get_weaviate_client() as client:
        # Delete existing collection if it exists (re-upload)
        try:
            client.collections.delete(collection_name)
        except Exception:
            pass
        
        WeaviateVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            client=client,
            index_name=collection_name,
        )
    
    logger.info(f"Indexed {len(chunks)} chunks for session {session_id}")
    return len(chunks)


def retrieve_relevant_chunks(
    session_id: str,
    query: str,
    k: int = None,
) -> list[Document]:
    """
    Retrieve the most semantically relevant chunks for a query.
    """
    k = k or settings.max_retrieved_chunks
    embeddings = _get_embeddings()
    collection_name = _get_collection_name(session_id)
    
    with _get_weaviate_client() as client:
        vectorstore = WeaviateVectorStore(
            client=client,
            index_name=collection_name,
            embedding=embeddings,
            text_key="text",
        )
        docs = vectorstore.similarity_search(query, k=k)
    
    return docs


def delete_session(session_id: str) -> bool:
    """Delete all indexed data for a session."""
    collection_name = _get_collection_name(session_id)
    try:
        with _get_weaviate_client() as client:
            client.collections.delete(collection_name)
        return True
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        return False
