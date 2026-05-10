"""
WhatsApp Chat Intelligence Agent
Uses LangChain + LangGraph to reason over conversation history.
Supports Anthropic (Claude) and OpenAI (GPT) models.
"""

import logging
import time
from typing import Any, Optional, Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.documents import Document
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from app.core.config import settings
from app.models.schemas import (
    ModelProvider, QueryRequest, QueryResponse,
    ConfidencedClaim, ConfidenceLevel, EvidenceItem,
)
from app.services.vectorstore import retrieve_relevant_chunks
from app.services.analyzer import ConversationStats, format_stats_for_context

logger = logging.getLogger(__name__)


# ── LangGraph State ─────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    session_id: str
    retrieved_docs: list[Document]
    stats_context: str
    final_answer: str
    retrieved_count: int


# ── System prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert WhatsApp Conversation Intelligence Analyst. You have deeply read and understood a complete WhatsApp conversation history between people and can answer virtually any question about it.

Your role is that of an **AI relationship historian + behavioral analyst + memory engine**.

## Your Capabilities
- Understand relationship dynamics, emotional patterns, and communication behaviors
- Identify phases, turning points, and changes over time
- Detect recurring themes, inside jokes, and communication habits
- Analyze conflict patterns, emotional reactions, and behavioral tendencies
- Provide timeline-based analysis and statistical insights

## Confidence Framework
You MUST classify every substantive claim into one of three confidence levels:

**[FACT]** - Directly verifiable from the chat data (e.g., "Person A sent 1,234 messages")
**[INFERENCE]** - Reasoned from observable patterns with medium confidence (e.g., "Person A appears more emotionally available based on their response patterns and supportive language")
**[SPECULATION]** - Interpretive or uncertain (e.g., "This silence may have been triggered by an external stressor we can't see in the chats")

## Response Guidelines
1. **Be nuanced, not absolute** - Avoid sweeping claims. Say "appears calmer based on X, Y, Z" not "is definitely calmer"
2. **Cite evidence** - Reference specific patterns, time periods, or examples from the chats
3. **Acknowledge limitations** - Be transparent when data is insufficient for confident claims
4. **Be psychologically sophisticated** - Understand that behavior patterns reflect deeper dynamics
5. **Respect both parties** - Maintain objectivity; this is analysis, not judgment
6. **Structure your answer** - Use clear sections for complex questions

## Response Format
For complex behavioral/relationship questions:
- Lead with a nuanced summary
- Break down key observations with confidence tags
- Note what the data does/doesn't show
- Suggest related angles the user might want to explore

For factual/statistical questions:
- Lead with the numbers
- Add behavioral context
- Note patterns or anomalies

Always end with 2-3 follow-up question suggestions that would deepen the analysis.

Remember: The goal is to make the person feel "this AI truly understands the entire history of this relationship."
"""

RETRIEVAL_PROMPT = """You are helping retrieve the most relevant context from a WhatsApp conversation to answer a user's question.

The question is: {question}

Based on this question, what specific aspects of the conversation are most relevant? Consider:
- Time periods mentioned
- People's names
- Emotional states or behaviors
- Specific events or topics
- Communication patterns

Provide a search-optimized version of the question that will retrieve the best context chunks.
"""

# ── LLM Factory ─────────────────────────────────────────────────────────────

def get_llm(provider: ModelProvider, model_name: Optional[str] = None) -> BaseChatModel:
    """Get the appropriate LLM based on provider."""
    if provider == ModelProvider.ANTHROPIC:
        if not settings.anthropic_api_key:
            raise ValueError("Anthropic API key not configured")
        model = model_name or settings.default_anthropic_model
        return ChatAnthropic(
            model=model,
            api_key=settings.anthropic_api_key,
            max_tokens=4096,
            temperature=0.3,
        )
    elif provider == ModelProvider.OPENAI:
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        model = model_name or settings.default_openai_model
        return ChatOpenAI(
            model=model,
            api_key=settings.openai_api_key,
            max_tokens=4096,
            temperature=0.3,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ── LangGraph Nodes ──────────────────────────────────────────────────────────

def retrieve_context_node(state: AgentState) -> dict:
    """Retrieve relevant conversation chunks from Weaviate."""
    question = state["question"]
    session_id = state["session_id"]
    
    try:
        docs = retrieve_relevant_chunks(
            session_id=session_id,
            query=question,
            k=settings.max_retrieved_chunks,
        )
        logger.info(f"Retrieved {len(docs)} chunks for question: {question[:50]}...")
        return {
            "retrieved_docs": docs,
            "retrieved_count": len(docs),
        }
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return {"retrieved_docs": [], "retrieved_count": 0}


def analyze_node(state: AgentState, llm: BaseChatModel) -> dict:
    """Generate the final analysis using retrieved context + stats."""
    question = state["question"]
    docs = state.get("retrieved_docs", [])
    stats_context = state.get("stats_context", "")
    
    # Build context from retrieved docs
    retrieved_context = ""
    if docs:
        retrieved_context = "\n\n=== RELEVANT CONVERSATION EXCERPTS ===\n"
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            start = meta.get("start_time", "")[:10]
            end = meta.get("end_time", "")[:10]
            retrieved_context += f"\n--- Excerpt {i} ({start} to {end}) ---\n"
            retrieved_context += doc.page_content
            retrieved_context += "\n"
    
    # Combine all context
    full_context = ""
    if stats_context:
        full_context += stats_context + "\n\n"
    if retrieved_context:
        full_context += retrieved_context
    
    if not full_context:
        full_context = "No conversation data available for this session."
    
    # Build messages for the LLM
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""Here is the conversation data for analysis:

{full_context}

---

User's Question: {question}

Please provide a thorough, nuanced analysis. Remember to:
1. Use [FACT], [INFERENCE], and [SPECULATION] tags for your claims
2. Cite specific evidence from the excerpts
3. Be psychologically sophisticated and nuanced
4. End with 2-3 follow-up question suggestions""")
    ]
    
    response = llm.invoke(messages)
    
    return {
        "final_answer": response.content,
        "messages": [HumanMessage(content=question), response],
    }


def parse_follow_ups(answer: str) -> list[str]:
    """Extract follow-up questions from the answer text."""
    import re
    
    # Look for numbered follow-up questions or bullet points at the end
    patterns = [
        r'\d+\.\s*"([^"]+)"',
        r'\d+\.\s*\*\*([^*]+)\*\*',
        r'[-•]\s*"([^"]+)"',
        r'Follow.up[^:]*:\s*(.+?)(?:\n|$)',
    ]
    
    questions = []
    for pattern in patterns:
        matches = re.findall(pattern, answer, re.IGNORECASE)
        questions.extend(matches)
        if len(questions) >= 3:
            break
    
    # Fallback: look for question marks
    if not questions:
        lines = answer.split('\n')
        for line in reversed(lines[-15:]):  # Check last 15 lines
            line = line.strip().strip('•-123456789. *"')
            if '?' in line and len(line) > 15 and len(line) < 150:
                questions.append(line)
            if len(questions) >= 3:
                break
    
    return questions[:3]


# ── Main Agent Function ──────────────────────────────────────────────────────

def build_agent_graph(llm: BaseChatModel):
    """Build the LangGraph agent workflow."""
    
    # Bind LLM to analyze node
    def analyze_with_llm(state: AgentState) -> dict:
        return analyze_node(state, llm)
    
    builder = StateGraph(AgentState)
    builder.add_node("retrieve", retrieve_context_node)
    builder.add_node("analyze", analyze_with_llm)
    
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "analyze")
    builder.add_edge("analyze", END)
    
    return builder.compile()


async def run_agent(
    request: QueryRequest,
    stats: Optional[ConversationStats] = None,
) -> QueryResponse:
    """
    Main entry point: run the full agent pipeline for a user query.
    """
    start_time = time.time()
    
    # Get LLM
    llm = get_llm(request.model_provider, request.model_name)
    model_name = request.model_name or (
        settings.default_anthropic_model
        if request.model_provider == ModelProvider.ANTHROPIC
        else settings.default_openai_model
    )
    
    # Format stats for context
    stats_context = format_stats_for_context(stats) if stats else ""
    
    # Build and run graph
    graph = build_agent_graph(llm)
    
    initial_state: AgentState = {
        "messages": [],
        "question": request.question,
        "session_id": request.session_id,
        "retrieved_docs": [],
        "stats_context": stats_context,
        "final_answer": "",
        "retrieved_count": 0,
    }
    
    result = graph.invoke(initial_state)
    
    elapsed_ms = int((time.time() - start_time) * 1000)
    final_answer = result.get("final_answer", "I could not generate an answer.")
    follow_ups = parse_follow_ups(final_answer)
    
    return QueryResponse(
        session_id=request.session_id,
        question=request.question,
        answer=final_answer,
        confidence_breakdown=[],  # Could be parsed from answer tags
        retrieved_context_count=result.get("retrieved_count", 0),
        model_used=f"{request.model_provider.value}/{model_name}",
        processing_time_ms=elapsed_ms,
        follow_up_suggestions=follow_ups,
    )


def get_available_models() -> list[dict]:
    """Return list of available models based on configured API keys."""
    models = []
    
    if settings.has_anthropic:
        models.extend([
            {
                "provider": "anthropic",
                "model_id": "claude-opus-4-20250514",
                "display_name": "Claude Opus 4",
                "description": "Most capable — best for deep relationship analysis",
            },
            {
                "provider": "anthropic",
                "model_id": "claude-sonnet-4-20250514",
                "display_name": "Claude Sonnet 4",
                "description": "Balanced — excellent analysis with fast responses",
            },
            {
                "provider": "anthropic",
                "model_id": "claude-haiku-4-5-20251001",
                "display_name": "Claude Haiku 4.5",
                "description": "Fast — quick answers and simple queries",
            },
        ])
    
    if settings.has_openai:
        models.extend([
            {
                "provider": "openai",
                "model_id": "gpt-4o",
                "display_name": "GPT-4o",
                "description": "OpenAI's flagship — powerful multi-modal analysis",
            },
            {
                "provider": "openai",
                "model_id": "gpt-4o-mini",
                "display_name": "GPT-4o Mini",
                "description": "Fast & affordable — great for quick questions",
            },
        ])
    
    return models
