# GOAT (Generative Online Agentic Tool) - An Advanced AI Research Agent
# Version 4.0: Ultra-Features & Hyper-Engineered Prompts

# --- Core Imports ---
import os
import asyncio
import httpx
import random
import uuid
import io
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- FastAPI and Related Imports ---
from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# --- External Service & API Integrations ---
from serpapi import GoogleSearch
import arxiv

# --- Application Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="GOAT - The Superior Research Agent",
    description="An AI agent with a multi-pass reasoning engine for advanced research.",
    version="4.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# --- API Keys & Environment Variables ---
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")


# --- Pydantic Models ---
class Citation(BaseModel):
    title: str
    url: Optional[str] = None
    source_type: str = "Web"
    year: Optional[int] = None

class VerificationMetrics(BaseModel):
    confidence_score: float
    bias_score: float
    sources_cross_checked: int
    reasoning_trace: str # Ultra-Feature: Explains the reasoning behind the scores.

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    mode: str
    content: str
    citations: List[Citation]
    verification: VerificationMetrics
    timestamp: str
    share_id: Optional[str] = None
    follow_ups: List[str]

# --- LLM Configuration & Core Functions ---
LLMS = {
    "primary": { # For advanced reasoning, synthesis, and creative tasks
        "url": f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={GEMINI_KEY}",
        "headers": lambda: {"Content-Type": "application/json"},
        "payload": lambda p: {"contents": [{"parts": [{"text": p}]}]},
        "extract": lambda j: j.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Error: No content found")
    },
    "technical": { # For classification, critique, and structured data tasks
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "headers": lambda: {"Authorization": f"Bearer {OPENROUTER_KEY}"},
        "payload": lambda p: {"model": "deepseek/deepseek-coder", "messages": [{"role": "user", "content": p}]},
        "extract": lambda j: j["choices"][0]["message"]["content"]
    }
}

async def call_api(url: str, headers: dict, payload: dict, timeout: int = 90):
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            res = await client.post(url, headers=headers, json=payload)
            res.raise_for_status()
            return res.json()
        except httpx.HTTPStatusError as e:
            logging.error(f"API call failed with status {e.response.status_code}: {e.response.text}")
        except Exception as e:
            logging.error(f"API call error: {e}")
    return None

async def ask_llm(model: str, prompt: str) -> str:
    if not GEMINI_KEY and not OPENROUTER_KEY: return "Error: API keys not set."
    cfg = LLMS.get(model)
    if not cfg: return f"Error: Model '{model}' not configured."
    data = await call_api(cfg["url"], cfg["headers"](), cfg["payload"](prompt))
    return cfg["extract"](data) if data else f"Error: Failed to get response from {model}."


# --- Data Fetching Service ---
class FetchService:
    @staticmethod
    async def search_sources(query: str) -> (str, List[Citation]):
        logging.info(f"Fetching sources for query: {query}")
        
        async def fetch_web():
            if not SERPAPI_KEY: return "Web search disabled.", []
            try:
                search = GoogleSearch({"api_key": SERPAPI_KEY, "q": query, "num": 5})
                results = search.get_dict().get("organic_results", [])
                web_sources = "\n".join([f"Source Title: {r.get('title')}\nURL: {r.get('link')}\nSnippet: {r.get('snippet', '')}" for r in results if r.get('snippet')])
                web_citations = [Citation(title=r.get('title'), url=r.get('link'), source_type="Web") for r in results]
                return web_sources, web_citations
            except Exception as e:
                logging.error(f"SerpAPI search failed: {e}")
                return "", []

        async def fetch_arxiv():
            try:
                search = arxiv.Search(query=query, max_results=3, sort_by=arxiv.SortCriterion.Relevance)
                results = list(search.results())
                arxiv_sources = "\n".join([f"Paper Title: {r.title}\nURL: {r.pdf_url}\nPublished: {r.published.date()}\nAbstract: {r.summary}" for r in results])
                arxiv_citations = [Citation(title=r.title, url=r.pdf_url, source_type="arXiv", year=r.published.year) for r in results]
                return arxiv_sources, arxiv_citations
            except Exception as e:
                logging.error(f"ArXiv search failed: {e}")
                return "", []
        
        web_task = fetch_web()
        arxiv_task = fetch_arxiv()
        (web_sources, web_citations), (arxiv_sources, arxiv_citations) = await asyncio.gather(web_task, arxiv_task)
        
        all_sources_text = f"--- WEB SOURCES ---\n{web_sources}\n\n--- ARXIV SOURCES ---\n{arxiv_sources}"
        all_citations = web_citations + arxiv_citations
        return all_sources_text, all_citations


# --- Ultra-Features: The Multi-pass Reasoning Engine & Prompts ---

# The hyper-engineered prompts are stored as constants for clarity and maintainability.
QUERY_ANALYSIS_PROMPT = """
You are a dispatch agent responsible for routing research queries. Analyze the user's query and classify it into one of three categories:
1.  'Simple_RAG': For straightforward, fact-based questions (e.g., "What is the boiling point of nitrogen?").
2.  'Standard_Agentic': For broad summary requests that require searching multiple sources but likely have no major contradictions (e.g., "Summarize the history of the Eiffel Tower.").
3.  'Deep_Reasoning': For complex, controversial, niche, or very recent topics that require in-depth analysis, timeline establishment, and resolution of conflicting sources (e.g., analyzing a new scientific discovery or a nuanced geopolitical event).
Respond with ONLY the category name ('Simple_RAG', 'Standard_Agentic', or 'Deep_Reasoning').
USER QUERY: "{query}"
"""

SYNTHESIS_PROMPT = """
You are the Skeptic-in-Chief, a world-class research analyst renowned for your ability to cut through noise and find the ground truth. Your task is to produce a definitive intelligence brief based on the provided sources in response to the user's query.
USER QUERY: "{query}"
SOURCES:
{sources_text}
INSTRUCTIONS:
1.  **Evidentiary Hierarchy:** You must weigh the provided sources. Give the highest weight to peer-reviewed papers and reports from major institutions. Give lower weight to pre-prints, press releases, and blog posts. Note any sources that seem unreliable.
2.  **Synthesize the Timeline of Truth:** Do not simply summarize each source. Create a single, cohesive narrative that explains how the understanding of this topic has evolved.
3.  **Find the Consensus & Explain Deviations:** Clearly state the current expert consensus. If there are contradictory claims, do not ignore them. Instead, explain *why* they likely exist (e.g., "Initial reports claimed X, but this was later disproven by study Y which found that confounding factor Z was responsible.").
4.  **Key Insights First:** Begin your report with a 2-3 bullet point "Executive Summary" that distills the most critical conclusions.
5.  **Cite Inline:** Cite sources immediately after the information they support, using the format: (Source: https://www.oregon.gov/odot/dmv/pages/vehicle/titlereg.aspx).
Produce the intelligence brief.
"""

CRITIQUE_PROMPT = """
You are an adversarial "Red Team" analyst. Your sole purpose is to find the weakest point in the following DRAFT REPORT. Your critique must be aggressive, intelligent, and aimed at exposing any potential flaw.
DRAFT REPORT: "{draft_report}"
ADVERSARIAL TASKS:
1.  **Identify the Weakest Link:** Find the single most poorly supported claim or logical leap in the draft. State it clearly.
2.  **Propose an Alternative Hypothesis:** Based on the information, propose a plausible alternative interpretation or explanation that the draft fails to consider.
3.  **Stress-Test the Conclusion:** Challenge the main conclusion. What new piece of information could completely invalidate it?
Provide your Red Team critique in a concise, hard-hitting format. If the draft is logically airtight, state "Red Team analysis concludes the report is robust against adversarial critique."
"""

REVISION_PROMPT = """
You are a research editor. Revise the following DRAFT REPORT using the provided CRITIQUE to produce a final, high-quality version. Seamlessly integrate the revisions without explicitly stating 'the critique said...' Enhance the narrative flow and intellectual rigor of the final piece.
DRAFT REPORT: "{draft_report}"
CRITIQUE: "{critique}"
Produce the FINAL REPORT.
"""

VERIFICATION_PROMPT = """
Analyze the provided research report. Based on its content and the fact it has been cross-referenced against sources, provide:
1.  A confidence score (0.0-1.0) on its factual accuracy.
2.  A bias score (0.0 for neutral, 1.0 for highly biased).
3.  A brief 'Reasoning Trace' (1-2 sentences) explaining your scores. Example: "Confidence is high due to consensus in sources, but a slight optimistic bias is noted."
Respond in the format: CONFIDENCE | BIAS | TRACE
REPORT: "{content}"
"""

FOLLOW_UP_PROMPT = """
You are a research strategist. Based on the key conclusions of the FINAL REPORT below, generate exactly three follow-up questions designed to push the boundaries of the research. Each question must come from a different intellectual vector.
FINAL REPORT: "{content}"
QUESTION VECTORS:
1.  **The Foundational Challenge:** A question that challenges a core assumption or methodology revealed in the report.
2.  **The Unorthodox Connection:** A question that connects the report's findings to a seemingly unrelated field or technology to spark innovation.
3.  **The Ethical Frontier:** A question that explores a complex, unresolved ethical or societal dilemma stemming from the report's conclusions.
Return a Python list of strings.
"""

class ReasoningEngine:
    @staticmethod
    async def run_deep_reasoning(query: str):
        logging.info("Deep Reasoning workflow initiated.")
        # PASS 1: Fetch and pre-process sources
        sources_text, citations = await FetchService.search_sources(query)
        if not sources_text.strip(): return "Could not find sufficient sources to conduct research.", []

        # PASS 2: Contradiction-Aware Synthesis
        draft_report = await ask_llm("primary", SYNTHESIS_PROMPT.format(query=query, sources_text=sources_text))
        
        # PASS 3: Reflective Self-Correction
        critique = await ask_llm("technical", CRITIQUE_PROMPT.format(draft_report=draft_report))
        
        # PASS 4: Final Revision
        if "No significant weaknesses found" not in critique:
            final_content = await ask_llm("primary", REVISION_PROMPT.format(draft_report=draft_report, critique=critique))
        else:
            final_content = draft_report

        return final_content, citations

    @staticmethod
    async def run_simple_rag(query: str):
        # A simpler workflow for basic queries
        logging.info("Simple RAG workflow initiated.")
        sources_text, citations = await FetchService.search_sources(query)
        prompt = f"Using the provided sources, directly answer the user's query.\n\nSOURCES:\n{sources_text}\n\nQUERY:\n{query}"
        content = await ask_llm("primary", prompt)
        return content, citations

    @staticmethod
    async def run_verification_and_insights(content: str, citations: list) -> (VerificationMetrics, List[str]):
        # This sub-process generates the final metadata for the response
        verification_task = ask_llm("technical", VERIFICATION_PROMPT.format(content=content[:3000]))
        follow_up_task = ask_llm("primary", FOLLOW_UP_PROMPT.format(content=content[:3000]))
        
        verification_str, follow_ups_str = await asyncio.gather(verification_task, follow_up_task)

        try:
            conf, bias, trace = verification_str.split('|')
            metrics = VerificationMetrics(confidence_score=float(conf), bias_score=float(bias), sources_cross_checked=len(citations), reasoning_trace=trace.strip())
        except:
            metrics = VerificationMetrics(confidence_score=0.85, bias_score=0.1, sources_cross_checked=len(citations), reasoning_trace="Default scores assigned due to parsing error.")

        try:
            follow_ups = eval(follow_ups_str)
        except:
            follow_ups = ["What is the most significant unaddressed limitation of this research?"]

        return metrics, follow_ups


# --- Main FastAPI Endpoint ---

@app.post("/research", response_model=QueryResponse)
@limiter.limit("10/minute")
async def research_endpoint(req: QueryRequest, request: Request):
    logging.info(f"Received query: '{req.query}' from {request.client.host}")
    
    # Ultra-Feature: Dynamic Workflow Selection
    query_type = await ask_llm("technical", QUERY_ANALYSIS_PROMPT.format(query=req.query))
    query_type = query_type.strip()
    logging.info(f"Query classified as: {query_type}")

    if query_type == "Deep_Reasoning":
        final_content, citations = await ReasoningEngine.run_deep_reasoning(req.query)
    elif query_type == "Standard_Agentic": # Falls back to Deep Reasoning for this implementation
        final_content, citations = await ReasoningEngine.run_deep_reasoning(req.query)
    else: # Simple_RAG
        final_content, citations = await ReasoningEngine.run_simple_rag(req.query)
    
    # Final pass to generate metadata
    verification, follow_ups = await ReasoningEngine.run_verification_and_insights(final_content, citations)
    
    return QueryResponse(
        mode=query_type,
        content=final_content,
        citations=citations,
        verification=verification,
        timestamp=datetime.utcnow().isoformat(),
        share_id=str(uuid.uuid4())[:8],
        follow_ups=follow_ups
    )

@app.get("/health", summary="Health Check")
async def health_check():
    return {"status": f"GOAT v{app.version} is operational", "timestamp": datetime.utcnow().isoformat()}

# Mount static files (for a web UI)
app.mount("/", StaticFiles(directory=".", html=True), name="static")