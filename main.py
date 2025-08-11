# GOAT (Generative Online Agentic Tool) - An Advanced AI Research Agent
# Version 5.1: Final Corrected Code with Hybrid Engine and Rate Limiting

# --- Core Imports ---
import os
import asyncio
import httpx
import random
import uuid
import io
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Set

# --- FastAPI and Related Imports ---
from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# --- External Service & API Integrations ---
from serpapi import GoogleSearch
import arxiv

# --- Security & Scalability ---
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- Application Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="GOAT - The Superior Research Agent",
    description="An AI agent with a hybrid synthesis engine for unparalleled research.",
    version="5.1.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# --- API Keys & Environment Variables ---
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# --- Custom Exception ---
class LLMAPIError(Exception):
    pass

# --- Pydantic Models ---
class Citation(BaseModel):
    title: str
    url: Optional[str] = None
    source_type: str = "Web"
    year: Optional[int] = None
    
    # To make Citation objects hashable for de-duplication
    def __hash__(self):
        return hash((self.url, self.title))
    def __eq__(self, other):
        return self.url == other.url and self.title == other.title

class VerificationMetrics(BaseModel):
    confidence_score: float
    bias_score: float
    sources_cross_checked: int
    reasoning_trace: str

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
    "primary": { "url": f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={GEMINI_KEY}", "headers": lambda: {"Content-Type": "application/json"}, "payload": lambda p: {"contents": [{"parts": [{"text": p}]}]}, "extract": lambda j: j.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Error: No content found") },
    "technical": { "url": "https://openrouter.ai/api/v1/chat/completions", "headers": lambda: {"Authorization": f"Bearer {OPENROUTER_KEY}"}, "payload": lambda p: {"model": "deepseek/deepseek-r1:free", "messages": [{"role": "user", "content": p}]}, "extract": lambda j: j["choices"][0]["message"]["content"] }
}

async def call_api(url: str, headers: dict, payload: dict, timeout: int = 90):
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            res = await client.post(url, headers=headers, json=payload)
            res.raise_for_status()
            return res.json()
        except httpx.HTTPStatusError as e:
            logging.error(f"API call failed with status {e.response.status_code}: {e.response.text}")
            raise LLMAPIError(f"API Error: Status {e.response.status_code} - {e.response.text}")
        except Exception as e:
            logging.error(f"API call error: {e}")
            raise LLMAPIError(f"Network or other error during API call: {e}")

async def ask_llm(model: str, prompt: str) -> str:
    if not GEMINI_KEY and not OPENROUTER_KEY: raise LLMAPIError("Error: API keys not set.")
    cfg = LLMS.get(model)
    if not cfg: raise LLMAPIError(f"Error: Model '{model}' not configured.")
    data = await call_api(cfg["url"], cfg["headers"](), cfg["payload"](prompt))
    return cfg["extract"](data)


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


# --- Ultra-Features: Prompts for the Reasoning Engine ---
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

ULTRA_HYBRID_SYNTHESIS_PROMPT = """
You are a Master Synthesizer. Your task is to create a single, superior "Hybrid Intelligence Brief" by integrating two different AI-generated reports: a concise 'Simple Answer' and a detailed 'Deep Analysis'.
**Simple Answer (Direct, fact-based):**
{simple_content}
**Deep Analysis (Narrative, in-depth):**
{deep_content}
**Your Instructions:**
1.  **Integrate, Don't Just Append:** Do not simply stack the two reports. Your primary goal is to weave the key facts, statistics, and direct answers from the 'Simple Answer' into the narrative of the 'Deep Analysis' as supporting evidence.
2.  **Preserve Depth:** The final output should retain the narrative flow, critical insights, and logical structure of the 'Deep Analysis'.
3.  **Enhance with Facts:** Use the specific data points from the 'Simple Answer' to make the 'Deep Analysis' more concrete and authoritative.
4.  **Create a New Executive Summary:** Write a new, overarching executive summary at the beginning that reflects the integrated, hybrid nature of the report.
Produce the final, integrated Hybrid Intelligence Brief.
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
        sources_text, citations = await FetchService.search_sources(query)
        if not sources_text.strip(): return "Could not find sufficient sources for deep reasoning.", []

        logging.info("Deep Reasoning - Pass 1: Synthesis")
        draft_report = await ask_llm("primary", SYNTHESIS_PROMPT.format(query=query, sources_text=sources_text))
        
        await asyncio.sleep(2) # Added delay to respect API rate limits

        logging.info("Deep Reasoning - Pass 2: Critique")
        critique = await ask_llm("technical", CRITIQUE_PROMPT.format(draft_report=draft_report))
        
        if "No significant weaknesses found" not in critique:
            await asyncio.sleep(2) # Added delay
            logging.info("Deep Reasoning - Pass 3: Revision")
            final_content = await ask_llm("primary", REVISION_PROMPT.format(draft_report=draft_report, critique=critique))
        else:
            final_content = draft_report
            
        return final_content, citations

    @staticmethod
    async def run_simple_rag(query: str):
        logging.info("Simple RAG workflow initiated.")
        sources_text, citations = await FetchService.search_sources(query)
        if not sources_text.strip(): return "Could not find sufficient sources for simple search.", []
        prompt = f"Using ONLY the provided sources, give a direct, concise, fact-based answer to the user's query. Extract key data points and statistics.\n\nSOURCES:\n{sources_text}\n\nQUERY:\n{query}"
        content = await ask_llm("technical", prompt)
        return content, citations

    @staticmethod
    async def run_verification_and_insights(content: str, citations: list) -> (VerificationMetrics, List[str]):
        verification_task = ask_llm("technical", VERIFICATION_PROMPT.format(content=content[:3000]))
        follow_up_task = ask_llm("primary", FOLLOW_UP_PROMPT.format(content=content[:3000]))
        verification_str, follow_ups_str = await asyncio.gather(verification_task, follow_up_task)
        try:
            conf, bias, trace = verification_str.split('|')
            metrics = VerificationMetrics(confidence_score=float(conf), bias_score=float(bias), sources_cross_checked=len(citations), reasoning_trace=trace.strip())
        except:
            metrics = VerificationMetrics(confidence_score=0.85, bias_score=0.1, sources_cross_checked=len(citations), reasoning_trace="Default scores assigned.")
        try:
            follow_ups = eval(follow_ups_str)
        except:
            follow_ups = ["What is the most significant unaddressed limitation of this research?"]
        return metrics, follow_ups


# --- Main FastAPI Endpoint ---

@app.post("/research", response_model=QueryResponse)
@limiter.limit("5/minute")
async def research_endpoint(req: QueryRequest, request: Request):
    logging.info(f"Received HYBRID query: '{req.query}' from {request.client.host}")
    
    try:
        # Ultra-Feature: Parallel Execution of RAG and Deep Reasoning workflows
        simple_rag_task = asyncio.create_task(ReasoningEngine.run_simple_rag(req.query))
        deep_reasoning_task = asyncio.create_task(ReasoningEngine.run_deep_reasoning(req.query))
        (simple_content, simple_citations), (deep_content, deep_citations) = await asyncio.gather(simple_rag_task, deep_reasoning_task)

        # Merge and de-duplicate citations
        all_citations = list(set(simple_citations + deep_citations))
        
        # Final Pass: Ultra Hybrid Synthesis
        logging.info("Initiating Ultra Hybrid Synthesis pass.")
        final_content = await ask_llm("primary", ULTRA_HYBRID_SYNTHESIS_PROMPT.format(simple_content=simple_content, deep_content=deep_content))

        # Final pass to generate metadata on the hybrid content
        verification, follow_ups = await ReasoningEngine.run_verification_and_insights(final_content, all_citations)
        
        return QueryResponse(
            mode="Hybrid",
            content=final_content,
            citations=all_citations,
            verification=verification,
            timestamp=datetime.utcnow().isoformat(),
            share_id=str(uuid.uuid4())[:8],
            follow_ups=follow_ups
        )
    except LLMAPIError as e:
        logging.error(f"A critical LLM API call failed, terminating workflow. Error: {e}")
        raise HTTPException(status_code=503, detail=f"An upstream AI service is unavailable. {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred in the research endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.get("/health", summary="Health Check")
async def health_check():
    return {"status": f"GOAT v{app.version} is operational", "timestamp": datetime.utcnow().isoformat()}

# Mount static files (for a web UI)
app.mount("/", StaticFiles(directory=".", html=True), name="static")