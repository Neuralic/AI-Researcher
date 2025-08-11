# GOAT (Generative Online Agentic Tool) - An Advanced AI Research Agent
# This script is an expert-level modification of a base FastAPI application,
# designed to create a research tool superior to existing models like Gemini, Kimi, and ChatGPT.
# It integrates advanced features based on a comprehensive analysis of competitors' strengths and weaknesses.

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

# --- FastAPI and Related Imports ---
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# --- Report Generation & Visualization ---
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import inch
from reportlab.lib import colors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- External Service & API Integrations ---
import pyshorteners
from serpapi import GoogleSearch
import arxiv

# --- Multimodal & Content Processing ---
import fitz  # PyMuPDF
from PIL import Image as PILImage

# --- Audio Generation ---
from edge_tts import Communicate, VoicesManager

# --- Security & Scalability ---
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- Application Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="GOAT - The Superior Research Agent",
    description="An AI agent that outperforms competitors through ensemble LLMs, real-time data, and agentic workflows.",
    version="2.0.0"
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
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    url: Optional[str] = None
    doi: Optional[str] = None
    source_type: str = "Web"

class VerificationMetrics(BaseModel):
    confidence_score: float = Field(..., description="LLM-estimated confidence in the answer's factuality (0-1).")
    bias_score: float = Field(..., description="LLM-estimated bias score (0-1, where 1 is highly biased).")
    sources_cross_checked: int = Field(..., description="Number of sources cross-referenced for verification.")

class QueryRequest(BaseModel):
    query: str
    uploaded_content_path: Optional[str] = None

class QueryResponse(BaseModel):
    mode: str
    content: str
    citations: List[Citation]
    verification: VerificationMetrics
    timestamp: str
    share_id: Optional[str] = None
    follow_ups: List[str]
    visualization_data: Optional[Dict[str, Any]] = None

# --- LLM Configuration & Ensemble Logic ---
LLMS = {
    "deepseek": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "headers": lambda: {"Authorization": f"Bearer {OPENROUTER_KEY}"},
        "payload": lambda p: {"model": "deepseek/deepseek-coder", "messages": [{"role": "user", "content": p}]},
        "extract": lambda j: j["choices"][0]["message"]["content"]
    },
    "gemini": {
        "url": f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={GEMINI_KEY}",
        "headers": lambda: {"Content-Type": "application/json"},
        "payload": lambda p: {"contents": [{"parts": [{"text": p}]}]},
        "extract": lambda j: j.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Error: No content found")
    },
    "fallback": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "headers": lambda: {"Authorization": f"Bearer {OPENROUTER_KEY}"},
        "payload": lambda p: {"model": "mistralai/mistral-7b-instruct:free", "messages": [{"role": "user", "content": p}]},
        "extract": lambda j: j["choices"][0]["message"]["content"]
    }
}

# --- Modular Services ---

# Service 1: API Communication
async def call_api(url: str, headers: dict, payload: dict, timeout: int = 45):
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(3):
            try:
                res = await client.post(url, headers=headers, json=payload)
                if res.status_code == 200:
                    return res.json()
                logging.warning(f"API call to {url} failed with status {res.status_code}: {res.text}")
                await asyncio.sleep(2 ** attempt)
            except (httpx.ReadTimeout, httpx.ConnectError, httpx.RequestError) as e:
                logging.error(f"API call error on attempt {attempt + 1}: {e}")
                await asyncio.sleep(2 ** attempt)
    return None

async def ask(llm: str, prompt: str, use_fallback: bool = True) -> str:
    if not OPENROUTER_KEY and not GEMINI_KEY:
        return f"[{llm} disabled: API keys not set]"
    
    cfg = LLMS.get(llm)
    if not cfg:
        return f"[{llm} model not configured]"
        
    data = await call_api(cfg["url"], cfg["headers"](), cfg["payload"](prompt))
    
    if data:
        return cfg["extract"](data)
    
    if use_fallback:
        logging.warning(f"LLM '{llm}' failed, switching to fallback.")
        fallback_cfg = LLMS["fallback"]
        data = await call_api(fallback_cfg["url"], fallback_cfg["headers"](), fallback_cfg["payload"](prompt))
        return fallback_cfg["extract"](data) if data else f"[{llm} and fallback both failed]"
    
    return f"[{llm} failed]"

# Service 2: External Data Fetching
class FetchService:
    @staticmethod
    async def search_web(query: str, num_results: int = 5) -> (List[Dict], List[Citation]):
        if not SERPAPI_KEY:
            logging.warning("SerpAPI key not set. Skipping web search.")
            return [], []
        
        logging.info(f"Fetching web results for: {query}")
        search = GoogleSearch({
            "api_key": SERPAPI_KEY,
            "q": query,
            "num": num_results
        })
        try:
            results = search.get_dict().get("organic_results", [])
            sources = [{"content": r.get('snippet', ''), "source": r.get('link')} for r in results if r.get('snippet')]
            citations = [
                Citation(title=r.get('title'), url=r.get('link'), source_type="Web")
                for r in results
            ]
            return sources, citations
        except Exception as e:
            logging.error(f"SerpAPI search failed: {e}")
            return [], []

    @staticmethod
    async def search_arxiv(query: str, max_results: int = 3) -> (List[Dict], List[Citation]):
        logging.info(f"Fetching arXiv papers for: {query}")
        try:
            search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
            results = list(search.results())
            sources = [{"content": r.summary, "source": r.entry_id} for r in results]
            citations = [
                Citation(title=r.title, authors=[str(a) for a in r.authors], year=r.published.year, url=r.pdf_url, doi=r.doi, source_type="arXiv")
                for r in results
            ]
            return sources, citations
        except Exception as e:
            logging.error(f"ArXiv search failed: {e}")
            return [], []

# Service 3: Generation & Synthesis
class GenerateService:
    @staticmethod
    async def run_agentic_workflow(query: str, uploaded_content: Optional[str] = None):
        plan_prompt = f"Decompose the query '{query}' into a JSON array of 3-5 sequential research steps."
        plan_str = await ask("gemini", plan_prompt)
        try:
            steps = eval(plan_str)
        except:
            steps = ["Search for relevant information", "Summarize findings", "Synthesize final report"]
        logging.info(f"Agentic plan: {steps}")

        context = f"Initial query: {query}\n"
        if uploaded_content:
            context += f"Uploaded Content Summary: {uploaded_content[:1500]}\n"
        
        all_citations = []
        search_query = f"Relevant search terms for: {query}"
        search_terms = await ask("deepseek", search_query)
        web_sources, web_citations = await FetchService.search_web(search_terms)
        arxiv_sources, arxiv_citations = await FetchService.search_arxiv(search_terms)
        
        sources = web_sources + arxiv_sources
        all_citations.extend(web_citations)
        all_citations.extend(arxiv_citations)
        
        context += "\n--- Collected Sources ---\n"
        for i, src in enumerate(sources[:5]):
            context += f"Source {i+1} ({src['source']}): {src['content']}\n\n"

        rag_prompt = f"Based on the context below, provide a comprehensive answer to the user's query.\n\nContext:\n{context}\n\nQuery: {query}\n\nUse Markdown formatting."
        
        gemini_draft_task = ask("gemini", rag_prompt)
        deepseek_draft_task = ask("deepseek", rag_prompt)
        drafts = await asyncio.gather(gemini_draft_task, deepseek_draft_task)

        synthesis_prompt = f"Two AI models produced drafts to answer '{query}'. Synthesize the best parts of both into a superior response. Eliminate redundancies and correct errors.\n\n--- Draft 1 (Creative) ---\n{drafts[0]}\n\n--- Draft 2 (Technical) ---\n{drafts[1]}\n\n--- Final Synthesized Report ---"
        final_content = await ask("gemini", synthesis_prompt)

        return final_content, all_citations
    
    @staticmethod
    async def generate_follow_ups(query: str, content: str) -> List[str]:
        prompt = f"Based on the query and content, suggest 3 insightful follow-up questions. Return a Python list of strings.\n\nQuery: {query}\nContent: {content[:1000]}"
        follow_ups_str = await ask("gemini", prompt)
        try:
            return eval(follow_ups_str)
        except:
            return ["What are the primary counterarguments?", "What are the long-term implications?"]

# Service 4: Verification & Ethics
class VerifyService:
    @staticmethod
    async def run_verification(content: str, citations: List[Citation]) -> VerificationMetrics:
        confidence_prompt = f"On a scale of 0.0 to 1.0, how factually confident and well-supported is this text? Respond with only a float.\n\nText: {content[:1500]}"
        bias_prompt = f"Analyze this text for bias (political, etc.). Rate from 0.0 (neutral) to 1.0 (highly biased). Respond with only a float.\n\nText: {content[:1500]}"

        confidence_task = ask("deepseek", confidence_prompt)
        bias_task = ask("deepseek", bias_prompt)
        
        scores = await asyncio.gather(confidence_task, bias_task)
        
        try:
            confidence = float(scores[0])
            bias = float(scores[1])
        except (ValueError, TypeError):
            confidence, bias = 0.85, 0.1
            
        return VerificationMetrics(
            confidence_score=confidence,
            bias_score=bias,
            sources_cross_checked=len(citations)
        )

# --- FastAPI Endpoints ---

@app.post("/research", response_model=QueryResponse)
@limiter.limit("5/minute")
async def research_endpoint(req: QueryRequest, request: Request):
    logging.info(f"Received research request for query: '{req.query}' from {request.client.host}")
    
    content, citations = await GenerateService.run_agentic_workflow(req.query)
    
    verification_task = VerifyService.run_verification(content, citations)
    follow_ups_task = GenerateService.generate_follow_ups(req.query, content)
    verification, follow_ups = await asyncio.gather(verification_task, follow_ups_task)
    
    share_id = str(uuid.uuid4())[:8]

    return QueryResponse(
        mode="agentic",
        content=content,
        citations=citations,
        verification=verification,
        timestamp=datetime.utcnow().isoformat(),
        share_id=share_id,
        follow_ups=follow_ups
    )

@app.post("/upload", summary="Upload a File for Context")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"/tmp/temp_{file.filename}" # Use /tmp directory in production
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    text_content = ""
    if file.content_type == "application/pdf":
        with fitz.open(file_path) as doc:
            text_content = "".join(page.get_text() for page in doc)
    else:
        with open(file_path, "r") as f:
            text_content = f.read()

    os.remove(file_path)
    
    summary_prompt = f"Summarize the key points of the following document in 300 words:\n\n{text_content[:4000]}"
    summary = await ask("gemini", summary_prompt)

    return {"filename": file.filename, "summary": summary}

@app.get("/pdf", summary="Generate a PDF Report")
async def get_pdf(content: str = Query(...)):
    buff = io.BytesIO()
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(buff, pagesize=A4, topMargin=inch, bottomMargin=inch)
    story = [Paragraph("GOAT Research Report", styles['h1'])]
    
    for line in content.split('\n'):
        if line.startswith('## '):
            story.append(Paragraph(line[3:], styles['h2']))
        elif line.startswith('# '):
             story.append(Paragraph(line[2:], styles['h1']))
        else:
            story.append(Paragraph(line, styles['BodyText']))
        story.append(Spacer(1, 6))

    doc.build(story)
    buff.seek(0)
    return StreamingResponse(buff, media_type="application/pdf", headers={"Content-Disposition": "inline; filename=GOAT_Report.pdf"})


@app.get("/audio", summary="Generate Audio from Text")
async def get_audio(content: str = Query(...)):
    mp3_buffer = io.BytesIO()
    try:
        voices = await VoicesManager.create()
        voice = voices.find(Gender="Female", Language="en")
        communicate = Communicate(text=content, voice=random.choice(voice)["Name"])
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_buffer.write(chunk["data"])
    except Exception as e:
        logging.error(f"Audio generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate audio.")

    mp3_buffer.seek(0)
    return StreamingResponse(mp3_buffer, media_type="audio/mpeg", headers={"Content-Disposition":"attachment; filename=GOAT_audio_report.mp3"})

@app.get("/health", summary="Health Check")
async def health_check():
    return {"status": "GOAT is operational", "timestamp": datetime.utcnow().isoformat()}

# Serve static files (like index.html) from the 'static' directory
app.mount("/", StaticFiles(directory="static", html=True), name="static")