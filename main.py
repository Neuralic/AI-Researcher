import os, asyncio, json, random, time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
import httpx
from contextlib import asynccontextmanager
from academic_guard import AcademicGuard
from chaos_engine import ChaosEngine

class QueryRequest(BaseModel):
    query: str
    mode_override: Optional[str] = None

class QueryResponse(BaseModel):
    mode: str
    content: str
    citations: Optional[List[Dict]] = None
    chaos_score: Optional[float] = None
    timestamp: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.guards = AcademicGuard()
    app.state.chaos = ChaosEngine()
    yield

app = FastAPI(title="Hybrid Research Agent", lifespan=lifespan)

def select_mode(query: str) -> str:
    if "methodology" in query.lower() or "literature review" in query.lower():
        return "academic"
    elif "disrupt" in query.lower() or "innovative" in query.lower():
        return "chaos"
    else:
        return "hybrid"

async def call_api(url: str, headers: dict, payload: dict, timeout: int = 30) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(3):
            try:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    return resp.json()
                await asyncio.sleep(2**attempt)
            except Exception:
                await asyncio.sleep(2**attempt)
        raise HTTPException(503, "API unavailable")

async def academic_process(query: str) -> QueryResponse:
    guard = app.state.guards
    methodology = await guard.build_methodology(query)
    citations = await guard.validate_citations(query)
    ethics = await guard.ethics_screen(query)
    
    prompt = f"""
    Generate a TRIPOD+AI compliant academic response for: {query}
    Methodology: {methodology}
    Include citations: {[c['title'] for c in citations[:3]]}
    Ethics check: {ethics}
    Format in LaTeX.
    """
    
    headers = {"Authorization": f"Bearer {os.getenv('OPENROUTER_KEY')}"}
    payload = {
        "model": "kimi/k1-8k",
        "messages": [{"role": "user", "content": prompt}]
    }
    
    result = await call_api("https://openrouter.ai/api/v1/chat/completions", headers, payload)
    content = result['choices'][0]['message']['content']
    
    return QueryResponse(
        mode="academic",
        content=content,
        citations=citations,
        timestamp=datetime.utcnow().isoformat()
    )

async def chaos_process(query: str) -> QueryResponse:
    engine = app.state.chaos
    seed = await engine.quantum_seed()
    rdf = engine.reality_distortion(query)
    fused = await engine.adversarial_fusion(query)
    
    prompt = f"""
    Generate a chaos-driven innovation for: {query}
    Quantum seed: {seed}
    RDF score: {rdf}
    Adversarial fusion: {fused}
    Include provocation markers (!!CHAOS!!).
    """
    
    headers = {"Authorization": f"Bearer {os.getenv('GEMINI_KEY')}"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    result = await call_api("https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent", headers, payload)
    content = result['candidates'][0]['content']['parts'][0]['text']
    
    return QueryResponse(
        mode="chaos",
        content=content,
        chaos_score=rdf,
        timestamp=datetime.utcnow().isoformat()
    )

async def hybrid_process(query: str) -> QueryResponse:
    academic_task = academic_process(query)
    chaos_task = chaos_process(query)
    
    academic_res, chaos_res = await asyncio.gather(academic_task, chaos_task)
    
    combined = f"""
    ## Hybrid Analysis (60/40 split)
    
    ### Academic Foundation (60%)
    {academic_res.content[:500]}...
    
    ### Chaos Injection (40%)
    {chaos_res.content[:300]}...
    
    **Synthesis Score**: {(academic_res.citations.__len__() or 0) * 0.6 + (chaos_res.chaos_score or 0) * 0.4}
    """
    
    return QueryResponse(
        mode="hybrid",
        content=combined,
        citations=academic_res.citations,
        chaos_score=chaos_res.chaos_score,
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/research", response_model=QueryResponse)
async def research_endpoint(request: QueryRequest):
    mode = request.mode_override or select_mode(request.query)
    
    if mode == "academic":
        return await academic_process(request.query)
    elif mode == "chaos":
        return await chaos_process(request.query)
    else:
        return await hybrid_process(request.query)

@app.get("/health")
async def health():
    return {"status": "operational", "timestamp": datetime.utcnow().isoformat()}
