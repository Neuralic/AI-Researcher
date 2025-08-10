# main.py
import os
import asyncio
import json
import httpx
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
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

app = FastAPI(title="Hybrid Research Agent")

# Static frontend files
app.mount("/", StaticFiles(directory=".", html=True), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

# Dual-mode router
def select_mode(query: str) -> str:
    q = query.lower()
    if "methodology" in q or "literature review" in q:
        return "academic"
    if "disrupt" in q or "innovative" in q:
        return "chaos"
    return "hybrid"

# API clients
async def call_api(url: str, headers: dict, payload: dict, timeout: int = 30) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(3):
            try:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    return resp.json()
                await asyncio.sleep(2 ** attempt)
            except Exception:
                await asyncio.sleep(2 ** attempt)
        raise HTTPException(503, "API unavailable")

# Processors
async def academic_process(query: str) -> QueryResponse:
    guard = AcademicGuard()
    methodology = await guard.build_methodology(query)
    citations = await guard.validate_citations(query)
    ethics = await guard.ethics_screen(query)
    prompt = (
        f"Generate a TRIPOD+AI compliant academic response for: {query}\n"
        f"Methodology: {methodology}\n"
        f"Include citations: {[c['title'] for c in citations[:3]]}\n"
        f"Ethics check: {ethics}\n"
        "Format in LaTeX."
    )
    headers = {"Authorization": f"Bearer {os.getenv('OPENROUTER_KEY')}"}
    payload = {"model": "kimi/k1-8k", "messages": [{"role": "user", "content": prompt}]}
    data = await call_api("https://openrouter.ai/api/v1/chat/completions", headers, payload)
    return QueryResponse(
        mode="academic",
        content=data["choices"][0]["message"]["content"],
        citations=citations,
        timestamp=datetime.utcnow().isoformat(),
    )

async def chaos_process(query: str) -> QueryResponse:
    engine = ChaosEngine()
    seed = await engine.quantum_seed()
    rdf = engine.reality_distortion(query)
    fused = await engine.adversarial_fusion(query)
    prompt = (
        f"Generate a chaos-driven innovation for: {query}\n"
        f"Quantum seed: {seed}\n"
        f"RDF score: {rdf}\n"
        f"Adversarial fusion: {fused}\n"
        "Include provocation markers (!!CHAOS!!)."
    )
    headers = {"Authorization": f"Bearer {os.getenv('GEMINI_KEY')}"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    data = await call_api(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        headers,
        payload,
    )
    return QueryResponse(
        mode="chaos",
        content=data["candidates"][0]["content"]["parts"][0]["text"],
        chaos_score=rdf,
        timestamp=datetime.utcnow().isoformat(),
    )

async def hybrid_process(query: str) -> QueryResponse:
    academic_task = academic_process(query)
    chaos_task = chaos_process(query)
    academic_res, chaos_res = await asyncio.gather(academic_task, chaos_task)
    combined = (
        "## Hybrid Analysis (60/40 split)\n\n"
        "### Academic Foundation (60%)\n"
        f"{academic_res.content[:500]}...\n\n"
        "### Chaos Injection (40%)\n"
        f"{chaos_res.content[:300]}...\n\n"
        f"**Synthesis Score**: "
        f"{(len(academic_res.citations or []) * 0.6 + (chaos_res.chaos_score or 0) * 0.4)}"
    )
    return QueryResponse(
        mode="hybrid",
        content=combined,
        citations=academic_res.citations,
        chaos_score=chaos_res.chaos_score,
        timestamp=datetime.utcnow().isoformat(),
    )

# API endpoint
@app.post("/research", response_model=QueryResponse)
async def research_endpoint(request: QueryRequest):
    mode = request.mode_override or select_mode(request.query)
    if mode == "academic":
        return await academic_process(request.query)
    if mode == "chaos":
        return await chaos_process(request.query)
    return await hybrid_process(request.query)

@app.get("/health")
async def health():
    return {"status": "operational", "timestamp": datetime.utcnow().isoformat()}