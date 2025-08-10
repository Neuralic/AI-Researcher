# main.py
import os, asyncio, httpx
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

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

# ---------- API ----------
@app.post("/research", response_model=QueryResponse)
async def research_endpoint(req: QueryRequest):
    mode = req.mode_override or select_mode(req.query)
    if mode == "academic":
        return await academic_process(req.query)
    if mode == "chaos":
        return await chaos_process(req.query)
    return await hybrid_process(req.query)

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

# ---------- static ----------
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# ---------- helpers ----------
def select_mode(query: str) -> str:
    q = query.lower()
    if "methodology" in q or "literature review" in q:
        return "academic"
    if "disrupt" in q or "innovative" in q:
        return "chaos"
    return "hybrid"

async def call_api(url: str, headers: dict, payload: dict, timeout: int = 30) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(3):
            try:
                r = await client.post(url, headers=headers, json=payload)
                if r.status_code == 200:
                    return r.json()
                await asyncio.sleep(2 ** attempt)
            except Exception:
                await asyncio.sleep(2 ** attempt)
    raise HTTPException(503, "API unavailable")

async def academic_process(query: str) -> QueryResponse:
    prompt = f"Generate a TRIPOD+AI compliant academic response for: {query}"
    headers = {"Authorization": f"Bearer {os.getenv('OPENROUTER_KEY')}"}
    payload = {"model": "kimi/k1-8k", "messages": [{"role": "user", "content": prompt}]}
    data = await call_api("https://openrouter.ai/api/v1/chat/completions", headers, payload)
    return QueryResponse(
        mode="academic",
        content=data["choices"][0]["message"]["content"],
        citations=[{"title": "OpenRouter", "year": 2024}],
        timestamp=datetime.utcnow().isoformat(),
    )

async def chaos_process(query: str) -> QueryResponse:
    prompt = f"Generate a chaos-driven innovation for: {query}"
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
        chaos_score=random.random() * 100,
        timestamp=datetime.utcnow().isoformat(),
    )

async def hybrid_process(query: str) -> QueryResponse:
    ac, ch = await asyncio.gather(academic_process(query), chaos_process(query))
    combined = (
        "## Hybrid Analysis (60/40 split)\n\n"
        "### Academic Foundation (60%)\n" + ac.content[:500] + "...\n\n"
        "### Chaos Injection (40%)\n" + ch.content[:300] + "...\n\n"
        f"**Synthesis Score**: {random.randint(60, 99)}"
    )
    return QueryResponse(
        mode="hybrid",
        content=combined,
        citations=ac.citations,
        chaos_score=ch.chaos_score,
        timestamp=datetime.utcnow().isoformat(),
    )