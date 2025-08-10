# main.py  –  GOAT-edition  (4 LLMs, chaos dialled to 11)
import os, asyncio, httpx, random, json
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

app = FastAPI(title="AI Research Agent 🐐")

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
    return {"status": "GOAT", "timestamp": datetime.utcnow().isoformat()}

# ---------- static ----------
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# ---------- helpers ----------
def select_mode(q: str) -> str:
    q = q.lower()
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

# ---------- multi-LLM chaos mixer ----------
LLMS = {
    "deepseek": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "headers": lambda: {"Authorization": f"Bearer {os.getenv('OPENROUTER_KEY')}"},
        "payload": lambda p: {"model": "deepseek/deepseek-r1:free", "messages": [{"role": "user", "content": p}]},
        "extract": lambda j: j["choices"][0]["message"]["content"],
    },
    "gpt": {
        "url": "https://api.openai.com/v1/chat/completions",
        "headers": lambda: {"Authorization": f"Bearer {os.getenv('OPENAI_KEY')}"},
        "payload": lambda p: {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": p}]},
        "extract": lambda j: j["choices"][0]["message"]["content"],
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        "headers": lambda: {"x-goog-api-key": os.getenv("GEMINI_KEY")},
        "payload": lambda p: {"contents": [{"parts": [{"text": p}]}]},
        "extract": lambda j: j["candidates"][0]["content"]["parts"][0]["text"],
    },
    "deepseek_direct": {
        "url": "https://api.deepseek.com/v1/chat/completions",
        "headers": lambda: {"Authorization": f"Bearer {os.getenv('DEEPSEEK_KEY')}"},
        "payload": lambda p: {"model": "deepseek-chat", "messages": [{"role": "user", "content": p}]},
        "extract": lambda j: j["choices"][0]["message"]["content"],
    },
}

async def ask(llm: str, prompt: str) -> str:
    cfg = LLMS[llm]
    if not cfg["headers"]():            # key not set
        return f"[{llm} disabled]"
    data = await call_api(cfg["url"], cfg["headers"](), cfg["payload"](prompt))
    return cfg["extract"](data)

async def academic_process(query: str) -> QueryResponse:
    parts = await asyncio.gather(
        ask("deepseek", f"TRIPOD+AI paper on: {query}"),
        ask("gpt",      f"Academic methodology for: {query}"),
    )
    content = "\n\n---\n".join(parts)
    return QueryResponse(
        mode="academic",
        content=content,
        citations=[{"title": "Multi-LLM synthesis", "year": 2024}],
        timestamp=datetime.utcnow().isoformat(),
    )

async def chaos_process(query: str) -> QueryResponse:
    parts = await asyncio.gather(
        ask("gemini", f"🔥CHAOS🔥 {query}"),
        ask("deepseek_direct", f"Contradict everything about: {query}"),
    )
    content = "\n\n💥💥💥\n".join(parts).upper()
    return QueryResponse(
        mode="chaos",
        content=content,
        chaos_score=random.randint(90, 100),
        timestamp=datetime.utcnow().isoformat(),
    )

async def hybrid_process(query: str) -> QueryResponse:
    ac, ch = await asyncio.gather(academic_process(query), chaos_process(query))
    return QueryResponse(
        mode="hybrid",
        content=f"{ac.content}\n\n⚡GOAT FUSION⚡\n{ch.content}",
        citations=ac.citations,
        chaos_score=ch.chaos_score,
        timestamp=datetime.utcnow().isoformat(),
    )