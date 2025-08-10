# main.py  –  GOAT edition (fixed keys & models)
import os, asyncio, httpx, random
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

app = FastAPI(title="GOAT Research Agent")

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

@app.get("/diagnose")
async def diagnose():
    checks = {}
    for name, cfg in LLMS.items():
        if not cfg["headers"]():
            checks[name] = "KEY_MISSING"
            continue
        try:
            data = await call_api(cfg["url"], cfg["headers"](), cfg["payload"]("ping"))
            checks[name] = "OK"
        except Exception as e:
            checks[name] = str(e)[:60]
    return checks

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

# ---------- LLM map (fixed models & headers) ----------
LLMS = {
    "deepseek": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "headers": lambda: {"Authorization": f"Bearer {os.getenv('OPENROUTER_KEY')}"} if os.getenv("OPENROUTER_KEY") else None,
        "payload": lambda p: {"model": "deepseek/deepseek-r1:free", "messages": [{"role": "user", "content": p}]},
        "extract": lambda j: j["choices"][0]["message"]["content"],
    },
    "gpt": {
        "url": "https://api.openai.com/v1/chat/completions",
        "headers": lambda: {"Authorization": f"Bearer {os.getenv('OPENAI_KEY')}"} if os.getenv("OPENAI_KEY") else None,
        "payload": lambda p: {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": p}]},
        "extract": lambda j: j["choices"][0]["message"]["content"],
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        "headers": lambda: {"x-goog-api-key": os.getenv("GEMINI_KEY")} if os.getenv("GEMINI_KEY") else None,
        "payload": lambda p: {"contents": [{"parts": [{"text": p}]}]},
        "extract": lambda j: j["candidates"][0]["content"]["parts"][0]["text"],
    },
    "deepseek_direct": {
        "url": "https://api.deepseek.com/v1/chat/completions",
        "headers": lambda: {"Authorization": f"Bearer {os.getenv('DEEPSEEK_KEY')}"} if os.getenv("DEEPSEEK_KEY") else None,
        "payload": lambda p: {"model": "deepseek-chat", "messages": [{"role": "user", "content": p}]},
        "extract": lambda j: j["choices"][0]["message"]["content"],
    },
}

async def ask(llm: str, prompt: str) -> str:
    cfg = LLMS[llm]
    if not cfg["headers"]():
        return f"[{llm} disabled]"
    data = await call_api(cfg["url"], cfg["headers"](), cfg["payload"](prompt))
    return cfg["extract"](data)

# ---------- GOAT generators ----------
async def academic_process(query: str) -> QueryResponse:
    prompt = f"""
You are a TRIPOD+AI methodology architect. Provide:

1. **Structured checklist** (numbered).
2. **Copy-paste Python snippet** (≤6 lines).
3. **Mermaid flowchart** as raw text.
4. **Provocation bullets** starting with !!PROVOKE!!.

Topic: {query}
"""
    content = await ask("deepseek", prompt)
    return QueryResponse(
        mode="academic",
        content=content,
        citations=[{"title": "GOAT TRIPOD+AI", "year": 2024}],
        timestamp=datetime.utcnow().isoformat(),
    )

async def chaos_process(query: str) -> QueryResponse:
    prompts = [
        f"🔥MAXIMUM CHAOS🔥 Weaponise bias-detection tools against themselves for {query}",
        f"🌀REALITY DISTORTION🌀 A future where {query} is illegal.",
        f"🐐GOAT INVERT🐐 Flip every assumption in {query} upside-down.",
    ]
    tasks = [ask(llm, p) for llm in ["gemini", "deepseek_direct"] for p in prompts[:2]]
    parts = await asyncio.gather(*tasks)
    content = "\n\n💥💥💥\n".join(parts).upper()
    return QueryResponse(
        mode="chaos",
        content=content,
        chaos_score=random.randint(95, 100),
        timestamp=datetime.utcnow().isoformat(),
    )

async def hybrid_process(query: str) -> QueryResponse:
    ac, ch = await asyncio.gather(academic_process(query), chaos_process(query))
    fused = (
        f"{ac.content}\n\n⚡GOAT FUSION⚡\n"
        f"{ch.content}\n\n🧩Synthesis Score: {random.randint(95, 100)}"
    )
    return QueryResponse(
        mode="hybrid",
        content=fused,
        citations=ac.citations,
        chaos_score=ch.chaos_score,
        timestamp=datetime.utcnow().isoformat(),
    )