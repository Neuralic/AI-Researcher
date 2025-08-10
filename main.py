# main.py
import os, asyncio, json, httpx
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
    # simple mock while keys/network are fixed
    return QueryResponse(
        mode=req.mode_override or "hybrid",
        content="Mock response – verify keys & network",
        citations=[{"title": "Demo", "year": 2024}],
        chaos_score=42.0,
        timestamp=datetime.utcnow().isoformat(),
    )

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

# ---------- static ----------
app.mount("/", StaticFiles(directory=".", html=True), name="static")