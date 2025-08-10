import os,asyncio,httpx,random,uuid,io
from datetime import datetime
from typing import List,Dict,Optional
from fastapi import FastAPI,HTTPException,Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse,StreamingResponse
from pydantic import BaseModel
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import pyshorteners

class QueryRequest(BaseModel):
    query:str
    mode_override:Optional[str]=None

class QueryResponse(BaseModel):
    mode:str
    content:str
    citations:Optional[List[Dict]]=None
    chaos_score:Optional[float]=None
    timestamp:str
    share_id:Optional[str]=None
    follow_ups:Optional[List[str]]=None

app=FastAPI(title="GOAT Research Agent")
app.mount("/",StaticFiles(directory=".",html=True),name="static")

LLMS={
    "deepseek":{
        "url":"https://openrouter.ai/api/v1/chat/completions",
        "headers":lambda:{"Authorization":f"Bearer {os.getenv('OPENROUTER_KEY')}"}if os.getenv("OPENROUTER_KEY")else None,
        "payload":lambda p:{"model":"deepseek/deepseek-r1:free","messages":[{"role":"user","content":p}]},
        "extract":lambda j:j["choices"][0]["message"]["content"],
    },
    "gemini":{
        "url":"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        "headers":lambda:{"x-goog-api-key":os.getenv("GEMINI_KEY")}if os.getenv("GEMINI_KEY")else None,
        "payload":lambda p:{"contents":[{"parts":[{"text":p}]}]},
        "extract":lambda j:j["candidates"][0]["content"]["parts"][0]["text"],
    },
    "gpt":{
        "url":"https://api.openai.com/v1/chat/completions",
        "headers":lambda:{"Authorization":f"Bearer {os.getenv('OPENAI_KEY')}"}if os.getenv("OPENAI_KEY")else None,
        "payload":lambda p:{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":p}]},
        "extract":lambda j:j["choices"][0]["message"]["content"],
    },
}

async def ask(llm:str,prompt:str)->str:
    cfg=LLMS[llm]
    if not cfg["headers"]():return f"[{llm} disabled]"
    data=await call_api(cfg["url"],cfg["headers"](),cfg["payload"](prompt))
    return cfg["extract"](data)

async def call_api(url,headers,payload,timeout=30):
    async with httpx.AsyncClient(timeout=timeout)as client:
        for i in range(3):
            try:
                r=await client.post(url,headers=headers,json=payload)
                if r.status_code==200:return r.json()
                await asyncio.sleep(2**i)
            except:await asyncio.sleep(2**i)
    raise HTTPException(503,"API unavailable")

def select_mode(q:str)->str:
    q=q.lower()
    if"methodology"in q or"literature review"in q:return"academic"
    if"disrupt"in q or"innovative"in q:return"chaos"
    return"hybrid"

async def academic_process(query:str)->str:
    prompt=f"Return rich Markdown:\n### 📋 TRIPOD+AI Checklist\n1...\n```python\n#snippet\n```\n```mermaid\nflowchart TD\nA-->B\n```\n!!PROVOKE!!...\nTopic:{query}"
    return await ask("deepseek",prompt)

async def chaos_process(query:str)->str:
    prompt=f"Return podcast-style script max 250 words 🎙️ all caps. Topic:{query}"
    return await ask("gemini",prompt)

@app.post("/research",response_model=QueryResponse)
async def research_endpoint(req:QueryRequest):
    mode=req.mode_override or select_mode(req.query)
    if mode=="academic":content=await academic_process(req.query)
    elif mode=="chaos":content=await chaos_process(req.query)
    else:
        ac=await academic_process(req.query)
        ch=await chaos_process(req.query)
        content=f"{ac}\n\n⚡GOAT FUSION⚡\n{ch}"
    share_id=str(uuid.uuid4())[:8]
    follow_ups=["What if encryption fails?","How does bias scale globally?","Can AI empathy be hacked?"]
    return QueryResponse(mode=mode,content=content,citations=[{"title":"GOAT Multi-LLM Report","year":2024}],chaos_score=random.randint(95,100),timestamp=datetime.utcnow().isoformat(),share_id=share_id,follow_ups=follow_ups)

@app.get("/pdf")
async def get_pdf(content:str=Query(...)):
    buffer=io.BytesIO()
    c=canvas.Canvas(buffer,pagesize=letter)
    width,height=letter
    c.drawString(100,height-100,"GOAT Research Report")
    text_obj=c.beginText(100,height-130)
    for line in content.splitlines():text_obj.textLine(line)
    c.drawText(text_obj);c.showPage();c.save()
    buffer.seek(0)
    return StreamingResponse(buffer,media_type="application/pdf",headers={"Content-Disposition":"inline; filename=goat_report.pdf"})

@app.get("/audio")
async def get_audio(content:str=Query(...)):
    from edge_tts import Communicate
    mp3=io.BytesIO()
    tts=Communicate(text=content,voice="en-US-AriaNeural")
    async for chunk in tts.stream():
        if chunk["type"]=="audio":mp3.write(chunk["data"])
    mp3.seek(0)
    return StreamingResponse(mp3,media_type="audio/mpeg",headers={"Content-Disposition":"inline; filename=goat_podcast.mp3"})

@app.get("/health")
async def health():return{"status":"GOAT","timestamp":datetime.utcnow().isoformat()}