import os,asyncio,httpx,random,uuid,io
from datetime import datetime
from fastapi import FastAPI,HTTPException,Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse,StreamingResponse
from pydantic import BaseModel
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer
from reportlab.lib.units import inch
from reportlab.lib import colors
import pyshorteners

class QueryRequest(BaseModel):
    query:str
    mode_override:Optional[str]=None

class QueryResponse(BaseModel):
    mode:str
    content:str
    citations:Optional[list]=None
    chaos_score:Optional[float]=None
    timestamp:str
    share_id:Optional[str]=None
    follow_ups:Optional[list]=None

app=FastAPI(title="GOAT")
app.mount("/",StaticFiles(directory=".",html=True),name="static")

LLMS={
    "deepseek":{"url":"https://openrouter.ai/api/v1/chat/completions","headers":lambda:{"Authorization":f"Bearer {os.getenv('OPENROUTER_KEY')}"}if os.getenv("OPENROUTER_KEY")else None,"payload":lambda p:{"model":"deepseek/deepseek-r1:free","messages":[{"role":"user","content":p}]},"extract":lambda j:j["choices"][0]["message"]["content"]},
    "gemini":{"url":"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent","headers":lambda:{"x-goog-api-key":os.getenv("GEMINI_KEY")}if os.getenv("GEMINI_KEY")else None,"payload":lambda p:{"contents":[{"parts":[{"text":p}]}]},"extract":lambda j:j["candidates"][0]["content"]["parts"][0]["text"]},
}

async def ask(llm,prompt):
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

def select_mode(q):
    return"chaos"if{"disrupt","innovative"}&{*q.lower()}else"academic"if{"methodology","literature"}&{*q.lower()}else"hybrid"

async def academic_process(query):
    prompt=f"Craft premium Markdown:\n## 🎯 Executive Summary\n…\n## 🔍 Key Findings\n1. …\n## 🚨 Implications\n*…*\n## 📈 Next Steps\n| Action | Owner | Deadline |\n|--------|--------|----------|\n| …      | …      | …        |\n## 🔗 References\n[link](url)\nTopic:{query}"
    return await ask("deepseek",prompt)

async def chaos_process(query):
    prompt=f"Podcast script (200 w) 🎙️ Host+Guest+SFX+cliff-hanger. Topic:{query}"
    return await ask("gemini",prompt)

@app.post("/research",response_model=QueryResponse)
async def research_endpoint(req:QueryRequest):
    mode=req.mode_override or select_mode(req.query)
    if mode=="academic":content=await academic_process(req.query)
    elif mode=="chaos":content=await chaos_process(req.query)
    else:ac=await academic_process(req.query);ch=await chaos_process(req.query);content=f"{ac}\n\n⚡GOAT FUSION⚡\n{ch}"
    share_id=str(uuid.uuid4())[:8]
    follow_ups=["What if encryption fails?","How does bias scale globally?","Can AI empathy be hacked?"]
    return QueryResponse(mode=mode,content=content,citations=[{"title":"GOAT Report","year":2024}],chaos_score=random.randint(95,100),timestamp=datetime.utcnow().isoformat(),share_id=share_id,follow_ups=follow_ups)

@app.get("/pdf")
async def get_pdf(content:str=Query(...)):
    buff=io.BytesIO()
    styles=getSampleStyleSheet()
    title=ParagraphStyle("Title",parent=styles["Heading1"],fontSize=20,alignment=1,textColor=colors.HexColor("#6366f1"))
    body=ParagraphStyle("Body",parent=styles["Normal"],fontSize=11,spaceAfter=6)
    doc=SimpleDocTemplate(buff,pagesize=A4,topMargin=1*inch,bottomMargin=1*inch)
    story=[Paragraph("GOAT Research Report",title),Spacer(1,12)]
    for line in content.splitlines():
        if line.startswith("## "):story.append(Paragraph(line[3:],styles["Heading2"]))
        elif line.startswith("### "):story.append(Paragraph(line[4:],styles["Heading3"]))
        elif line.startswith("```"):story.append(Paragraph(line,ParagraphStyle("Code",fontSize=9,leftIndent=10)))
        else:story.append(Paragraph(line,body))
    doc.build(story)
    buff.seek(0)
    return StreamingResponse(buff,media_type="application/pdf",headers={"Content-Disposition":"inline; filename=GOAT_Report.pdf"})

@app.get("/audio")
async def get_audio(content:str=Query(...)):
    from edge_tts import Communicate
    mp3=io.BytesIO()
    tts=Communicate(text=content,voice="en-US-JennyNeural")
    async for ck in tts.stream():
        if ck["type"]=="audio":mp3.write(ck["data"])
    mp3.seek(0)
    return StreamingResponse(mp3,media_type="audio/mpeg")

@app.get("/health")
async def health():return{"status":"GOAT","timestamp":datetime.utcnow().isoformat()}