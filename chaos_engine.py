import numpy as np
from astropy.cosmology import Planck18
from typing import Tuple
import httpx, os, random

class ChaosEngine:
    def __init__(self):
        self.cosmic_bg = Planck18.Tcmb0.value
        
    async def quantum_seed(self) -> float:
        noise = np.random.normal(self.cosmic_bg, 0.001)
        return float(noise)
    
    def reality_distortion(self, query: str) -> float:
        base = hash(query) % 100
        quantum = (self.cosmic_bg * 1000) % 1
        return min(100.0, base + quantum * 30)
    
    async def adversarial_fusion(self, query: str) -> str:
        apis = ["deepseek", "openai"]
        selected = random.choice(apis)
        
        headers = {"Authorization": f"Bearer {os.getenv(f'{selected.upper()}_KEY')}"}
        prompt = f"Contradict everything about: {query}"
        payload = {"messages": [{"role": "user", "content": prompt}]}
        
        if selected == "deepseek":
            url = "https://api.deepseek.com/v1/chat/completions"
        else:
            url = "https://api.openai.com/v1/chat/completions"
            
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(url, headers=headers, json=payload)
                return resp.json()['choices'][0]['message']['content'][:200]
            except:
                return "Contradiction unavailable"