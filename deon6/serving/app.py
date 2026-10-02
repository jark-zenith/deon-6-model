from fastapi import FastAPI
from pydantic import BaseModel

app=FastAPI(title="DEON 6 Inference API",version="0.1.0")

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 64

@app.get("/health")
def health():
    return {"status":"ok","model":"DEON-6","trained":False}

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    return {"id":"deon6-placeholder","object":"chat.completion","model":"deon-6","choices":[{"index":0,"message":{"role":"assistant","content":"DEON 6 inference is not loaded yet. Train and load a checkpoint first."},"finish_reason":"stop"}]}
