# app/main.py
import os, json, uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import redis

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
r = redis.from_url(REDIS_URL, decode_responses=True)

app = FastAPI(title="watsonx chat API (wrapped)")

# ---------- Config (from your snippet) ----------
WATSONX_URL = os.getenv("WATSONX_URL")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
MODEL_ID = os.getenv("MODEL_ID", "ibm/granite-3-8b-instruct")
VERIFY_TLS = os.getenv("VERIFY_TLS", "false").lower() in ("1", "true", "yes")

if not WATSONX_URL or not WATSONX_API_KEY:
    raise RuntimeError("Set WATSONX_URL and WATSONX_API_KEY environment variables")

credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
client = APIClient(credentials)


# NOTE: fixed param name -> max_new_tokens
model_params = {
    "time_limit": 1000,
    "max_new_tokens": 300
}

# create ModelInference once at startup (re-used)
model = ModelInference(
    model_id=MODEL_ID,
    api_client=client,
    params=model_params,
    project_id=PROJECT_ID,
    space_id=None,
    verify=VERIFY_TLS,
)

# threadpool to keep FastAPI event loop responsive
executor = ThreadPoolExecutor(max_workers=4)


# ---------- Pydantic models (request/response) ----------
class MessageChunk(BaseModel):
    type: Optional[str] = "text"
    text: str

class Message(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: Optional[List[MessageChunk]] = None
    plain: Optional[str] = None  # convenience: allow plain text

class ChatRequest(BaseModel):
    messages: List[Message]
    model_id: Optional[str] = None
    max_new_tokens: Optional[int] = None

class ChatResponse(BaseModel):
    reply: str
    raw: dict


# ---------- Helpers ----------
def build_sdk_messages(messages: List[Message]):
    sdk_msgs = []
    for m in messages:
        if m.content:
            sdk_msgs.append({"role": m.role, "content": [c.dict() for c in m.content]})
        elif m.plain:
            sdk_msgs.append({"role": m.role, "content": m.plain})
        else:
            sdk_msgs.append({"role": m.role, "content": ""})
    return sdk_msgs


# ---------- Endpoint ----------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Send chat conversation to watsonx foundation model and return assistant reply.
    """
    # Allow per-request model override (creates a lightweight ModelInference)
    if req.model_id and req.model_id != MODEL_ID:
        local_params = dict(model_params)
        if req.max_new_tokens:
            local_params["max_new_tokens"] = req.max_new_tokens
        used_model = ModelInference(
            model_id=req.model_id,
            api_client=client,
            params=local_params,
            project_id=PROJECT_ID,
            verify=VERIFY_TLS,
        )
    else:
        used_model = model
        if req.max_new_tokens:
            used_model.params["max_new_tokens"] = req.max_new_tokens

    sdk_messages = build_sdk_messages(req.messages)

    loop = asyncio.get_running_loop()
    try:
        # run in threadpool because SDK is synchronous
        result = await loop.run_in_executor(executor, lambda: used_model.chat(messages=sdk_messages))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Model call failed: {e}")

    # parse typical SDK response shapes to extract assistant text
    reply_text = ""
    try:
        choices = result.get("choices") or []
        if choices:
            first = choices[0].get("message", {})
            content = first.get("content")
            if isinstance(content, str):
                reply_text = content
            elif isinstance(content, list):
                parts = []
                for blk in content:
                    if isinstance(blk, dict) and blk.get("text"):
                        parts.append(blk["text"])
                    elif isinstance(blk, str):
                        parts.append(blk)
                reply_text = "\n".join(parts)
            else:
                # fallback: stringify content
                reply_text = str(content)
        else:
            # fallback for other shapes
            reply_text = result.get("message", {}).get("content", "") or ""
    except Exception:
        reply_text = ""

    return ChatResponse(reply=reply_text, raw=result)