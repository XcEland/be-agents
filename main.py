# app/main.py
import os
import json
import uuid
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Any, Dict
from threading import Lock
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# NOTE: removed redis import for in-memory storage

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

load_dotenv()

app = FastAPI(title="watsonx chat API (in-memory)")

# ---------- WatsonX config ----------
WATSONX_URL = os.getenv("WATSONX_URL")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
MODEL_ID = os.getenv("MODEL_ID", "ibm/granite-3-8b-instruct")
VERIFY_TLS = os.getenv("VERIFY_TLS", "false").lower() in ("1", "true", "yes")

if not WATSONX_URL or not WATSONX_API_KEY:
    raise RuntimeError("Set WATSONX_URL and WATSONX_API_KEY environment variables")

credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
client = APIClient(credentials)

# NOTE: parameter name used by this code
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

# Threadpool to avoid blocking the event loop for sync SDK calls
executor = ThreadPoolExecutor(max_workers=4)

# ---------- In-memory conversation store ----------
# Structure: conversations[cid] = {"messages": [msg_dict...], "created_at": ts, "last_updated": ts}
conversations: Dict[str, Dict[str, Any]] = {}
conv_lock = Lock()  # thread-safe access for the dict

# Optional TTL and trimming
CONV_TTL_SECONDS = int(os.getenv("CONV_TTL_SECONDS", 60 * 60 * 24 * 7))  # default 7 days
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", 200))


# ---------- Pydantic models ----------
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

class CreateConversationResponse(BaseModel):
    conversation_id: str

class SendMessageRequest(BaseModel):
    role: str  # "user" or "system"
    content: Optional[List[MessageChunk]] = None
    plain: Optional[str] = None
    max_new_tokens: Optional[int] = None

class SendMessageResponse(BaseModel):
    conversation_id: str
    reply: str
    raw: dict


# ---------- Helper functions for in-memory store ----------
def append_message_in_memory(cid: str, msg: dict):
    now = int(time.time())
    with conv_lock:
        conv = conversations.get(cid)
        if conv is None:
            conv = {"messages": [], "created_at": now, "last_updated": now}
            conversations[cid] = conv
        conv["messages"].append(msg)
        # trim to last MAX_HISTORY_MESSAGES
        if MAX_HISTORY_MESSAGES > 0 and len(conv["messages"]) > MAX_HISTORY_MESSAGES:
            conv["messages"] = conv["messages"][-MAX_HISTORY_MESSAGES:]
        conv["last_updated"] = now

def get_history_in_memory(cid: str, limit: int = MAX_HISTORY_MESSAGES) -> List[dict]:
    with conv_lock:
        conv = conversations.get(cid)
        if not conv:
            return []
        msgs = conv["messages"][:limit]
        # return a deep-ish copy to avoid accidental mutation
        return json.loads(json.dumps(msgs))

def cleanup_expired_conversations_task():
    """
    Periodic cleanup that removes conversations older than CONV_TTL_SECONDS.
    Runs in the background when the app starts. Simple and optional.
    """
    async def _cleanup():
        while True:
            now = int(time.time())
            if CONV_TTL_SECONDS > 0:
                with conv_lock:
                    keys_to_delete = [cid for cid, conv in conversations.items()
                                      if (now - conv.get("last_updated", conv.get("created_at", now))) > CONV_TTL_SECONDS]
                    for cid in keys_to_delete:
                        del conversations[cid]
            # sleep (check hourly by default)
            await asyncio.sleep(60 * 60)
    return _cleanup


# ---------- SDK message builder & parser (same as before) ----------
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

def build_sdk_messages_from_stored(stored_messages: List[dict]) -> List[dict]:
    sdk_msgs = []
    for m in stored_messages:
        sdk_msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})
    return sdk_msgs

def parse_assistant_content_from_sdk_result(result: dict) -> str:
    reply_text = ""
    try:
        choices = result.get("choices") or []
        if choices:
            first = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            content = first.get("content") if isinstance(first, dict) else None
            if content is None:
                content = choices[0].get("content") if isinstance(choices[0], dict) else None

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
                reply_text = str(content)
        else:
            reply_text = result.get("message", {}).get("content", "") or ""
    except Exception:
        reply_text = ""
    return reply_text


# ---------- Stateless /chat endpoint ----------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Stateless chat: client sends full message history in the request.
    """
    # Optionally override model per-request
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
        result = await loop.run_in_executor(executor, lambda: used_model.chat(messages=sdk_messages))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Model call failed: {e}")

    reply_text = parse_assistant_content_from_sdk_result(result)
    return ChatResponse(reply=reply_text, raw=result)


# ---------- Conversation endpoints (in-memory) ----------
@app.post("/conversations", response_model=CreateConversationResponse)
def create_conversation(seed_system_message: Optional[str] = "You are a helpful assistant."):
    """
    Create a new conversation and seed a system message (default helpful assistant).
    Returns conversation_id.
    """
    cid = str(uuid.uuid4())
    system_msg = {"role": "system", "content": seed_system_message}
    append_message_in_memory(cid, system_msg)
    return CreateConversationResponse(conversation_id=cid)


@app.post("/conversations/{conversation_id}/send", response_model=SendMessageResponse)
async def send_conversation_message(conversation_id: str, req: SendMessageRequest):
    """
    Append the provided message to stored history, call model with stored history,
    append assistant reply to history and return the reply.
    """
    incoming_msg = {"role": req.role, "content": ( [c.dict() for c in req.content] if req.content else (req.plain or "")) }
    append_message_in_memory(conversation_id, incoming_msg)

    stored = get_history_in_memory(conversation_id, limit=MAX_HISTORY_MESSAGES)
    sdk_messages = build_sdk_messages_from_stored(stored)

    # Optionally clone model with per-request tokens
    used_model = model
    if req.max_new_tokens:
        new_params = dict(model.params or {})
        new_params["max_new_tokens"] = req.max_new_tokens
        used_model = ModelInference(
            model_id=model.model_id,
            api_client=client,
            params=new_params,
            project_id=PROJECT_ID,
            verify=VERIFY_TLS,
        )

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(executor, lambda: used_model.chat(messages=sdk_messages))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Model call failed: {e}")

    reply_text = parse_assistant_content_from_sdk_result(result)
    assistant_msg = {"role": "assistant", "content": reply_text}
    append_message_in_memory(conversation_id, assistant_msg)

    return SendMessageResponse(conversation_id=conversation_id, reply=reply_text, raw=result)


@app.get("/conversations/{conversation_id}/history")
def get_conversation_history(conversation_id: str, limit: int = MAX_HISTORY_MESSAGES) -> List[Any]:
    return get_history_in_memory(conversation_id, limit=limit)


# ---------- Startup: optional background cleanup task ----------
@app.on_event("startup")
async def startup_event():
    # start background cleanup task only if TTL configured
    if CONV_TTL_SECONDS > 0:
        asyncio.create_task(cleanup_expired_conversations_task()())