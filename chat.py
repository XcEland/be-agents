# app/chat.py
import json
import uuid
import time
import asyncio
from typing import List, Optional, Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Import shared resources from main (main must define them before importing chat)
import main as app_main

router = APIRouter()  # will be included by main.app

# Aliases to shared objects (same object identities)
client = app_main.client
model = app_main.model
executor = app_main.executor
MODEL_ID = app_main.MODEL_ID
model_params = app_main.model_params

conversations = app_main.conversations
conv_lock = app_main.conv_lock
CONV_TTL_SECONDS = app_main.CONV_TTL_SECONDS
MAX_HISTORY_MESSAGES = app_main.MAX_HISTORY_MESSAGES

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


async def cleanup_expired_conversations_task():
    """
    Periodic cleanup that removes conversations older than CONV_TTL_SECONDS.
    This coroutine loops forever; main creates a task for it on startup.
    """
    while True:
        now = int(time.time())
        if CONV_TTL_SECONDS > 0:
            with conv_lock:
                keys_to_delete = [
                    cid for cid, conv in conversations.items()
                    if (now - conv.get("last_updated", conv.get("created_at", now))) > CONV_TTL_SECONDS
                ]
                for cid in keys_to_delete:
                    del conversations[cid]
        # sleep (check hourly by default)
        await asyncio.sleep(60 * 60)


# ---------- SDK message builder & parser ----------
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
@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Stateless chat: client sends full message history in the request.
    """
    # Optionally override model per-request
    if req.model_id and req.model_id != MODEL_ID:
        local_params = dict(model_params)
        if req.max_new_tokens:
            local_params["max_new_tokens"] = req.max_new_tokens
        used_model = app_main.ModelInference(
            model_id=req.model_id,
            api_client=client,
            params=local_params,
            project_id=app_main.PROJECT_ID,
            verify=app_main.VERIFY_TLS,
        )
    else:
        used_model = model
        if req.max_new_tokens:
            # mutate shared model.params as before (same behaviour as original)
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
@router.post("/conversations", response_model=CreateConversationResponse)
def create_conversation(seed_system_message: Optional[str] = "You are a helpful assistant."):
    """
    Create a new conversation and seed a system message (default helpful assistant).
    Returns conversation_id.
    """
    cid = str(uuid.uuid4())
    system_msg = {"role": "system", "content": seed_system_message}
    append_message_in_memory(cid, system_msg)
    return CreateConversationResponse(conversation_id=cid)


@router.post("/conversations/{conversation_id}/send", response_model=SendMessageResponse)
async def send_conversation_message(conversation_id: str, req: SendMessageRequest):
    """
    Append the provided message to stored history, call model with stored history,
    append assistant reply to history and return the reply.
    """
    incoming_msg = {
        "role": req.role,
        "content": ([c.dict() for c in req.content] if req.content else (req.plain or ""))
    }
    append_message_in_memory(conversation_id, incoming_msg)

    stored = get_history_in_memory(conversation_id, limit=MAX_HISTORY_MESSAGES)
    sdk_messages = build_sdk_messages_from_stored(stored)

    # Optionally clone model with per-request tokens
    used_model = model
    if req.max_new_tokens:
        new_params = dict(model.params or {})
        new_params["max_new_tokens"] = req.max_new_tokens
        used_model = app_main.ModelInference(
            model_id=model.model_id,
            api_client=client,
            params=new_params,
            project_id=app_main.PROJECT_ID,
            verify=app_main.VERIFY_TLS,
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


@router.get("/conversations/{conversation_id}/history")
def get_conversation_history(conversation_id: str, limit: int = MAX_HISTORY_MESSAGES) -> List[Any]:
    return get_history_in_memory(conversation_id, limit=limit)
