# app/tool_calling.py
import json
from typing import List, Dict, Any, Optional
import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Import shared resources from main
import main as app_main

router = APIRouter(prefix="/tool-calling", tags=["tool-calling"])

# Aliases to shared objects
client = app_main.client
model = app_main.model
executor = app_main.executor
MODEL_ID = app_main.MODEL_ID
model_params = app_main.model_params
PROJECT_ID = app_main.PROJECT_ID
VERIFY_TLS = app_main.VERIFY_TLS

# ---------- Pydantic Models ----------
class MessageChunk(BaseModel):
    type: Optional[str] = "text"
    text: str

class Message(BaseModel):
    role: str  # "system" | "user" | "assistant" | "tool"
    content: Optional[List[MessageChunk]] = None
    plain: Optional[str] = None
    tool_call_id: Optional[str] = None  # For tool responses

class ToolFunctionSpec(BaseModel):
    type: str
    function: Dict[str, Any]

class ToolCallingRequest(BaseModel):
    messages: List[Message]
    tools: List[ToolFunctionSpec]
    model_id: Optional[str] = None
    max_new_tokens: Optional[int] = None
    time_limit: Optional[int] = None

class ToolCall(BaseModel):
    id: str
    type: str
    function: Dict[str, Any]

class ToolCallingResponse(BaseModel):
    message: Optional[str] = None
    tool_calls: List[ToolCall] = []
    raw: dict

# ---------- Helper Functions ----------
def build_sdk_messages(messages: List[Message]) -> List[dict]:
    sdk_msgs = []
    for m in messages:
        msg = {"role": m.role}
        
        if m.content:
            msg["content"] = [c.dict() for c in m.content]
        elif m.plain:
            msg["content"] = m.plain
        else:
            msg["content"] = ""
        
        if m.role == "tool" and m.tool_call_id:
            msg["tool_call_id"] = m.tool_call_id
        
        sdk_msgs.append(msg)
    return sdk_msgs

def extract_tool_calls(result: dict) -> List[dict]:
    choices = result.get("choices") or []
    if not choices:
        return []
    
    first = choices[0]
    msg = first.get("message", {}) if isinstance(first, dict) else {}
    tool_calls = msg.get("tool_calls") or msg.get("tool_call") or first.get("tool_calls")
    
    if not tool_calls:
        return []
    
    return tool_calls if isinstance(tool_calls, list) else [tool_calls]

# ---------- Tool Calling Endpoint ----------
@router.post("/chat", response_model=ToolCallingResponse)
async def tool_calling_chat(req: ToolCallingRequest):
    """
    Single-step tool calling: returns either a natural language response
    or tool calls that need to be executed by the client.
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
            project_id=PROJECT_ID,
            verify=VERIFY_TLS,
        )
    else:
        used_model = model
        if req.max_new_tokens:
            used_model.params["max_new_tokens"] = req.max_new_tokens
        if req.time_limit:
            used_model.params["time_limit"] = req.time_limit

    # Build SDK-compatible messages and tools
    sdk_messages = build_sdk_messages(req.messages)
    sdk_tools = [t.dict() for t in req.tools]

    # Call the model
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            executor, 
            lambda: used_model.chat(messages=sdk_messages, tools=sdk_tools)
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Model call failed: {e}")

    # Parse response
    tool_calls = extract_tool_calls(result)
    response = ToolCallingResponse(raw=result)
    
    if tool_calls:
        # If tool calls are present, return them
        response.tool_calls = [
            ToolCall(
                id=tc.get("id", ""),
                type=tc.get("type", "function"),
                function=tc.get("function", {})
            ) for tc in tool_calls
        ]
    else:
        # If no tool calls, return the natural language response
        choices = result.get("choices") or []
        if choices:
            first = choices[0]
            msg = first.get("message", {}) if isinstance(first, dict) else {}
            content = msg.get("content", "")
            
            if isinstance(content, str):
                response.message = content
            elif isinstance(content, list):
                parts = []
                for blk in content:
                    if isinstance(blk, dict) and blk.get("text"):
                        parts.append(blk["text"])
                    elif isinstance(blk, str):
                        parts.append(blk)
                response.message = "\n".join(parts)
            else:
                response.message = str(content)
    
    return response