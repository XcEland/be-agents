# app/tool_calling.py
import json
import uuid
from typing import List, Dict, Any, Optional
import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Import shared resources from main
import main as app_main
from chat import append_message_in_memory, get_history_in_memory

router = APIRouter(prefix="/tool-calling", tags=["tool-calling"])

# Aliases to shared objects
client = app_main.client
model = app_main.model
executor = app_main.executor
MODEL_ID = app_main.MODEL_ID
model_params = app_main.model_params
PROJECT_ID = app_main.PROJECT_ID
VERIFY_TLS = app_main.VERIFY_TLS
conversations = app_main.conversations
conv_lock = app_main.conv_lock

# ---------- Constants ----------
DEFAULT_SYSTEM_PROMPT = (
    "You are Granite, developed by IBM. You are a helpful assistant with access to the following tools. "
    "For arithmetic and numeric calculations you MUST call the appropriate tool and must NOT answer directly. "
    "When a tool is required, respond with <|tool_call|> followed by a JSON list of the tool call(s). "
    "If a requested tool is not available, explicitly state that it is unavailable."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Adds the values a and b to get a sum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"description": "A number value", "type": "number"},
                    "b": {"description": "A number value", "type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiplies the values a and b.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"description": "A number value", "type": "number"},
                    "b": {"description": "A number value", "type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    }
]

# ---------- Tool Functions ----------
def _tool_add(params: Dict[str, Any]) -> Dict[str, Any]:
    a = float(params.get("a", 0))
    b = float(params.get("b", 0))
    return {"result": a + b}

def _tool_multiply(params: Dict[str, Any]) -> Dict[str, Any]:
    a = float(params.get("a", 0))
    b = float(params.get("b", 0))
    return {"result": a * b}

LOCAL_TOOL_REGISTRY = {
    "add": _tool_add,
    "multiply": _tool_multiply,
}

# ---------- Pydantic Models ----------
class CreateConversationResponse(BaseModel):
    conversation_id: str

class SendMessageRequest(BaseModel):
    query: str
    max_new_tokens: Optional[int] = None
    time_limit: Optional[int] = None

class SendMessageResponse(BaseModel):
    conversation_id: str
    reply: str
    raw: dict

class ToolCall(BaseModel):
    id: str
    type: str
    function: Dict[str, Any]

# ---------- Helper Functions ----------
def build_sdk_messages_from_stored(stored_messages: List[dict]) -> List[dict]:
    sdk_msgs = []
    for m in stored_messages:
        sdk_msgs.append({
            "role": m.get("role", "user"),
            "content": m.get("content", "")
        })
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

def parse_assistant_content(result: dict) -> str:
    choices = result.get("choices") or []
    if not choices:
        return ""
    
    first = choices[0]
    msg = first.get("message", {}) if isinstance(first, dict) else {}
    content = msg.get("content", "")
    
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for blk in content:
            if isinstance(blk, dict) and blk.get("text"):
                parts.append(blk["text"])
            elif isinstance(blk, str):
                parts.append(blk)
        return "\n".join(parts)
    return str(content)

# ---------- Core Agent Execution ----------
async def execute_agent_loop(
    sdk_messages: List[dict],
    base_model,
    max_new_tokens: Optional[int] = None,
    time_limit: Optional[int] = None
) -> dict:
    used_model = base_model
    if max_new_tokens:
        used_model.params["max_new_tokens"] = max_new_tokens
    if time_limit:
        used_model.params["time_limit"] = time_limit

    def call_model(messages):
        return used_model.chat(
            messages=messages,
            tools=TOOLS,
            tool_choice_option="auto"
        )

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, lambda: call_model(sdk_messages))

    while True:
        tool_calls = extract_tool_calls(result)
        finish_reason = (result.get("choices") or [{}])[0].get("finish_reason")
        if not tool_calls or finish_reason != "tool_calls":
            break

        for tc in tool_calls:
            try:
                function_block = tc.get("function", {})
                func_name = function_block.get("name")
                tool_call_id = tc.get("id")
                raw_args = function_block.get("arguments", "{}")
                func_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args or {}
                
                if func_name not in LOCAL_TOOL_REGISTRY:
                    tool_output = {"error": f"Tool '{func_name}' not available."}
                else:
                    try:
                        tool_output = LOCAL_TOOL_REGISTRY[func_name](func_args)
                    except Exception as ex:
                        tool_output = {"error": f"Tool error: {ex}"}
                
                sdk_messages.append({
                    "role": "tool",
                    "content": json.dumps(tool_output),
                    "tool_call_id": tool_call_id
                })
            except Exception as e:
                sdk_messages.append({
                    "role": "tool",
                    "content": json.dumps({"error": str(e)}),
                    "tool_call_id": "error"
                })

        result = await loop.run_in_executor(executor, lambda: call_model(sdk_messages))
    
    return result

# ---------- Stateful Endpoints ----------
@router.post("/conversations", response_model=CreateConversationResponse)
def create_conversation():
    """Create a new tool-calling conversation with system prompt"""
    cid = str(uuid.uuid4())
    system_msg = {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
    append_message_in_memory(cid, system_msg)
    return CreateConversationResponse(conversation_id=cid)

@router.post("/conversations/{conversation_id}/send", response_model=SendMessageResponse)
async def send_message(conversation_id: str, req: SendMessageRequest):
    """Process user query in a tool-calling conversation"""
    with conv_lock:
        if conversation_id not in conversations:
            raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Append user message
    user_msg = {"role": "user", "content": req.query}
    append_message_in_memory(conversation_id, user_msg)
    
    # Retrieve full history
    stored_messages = get_history_in_memory(conversation_id)
    sdk_messages = build_sdk_messages_from_stored(stored_messages)
    
    # Run agent loop
    try:
        result = await execute_agent_loop(
            sdk_messages,
            base_model=model,
            max_new_tokens=req.max_new_tokens,
            time_limit=req.time_limit
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Tool calling failed: {e}")
    
    # Extract and store assistant response
    reply_text = parse_assistant_content(result)
    assistant_msg = {"role": "assistant", "content": reply_text}
    append_message_in_memory(conversation_id, assistant_msg)
    
    return SendMessageResponse(
        conversation_id=conversation_id,
        reply=reply_text,
        raw=result
    )

# ---------- Stateless Endpoint (for backward compatibility) ----------
class MessageChunk(BaseModel):
    type: Optional[str] = "text"
    text: str

class Message(BaseModel):
    role: str
    content: Optional[List[MessageChunk]] = None
    plain: Optional[str] = None
    tool_call_id: Optional[str] = None

class ToolFunctionSpec(BaseModel):
    type: str
    function: Dict[str, Any]

class ToolCallingRequest(BaseModel):
    messages: List[Message]
    tools: List[ToolFunctionSpec]
    model_id: Optional[str] = None
    max_new_tokens: Optional[int] = None
    time_limit: Optional[int] = None

class ToolCallingResponse(BaseModel):
    message: Optional[str] = None
    tool_calls: List[ToolCall] = []
    raw: dict

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

@router.post("/chat", response_model=ToolCallingResponse)
async def tool_calling_chat(req: ToolCallingRequest):
    """
    Stateless tool calling: returns either a natural language response
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
        response.message = parse_assistant_content(result)
    
    return response