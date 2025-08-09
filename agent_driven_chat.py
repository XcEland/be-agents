# app/agent_driven_chat.py
import json
import uuid
import time
from typing import List, Optional, Any, Dict
import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# import shared resources from main (main.py must define them before importing this module)
import main as app_main

router = APIRouter(prefix="/agent", tags=["agent-driven"])

# aliases to shared resources
client = app_main.client
model = app_main.model
executor = app_main.executor
MODEL_ID = app_main.MODEL_ID
model_params = app_main.model_params
PROJECT_ID = app_main.PROJECT_ID
VERIFY_TLS = app_main.VERIFY_TLS

conversations = app_main.conversations
conv_lock = app_main.conv_lock
CONV_TTL_SECONDS = app_main.CONV_TTL_SECONDS
MAX_HISTORY_MESSAGES = app_main.MAX_HISTORY_MESSAGES

# Import helper functions from chat
from chat import append_message_in_memory, get_history_in_memory


# ---- Constants ----
DEFAULT_AGENT_SYSTEM_PROMPT = (
    "You are Granite, developed by IBM. You are a helpful assistant with access to the following tools. "
    "For arithmetic and numeric calculations you MUST call the appropriate tool and must NOT answer directly. "
    "When a tool is required, respond with <|tool_call|> followed by a JSON list of the tool call(s). "
    "If a requested tool is not available, explicitly state that it is unavailable."
)

AGENT_TOOLS = [
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

class CreateAgentConversationResponse(BaseModel):
    conversation_id: str

class SendAgentMessageRequest(BaseModel):
    query: str
    max_new_tokens: Optional[int] = None
    time_limit: Optional[int] = None

class SendAgentMessageResponse(BaseModel):
    conversation_id: str
    reply: str
    raw: dict

# ---- Pydantic request/response models (minimal / compatible with chat.py) ----
class MessageChunk(BaseModel):
    type: Optional[str] = "text"
    text: str

class Message(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: Optional[List[MessageChunk]] = None
    plain: Optional[str] = None

class ToolFunctionSpec(BaseModel):
    # Accept the same shape your IBM example uses for a tool entry:
    # {"type":"function","function": { "name": "...", "description": "...", "parameters": { ... } } }
    type: str
    function: Dict[str, Any]

class AgentChatRequest(BaseModel):
    messages: List[Message]
    tools: Optional[List[ToolFunctionSpec]] = None
    tool_choice_option: Optional[str] = "auto"  # "auto" or "manual"
    model_id: Optional[str] = None
    max_new_tokens: Optional[int] = None
    time_limit: Optional[int] = None

class AgentChatResponse(BaseModel):
    reply: str
    raw: dict


# ---------- SDK message builders (same format as chat.py) ----------
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


def parse_assistant_content_from_sdk_result(result: dict) -> str:
    reply_text = ""
    try:
        choices = result.get("choices") or []
        if choices:
            first_choice = choices[0]
            msg = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
            content = msg.get("content") if isinstance(msg, dict) else None
            if content is None:
                content = first_choice.get("content")
            
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

# ---------- Helper functions ----------
def build_sdk_messages_from_stored(stored_messages: List[dict]) -> List[dict]:
    sdk_msgs = []
    for m in stored_messages:
        sdk_msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})
    return sdk_msgs


# ---------- Core agent execution ----------
async def execute_agent_loop(
    sdk_messages: List[dict],
    tools_payload: List[dict],
    base_model,
    max_new_tokens: Optional[int] = None,
    time_limit: Optional[int] = None
) -> dict:
    used_model = base_model
    if max_new_tokens:
        used_model.params["max_new_tokens"] = max_new_tokens
    if time_limit:
        used_model.params["time_limit"] = time_limit

    def call_model_once(messages, tools=None):
        kwargs = {"messages": messages}
        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice_option"] = "auto"
        return used_model.chat(**kwargs)

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        executor, lambda: call_model_once(sdk_messages, tools_payload)
    )

    def _extract_tool_calls(res: dict):
        choices = res.get("choices") or []
        if not choices:
            return []
        first = choices[0]
        msg = first.get("message", {}) if isinstance(first, dict) else {}
        tool_calls = msg.get("tool_calls") or msg.get("tool_call") or first.get("tool_calls")
        return tool_calls if isinstance(tool_calls, list) else [tool_calls] if tool_calls else []

    while True:
        tool_calls = _extract_tool_calls(result)
        finish_reason = (result.get("choices") or [{}])[0].get("finish_reason")
        if not tool_calls or finish_reason != "tool_calls":
            break

        for tc in tool_calls:
            try:
                function_block = tc.get("function", {})
                func_name = function_block.get("name")
                tool_call_id = tc.get("id")  # Get tool call ID
                raw_args = function_block.get("arguments", "{}")
                func_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args or {}
                
                if func_name not in LOCAL_TOOL_REGISTRY:
                    tool_output = {"error": f"Tool '{func_name}' not available."}
                else:
                    try:
                        tool_output = LOCAL_TOOL_REGISTRY[func_name](func_args)
                    except Exception as ex:
                        tool_output = {"error": f"Tool error: {ex}"}
                
                # Add tool response message with required tool_call_id
                sdk_messages.append({
                    "role": "tool",
                    "content": json.dumps(tool_output),
                    "tool_call_id": tool_call_id  # Required by Watsonx API
                })
            except Exception as e:
                sdk_messages.append({
                    "role": "tool",
                    "content": json.dumps({"error": str(e)}),
                    "tool_call_id": "error"  # Placeholder for errors
                })

        result = await loop.run_in_executor(
            executor, lambda: call_model_once(sdk_messages, tools_payload))
    
    return result

# ---------- Endpoints ----------
@router.post("/conversations", response_model=CreateAgentConversationResponse)
def create_agent_conversation():
    cid = str(uuid.uuid4())
    system_msg = {"role": "system", "content": DEFAULT_AGENT_SYSTEM_PROMPT}
    append_message_in_memory(cid, system_msg)
    return CreateAgentConversationResponse(conversation_id=cid)

@router.post("/conversations/{conversation_id}/send", response_model=SendAgentMessageResponse)
async def send_agent_message(conversation_id: str, req: SendAgentMessageRequest):
    with conv_lock:
        if conversation_id not in conversations:
            raise HTTPException(status_code=404, detail="Conversation not found")
    
    user_msg = {"role": "user", "content": req.query}
    append_message_in_memory(conversation_id, user_msg)
    
    stored_messages = get_history_in_memory(conversation_id)
    sdk_messages = build_sdk_messages_from_stored(stored_messages)
    
    try:
        result = await execute_agent_loop(
            sdk_messages,
            tools_payload=AGENT_TOOLS,
            base_model=model,
            max_new_tokens=req.max_new_tokens,
            time_limit=req.time_limit
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Agent processing failed: {e}")
    
    reply_text = parse_assistant_content_from_sdk_result(result)
    assistant_msg = {"role": "assistant", "content": reply_text}
    append_message_in_memory(conversation_id, assistant_msg)
    
    return SendAgentMessageResponse(
        conversation_id=conversation_id,
        reply=reply_text,
        raw=result
    )

# ---------- Core agent endpoint ----------
@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(req: AgentChatRequest):
    """
    Agent-driven chat: accepts a tools list (definitions) and will execute tool calls
    the model requests, feeding results back to the model until the model completes.
    """

    # choose model / params same as chat.py
    used_model = model
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
        if req.max_new_tokens:
            used_model.params["max_new_tokens"] = req.max_new_tokens
        if req.time_limit:
            used_model.params["time_limit"] = req.time_limit

    # build initial messages
    sdk_messages = build_sdk_messages(req.messages)

    # translate Pydantic tools into a structure we pass to the SDK (if SDK supports it)
    tools_payload = None
    if req.tools:
        tools_payload = [t.dict() for t in req.tools]

    # helper: call model once in executor, passing tools if provided
    def call_model_once(messages, tools=None, tool_choice_option=req.tool_choice_option):
        kwargs = {"messages": messages}
        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice_option"] = tool_choice_option
        # allow additional params from model params
        # many SDKs expect max tokens / time limit in model.params; we've set them on used_model
        return used_model.chat(**kwargs)

    loop = asyncio.get_running_loop()

    # iterative flow: call model -> if it requests tool_calls, run tool(s) locally and feed results back -> repeat
    try:
        result = await loop.run_in_executor(executor, lambda: call_model_once(sdk_messages, tools_payload))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Model call failed: {e}")

    # convenience function to extract tool_calls from result
    def _extract_tool_calls(res: dict):
        choices = res.get("choices") or []
        if not choices:
            return []
        first = choices[0]
        # IBM returns tool-calls under message.tool_calls or message.tool_calls[] or message.tool_calls
        msg = first.get("message", {}) if isinstance(first, dict) else {}
        tool_calls = msg.get("tool_calls") or msg.get("tool_call") or first.get("tool_calls")
        # normalize to list
        if tool_calls is None:
            return []
        return tool_calls if isinstance(tool_calls, list) else [tool_calls]

    # loop while model asked for tool calls
    while True:
        tool_calls = _extract_tool_calls(result)
        # some models also set finish_reason == "tool_calls"
        finish_reason = (result.get("choices") or [{}])[0].get("finish_reason")
        if not tool_calls or finish_reason != "tool_calls":
            break  # done; model didn't request tools or is finished

        # For each tool call, execute if local tool exists, then append tool output as a 'tool' message and re-call model
        for tc in tool_calls:
            try:
                # tc structure from IBM example:
                # {
                #   "id": "...",  # This is the tool_call_id we need
                #   "type": "function",
                #   "function": {
                #       "name": "multiply",
                #       "arguments": "{\"a\": 2, \"b\": 4}"
                #   }
                # }
                function_block = tc.get("function", {})
                func_name = function_block.get("name")
                tool_call_id = tc.get("id")  # Get tool call ID
                raw_args = function_block.get("arguments", "{}")
                # arguments sometimes come as JSON string; parse if so
                if isinstance(raw_args, str):
                    try:
                        func_args = json.loads(raw_args)
                    except Exception:
                        # fallback: try to treat as simple string
                        func_args = {"raw": raw_args}
                else:
                    func_args = raw_args or {}

                # check registry
                if func_name not in LOCAL_TOOL_REGISTRY:
                    # append a "tool" message saying tool unavailable and re-call model (model will see tool output)
                    tool_output = {"error": f"Tool '{func_name}' not available on server."}
                else:
                    # execute tool
                    executor_fn = LOCAL_TOOL_REGISTRY[func_name]
                    # call the function (synchronous) — safe to run inline
                    try:
                        tool_output = executor_fn(func_args)
                    except Exception as ex:
                        tool_output = {"error": f"Tool '{func_name}' raised an error: {ex}"}

                # append the tool result as a message the model can consume
                # Watsonx requires tool_call_id in the response
                tool_result_msg = {
                    "role": "tool",
                    "content": json.dumps(tool_output),
                    "tool_call_id": tool_call_id  # Required by Watsonx API
                }
                sdk_messages.append(tool_result_msg)

            except Exception as e:
                # if something unexpected happens, record it into sdk_messages so model can see
                sdk_messages.append({
                    "role": "tool",
                    "content": json.dumps({"error": str(e)}),
                    "tool_call_id": "error"  # Placeholder for errors
                })

        # re-call model with appended tool outputs. Keep passing tools payload if originally provided.
        try:
            result = await loop.run_in_executor(executor, lambda: call_model_once(sdk_messages, tools_payload))
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Model call failed on tool-result iteration: {e}")

        # continue loop — if model again requests tools, the above will repeat until completion

    # final: parse reply and return
    reply_text = parse_assistant_content_from_sdk_result(result)
    return AgentChatResponse(reply=reply_text, raw=result)