import os
from dotenv import load_dotenv
import asyncio

from fastapi import FastAPI

# load env
load_dotenv()

# create FastAPI app
app = FastAPI(title="watsonx chat API (in-memory)")

# ---------- WatsonX config & SDK init ----------
WATSONX_URL = os.getenv("WATSONX_URL")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
MODEL_ID = os.getenv("MODEL_ID", "ibm/granite-3-8b-instruct")
VERIFY_TLS = os.getenv("VERIFY_TLS", "false").lower() in ("1", "true", "yes")

if not WATSONX_URL or not WATSONX_API_KEY:
    raise RuntimeError("Set WATSONX_URL and WATSONX_API_KEY environment variables")

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from concurrent.futures import ThreadPoolExecutor

credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
client = APIClient(credentials)

# NOTE: parameter name used by this code
model_params = {
    "time_limit": 10000,
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

# ---------- In-memory conversation store (shared) ----------
from typing import Dict, Any
import time
from threading import Lock

conversations: Dict[str, Dict[str, Any]] = {}
conv_lock = Lock()  # thread-safe access for the dict

CONV_TTL_SECONDS = int(os.getenv("CONV_TTL_SECONDS", 60 * 60 * 24 * 7))  # default 7 days
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", 200))

# ---------- Include chat and agent routers (import after shared resources exist) ----------
# chat.py and agent_driven_chat.py import this module (main) to access the shared resources
import chat  # local module - must be after shared resources above
app.include_router(chat.router)

import agent_driven_chat  # local module - must be after shared resources above
app.include_router(agent_driven_chat.router)

import tool_calling
app.include_router(tool_calling.router)

import text_generation  # new module
app.include_router(text_generation.router)

import generation  # new module
app.include_router(generation.router)

# ---------- Startup event: start cleanup task if configured ----------
@app.on_event("startup")
async def startup_event():
    # start background cleanup task only if TTL configured
    if CONV_TTL_SECONDS > 0:
        # chat.cleanup_expired_conversations_task is an async coroutine function
        # it references conversations and conv_lock which are defined above
        asyncio.create_task(chat.cleanup_expired_conversations_task())

# Optional: allow running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)