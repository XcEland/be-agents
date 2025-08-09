# app/text_generation.py
import asyncio
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import main as app_main

router = APIRouter(prefix="/text", tags=["text-generation"])

client = app_main.client
model = app_main.model
executor = app_main.executor
MODEL_ID = app_main.MODEL_ID
model_params = app_main.model_params
PROJECT_ID = app_main.PROJECT_ID
VERIFY_TLS = app_main.VERIFY_TLS

class TextGenerationRequest(BaseModel):
    prompt: str
    model_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    guardrails: Optional[bool] = False
    guardrails_hap_params: Optional[Dict[str, Any]] = None
    guardrails_pii_params: Optional[Dict[str, Any]] = None

class TextGenerationResponse(BaseModel):
    generated_text: str
    raw: dict

@router.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    # Select model
    used_model = model
    if request.model_id and request.model_id != MODEL_ID:
        local_params = dict(model_params)
        if request.parameters:
            local_params.update(request.parameters)
        used_model = app_main.ModelInference(
            model_id=request.model_id,
            api_client=client,
            params=local_params,
            project_id=PROJECT_ID,
            verify=VERIFY_TLS,
        )
    elif request.parameters:
        # Create a copy with updated parameters
        new_params = dict(model_params)
        new_params.update(request.parameters)
        used_model = app_main.ModelInference(
            model_id=MODEL_ID,
            api_client=client,
            params=new_params,
            project_id=PROJECT_ID,
            verify=VERIFY_TLS,
        )

    # Prepare arguments
    kwargs = {
        "prompt": request.prompt,
        "guardrails": request.guardrails,
    }
    if request.guardrails_hap_params:
        kwargs["guardrails_hap_params"] = request.guardrails_hap_params
    if request.guardrails_pii_params:
        kwargs["guardrails_pii_params"] = request.guardrails_pii_params

    # Execute generation
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            executor, 
            lambda: used_model.generate(**kwargs)
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Text generation failed: {e}")

    # Extract generated text
    generated_text = ""
    if "results" in result and result["results"]:
        generated_text = result["results"][0].get("generated_text", "")

    return TextGenerationResponse(
        generated_text=generated_text,
        raw=result
    )