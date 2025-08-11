# text_generation.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional
import asyncio
from starlette.concurrency import iterate_in_threadpool

# Import shared resources from main
from main import model, executor

router = APIRouter()

@router.post("/generation")
async def generate_text(
    prompt: str,
    params: Optional[Dict[str, Any]] = None,
    guardrails: Optional[Dict[str, Any]] = None
):
    """
    Generate text based on a prompt
    """
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    # Set default parameters if not provided
    gen_params = params or {}
    guardrails_config = guardrails or {}
    
    try:
        # Prepare guardrails parameters if enabled
        guardrails_enabled = guardrails_config.get("enable", False)
        hap_params = guardrails_config.get("hap_params", {})
        pii_params = guardrails_config.get("pii_params", {})

        if guardrails_enabled:
            # Generate with guardrails
            def sync_generate_with_guardrails():
                return model.generate(
                    prompt,
                    params=gen_params,
                    guardrails=True,
                    guardrails_hap_params=hap_params,
                    guardrails_pii_params=pii_params
                )
            
            response = await asyncio.get_event_loop().run_in_executor(
                executor, 
                sync_generate_with_guardrails
            )
            generated_text = response['results'][0]['generated_text']
        else:
            # Generate without guardrails
            def sync_generate_text():
                return model.generate_text(prompt, params=gen_params)
            
            generated_text = await asyncio.get_event_loop().run_in_executor(
                executor, 
                sync_generate_text
            )
        
        return {"generated_text": generated_text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generation/stream")
async def generate_text_stream(
    prompt: str,
    params: Optional[Dict[str, Any]] = None
):
    """
    Stream generated text based on a prompt
    """
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    # Set default parameters if not provided
    gen_params = params or {}
    
    try:
        def sync_stream_generator():
            stream = model.generate_text_stream(prompt, params=gen_params)
            for chunk in stream:
                yield chunk
        
        async def async_stream_generator():
            async for chunk in iterate_in_threadpool(sync_stream_generator()):
                yield chunk
        
        return StreamingResponse(
            async_stream_generator(),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))