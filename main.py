
from fastapi import FastAPI
from pydantic import BaseModel
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
import os
from dotenv import load_dotenv


load_dotenv()
app = FastAPI()

credentials = Credentials(
    url=os.getenv("WATSONX_URL"),
    api_key=os.getenv("WATSONX_API_KEY")
)
client = APIClient(credentials)
params = {
    "time_limit": 1000,
    "max_new_token": 300
}
model_id = "ibm/granite-3-8b-instruct"
project_id = os.getenv("WATSONX_PROJECT_ID")
space_id = None
verify = False

model = ModelInference(
    model_id=model_id,
    api_client=client,
    params=params,
    project_id=project_id,
    space_id=space_id,
    verify=verify,
)

class EssayRequest(BaseModel):
    essay: str

@app.post(
    "/grade-essay/",
    summary="Automated Essay Grader",
    description="Submit an essay for AI grading and feedback using the Granite model."
)
def grade_essay(request: EssayRequest):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Grade this essay and provide feedback: {request.essay}"
                }
            ]
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "automated_essay_grader",
                "description": "Automatically grades an essay and provides detailed feedback on content, structure, grammar, and style.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "essay_text": {
                            "description": "The full text of the student's essay.",
                            "type": "string"
                        }
                    },
                    "required": ["essay_text"]
                }
            }
        }
    ]
    try:
        # result = model.chat(messages=messages, tools=tools)
        result = model.chat(messages=messages)
        return {"feedback": result}
    except Exception as e:
        return {"error": str(e)}
