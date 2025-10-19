from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import List, Dict, Any

# Vertex AI stable imports
from vertexai import init as vertex_init
from vertexai.generative_models import GenerativeModel

# --- Config (env with safe defaults) ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "lovable-vitana-vers1")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "global")   # Use 'global' for Gemini 2.5
MODEL_ID = os.getenv("MODEL_ID", "gemini-2.5-pro")        # Stable, non-suffixed

# Init Vertex AI once
vertex_init(project=PROJECT_ID, location=VERTEX_LOCATION)
gemini = GenerativeModel(MODEL_ID)

# --- Schemas ---
class WorkItem(BaseModel):
    work_item_id: str
    description: str

class TaskPack(BaseModel):
    work_item_id: str
    prompt: str
    tests: List[Dict[str, Any]]
    acceptance: List[str]
    metadata: Dict[str, Any]

# --- App ---
app = FastAPI(title="CrewAI Prompt Synth")

@app.on_event("startup")
def _startup_healthcheck():
    try:
        _ = gemini.generate_content("healthcheck").text
    except Exception as e:
        raise RuntimeError(
            f"Vertex AI model check failed for {MODEL_ID} @ {VERTEX_LOCATION}: {e}"
        ) from e

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "location": VERTEX_LOCATION, "project": PROJECT_ID}

@app.post("/run", response_model=TaskPack)
def run_crew(item: WorkItem) -> TaskPack:
    try:
        prompt_text = gemini.generate_content(
            f"Create Task Pack JSON for: {item.description}"
        ).text
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Gemini generate_content failed: {str(e)}")

    return TaskPack(
        work_item_id=item.work_item_id,
        prompt=prompt_text,
        tests=[{"type": "contract", "code": "test('dark-mode', () => {})"}],
        acceptance=["Toggle works", "Persists in DB"],
        metadata={
            "max_tokens": 100000,
            "deadline": "2025-10-20",
            "model": MODEL_ID,
            "location": VERTEX_LOCATION,
            "project": PROJECT_ID
        }
    )
