# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict
import logging

# Ensure crews/ is a package with __init__.py
from crews.prompt_synth_crew import kickoff  # expects a function we define below

app = FastAPI(title="Prompt Synth Service", version="1.0.0")

# Optional CORS if you’ll call it from the web app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

class WorkItem(BaseModel):
    work_item_id: str
    description: str

class TaskPack(BaseModel):
    # shape this to what kickoff returns
    # minimal example:
    work_item_id: str
    knowledge_proof: Dict[str, Any]
    files: Dict[str, str]  # e.g., {"prompt.md": "...", "diff_plan.yaml": "..."}
    artifacts_url: str | None = None

@app.post("/run", response_model=TaskPack)
def run_crew(item: WorkItem) -> TaskPack:
    try:
        pack = kickoff(item.work_item_id, item.description)
        # Accept both dict or TaskPack from kickoff
        if isinstance(pack, dict):
            # validate/normalize via Pydantic
            return TaskPack(**pack)
        elif isinstance(pack, TaskPack):
            return pack
        else:
            logging.error("kickoff() returned unsupported type: %s", type(pack))
            raise HTTPException(status_code=500, detail="Invalid TaskPack from crew.")
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Crew execution failed")
        raise HTTPException(status_code=500, detail=f"Crew error: {e}")

@app.get("/")
def health():
    return {"status": "ok"}
