# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional
import logging

# Import kickoff() from your root-level file
from prompt_synth_crew import kickoff

# Basic logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="Prompt Synth Service", version="1.0.0")

# CORS (loosen now, tighten to your domains later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# ----- Models -----
class WorkItem(BaseModel):
    work_item_id: str
    description: str

class TaskPack(BaseModel):
    work_item_id: str
    knowledge_proof: Dict[str, Any]
    files: Dict[str, str]                 # e.g. {"prompt.md": "...", "diff_plan.yaml": "..."}
    artifacts_url: Optional[str] = None   # e.g. Supabase Storage URL

# ----- Routes -----
@app.post("/run", response_model=TaskPack)
def run_crew(item: WorkItem) -> TaskPack:
    """
    Build and return a Task Pack for the given work item.
    """
    try:
        pack = kickoff(item.work_item_id, item.description)

        # Accept dict or already-validated TaskPack
        if isinstance(pack, dict):
            return TaskPack(**pack)
        if isinstance(pack, TaskPack):
            return pack

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

# Optional local dev entrypoint (not used by Cloud Run)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(8080), reload=True)
