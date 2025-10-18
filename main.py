from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json, os
from crews.prompt_synth_crew import kickoff

app = FastAPI()

class WorkItem(BaseModel):
    work_item_id: str
    description: str

@app.post("/run")
def run_crew(item: WorkItem):
    try:
        pack = kickoff(item.work_item_id, item.description)
        return pack.dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check for Cloud Run
@app.get("/")
def health():
    return {"status": "ok"}
