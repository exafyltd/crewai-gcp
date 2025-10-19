# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import List, Dict, Any

# --- Vertex AI (stable path) ---
from vertexai import init as vertex_init
from vertexai.generative_models import GenerativeModel

# --- CrewAI (via LiteLLM adapter) ---
# CrewAI's LLM wrapper delegates to LiteLLM; "vertex_ai/<model>" uses Vertex with ADC
from crewai import Agent, Task, Crew, LLM

# --------- Config (env-driven with safe defaults) ---------
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "lovable-vitana-vers1")
# Keep Vertex GLOBAL for Gemini 2.5 to avoid regional model 404s
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "global")
MODEL_ID = os.getenv("MODEL_ID", "gemini-2.5-pro")

# Initialize Vertex once at import (for direct Gemini calls)
vertex_init(project=PROJECT_ID, location=VERTEX_LOCATION)
gemini = GenerativeModel(MODEL_ID)

# CrewAI LLM configured to use Vertex via LiteLLM backend
# Requires only ADC (default on Cloud Run) + project/region already set.
# See: model naming "vertex_ai/<model-id>"
crew_llm = LLM(model=f"vertex_ai/{MODEL_ID}", temperature=0.2)

# --------- Schemas ---------
class WorkItem(BaseModel):
    work_item_id: str
    description: str

class TaskPack(BaseModel):
    work_item_id: str
    prompt: str
    tests: List[Dict[str, Any]]
    acceptance: List[str]
    metadata: Dict[str, Any]

# --------- App ---------
app = FastAPI(title="CrewAI Prompt Synth")

@app.on_event("startup")
def _startup_healthcheck():
    """
    Fail fast if model/location/permissions are wrong,
    so Cloud Run won't shift traffic to a bad revision.
    """
    try:
        _ = gemini.generate_content("healthcheck").text
    except Exception as e:
        raise RuntimeError(
            f"Vertex AI model check failed for {MODEL_ID} @ {VERTEX_LOCATION}: {e}"
        ) from e

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "location": VERTEX_LOCATION,
        "project": PROJECT_ID
    }

# --- Existing simple endpoint (kept intact) ---
@app.post("/run", response_model=TaskPack)
def run_crew(item: WorkItem) -> TaskPack:
    try:
        prompt_text = gemini.generate_content(
            f"Create Task Pack JSON for: {item.description}"
        ).text
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Gemini generate_content failed: {str(e)}"
        )

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

# --- New: CrewAI-powered endpoint (uses Crew -> Agent -> Task flow) ---
@app.post("/crew", response_model=TaskPack)
def crew_pipeline(item: WorkItem) -> TaskPack:
    """
    Minimal CrewAI integration that uses Vertex Gemini 2.5 via LiteLLM backend.
    Keeps behavior similar to /run, but goes through Agent+Task orchestration.
    """
    try:
        # 1) Define an agent that “thinks” like a product/eng synthesis lead
        synthesizer = Agent(
            role="Task Pack Synthesizer",
            goal="Produce an actionable Task Pack JSON with clear tests & acceptance criteria.",
            backstory=(
                "You lead Vitana's engineering planning. You convert product intents "
                "into execution-ready task packs with IDs, acceptance criteria, and tests."
            ),
            llm=crew_llm,
            allow_delegation=False,
            verbose=False,
        )

        # 2) Define the task the agent will perform
        task = Task(
            description=(
                "Create a Task Pack JSON for the following work item.\n\n"
                f"Work Item Description:\n{item.description}\n\n"
                "Output must be a concise JSON object (no markdown fences) with keys:\n"
                "epicTitle, epicDescription, taskPack[]. Each task in taskPack must have:\n"
                "id, title, description, status (To Do/In Progress/Done), acceptanceCriteria[], tests{unit[],integration[],e2e[],manual[]?}."
            ),
            expected_output="A single valid JSON object as specified (no extra prose).",
            agent=synthesizer,
        )

        # 3) Run the crew
        crew = Crew(agents=[synthesizer], tasks=[task])
        result_text = crew.kickoff()  # returns a string

        # 4) Return in same TaskPack envelope as /run for compatibility
        return TaskPack(
            work_item_id=item.work_item_id,
            prompt=str(result_text),
            tests=[{"type": "contract", "code": "test('crewai-dark-mode', () => {})"}],
            acceptance=["JSON valid", "Criteria precise", "Tests runnable"],
            metadata={
                "engine": "CrewAI",
                "model": MODEL_ID,
                "location": VERTEX_LOCATION,
                "project": PROJECT_ID
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"CrewAI pipeline failed: {str(e)}"
        )
