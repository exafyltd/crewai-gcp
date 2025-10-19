from crewai import Agent, Task, Crew, Process
from pydantic import BaseModel
import json

# ✅ MODEL & PROVIDER SETTINGS FOR GEMINI via Google Vertex AI
GEMINI_MODEL = "gemini-2.5-pro-exp-03-25"  # Your selected model
PROVIDER = "google"  # Explicitly avoids OpenAI

class TaskPack(BaseModel):
    work_item_id: str
    prompt: str | None = None
    tests: list | None = None
    acceptance: list | None = None
    metadata: dict | None = None

# ---------- AGENTS ----------
pm = Agent(
    role="Senior Product Manager",
    goal="Turn a one-liner ticket into a full Task Pack",
    backstory="You excel at writing acceptance criteria and test cases.",
    model=GEMINI_MODEL,
    provider=PROVIDER,
    verbose=True,
    allow_delegation=False,
)

designer = Agent(
    role="Test Designer",
    goal="Generate executable test code (Jest, SQL, Playwright, Lighthouse)",
    backstory="You never write prose—only runnable code blocks.",
    model=GEMINI_MODEL,
    provider=PROVIDER,
    verbose=True,
    allow_delegation=False,
)

engineer = Agent(
    role="Prompt Engineer",
    goal="Write a concise mega-prompt ≤ 8000 tokens",
    backstory="You compress requirements into clear instructions.",
    model=GEMINI_MODEL,
    provider=PROVIDER,
    verbose=True,
    allow_delegation=False,
)

assembler = Agent(
    role="Pack Assembler",
    goal="Validate output and produce final JSON",
    backstory="You ensure schema correctness and return valid TaskPack.",
    model=GEMINI_MODEL,
    provider=PROVIDER,
    verbose=True,
    allow_delegation=False,
)

# ---------- TASKS ----------
analyze = Task(
    description="Analyze the work-item description.",
    expected_output="Bullet list of requirements and edge-cases.",
    agent=pm,
)

design_tests = Task(
    description="Generate executable test code.",
    expected_output="""{
      "tests": [
        {"contract": "<jest code>"},
        {"migration": "<sql>"},
        {"rls_policy": "<sql>"},
        {"ui_test": "<playwright>"},
        {"performance": "<lighthouse>"}
      ]
    }""",
    agent=designer,
    output_json=True,
)

create_prompt = Task(
    description="Write the mega-prompt for Lovable.",
    expected_output="Markdown string ≤ 8000 tokens, ends with END_PROMPT.",
    agent=engineer,
    output_json=True,
)

assemble_pack = Task(
    description="Assemble final Task Pack JSON.",
    expected_output="Valid JSON matching TaskPack schema.",
    agent=assembler,
    output_json=True,
)

# ---------- CREW ----------
crew = Crew(
    agents=[pm, designer, engineer, assembler],
    tasks=[analyze, design_tests, create_prompt, assemble_pack],
    process=Process.sequential,
    verbose=True,
    memory=False,
)

def kickoff(work_item_id: str, description: str) -> TaskPack:
    """
    Run crew with inputs, return TaskPack.
    """
    result = crew.kickoff(inputs={"work_item_id": work_item_id, "description": description})
    parsed = json.loads(result.raw)
    return TaskPack(**parsed)
