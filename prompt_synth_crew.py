from crewai import Agent, Task, Crew, Process
from pydantic import BaseModel
import json, os
# from vertexai.preview.generative_models import GenerativeModel  # Google Cloud
# model = GenerativeModel("gemini-2.5-pro-exp-03-25")  # latest 2.5 Pro

class TaskPack(BaseModel):
    work_item_id: str
    prompt: str
    tests: list
    acceptance: list
    metadata: dict

# ---------- AGENTS ----------
pm = Agent(
    role="Senior Product Manager",
    goal="Turn a one-liner ticket into a full Task Pack",
    backstory="You excel at writing acceptance criteria and test cases.",
    verbose=False,
    allow_delegation=False,
)

designer = Agent(
    role="Test Designer",
    goal="Generate executable test code (Jest, SQL, Playwright, Lighthouse)",
    backstory="You never write prose—only runnable snippets.",
    verbose=False,
    allow_delegation=False,
)

engineer = Agent(
    role="Prompt Engineer",
    goal="Write a concise mega-prompt ≤ 8 000 tokens",
    backstory="You compress requirements into clear instructions.",
    verbose=False,
    allow_delegation=False,
)

assembler = Agent(
    role="Pack Assembler",
    goal="Output valid JSON that passes Zod schema",
    backstory="You validate once and return pure JSON.",
    verbose=False,
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
    expected_output="""JSON array under key "tests" with:
  1. contract test (Jest + supertest)
  2. migration SQL string
  3. RLS policy SQL string
  4. UI test (Playwright)
  5. performance test (Lighthouse)
Each element must be runnable code, NOT descriptions.""",
    agent=designer,
    output_json=True,
)

create_prompt = Task(
    description="Write the mega-prompt for Lovable.",
    expected_output="Markdown string ≤ 8 000 tokens, no repetition, ends with END_PROMPT.",
    agent=engineer,
    output_json=True,
)

assemble_pack = Task(
    description="Assemble final Task Pack JSON.",
    expected_output="Valid JSON that passes json_schema_validator exactly once.",
    agent=assembler,
    output_json=True,
)

# ---------- CREW ----------
crew = Crew(
    agents=[pm, designer, engineer, assembler],
    tasks=[analyze, design_tests, create_prompt, assemble_pack],
    process=Process.sequential,
    verbose=False,
    memory=False,
)

def kickoff(work_item_id: str, description: str) -> TaskPack:
    result = crew.kickoff(inputs={"work_item_id": work_item_id, "description": description})
    return TaskPack(**json.loads(result.raw))
