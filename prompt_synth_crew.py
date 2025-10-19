from crewai import Agent, Task, Crew, Process
from pydantic import BaseModel
import json, os
from vertexai.preview.generative_models import GenerativeModel

gemini = GenerativeModel("gemini-2.5-pro-exp-03-25")

class TaskPack(BaseModel):
    work_item_id: str
    prompt: str
    tests: list
    acceptance: list
    metadata: dict

def gemini_generate(prompt: str) -> str:
    """Native Gemini call – no LiteLLM."""
    return gemini.generate_content(prompt).text

# Agents (no llm= param – we call gemini_generate manually)
pm = Agent(role="PM", goal="clarify requirements", backstory="Senior PM", verbose=False)
designer = Agent(role="Test Designer", goal="output runnable test code", backstory="QA lead", verbose=False)
engineer = Agent(role="Prompt Engineer", goal="write concise mega-prompt ≤ 8 k tokens", backstory="Staff engineer", verbose=False)
assembler = Agent(role="Pack Assembler", goal="output valid JSON", backstory="DevOps", verbose=False)

# Tasks – we inject the Gemini response ourselves
analyze = Task(
    description="Analyze ticket and produce bullet requirements.",
    expected_output="Bullet list",
    agent=pm,
    tools=[gemini_generate],
)
design_tests = Task(
    description="Generate executable test code (Jest, SQL, Playwright, Lighthouse).",
    expected_output="JSON array under key tests with 5 runnable code blocks",
    agent=designer,
    tools=[gemini_generate],
)
create_prompt = Task(
    description="Write mega-prompt ≤ 8 k tokens.",
    expected_output="Markdown string",
    agent=engineer,
    tools=[gemini_generate],
)
assemble_pack = Task(
    description="Assemble final JSON that passes schema.",
    expected_output="Valid JSON",
    agent=assembler,
    tools=[gemini_generate],
)

crew = Crew(agents=[pm, designer, engineer, assembler], tasks=[analyze, design_tests, create_prompt, assemble_pack], process=Process.sequential, verbose=False)

def kickoff(work_item_id: str, description: str) -> TaskPack:
    result = crew.kickoff(inputs={"work_item_id": work_item_id, "description": description})
    return TaskPack(**json.loads(result.raw))
