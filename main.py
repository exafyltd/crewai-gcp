# --- New: CrewAI-powered endpoint (uses Crew -> Agent -> Task flow) ---
@app.post("/crew", response_model=TaskPack)
def crew_pipeline(item: WorkItem) -> TaskPack:
    """
    Minimal CrewAI integration that uses Vertex Gemini 2.5 via LiteLLM backend.
    Sanitizes output to plain JSON (no markdown fences) for downstream use.
    """
    import re, json

    def to_plain_json(text: str) -> str:
        # 1) Strip fenced blocks ```json ... ``` or ``` ... ```
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)

        # 2) If there's extra prose, attempt to extract the first JSON object
        #    by locating the first '{' to the matching '}' using a simple stack.
        if not text.lstrip().startswith("{"):
            start = text.find("{")
            if start != -1:
                stack = 0
                for i in range(start, len(text)):
                    if text[i] == "{":
                        stack += 1
                    elif text[i] == "}":
                        stack -= 1
                        if stack == 0:
                            candidate = text[start:i+1]
                            text = candidate
                            break

        # 3) Validate it’s JSON; if not, fall back to raw text
        try:
            obj = json.loads(text)
            return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return text  # return best-effort string; clients can decide

    try:
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

        task = Task(
            description=(
                "Create a Task Pack JSON for the following work item.\n\n"
                f"Work Item Description:\n{item.description}\n\n"
                "Return ONLY a single valid JSON object with keys:\n"
                "epicTitle, epicDescription, taskPack[]. Each task in taskPack must have:\n"
                "id, title, description, status (To Do/In Progress/Done), "
                "acceptanceCriteria[], tests{unit[],integration[],e2e[],manual[]?}.\n"
                "No markdown fences, no extra prose."
            ),
            expected_output="A single valid JSON object (no markdown fences, no extra text).",
            agent=synthesizer,
        )

        crew = Crew(agents=[synthesizer], tasks=[task])
        result_text = str(crew.kickoff())
        cleaned = to_plain_json(result_text)

        return TaskPack(
            work_item_id=item.work_item_id,
            prompt=cleaned,
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
        raise HTTPException(status_code=503, detail=f"CrewAI pipeline failed: {str(e)}")
