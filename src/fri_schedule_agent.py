from __future__ import annotations

from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    TransformersModel,
    tool,
)
from langchain_core.runnables import Runnable
from schedule import fetch_timetable, Course, Session

print("Loading timetable...")
_COURSES: dict[str, Course] = fetch_timetable()
print(f"Loaded {len(_COURSES)} courses.")

class AgentLLM(Runnable):
    def __init__(self, agent):
        self.agent = agent
    
    def invoke(self, prompt: str, config=None) -> str:
        """Call the agent with the prompt."""
        if isinstance(prompt, dict):
            # If it's a dict (from LangChain piping), try to extract the actual prompt
            prompt = prompt.get("question", str(prompt))
        return self.agent.run(prompt)

def _time_to_minutes(t: str) -> int:
    h, m = map(int, t.split(":"))
    return h * 60 + m

def _sessions_overlap(a: Session, b: Session) -> bool:
    if a.day != b.day:
        return False
    a_start, a_end = _time_to_minutes(a.start), _time_to_minutes(a.end)
    b_start, b_end = _time_to_minutes(b.start), _time_to_minutes(b.end)
    return a_start < b_end and b_start < a_end


def _find_course(query: str) -> tuple[str, list[Session]]:
    q = query.strip().lower()
    for key, course in _COURSES.items():
        if q in key.lower() or q in course.full_name.lower():
            return course.full_name, course.sessions
    return "", []


@tool
def list_courses() -> str:
    """
    Return a list of all available course short names in the timetable.
    Use this to discover what courses exist before querying them.
    """
    return "\n".join(sorted(_COURSES.keys()))


@tool
def get_course_schedule(course_query: str) -> str:
    """
    Return the full schedule for a course matching the given query string.
    Matched case-insensitively against short name and full name.

    Args:
        course_query: Partial or full course name or code, e.g. "APS2", "Analiza omrežij".
    """
    query = course_query.strip().lower()
    matches = [
        course for key, course in _COURSES.items()
        if query in key.lower() or query in course.full_name.lower()
    ]
    if not matches:
        return f"No course found matching '{course_query}'."

    lines = []
    for course in matches:
        lines.append(f"{course.full_name}  [{course.short_name}]")
        for s in course.sessions:
            degrees = ", ".join(sorted(set(s.degree_types))) or "—"
            lines.append(
                f"  {s.day:12s}  {s.start}–{s.end}"
                f"  {s.session_type:18s}  [{s.type_code}]"
                f"  {s.classroom}  {s.teacher}  | {degrees}"
            )
        lines.append("")
    return "\n".join(lines)


@tool
def check_overlap(course_a: str, course_b: str) -> str:
    """
    Check whether any sessions of two courses overlap in time.

    Args:
        course_a: Partial or full name/code of the first course, e.g. "APS2".
        course_b: Partial or full name/code of the second course, e.g. "AAHRP".
    """
    name_a, sessions_a = _find_course(course_a)
    name_b, sessions_b = _find_course(course_b)

    if not sessions_a:
        return f"Course not found: '{course_a}'"
    if not sessions_b:
        return f"Course not found: '{course_b}'"

    overlaps = [
        f"  {sa.day}: {sa.start}–{sa.end} ({name_a}) overlaps {sb.start}–{sb.end} ({name_b})"
        for sa in sessions_a
        for sb in sessions_b
        if _sessions_overlap(sa, sb)
    ]

    if overlaps:
        return f"YES — {len(overlaps)} overlap(s):\n" + "\n".join(overlaps)
    return f"NO — '{name_a}' and '{name_b}' have no overlapping sessions."


@tool
def courses_on_day(day: str) -> str:
    """
    List all courses on a given weekday ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday), sorted by start time.

    Args:
        day: Day in English: "Monday", "Tuesday", "Wednesday", "Thursday", "Friday".
    """
    day_norm = day.strip().capitalize()
    results = []
    for course in sorted(_COURSES.values(), key=lambda c: c.full_name):
        for s in sorted(
            (s for s in course.sessions if s.day == day_norm),
            key=lambda s: s.start
        ):
            results.append(
                f"{s.start}–{s.end}  [{s.type_code:3s}]  "
                f"{course.full_name:25s}  {s.classroom}  {s.teacher}"
            )
    if not results:
        return f"No courses found on {day_norm}."
    return f"Courses on {day_norm} ({len(results)} sessions):\n" + "\n".join(results)


@tool
def courses_by_degree(degree: str) -> str:
    """
    List all courses for a given degree programme.

    Args:
        degree: One of: "Bachelor (UN)", "Bachelor (VSS)", "Master", "PhD".
    """
    degree_norm = degree.strip()
    matches = [
        f"  {c.short_name:30s}  {c.full_name}"
        for c in sorted(_COURSES.values(), key=lambda c: c.full_name)
        if any(degree_norm in s.degree_types for s in c.sessions)
    ]
    if not matches:
        return f"No courses found for degree '{degree_norm}'."
    return f"Courses for {degree_norm} ({len(matches)}):\n" + "\n".join(matches)


TIMETABLE_TOOLS = [
    list_courses,
    get_course_schedule,
    check_overlap,
    courses_on_day,
    courses_by_degree,
]

def build_agent(cache_dir):
    import os
    from transformers import AutoTokenizer
    
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
    
    # Set HuggingFace cache directory
    os.environ['HF_HOME'] = cache_dir

    model = TransformersModel(
        model_id=MODEL_ID,
        device_map="auto",                          # Distribute across available GPUs automatically
        max_new_tokens=8096,
        trust_remote_code=True,
    )
    
    # Properly configure the tokenizer for tool calling
    tokenizer = model.model.get_tokenizer() if hasattr(model.model, 'get_tokenizer') else None
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    agent = ToolCallingAgent(
        tools=TIMETABLE_TOOLS,
        model=model,
        max_steps=5,
    )

    return AgentLLM(agent)