import re
import pickle
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from datetime import datetime, timedelta

URL = "https://urnik.fri.uni-lj.si/timetable/fri-2025_2026-letni/allocations"
CACHE_FILE = Path(__file__).parent / "timetable_cache.pkl"

DAYS = {
    "MON": "Monday",
    "TUE": "Tuesday",
    "WED": "Wednesday",
    "THU": "Thursday",
    "FRI": "Friday",
    "SAT": "Saturday",
}

SESSION_TYPE_BY_DURATION = {
    2: "Lab / Tutorial",
    3: "Lecture",
}

DEGREE_MAP = {
    "univerzitetni":        "Bachelor (UN)",
    "visokošolski strokovni": "Bachelor (VSS)",
    "magistrski":           "Master",
    "doktorski":            "PhD",
}


@dataclass
class Session:
    day: str            # "Monday"
    start: str          # "10:00"
    end: str            # "12:00"
    duration: int       # hours (2 or 3)
    session_type: str   # "Lab / Tutorial" or "Lecture" (derived from duration)
    type_code: str      # original code from HTML: "LV", "P", "AV", ...
    classroom: str      # "PR10"
    teacher: str        # "Rok Gomišček"
    degree_types: list[str] = field(default_factory=list)   # ["Bachelor (UN)", "Master", ...]
    groups: list[str]   = field(default_factory=list)


@dataclass
class Course:
    short_name: str         # "APS2(63280)_LV"
    full_name: str          # "Algoritmi in podatkovne strukture 2(63280)_LV"
    sessions: list[Session] = field(default_factory=list)

    def __repr__(self):
        lines = [f"{self.full_name}  [{self.short_name}]"]
        for s in self.sessions:
            degrees = ", ".join(sorted(set(s.degree_types))) or "—"
            lines.append(
                f"   {s.day:12s}  {s.start} - {s.end}"
                f"  {s.session_type:18s}  [{s.type_code:3s}]"
                f"  {s.classroom:6s}  {s.teacher}"
                f"  | {degrees}"
            )
        return "\n".join(lines)


def _calculate_end(start: str, duration_hours: int) -> str:
    """Calculate end time from start time and duration in hours."""
    dt = datetime.strptime(start, "%H:%M") + timedelta(hours=duration_hours)
    return dt.strftime("%H:%M")


def _get_full_name(hover_div) -> str:
    if not hover_div:
        return ""
    lines = [
        line.strip()
        for line in hover_div.get_text(separator="\n").split("\n")
        if line.strip()
    ]
    if len(lines) >= 3:
        return lines[2]
    return ""


def _extract_degree_types(hover_div) -> list[str]:
    if not hover_div:
        return []

    text = hover_div.get_text(separator="\n")
    found = set()

    for match in re.finditer(r"stopnja:\s*([^\n,<]+)", text):
        raw = match.group(1).strip().lower()
        for keyword, label in DEGREE_MAP.items():
            if keyword in raw:
                found.add(label)
                break

    return sorted(found)


def _derive_session_type(duration: int) -> str:
    return SESSION_TYPE_BY_DURATION.get(duration, "Unknown")


def fetch_timetable(url: str = URL) -> dict[str, Course]:
    """
    Fetch and parse all course sessions from the FRI timetable.

    Returns a dict:  { short_name: Course }

    Each Course contains a list of Session objects with:
      - day           -> "Monday" / "Tuesday" / ...
      - start         -> "10:00"
      - end           -> "12:00"
      - duration      -> 2 or 3 (hours)
      - session_type  -> "Lab / Tutorial" or "Lecture" (derived from duration)
      - type_code     -> original HTML type: "LV", "P", "AV", ...
      - classroom     -> "PR10"
      - teacher       -> "Rok Gomišček"
      - degree_types  -> ["Bachelor (UN)", "Master", ...] (from hover text)
      - groups        -> ["2_BUN-RI_LV_01", ...]
    """
    if CACHE_FILE.exists():
        print("Loading timetable from cache...")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    
    # Retry logic with exponential backoff
    max_retries = 3
    timeout = 60  # Increased from 20 seconds
    
    for attempt in range(max_retries):
        try:
            print(f"Fetching timetable... (attempt {attempt + 1}/{max_retries})")
            response = requests.get(
                url, 
                headers={"User-Agent": "Mozilla/5.0"}, 
                timeout=timeout
            )
            response.raise_for_status()
            print("Timetable fetched successfully!")
            break
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"  Timeout. Retrying with longer timeout ({timeout * 1.5:.0f}s)...")
                timeout *= 1.5
            else:
                print(f"  Failed to fetch timetable after {max_retries} attempts. Using empty timetable.")
                return {}
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"  Error: {e}. Retrying...")
            else:
                print(f"  Failed to fetch timetable after {max_retries} attempts. Using empty timetable.")
                return {}

    soup = BeautifulSoup(response.text, "html.parser")
    courses: dict[str, Course] = {}

    for entry in soup.find_all("div", class_="grid-entry"):
        # --- data attributes ---
        day_raw  = entry.get("data-day", "")
        start    = entry.get("data-start", "")
        duration = int(entry.get("data-duration", 1))

        day = DAYS.get(day_raw, day_raw)
        end = _calculate_end(start, duration) if start else ""

        # --- short name and type code ---
        link_subject = entry.find("a", class_="link-subject")
        short_name = link_subject.get_text(strip=True) if link_subject else ""

        type_span = entry.find("span", class_="entry-type")
        type_code = type_span.get_text(strip=True).lstrip("| ").strip() if type_span else ""

        # --- full name and degree types from hover div ---
        hover = entry.find("div", class_="entry-hover")
        full_name    = _get_full_name(hover)
        degree_types = _extract_degree_types(hover)

        # --- classroom ---
        link_classroom = entry.find("a", class_="link-classroom")
        classroom = link_classroom.get_text(strip=True) if link_classroom else ""

        # --- teacher ---
        link_teacher = entry.find("a", class_="link-teacher")
        teacher = link_teacher.get_text(strip=True) if link_teacher else ""

        # --- groups ---
        groups = [
            a.get_text(strip=True)
            for a in entry.find_all("a", class_="link-group")
        ]

        # --- session type derived from duration ---
        session_type = _derive_session_type(duration)

        if short_name not in courses:
            courses[short_name] = Course(short_name=short_name, full_name=full_name or short_name)

        courses[short_name].sessions.append(Session(
            day=day,
            start=start,
            end=end,
            duration=duration,
            session_type=session_type,
            type_code=type_code,
            classroom=classroom,
            teacher=teacher,
            degree_types=degree_types,
            groups=groups,
        ))

    print(f"Saving timetable to cache: {CACHE_FILE}")
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(courses, f)

    return courses


if __name__ == "__main__":
    print("Fetching timetable...\n")
    courses = fetch_timetable()
    print(f"Found {len(courses)} courses:\n")
    for course in sorted(courses.values(), key=lambda c: c.full_name):
        print(course)
        print()