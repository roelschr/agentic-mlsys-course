"""Read the authoritative course Markdown into a small presentation model."""

import hashlib
import re
from pathlib import Path


def sections(markdown, level):
    """Return (heading, body) pairs without consuming nested headings."""
    matches = list(re.finditer(rf"^{'#' * level} (.+)$", markdown, re.MULTILINE))
    return [
        (match[1], markdown[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(markdown)].strip())
        for i, match in enumerate(matches)
    ]


def reading_assignments(markdown):
    """Extract budgeted bullets, retaining their exact assignment text."""
    resources = []
    for match in re.finditer(r"^- (\d+)m: (.*(?:\n(?!\n|[-#]).+)*)", markdown, re.MULTILINE):
        text = re.sub(r"\s*\n\s*", " ", match[2]).strip()
        link = re.search(r"\[([^]]+)\]\((https?://[^\s)]+)\)", text)
        title = link[1] if link else "Revisit your earlier work"
        url = link[2] if link else None
        pdf = None
        if url and re.fullmatch(r"https://arxiv\.org/abs/[\d.v]+", url):
            pdf = url.replace("/abs/", "/pdf/")
        detail = text
        if link:
            detail = text[link.end():].lstrip(" ,;:")
            if text[:link.start()].strip():
                title = f"Revisit · {title}"
        resources.append({
            "id": hashlib.sha256(match[0].encode()).hexdigest()[:16],
            "title": title,
            "minutes": int(match[1]),
            "assignment": text,
            "detail": detail,
            "url": url,
            "pdf": pdf,
        })
    return resources


def load_course(repo):
    """Load all ten chapters, their contracts, and the shared capstone scope."""
    markdown = (Path(repo) / "SYLLABUS.md").read_text(encoding="utf-8")
    chapters = sections(markdown, 2)
    weeks = []
    for title, body in chapters:
        match = re.match(r"Week (\d+): (.+)", title)
        if not match:
            continue
        number = int(match[1])
        parts = sections(body, 3)
        by_title = dict(parts)
        readings = next((value for key, value in parts if key.startswith("Reading &")), "")
        verification = by_title.get("PyTest verification target", "")
        commands = re.findall(r"`(python -m pytest [^`]+)`", verification)
        resources = reading_assignments(readings)
        if not resources or sum(item["minutes"] for item in resources) != 180:
            raise ValueError(f"Week {number} must contain its full 180-minute reading allocation.")
        if not commands:
            raise ValueError(f"Week {number} has no pytest verification command.")
        checkpoint = re.search(r"\*\*(?:Final checkpoint|Checkpoint):\*\* ([\s\S]+)$", body)
        weeks.append({
            "number": number,
            "title": match[2],
            "budget": {"reading": 180, "implementation": 360, "review": 120},
            "resources": resources,
            "concepts": by_title.get("Core concepts", ""),
            "implementation": next((value for key, value in parts if key.startswith(("Target implementation", "Integration deliverable"))), ""),
            "verification": verification,
            "review": next((value.split("**Checkpoint:")[0].split("**Final checkpoint:")[0].strip()
                            for key, value in parts if "Grill Yourself" in key), ""),
            "checkpoint": checkpoint[1].replace("[ ] ", "", 1).strip() if checkpoint else "",
            "commands": [f"uv run --locked --extra dev {command}" for command in commands],
            "exercise": f"drills/week{number:02}.py" if number < 9 else "capstone/project.py",
            "tests": f"tests/test_week{number:02}.py" if number < 9 else "tests/test_capstone.py",
        })
    if [week["number"] for week in weeks] != list(range(1, 11)):
        raise ValueError("The syllabus must contain Weeks 1–10 in order.")
    pacing = next(body for title, body in chapters if title == "Contract and pacing")
    review_format = next(body for title, body in sections(pacing, 3) if title.startswith("Reusable review"))
    return {
        "title": "MLSys Field Guide",
        "revision": hashlib.sha256(markdown.encode()).hexdigest()[:12],
        "pacing": pacing.split("### Tracking")[0].strip(),
        "review_format": review_format,
        "capstone": next(body for title, body in chapters if title.startswith("Weeks 9")),
        "weeks": weeks,
    }
