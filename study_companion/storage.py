"""Atomic local persistence, with revision checks for notes and progress."""

import hashlib
import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path


MAX_NOTE_BYTES = 200_000


class ConflictError(Exception):
    """The file changed since the client loaded it."""


class StorageError(Exception):
    """Stored user work is unavailable or malformed; never silently reset it."""


def revision(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def empty_progress():
    return {
        "readings": [],
        "attempted": False,
        "explained": False,
        "minutes": {"reading": 0, "implementation": 0, "review": 0},
        "evidence": {"status": "not_run", "command": "", "summary": "", "recorded_at": None},
        "active_resource": None,
    }


def validate_progress(data):
    if not isinstance(data, dict) or data.keys() != empty_progress().keys():
        raise ValueError("Progress must contain the documented fields only.")
    readings = data["readings"]
    if (not isinstance(readings, list) or len(readings) > 100
            or any(not isinstance(item, str) or not re.fullmatch(r"[a-f0-9]{16}", item) for item in readings)
            or len(set(readings)) != len(readings)):
        raise ValueError("Reading selections must be unique resource IDs.")
    for field in ("attempted", "explained"):
        if type(data[field]) is not bool:
            raise ValueError(f"{field} must be a boolean.")
    minutes = data["minutes"]
    if not isinstance(minutes, dict) or minutes.keys() != {"reading", "implementation", "review"}:
        raise ValueError("Provide all three time allocations.")
    if any(type(value) is not int or not 0 <= value <= 100_000 for value in minutes.values()):
        raise ValueError("Logged minutes must be nonnegative whole numbers, at most 100000.")
    evidence = data["evidence"]
    if not isinstance(evidence, dict) or evidence.keys() != {"status", "command", "summary", "recorded_at"}:
        raise ValueError("Invalid evidence fields.")
    if evidence["status"] not in ("not_run", "failed", "passed"):
        raise ValueError("Unknown verification status.")
    if any(not isinstance(evidence[key], str) or len(evidence[key]) > limit
           for key, limit in (("command", 2000), ("summary", 20000))):
        raise ValueError("Evidence must be bounded text.")
    if evidence["status"] != "not_run" and not all(evidence[key].strip() for key in ("command", "summary")):
        raise ValueError("A recorded test outcome needs its command and observed results.")
    timestamp = evidence["recorded_at"]
    if timestamp is not None and (not isinstance(timestamp, str) or len(timestamp) > 40):
        raise ValueError("Invalid evidence timestamp.")
    active = data["active_resource"]
    if active is not None and (not isinstance(active, str) or not re.fullmatch(r"[a-f0-9]{16}", active)):
        raise ValueError("Invalid active resource.")


def atomic_write(path, text):
    path = Path(path)
    if path.is_symlink():
        raise StorageError(f"Refusing to replace a symlink: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            temporary.chmod(path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class Store:
    def __init__(self, data_dir, notes_dir):
        self.path = Path(data_dir) / "progress.json"
        self.notes_dir = Path(notes_dir)
        self.lock = threading.RLock()

    def _read(self):
        if not self.path.exists():
            return {"version": 1, "current_week": 1, "weeks": {str(i): empty_progress() for i in range(1, 11)}}
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
            if (state.keys() != {"version", "current_week", "weeks"} or type(state["version"]) is not int or state["version"] != 1
                    or type(state["current_week"]) is not int or not 1 <= state["current_week"] <= 10
                    or not isinstance(state["weeks"], dict) or state["weeks"].keys() != {str(i) for i in range(1, 11)}):
                raise ValueError("Unknown progress format")
            for progress in state["weeks"].values():
                validate_progress(progress)
            return state
        except (ValueError, TypeError, AttributeError, KeyError, UnicodeError) as error:
            raise StorageError("progress.json is malformed. Restore or repair it; it has not been overwritten.") from error

    @staticmethod
    def _progress_revision(progress):
        return revision(json.dumps(progress, sort_keys=True, ensure_ascii=False))

    def state(self):
        with self.lock:
            return self._read()

    def visit(self, week):
        with self.lock:
            state = self._read()
            state["current_week"] = week
            atomic_write(self.path, json.dumps(state, indent=2, ensure_ascii=False) + "\n")

    def note(self, week):
        path = self.notes_dir / f"week{week:02}.md"
        if path.is_symlink():
            raise StorageError(f"{path.name} is a symlink; choose a regular notes directory.")
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        return {"text": text, "revision": revision(text)}

    def week(self, week):
        with self.lock:
            progress = self._read()["weeks"][str(week)]
            return {"progress": progress, "progress_revision": self._progress_revision(progress), "note": self.note(week)}

    def save_note(self, week, text, expected_revision):
        if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_NOTE_BYTES:
            raise ValueError(f"A note must be text, at most {MAX_NOTE_BYTES} UTF-8 bytes.")
        with self.lock:
            if self.note(week)["revision"] != expected_revision:
                raise ConflictError("This note changed on disk or in another tab. Export your draft, or load the disk version before editing again.")
            atomic_write(self.notes_dir / f"week{week:02}.md", text)
            return {"text": text, "revision": revision(text)}

    def save_progress(self, week, data, expected_revision):
        validate_progress(data)
        with self.lock:
            state = self._read()
            old = state["weeks"][str(week)]
            if self._progress_revision(old) != expected_revision:
                raise ConflictError("This week's progress changed in another tab. Reload progress before making another change.")
            old_evidence = {key: old["evidence"][key] for key in ("status", "command", "summary")}
            new_evidence = {key: data["evidence"][key] for key in old_evidence}
            data["evidence"]["recorded_at"] = old["evidence"]["recorded_at"]
            if old_evidence != new_evidence:
                data["evidence"]["recorded_at"] = (
                    datetime.now(timezone.utc).isoformat(timespec="seconds")
                    if data["evidence"]["status"] != "not_run" else None
                )
            state["weeks"][str(week)] = data
            atomic_write(self.path, json.dumps(state, indent=2, ensure_ascii=False) + "\n")
            return {"progress": data, "progress_revision": self._progress_revision(data)}
