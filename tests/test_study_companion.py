"""Contracts for course tooling only; no learner exercise is executed here."""

import copy
import http.client
import json
import threading
from pathlib import Path

import pytest

from study_companion.content import load_course
from study_companion.server import StudyServer
from study_companion.storage import ConflictError, StorageError, Store, empty_progress, revision


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "state", tmp_path / "notes")


@pytest.fixture
def server(tmp_path):
    httpd = StudyServer(("127.0.0.1", 0), REPO, tmp_path / "state", tmp_path / "notes")
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()
    worker.join(timeout=5)


def request(server, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    base_headers = {"Content-Type": "application/json"}
    base_headers.update(headers or {})
    connection.request(method, path, body=json.dumps(body) if body is not None else None, headers=base_headers)
    response = connection.getresponse()
    data = response.read()
    status, content_type = response.status, response.getheader("Content-Type")
    connection.close()
    return status, json.loads(data) if content_type.startswith("application/json") else data


def test_all_ten_weeks_are_generated_from_the_syllabus():
    course = load_course(REPO)
    assert [week["number"] for week in course["weeks"]] == list(range(1, 11))
    assert "TinyDecoder" in course["capstone"]
    assert "TinyDecoder" not in course["weeks"][7]["review"]
    assert "30m: derive" in course["review_format"]
    for week in course["weeks"]:
        assert sum(resource["minutes"] for resource in week["resources"]) == 480
        assert week["budget"] == {"reading": 480, "implementation": 120, "review": 120}
        assert week["concepts"] and week["implementation"] and week["review"] and week["checkpoint"]
        assert all(command.startswith("uv run --locked --extra dev python -m pytest") for command in week["commands"])
        assert len({resource["id"] for resource in week["resources"]}) == len(week["resources"])
        assert 7 <= len(week["resources"]) <= 8
        assert week["resources"][0]["url"] and week["resources"][0]["pdf"] is None
        assert all(all(label in resource["assignment"] for label in ("Prerequisite:", "Purpose:", "**Check:**"))
                   for resource in week["resources"])
        urls = [resource["url"] for resource in week["resources"]]
        assert any(url and "youtube.com" in url for url in urls)
        assert 1 <= sum(bool(url and "arxiv.org" in url) for url in urls) <= 2
        assert any(url and "youtube.com" not in url and "arxiv.org" not in url for url in urls)
        assert any(url is None for url in urls)
    attention = course["weeks"][2]["resources"][0]
    assert attention["url"] == "https://www.youtube.com/watch?v=eMlx5fFNoYc"
    assert "00:00–26:09" in attention["assignment"]
    assert "https://www.3blue1brown.com/lessons/attention/" in attention["assignment"]
    gqa = course["weeks"][2]["resources"][3]
    assert gqa["url"] == "https://arxiv.org/abs/2305.13245"
    assert gqa["pdf"] == "https://arxiv.org/pdf/2305.13245"
    assert "Figure 2" in gqa["assignment"]
    assert course["weeks"][8]["resources"][5]["url"] is None
    assert len(course["weeks"][9]["commands"]) == 2


def test_source_edits_change_content_without_touching_saved_work(tmp_path):
    source = (REPO / "SYLLABUS.md").read_text()
    (tmp_path / "SYLLABUS.md").write_text(source)
    before = load_course(tmp_path)
    (tmp_path / "SYLLABUS.md").write_text(source.replace("Read §2.2 and\n  Figure 2", "Read §2.2 and\n  Figure 2; annotate the shapes"))
    after = load_course(tmp_path)
    assert before["revision"] != after["revision"]
    assert before["weeks"][2]["resources"][3]["id"] != after["weeks"][2]["resources"][3]["id"]
    assert before["weeks"][0] == after["weeks"][0]


def test_initial_state_is_real_zero_progress_and_reading_never_marks_a_gate(store):
    state = store.state()
    assert state["current_week"] == 1
    assert not store.path.exists()
    assert all(value == empty_progress() for value in state["weeks"].values())
    week = store.week(3)
    progress = week["progress"]
    progress["readings"] = [item["id"] for item in load_course(REPO)["weeks"][2]["resources"]]
    store.save_progress(3, progress, week["progress_revision"])
    result = store.week(3)["progress"]
    assert result["evidence"]["status"] == "not_run"
    assert not result["explained"]
    assert result["minutes"] == {"reading": 0, "implementation": 0, "review": 0}


def test_notes_preserve_unicode_existing_content_and_survive_restart(store):
    store.notes_dir.mkdir()
    original = "# My own note\n\nPrediction: θ changes; ε matters.\n"
    path = store.notes_dir / "week03.md"
    path.write_text(original)
    loaded = store.week(3)["note"]
    assert loaded["text"] == original
    updated = original + "\nObserved: still investigating.\n"
    result = store.save_note(3, updated, loaded["revision"])
    assert result["text"] == updated
    restarted = Store(store.path.parent, store.notes_dir)
    assert restarted.week(3)["note"] == result
    assert not (store.notes_dir / "week02.md").exists()
    assert list(store.notes_dir.iterdir()) == [path]


def test_external_note_edit_is_not_overwritten_by_stale_browser(store):
    loaded = store.week(2)["note"]
    store.notes_dir.mkdir()
    path = store.notes_dir / "week02.md"
    path.write_text("Edited in my IDE.\n")
    with pytest.raises(ConflictError):
        store.save_note(2, "Stale browser edit", loaded["revision"])
    assert path.read_text() == "Edited in my IDE.\n"


def test_concurrent_progress_revisions_and_unrelated_weeks(store):
    first = store.week(3)
    stale = store.week(3)
    other = store.week(4)
    first["progress"]["attempted"] = True
    store.save_progress(3, first["progress"], first["progress_revision"])
    stale["progress"]["explained"] = True
    with pytest.raises(ConflictError):
        store.save_progress(3, stale["progress"], stale["progress_revision"])
    other["progress"]["minutes"]["reading"] = 75
    store.save_progress(4, other["progress"], other["progress_revision"])
    store.visit(4)
    assert store.week(3)["progress"]["attempted"]
    assert not store.week(3)["progress"]["explained"]
    assert store.week(4)["progress"]["minutes"]["reading"] == 75
    assert Store(store.path.parent, store.notes_dir).state()["current_week"] == 4


def test_reported_evidence_requires_command_and_observation_and_gets_timestamp(store):
    loaded = store.week(1)
    progress = loaded["progress"]
    progress["evidence"]["status"] = "passed"
    with pytest.raises(ValueError, match="command and observed results"):
        store.save_progress(1, progress, loaded["progress_revision"])
    assert store.week(1)["progress"]["evidence"]["status"] == "not_run"
    progress["evidence"].update(command="python -m pytest tests/test_week01.py -q", summary="Learner-reported result, CPU float64.")
    result = store.save_progress(1, progress, loaded["progress_revision"])
    recorded = result["progress"]["evidence"]["recorded_at"]
    assert recorded and recorded.endswith("+00:00")
    result["progress"]["explained"] = True
    next_result = store.save_progress(1, result["progress"], result["progress_revision"])
    assert next_result["progress"]["evidence"]["recorded_at"] == recorded


@pytest.mark.parametrize("change", [
    lambda p: p["minutes"].update(reading=-1),
    lambda p: p["minutes"].update(reading=True),
    lambda p: p["minutes"].update(reading=1.5),
    lambda p: p.update(attempted="yes"),
    lambda p: p.update(readings=["../SYLLABUS.md"]),
    lambda p: p.update(readings=["a" * 16, "a" * 16]),
    lambda p: p.update(extra="not a valid field"),
    lambda p: p["evidence"].update(status="pretend"),
])
def test_invalid_progress_never_changes_persisted_state(store, change):
    store.visit(1)
    before = store.path.read_bytes()
    loaded = store.week(1)
    invalid = copy.deepcopy(loaded["progress"])
    change(invalid)
    with pytest.raises(ValueError):
        store.save_progress(1, invalid, loaded["progress_revision"])
    assert store.path.read_bytes() == before


def test_malformed_progress_is_not_replaced_with_defaults(store):
    store.path.parent.mkdir()
    store.path.write_text("{ damaged progress")
    with pytest.raises(StorageError, match="has not been overwritten"):
        store.visit(1)
    assert store.path.read_text() == "{ damaged progress"


def test_oversized_note_and_symlink_do_not_replace_user_files(store, tmp_path):
    with pytest.raises(ValueError):
        store.save_note(1, "x" * 200_001, revision(""))
    store.notes_dir.mkdir()
    target = tmp_path / "original.md"
    target.write_text("Keep me")
    (store.notes_dir / "week01.md").symlink_to(target)
    with pytest.raises(StorageError):
        store.save_note(1, "New content", revision("Keep me"))
    assert target.read_text() == "Keep me"


def test_http_course_assets_and_note_roundtrip(server):
    status, course = request(server, "GET", "/api/course")
    assert status == 200 and len(course["weeks"]) == 10
    for path in ("/", "/app.js", "/markdown.js", "/styles.css", "/source/SYLLABUS.md", "/source/drills/week03.py"):
        status, data = request(server, "GET", path)
        assert status == 200 and data
    status, week = request(server, "GET", "/api/weeks/3")
    assert status == 200
    payload = {"revision": week["note"]["revision"], "text": "# Prediction\n\nCache capacity ≠ valid length.\n"}
    status, saved = request(server, "PUT", "/api/weeks/3/note", payload)
    assert status == 200
    assert saved["text"] == payload["text"]
    assert (server.store.notes_dir / "week03.md").read_text() == payload["text"]
    payload["text"] = "Stale request"
    assert request(server, "PUT", "/api/weeks/3/note", payload)[0] == 409


def test_http_scope_origin_and_malformed_requests(server):
    for path in ("/source/.env", "/source/../.git/config", "/.git/config", "/design-preview/index.html", "/api/weeks/0", "/api/weeks/11"):
        assert request(server, "GET", path)[0] == 404
    assert request(server, "GET", "/api/state", headers={"Host": "evil.example"})[0] == 403
    assert request(server, "PUT", "/api/current", {"week": 3}, {"Origin": "https://evil.example"})[0] == 403
    assert request(server, "PUT", "/api/current", {"week": 3}, {"Content-Type": "text/plain"})[0] == 400
    for payload in ([], {"week": True}, {"week": 11}, {"week": 1, "extra": "x"}):
        assert request(server, "PUT", "/api/current", payload)[0] == 400
    assert server.store.state()["current_week"] == 1
    assert request(server, "PUT", "/api/current", {"week": 10})[0] == 200
    assert request(server, "GET", "/api/state")[1]["current_week"] == 10


def test_malformed_live_syllabus_returns_json_error(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = (REPO / "SYLLABUS.md").read_text()
    syllabus = repo / "SYLLABUS.md"
    syllabus.write_text(source)
    httpd = StudyServer(("127.0.0.1", 0), repo, tmp_path / "state", tmp_path / "notes")
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    try:
        syllabus.write_text(source.replace("## Contract and pacing", "## Pacing"))
        status, data = request(httpd, "GET", "/api/course")
        assert status == 503
        assert "Could not read course data" in data["error"]
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=5)


def test_static_ui_supports_narrow_viewports_and_budget_docs_agree():
    css = (REPO / "study_companion/static/styles.css").read_text()
    assert "min-width:1060px" not in css
    assert "@media (max-width:760px)" in css
    assert "11-hour week" not in (REPO / "README.md").read_text()
