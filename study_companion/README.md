# MLSys Field Guide

A desktop-first, local study companion: the Field Guide's editorial layout in a
dark palette, generated from `SYLLABUS.md`. The server uses only the Python
standard library; the browser UI has no package manager, CDN, or build step.

## Start

From the repository root:

```bash
python -m study_companion
```

Open **http://localhost:8765**. Stop the server with Ctrl+C. Python 3.10 or newer
is sufficient; launching the companion does not import PyTorch.

If the port is occupied:

```bash
python -m study_companion --port 8767
```

The server binds to `127.0.0.1`. On the existing WSL setup, open the localhost URL
in your Windows browser. Keep one server running per progress/notes directory.

## Study workflow

- **All ten weeks:** the chapters, readings, allocation, implementation contracts,
  verification commands, and review questions come from the syllabus. Refresh the
  page after editing that file; there is no separate content-generation step.
- **Reading room:** assigned arXiv PDFs open beside your weekly Markdown notebook.
  Use **Open original** for YouTube videos, visual articles, book chapters, and
  tutorials, or if a browser cannot embed a PDF. The assignment and your notebook
  stay in the reader. Follow the syllabus order and the named sections/timestamps;
  linked text alternatives replace videos within the same allowance. External
  resources require an internet connection; the course UI and notes are local.
- **Progress:** readings, implementation attempts, manual test evidence, and your
  independent explanation are separate. A checkpoint is recorded complete only
  with a reported passing run and an explanation. The site never executes pytest
  or verifies a learner's claim automatically.
- **Time:** optional, editable totals in minutes for reading, implementation, and
  review. “Reading” includes watching/listening, pauses, notes, and comprehension
  checks. Checking a resource does not log time. At 12 logged hours, record the
  blocker and resume the same week in the next available study slot.
- **Focus timer:** an optional 25-minute countdown in the reader. It pauses when
  the tab is hidden or the reader closes. It never changes logged time, and resets
  on page reload. Forgetting to pause cannot inflate the weekly time ledger.
- **Notebook:** both editors share the same weekly note. Markdown autosaves after
  a short pause; **Save now** and **Export .md** are also available. The evidence
  template contains prompts, not answers.

The exercise and test links display local source files as text. Work on them in
your editor; copy the verification commands to your repository terminal.

## Local files and preserving work

| Location | Contents |
|---|---|
| `notes/week01.md` … `notes/week10.md` | Your editable, ordinary Markdown notes. Existing contents are loaded intact. |
| `.study-companion/progress.json` | Reading selections, current week, optional minutes, and manual evidence. Ignored by Git. |
| Browser localStorage | Unsaved-note recovery drafts only. The server files are authoritative. |

Back up the notes directory and `progress.json` together to preserve your study
record. Notes stay ordinary user files that you can choose to version in Git.
Design-preview sample progress is a different storage key and is never imported.

Resource checkboxes identify exact assignments, including their text and time
allowance. A revised assignment gets a new ID and appears unchecked; old saved
selections are retained, but do not certify the new assignment. Syllabus edits
do not erase your notes, logged time, or manual test evidence.

Writes replace individual files atomically. Note and progress updates include
the loaded revision: a changed file or another browser tab produces a conflict
instead of silently overwriting it. Your note draft stays in the editor and can
be exported; **Load disk version** reads the external edit. If a recovery draft
differs from disk on a later visit, the notebook offers to restore it. Malformed
progress files produce an error and are never automatically reset.

The UI waits for a note save before switching weeks. If a save fails, keep the tab
open, restore the server connection, and use **Save now** or export your draft.
Notes are limited to 200,000 UTF-8 bytes per week. No private repository files
are served: source viewing is limited to the syllabus, README, and assigned
exercise/test files. The server accepts only its own loopback host and same-origin
JSON writes; it has no command-execution endpoint.

For a separate scratch profile or testing:

```bash
python -m study_companion --port 8766 \
  --data-dir /tmp/mlsys-study-state \
  --notes-dir /tmp/mlsys-study-notes
```

`--repo /path/to/agentic-mlsys-course` selects another course checkout. The default
is the current working directory. Alternate storage directories do not change
the notebook's logical `weekNN.md` filenames.

## Implementation and checks

- `content.py`: parse the authoritative syllabus into the ten-week presentation model.
- `storage.py`: independent file-backed notes and progress with optimistic concurrency.
- `server.py`: loopback HTTP API and allowlisted static/source files.
- `static/`: dark Field Guide UI, small escaped Markdown renderer, native PDF reader.

API reads: `/api/course`, `/api/state`, `/api/weeks/1` through `/api/weeks/10`.
API writes: `PUT /api/current`, `PUT /api/weeks/N/progress`, and
`PUT /api/weeks/N/note`. Note/progress writes require the loaded revision. Notes
and progress have independent revisions, so a note edit cannot erase a checklist.

Run the companion contracts independently of the unsolved course exercises:

```bash
uv run --locked --extra dev python -m pytest tests/test_study_companion.py -q
```

Contracts cover course parsing, zero initial progress, file persistence across
restarts, stale-write rejection, validation, preserved corrupt files, HTTP scope,
and note/API round trips. Browser verification additionally covers the reading
pane, notes, week/stage navigation, and progress controls. This tooling is an
optional course companion, not another learner implementation component.
