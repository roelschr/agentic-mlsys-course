"""Loopback-only HTTP server for the study companion; never executes exercises."""

import argparse
import json
import mimetypes
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .content import load_course
from .storage import ConflictError, StorageError, Store


ASSETS = Path(__file__).with_name("static")
MAX_BODY = 1_300_000  # Includes JSON escaping of a 200 KB note.


class StudyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, repo, data_dir=None, notes_dir=None):
        self.repo = Path(repo).resolve()
        self.course = load_course(self.repo)
        self.store = Store(data_dir or self.repo / ".study-companion", notes_dir or self.repo / "notes")
        self.sources = {"README.md", "SYLLABUS.md"}
        for week in self.course["weeks"]:
            self.sources.update((week["exercise"], week["tests"]))
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = "MLSysStudy/1.0"

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def send_data(self, status, body, content_type="application/json; charset=utf-8"):
        if isinstance(body, dict):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-src https://arxiv.org; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def error(self, status, message):
        self.send_data(status, {"error": message})

    def local_request(self):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:
            hosts.update(("127.0.0.1", "localhost"))
        host = self.headers.get("Host", "")
        if host not in hosts:
            self.error(HTTPStatus.FORBIDDEN, "Use the localhost URL printed by the server.")
            return False
        origin = self.headers.get("Origin")
        if (origin is not None and origin != f"http://{host}") or self.headers.get("Sec-Fetch-Site") == "cross-site":
            self.error(HTTPStatus.FORBIDDEN, "Cross-origin requests are not supported.")
            return False
        return True

    def do_GET(self):
        if not self.local_request():
            return
        path = urlsplit(self.path).path
        try:
            if path == "/api/course":
                # Reflect syllabus edits without a separate generation/build step.
                self.send_data(200, load_course(self.server.repo))
            elif path == "/api/state":
                self.send_data(200, self.server.store.state())
            elif match := re.fullmatch(r"/api/weeks/(10|[1-9])", path):
                self.send_data(200, self.server.store.week(int(match[1])))
            elif path.startswith("/source/") and path[8:] in self.server.sources:
                source = self.server.repo / path[8:]
                if source.resolve().is_relative_to(self.server.repo):
                    self.send_data(200, source.read_bytes(), "text/plain; charset=utf-8")
                else:
                    self.error(404, "Source not found.")
            elif path in ("/", "/index.html", "/styles.css", "/app.js", "/markdown.js"):
                asset = ASSETS / ("index.html" if path == "/" else path[1:])
                mime = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
                self.send_data(200, asset.read_bytes(), mime + "; charset=utf-8")
            elif path == "/favicon.ico":
                self.send_data(204, b"", "image/x-icon")
            else:
                self.error(404, "Not found.")
        except (StorageError, OSError, UnicodeError, ValueError) as error:
            self.error(503, f"Could not read course data: {error}")

    def read_json(self):
        if self.headers.get_content_type() != "application/json":
            raise ValueError("Send application/json.")
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("Chunked request bodies are not supported.")
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= MAX_BODY:
            raise ValueError(f"Request size must be between 1 and {MAX_BODY} bytes.")
        body = json.loads(self.rfile.read(length))
        if not isinstance(body, dict):
            raise ValueError("Send a JSON object.")
        return body

    def do_PUT(self):
        self.close_connection = True
        if not self.local_request():
            return
        path = urlsplit(self.path).path
        try:
            body = self.read_json()
            if path == "/api/current":
                week = body.get("week")
                if body.keys() != {"week"} or type(week) is not int or not 1 <= week <= 10:
                    raise ValueError("Choose a week from 1 to 10.")
                self.server.store.visit(week)
                self.send_data(200, {"week": week})
                return
            match = re.fullmatch(r"/api/weeks/(10|[1-9])/(note|progress)", path)
            if not match:
                self.error(404, "Not found.")
                return
            week, kind = int(match[1]), match[2]
            value_key = "text" if kind == "note" else "progress"
            if body.keys() != {"revision", value_key} or not isinstance(body["revision"], str):
                raise ValueError("Provide the loaded revision and the new value.")
            if kind == "note":
                result = self.server.store.save_note(week, body["text"], body["revision"])
            else:
                result = self.server.store.save_progress(week, body["progress"], body["revision"])
            self.send_data(200, result)
        except ConflictError as error:
            self.error(409, str(error))
        except (ValueError, UnicodeError, TypeError) as error:
            self.error(400, str(error))
        except (StorageError, OSError) as error:
            self.error(503, f"Could not save your work: {error}")


def main():
    parser = argparse.ArgumentParser(description="Run the local MLSys Field Guide (Python standard library only).")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository containing SYLLABUS.md (default: current directory)")
    parser.add_argument("--data-dir", type=Path, help="Progress directory (default: <repo>/.study-companion)")
    parser.add_argument("--notes-dir", type=Path, help="Notes directory (default: <repo>/notes)")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("Port must be between 0 and 65535.")
    try:
        server = StudyServer(("127.0.0.1", args.port), args.repo, args.data_dir, args.notes_dir)
    except (OSError, ValueError, StopIteration) as error:
        parser.exit(1, f"Cannot start the study companion: {error}\nRun from the repository root, or pass --repo.\n")
    with server:
        print(f"\nMLSys Field Guide → http://localhost:{server.server_port}\n", flush=True)
        print(f"Notes: {server.store.notes_dir}\nProgress: {server.store.path}\nPress Ctrl+C to stop.\n", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
