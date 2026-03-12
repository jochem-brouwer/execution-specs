"""Record engine API payload pairs to newline-delimited files."""

import json
import os
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List


class PayloadRecorder:
    """
    Record engine API payload pairs to disk for offline replay.

    Each recorded block produces two JSON-RPC lines (newPayload + FCU)
    written to phase-specific files under *output_dir*.

    A ``recording_context`` context manager sets the active phase and
    scenario so that ``_finalize_payload`` does not need to know about
    test identifiers.
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        (self._output_dir / "setup").mkdir(exist_ok=True)
        (self._output_dir / "testing").mkdir(exist_ok=True)

        self._lock = threading.Lock()
        self._scenario_index: Dict[str, int] = {}
        self._scenario_sequence: List[Dict[str, Any]] = []
        self._testing_seen_count: Dict[str, int] = {}

        # Thread-local storage for the active (phase, scenario) context.
        self._local = threading.local()

    # ------------------------------------------------------------------
    # Recording context
    # ------------------------------------------------------------------

    @contextmanager
    def recording_context(
        self, phase: str, scenario: str
    ) -> Generator[None, None, None]:
        """
        Set the active phase and scenario for the current thread.

        Nested contexts are supported; the previous context is restored
        on exit.
        """
        previous = getattr(self._local, "context", None)
        self._local.context = (phase, scenario)
        try:
            yield
        finally:
            self._local.context = previous

    @property
    def _current_context(self) -> tuple[str, str] | None:
        return getattr(self._local, "context", None)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_payload_pair(
        self,
        *,
        new_payload_method: str,
        new_payload_params: list,
        fcu_method: str,
        fcu_params: list,
    ) -> None:
        """
        Record a ``engine_newPayloadVX`` + ``engine_forkchoiceUpdatedVX``
        pair to the file for the current recording context.
        """
        ctx = self._current_context
        if ctx is None:
            return

        phase, scenario = ctx

        np_body = {
            "jsonrpc": "2.0",
            "id": int(time.time()),
            "method": new_payload_method,
            "params": new_payload_params,
        }
        fcu_body = {
            "jsonrpc": "2.0",
            "id": int(time.time()),
            "method": fcu_method,
            "params": fcu_params,
        }
        np_line = json.dumps(np_body, separators=(",", ":"))
        fcu_line = json.dumps(fcu_body, separators=(",", ":"))

        with self._lock:
            self._register_scenario(scenario)
            self._dump_pair(phase, scenario, np_line, fcu_line)

    # ------------------------------------------------------------------
    # File routing (mirrors mitm_addon._dump_pair_to_phase)
    # ------------------------------------------------------------------

    def _dump_pair(
        self,
        phase: str,
        scenario: str,
        np_line: str,
        fcu_line: str,
    ) -> None:
        """Route a payload pair to the correct file."""
        if phase == "setup":
            path = self._scenario_path("setup", scenario)
            self._append(path, np_line)
            self._append(path, fcu_line)
            return

        if phase == "cleanup":
            # Cleanup payloads are not needed for replay.
            return

        # -- testing phase --
        count = self._testing_seen_count.get(scenario, 0)
        testing_path = self._scenario_path("testing", scenario)
        setup_path = self._scenario_path("setup", scenario)

        if count == 0:
            self._overwrite(testing_path, [np_line, fcu_line])
        else:
            # Migrate previous testing payloads to setup.
            if testing_path.exists():
                prev_lines = [
                    ln
                    for ln in testing_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if ln.strip()
                ]
                for ln in prev_lines:
                    self._append(setup_path, ln)
            self._overwrite(testing_path, [np_line, fcu_line])

        self._testing_seen_count[scenario] = count + 1

    # ------------------------------------------------------------------
    # Scenario tracking
    # ------------------------------------------------------------------

    def _register_scenario(self, name: str) -> int:
        """Assign a sequential index to *name* (first-seen order)."""
        idx = self._scenario_index.get(name)
        if idx is not None:
            return idx
        idx = len(self._scenario_index) + 1
        self._scenario_index[name] = idx
        self._scenario_sequence.append({"index": idx, "name": name})
        self._write_scenario_order()
        return idx

    def _write_scenario_order(self) -> None:
        path = self._output_dir / "scenario_order.json"
        path.write_text(
            json.dumps(self._scenario_sequence, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Scenario name sanitization
    # ------------------------------------------------------------------

    @staticmethod
    def sanitize_scenario_name(node_id: str) -> str:
        """
        Convert a pytest ``node_id`` into a filesystem-safe scenario
        name matching the mitm addon convention.

        ``node_id`` has the form ``path/to/test.py::test_func[params]``.
        """
        if "::" in node_id:
            file_path_str, test_name = node_id.split("::", 1)
        else:
            file_path_str, test_name = node_id, "unknown_test"

        file_base = _sanitize_component(os.path.basename(file_path_str))
        test_name = _sanitize_component(test_name)

        suffix = ""
        match = re.search(r"-benchmark-gas-value_([^-]+)", test_name)
        if match:
            value = _sanitize_component(match.group(1))
            test_name = re.sub(
                r"-benchmark-gas-value_[^-]+",
                "-benchmark",
                test_name,
                count=1,
            )
            suffix = f"_{value}" if value else ""

        return f"{file_base}__{test_name}{suffix}"

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    def _scenario_path(self, phase: str, scenario: str) -> Path:
        return self._output_dir / phase / f"{scenario}.txt"

    @staticmethod
    def _append(path: Path, line: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    @staticmethod
    def _overwrite(path: Path, lines: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")


def _sanitize_component(s: str) -> str:
    """Make *s* safe for use as a filename component."""
    s = s.replace(os.sep, "_").replace("\\", "_").replace("/", "_")
    s = (
        s.replace("\x00", "")
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
    )
    return s.strip() or "unknown"
