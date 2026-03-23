#!/usr/bin/env python3
"""Experimental curses TUI for note extraction orchestration."""

from __future__ import annotations

import argparse
import curses
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experimental.extract_tui.core import (  # noqa: E402
    create_quick_test_note,
    DEFAULT_AGENT,
    DEFAULT_DB,
    DEFAULT_MODEL,
    DEFAULT_VAULT,
    NoteRow,
    RunResult,
    VaultSummary,
    clear_selection,
    discover_agents,
    discover_models,
    load_pending_rows,
    make_run_log_dir,
    relative_path,
    run_extraction_once,
    reset_tracker_db,
    safe_log_name,
    select_first_n,
    selected_count,
    short_time,
    toggle_select_all,
    verify_post_extract_status,
)


def _truncate(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


class ExtractTUI:
    def __init__(
        self,
        *,
        repo_root: Path,
        vault_path: Path,
        db_path: Path,
        agent: str,
        model: str,
    ):
        self.repo_root = repo_root
        self.vault_path = vault_path
        self.db_path = db_path

        self.summary = VaultSummary(
            total=0,
            needs_extraction=0,
            ok=0,
            first_extraction=0,
            content_changed=0,
        )
        self.rows: list[NoteRow] = []
        self.cursor = 0
        self.top = 0
        self.notice = ""
        self.last_run_dir: Path | None = None
        self.use_colors = False
        self.last_failed_paths: list[str] = []
        self.command_center_lines: list[str] = [
            "No command executed yet.",
            "Press e to run extraction for selected notes.",
        ]

        self.agents = discover_agents(repo_root)
        self.models = discover_models(repo_root)

        self.current_agent = agent if agent else DEFAULT_AGENT
        if self.current_agent not in self.agents:
            self.agents.insert(0, self.current_agent)

        self.current_model = model if model else DEFAULT_MODEL
        if self.current_model not in self.models:
            self.models.insert(0, self.current_model)

        self.refresh_rows(initial=True)

    def set_notice(self, text: str) -> None:
        self.notice = text

    def _init_colors(self) -> None:
        if not curses.has_colors():
            self.use_colors = False
            return
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_MAGENTA, -1)
        curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_WHITE)
        curses.init_pair(6, curses.COLOR_RED, -1)
        self.use_colors = True

    def _attr(self, pair: int, fallback: int = curses.A_NORMAL) -> int:
        if self.use_colors:
            return curses.color_pair(pair)
        return fallback

    def _label_value(self, label: str, value: str) -> str:
        return f"{label}:{value}"

    @staticmethod
    def _tail_lines(text: str, limit: int = 6) -> list[str]:
        if not text:
            return []
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return []
        return lines[-limit:]

    def _set_command_center(self, lines: list[str]) -> None:
        cleaned = [line.rstrip() for line in lines if line is not None]
        self.command_center_lines = cleaned if cleaned else ["(no output)"]

    def _command_center_height(self, terminal_height: int) -> int:
        if terminal_height >= 32:
            return 8
        if terminal_height >= 24:
            return 6
        return 4

    def _footer_reserved_lines(self, terminal_height: int) -> int:
        return self._command_center_height(terminal_height) + 3

    def refresh_rows(self, *, initial: bool = False) -> None:
        previous = {row.path for row in self.rows if row.selected}
        self.summary, new_rows = load_pending_rows(self.vault_path, self.db_path)
        for row in new_rows:
            if row.path in previous:
                row.selected = True
        self.rows = new_rows

        if not self.rows:
            self.cursor = 0
            self.top = 0
        else:
            self.cursor = max(0, min(self.cursor, len(self.rows) - 1))
            self.top = max(0, min(self.top, self.cursor))

        if not initial:
            self.set_notice("Refreshed vault state")

    def quick_create_note(self) -> None:
        note_path = create_quick_test_note(self.vault_path)
        self.refresh_rows(initial=True)
        note_path_str = str(note_path)
        for idx, row in enumerate(self.rows):
            if row.path == note_path_str:
                row.selected = True
                self.cursor = idx
                self.top = min(self.top, self.cursor)
                break
        rel = relative_path(note_path_str, self.vault_path)
        self.set_notice(f"Created test note: {rel}")

    def reset_db(self, stdscr: curses.window) -> None:
        confirm = self._prompt(stdscr, "Type RESET to clear tracker DB", "")
        if confirm != "RESET":
            self.set_notice("DB reset cancelled")
            return
        resolved = reset_tracker_db(self.db_path)
        self.last_failed_paths = []
        self.refresh_rows(initial=True)
        self.set_notice(f"Tracker DB reset: {resolved}")

    def move_cursor(self, delta: int) -> None:
        if not self.rows:
            return
        self.cursor = max(0, min(self.cursor + delta, len(self.rows) - 1))

    def toggle_current(self) -> None:
        if not self.rows:
            return
        self.rows[self.cursor].selected = not self.rows[self.cursor].selected

    def _prompt(self, stdscr: curses.window, label: str, default: str = "") -> str:
        height, width = stdscr.getmaxyx()
        prompt = label if not default else f"{label} [{default}]"
        prompt += ": "
        prompt = _truncate(prompt, width - 1)
        x_pos = min(len(prompt), width - 1)

        curses.echo()
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        try:
            stdscr.move(height - 1, 0)
            stdscr.clrtoeol()
            stdscr.addnstr(height - 1, 0, prompt, width - 1)
            stdscr.refresh()
            max_len = max(1, width - x_pos - 1)
            raw = stdscr.getstr(height - 1, x_pos, max_len)
            value = raw.decode("utf-8", errors="replace").strip()
        finally:
            curses.noecho()
            try:
                curses.curs_set(0)
            except curses.error:
                pass
        return value or default

    def choose_agent(self, stdscr: curses.window) -> None:
        preview = ", ".join(f"{i + 1}:{name}" for i, name in enumerate(self.agents[:8]))
        label = "Agent index/name"
        if preview:
            label += f" ({preview})"
        raw = self._prompt(stdscr, label, self.current_agent)

        chosen = raw
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(self.agents):
                chosen = self.agents[idx]

        self.current_agent = chosen
        if chosen not in self.agents:
            self.agents.insert(0, chosen)
        self.set_notice(f"Agent set: {self.current_agent}")

    def choose_model(self, stdscr: curses.window) -> None:
        preview = ", ".join(f"{i + 1}:{name}" for i, name in enumerate(self.models[:6]))
        label = "Model index/name"
        if preview:
            label += f" ({preview})"
        raw = self._prompt(stdscr, label, self.current_model)

        chosen = raw
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(self.models):
                chosen = self.models[idx]

        self.current_model = chosen
        if chosen not in self.models:
            self.models.insert(0, chosen)
        self.set_notice(f"Model set: {self.current_model}")

    def select_n(self, stdscr: curses.window) -> None:
        value = self._prompt(stdscr, "Select first N pending", "10")
        try:
            n = int(value)
        except ValueError:
            self.set_notice(f"Invalid integer: {value}")
            return
        count = select_first_n(self.rows, n)
        self.cursor = 0
        self.top = 0
        self.set_notice(f"Selected {count} notes")

    def _draw_header(self, stdscr: curses.window) -> int:
        _, width = stdscr.getmaxyx()
        selected = selected_count(self.rows)
        title = "Extract TUI (experimental)"
        stdscr.addnstr(
            0, 0, _truncate(title, width - 1), width - 1, self._attr(1, curses.A_BOLD)
        )

        stats = "  ".join(
            [
                self._label_value("total", str(self.summary.total)),
                self._label_value("needs", str(self.summary.needs_extraction)),
                self._label_value("first", str(self.summary.first_extraction)),
                self._label_value("changed", str(self.summary.content_changed)),
                self._label_value("ok", str(self.summary.ok)),
                self._label_value("selected", str(selected)),
            ]
        )
        stdscr.addnstr(1, 0, _truncate(stats, width - 1), width - 1, self._attr(2))

        vault_line = self._label_value("vault", str(self.vault_path))
        stdscr.addnstr(2, 0, _truncate(vault_line, width - 1), width - 1)

        run_cfg = (
            f"agent:{self.current_agent}  model:{self.current_model}  db:{self.db_path}"
        )
        stdscr.addnstr(3, 0, _truncate(run_cfg, width - 1), width - 1)

        stdscr.hline(4, 0, curses.ACS_HLINE, max(1, width - 1))
        stdscr.addnstr(
            5,
            0,
            "Sel  Why  Last Extracted             Note",
            width - 1,
            self._attr(1),
        )
        return 6

    def _draw_rows(self, stdscr: curses.window, start_line: int) -> None:
        height, width = stdscr.getmaxyx()
        footer_reserved = self._footer_reserved_lines(height)
        visible_rows = max(1, height - start_line - footer_reserved)

        if self.cursor < self.top:
            self.top = self.cursor
        elif self.cursor >= self.top + visible_rows:
            self.top = self.cursor - visible_rows + 1

        end = min(len(self.rows), self.top + visible_rows)
        for screen_idx, row_idx in enumerate(range(self.top, end)):
            row = self.rows[row_idx]
            line_no = start_line + screen_idx
            marker = ">" if row_idx == self.cursor else " "
            checkbox = "[x]" if row.selected else "[ ]"
            reason = "FIRST" if row.reason == "first_extraction" else "CHG"
            ts = _truncate(short_time(row.last_extracted_at), 24)
            rel = relative_path(row.path, self.vault_path)
            content = f"{marker}{checkbox}  {reason:<5} {ts:<26} {rel}"
            attr = curses.A_REVERSE if row_idx == self.cursor else curses.A_NORMAL
            if row.reason == "content_changed":
                attr |= self._attr(3)
            elif row.reason == "first_extraction":
                attr |= self._attr(4)
            stdscr.addnstr(line_no, 0, _truncate(content, width - 1), width - 1, attr)

        if not self.rows:
            stdscr.addnstr(start_line, 0, "No pending notes.", width - 1)

    def _draw_footer(self, stdscr: curses.window) -> None:
        height, width = stdscr.getmaxyx()
        if height < 3:
            return
        command_h = self._command_center_height(height)
        reserved = command_h + 3
        top = height - reserved

        if top >= 0:
            stdscr.hline(top, 0, curses.ACS_HLINE, max(1, width - 1))
            stdscr.addnstr(top, 2, "Command Center", max(1, width - 3), self._attr(1))

            lines = self.command_center_lines[-command_h:]
            pad = command_h - len(lines)
            for i in range(max(0, pad)):
                y = top + 1 + i
                stdscr.addnstr(y, 0, "", width - 1)
            for i, line in enumerate(lines):
                y = top + 1 + pad + i
                stdscr.addnstr(y, 0, _truncate(line, width - 1), width - 1)

        controls = (
            "q quit | r refresh | j/k move | space toggle | a all | u clear | "
            "n first N | c test-note | x reset-db | g agent | m model | e extract | f rerun-failed"
        )
        stdscr.hline(height - 3, 0, curses.ACS_HLINE, max(1, width - 1))
        stdscr.addnstr(
            height - 2, 0, _truncate(controls, width - 1), width - 1, self._attr(1)
        )
        notice_attr = (
            self._attr(6)
            if "fail=" in self.notice or "Invalid" in self.notice
            else curses.A_NORMAL
        )
        stdscr.addnstr(
            height - 1, 0, _truncate(self.notice, width - 1), width - 1, notice_attr
        )

    def draw(self, stdscr: curses.window) -> None:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        if height < 16 or width < 60:
            stdscr.addnstr(
                0,
                0,
                "Terminal too small (need at least 60x16).",
                max(1, width - 1),
                self._attr(6, curses.A_BOLD),
            )
            stdscr.addnstr(
                1, 0, "Resize terminal or press q to quit.", max(1, width - 1)
            )
            self._draw_footer(stdscr)
            stdscr.refresh()
            return
        start = self._draw_header(stdscr)
        self._draw_rows(stdscr, start)
        self._draw_footer(stdscr)
        stdscr.refresh()

    def _run_note_paths(
        self,
        stdscr: curses.window,
        note_paths: list[str],
        *,
        mode_label: str,
    ) -> None:
        if not note_paths:
            self.set_notice(f"No notes to run ({mode_label})")
            return

        run_dir = make_run_log_dir(self.repo_root)
        self.last_run_dir = run_dir

        success = 0
        failed = 0
        failed_notes: list[str] = []
        failed_paths: list[str] = []

        total = len(note_paths)
        for idx, path in enumerate(note_paths, start=1):
            rel = relative_path(path, self.vault_path)
            self.set_notice(f"{mode_label} {idx}/{total}: {rel}")
            self._set_command_center(
                [
                    f"[{idx}/{total}] {rel}",
                    "Running opencode extraction...",
                    f"agent={self.current_agent} model={self.current_model}",
                ]
            )
            self.draw(stdscr)

            log_name = safe_log_name(rel, idx)
            result: RunResult = run_extraction_once(
                repo_root=self.repo_root,
                note_path=path,
                agent=self.current_agent,
                model=self.current_model,
                log_path=run_dir / log_name,
            )

            if result.success:
                post_status = verify_post_extract_status(path, self.db_path)
                result.post_status = post_status
                if post_status == "ok":
                    success += 1
                else:
                    failed += 1
                    failed_notes.append(f"{rel} (post-status={post_status})")
                    failed_paths.append(path)
            else:
                failed += 1
                failed_notes.append(f"{rel} (exit={result.returncode})")
                failed_paths.append(path)

            cc_lines = [
                f"[{idx}/{total}] {rel}",
                f"$ {' '.join(result.command)}",
                f"exit={result.returncode} timed_out={result.timed_out} post_status={result.post_status}",
            ]

            stdout_tail = self._tail_lines(result.stdout, limit=4)
            stderr_tail = self._tail_lines(result.stderr, limit=4)

            if stdout_tail:
                cc_lines.append("stdout:")
                cc_lines.extend(stdout_tail)
            if stderr_tail:
                cc_lines.append("stderr:")
                cc_lines.extend(stderr_tail)

            self._set_command_center(cc_lines)

        self.last_failed_paths = failed_paths
        self.refresh_rows(initial=True)
        clear_selection(self.rows)

        summary = f"{mode_label} done ok={success} fail={failed} logs={run_dir}"
        if failed_notes:
            summary += f" | first_fail={failed_notes[0]}"
        self.set_notice(summary)

    def run_selected(self, stdscr: curses.window) -> None:
        selected_rows = [row for row in self.rows if row.selected]
        if not selected_rows:
            self.set_notice("No selected notes")
            return
        note_paths = [row.path for row in selected_rows]
        self._run_note_paths(stdscr, note_paths, mode_label="run")

    def rerun_failed(self, stdscr: curses.window) -> None:
        if not self.last_failed_paths:
            self.set_notice("No failed notes to re-run")
            return
        self._run_note_paths(stdscr, self.last_failed_paths, mode_label="rerun")

    def event_loop(self, stdscr: curses.window) -> None:
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        self._init_colors()
        stdscr.keypad(True)

        while True:
            self.draw(stdscr)
            key = stdscr.getch()

            if key in (ord("q"), 27):
                break
            if key in (curses.KEY_DOWN, ord("j")):
                self.move_cursor(1)
                continue
            if key in (curses.KEY_UP, ord("k")):
                self.move_cursor(-1)
                continue
            if key == ord(" "):
                self.toggle_current()
                continue
            if key == ord("a"):
                toggle_select_all(self.rows)
                self.set_notice("Toggled all")
                continue
            if key == ord("u"):
                clear_selection(self.rows)
                self.set_notice("Selection cleared")
                continue
            if key == ord("r"):
                self.refresh_rows()
                continue
            if key == ord("n"):
                self.select_n(stdscr)
                continue
            if key == ord("c"):
                self.quick_create_note()
                continue
            if key == ord("x"):
                self.reset_db(stdscr)
                continue
            if key == ord("g"):
                self.choose_agent(stdscr)
                continue
            if key == ord("m"):
                self.choose_model(stdscr)
                continue
            if key == ord("e"):
                self.run_selected(stdscr)
                continue
            if key == ord("f"):
                self.rerun_failed(stdscr)
                continue


def run_self_check(repo_root: Path, vault: Path, db: Path) -> int:
    agents = discover_agents(repo_root)
    models = discover_models(repo_root)
    summary, rows = load_pending_rows(vault, db)

    print(f"vault={vault}")
    print(f"db={db}")
    print(
        "summary="
        f"total:{summary.total} "
        f"needs_extraction:{summary.needs_extraction} "
        f"first_extraction:{summary.first_extraction} "
        f"content_changed:{summary.content_changed} "
        f"ok:{summary.ok}"
    )
    print(f"agents={len(agents)} first={agents[0] if agents else 'none'}")
    print(f"models={len(models)} first={models[0] if models else 'none'}")
    print(f"rows={len(rows)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Experimental extraction TUI")
    parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--agent", type=str, default=DEFAULT_AGENT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run data/model/agent checks without opening curses UI",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    vault = args.vault.expanduser().resolve()
    db = args.db.expanduser()

    if args.self_check:
        return run_self_check(repo_root, vault, db)

    app = ExtractTUI(
        repo_root=repo_root,
        vault_path=vault,
        db_path=db,
        agent=args.agent,
        model=args.model,
    )
    curses.wrapper(app.event_loop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
