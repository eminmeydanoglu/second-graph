# Experimental Extract TUI

Minimal, modern-style terminal UI to inspect vault extraction state and run note extraction with a chosen `opencode` agent/model.

## Run

```bash
uv run python experimental/extract_tui/main.py
```

Options:

```bash
uv run python experimental/extract_tui/main.py \
  --vault ~/obsidian_emin/obsidian_vault \
  --db data/vectors.experimental.db \
  --agent graph-agent \
  --model openai/gpt-5.3-codex
```

## Controls

- `q` or `Esc`: quit
- `r`: refresh vault status
- `Up/Down` or `j/k`: move cursor
- `Space`: toggle current note selection
- `a`: toggle select all / unselect all
- `u`: clear selection
- `n`: select first N pending notes
- `c`: create a quick test note under `Inbox/`
- `x`: reset tracker DB (asks for `RESET` confirmation)
- `g`: set agent (index or name)
- `m`: set model (index or name)
- `e`: run extraction for selected notes
- `f`: re-run last failed extraction set

## What it shows

- Vault totals: `total`, `needs_extraction`, `first_extraction`, `content_changed`, `ok`
- Pending note list with extraction reason (`FIRST` or `CHG`) and `last_extracted_at`
- Current execution config: `agent`, `model`
- Run result summary with log directory
- Last failed set memory for one-key retry
- Command Center panel with latest `opencode run` command + stdout/stderr tail
- Colorized reason rows (`CHG`, `FIRST`) and compact command bar

Logs are written under:

- `data/extraction-logs/tui_<timestamp>/`

Default tracker DB is isolated for the experimental app:

- `data/vectors.experimental.db`

Each note gets its own `.log` file containing command, stdout, and stderr.

## Non-interactive health check

```bash
uv run python experimental/extract_tui/main.py --self-check
```

This validates tracker access, pending-note loading, and model/agent discovery.
