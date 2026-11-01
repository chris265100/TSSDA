# TSSDA

TSSDA is an automated data generation framework for Text-to-SQL tasks in time-series databases.

## Setup

Use Python 3.10 or newer and an accessible Apache IoTDB database:

```sh
python -m pip install -e .
```

Set the environment variables shown in `.env.example` in your shell. Provide
your own database credentials, model identifier and API key; the file is a
reference and is not loaded automatically. `TSSDA_SCHEMA_ROOT` selects the
database when `--database` is omitted. No database contents or learned state
are bundled.

For a POSIX shell:

```sh
export IOTDB_HOST=localhost
export IOTDB_USERNAME='your-database-user'
export IOTDB_PASSWORD='your-database-password'
export DEEPSEEK_API_KEY='your-api-key'
export DEEPSEEK_MODEL='your-model-id'
```

For PowerShell:

```powershell
$env:IOTDB_HOST = 'localhost'
$env:IOTDB_USERNAME = 'your-database-user'
$env:IOTDB_PASSWORD = 'your-database-password'
$env:DEEPSEEK_API_KEY = 'your-api-key'
$env:DEEPSEEK_MODEL = 'your-model-id'
```

Generate pairs using your selected database:

```sh
python run_tssda.py --database root.example --num-trajectories 1 --output runtime/nl2sql.json
```

The installed `tssda` command accepts the same arguments. Use `--provider qwen`
with `QWEN_API_KEY` and `QWEN_MODEL` to select the alternative client.
`--user-goal` supplies an optional domain request. `--max-samples` sets the
per-trajectory output ceiling; it defaults to three.

Runtime state is created under `runtime/knowledge/`. `--runtime-dir` selects a
different relative directory. `--log runtime/trajectories.jsonl` enables an
audit log. Generated files and credentials are excluded by `.gitignore`.
Output is a JSON array of objects with `sql` and `nl` fields; repeated runs
append new SQL statements without duplicating existing ones. Exit status is
zero when new pairs were exported and one when none were exported. Connection,
configuration and unrecoverable generation errors stop the command.

## Code layout

- `tssda/`: runner, agents, state, memory, learning and validation.
- `skills/`: P0–P10 packages, prompts, actions, masks and typed function catalog.
- `iotdb_tools/`: schema and time-bound access used by the framework.
- `run_tssda.py`: command-line entry point.
- `tests/`: offline checks using synthetic schemas and mocked services.

## Offline checks

```sh
python -m pip install -e '.[dev]'
python -m pytest
```

Tests do not require model credentials or a running database. A live generation
run additionally requires the configured model endpoint and IoTDB server.
