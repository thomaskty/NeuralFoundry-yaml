# NeuralFoundry YAML Runner

NeuralFoundry is a YAML-driven RAG pipeline that ingests knowledge bases and chat attachments into Postgres + pgvector, then runs one or more queries and prints answers in the terminal (and optionally to an output file). This repo is intentionally simple and CLI-first.

## Folder Layout
- `configs/` YAML run files
- `documents/` sample inputs (optional)
- `outputs/` generated answers (one file per run)
- `logs/` run logs (one file per run)

## Requirements
- Python 3.11+ for local runs
- Docker Desktop (for Postgres + pgAdmin)
- OpenAI API key

## Setup

### 1) Configure environment
Edit `/Users/thomaskuttyreji/Documents/GitHub/NeuralFoundry-yaml/.env`:
```env
POSTGRES_USER=neuralfoundry_yaml
POSTGRES_PASSWORD=neuralfoundry_yaml_pw
POSTGRES_DB=neuralfoundry_yaml_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
# OPENAI_API_KEY=sk-your-key
```

Set your OpenAI key in your shell (recommended):
```bash
export OPENAI_API_KEY="your_key_here"
```

### 2) Install Python dependencies (local runs only)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Options

### Option A: Local run (recommended for development)
This uses Docker only for the database.

```bash
# start db
docker compose up -d db

# run the yaml
python run.py configs/run.yaml
```

### Option B: Docker run (db + runner)
This runs the YAML runner inside Docker.

```bash
docker compose up --build
```

## Database Connection Details

These are the default credentials used by both local and docker runs:

- Host: `localhost` (from your Mac)
- Port: `5432`
- Database: `neuralfoundry_yaml_db`
- Username: `neuralfoundry_yaml`
- Password: `neuralfoundry_yaml_pw`

## pgAdmin (Docker)
If you want to inspect the database in pgAdmin, use these settings:

- Host: `db`
- Port: `5432`
- Database: `neuralfoundry_yaml_db`
- Username: `neuralfoundry_yaml`
- Password: `neuralfoundry_yaml_pw`

## YAML Configs

Example YAML files are in `configs/`:
- `configs/run.yaml` (basic)
- `configs/one_kb.yaml`
- `configs/multiple_kbs.yaml`
- `configs/with_attachments.yaml`
- `configs/attachments_only.yaml`

Minimal example:
```yaml
user:
  username: "demo_user"

chat:
  title: "YAML Run"
  system_prompt: "You are a helpful assistant. Use the provided context when available."

knowledge_bases:
  - title: "Sample KB"
    description: "Sample documents for a basic run"
    replace_if_changed: true
    files:
      - path: "documents/dataset.txt"

attachments: []

messages:
  - "What does the document say about PAN cards and their purpose?"

output_file: true
```

## Output Files and Logs
- Logs: `logs/<yaml_name>.log`
- Outputs: `outputs/<yaml_name>.out.txt`

## Notes
- Knowledge base documents are cached globally by content hash, so reusing the same file across KBs does not re-chunk.
- If you edit a file and want it reprocessed, set `replace_if_changed: true` in that KB entry.

