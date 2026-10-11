# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

HumAIne Swarm Assistant: a Chainlit chat app (OpenAI Responses API + function tools) that lets HumAIne EU-project pilots talk to Kubeflow, MinIO, a Pinecone RAG index, the HAIC Benchmark Suite and a Smart Manufacturing knowledge graph. Developer docs live in `docs/developers/` (architecture, app flow, tools, auth, data injection, config, tool reference). Read the relevant one before a non-trivial change, and update it when you change the behaviour it describes.

## Commands

Python >=3.11,<3.13, Poetry (`package-mode = false`, so it is not an installable package; run everything from the repo root).

```bash
poetry install
chainlit run app.py -w                      # local dev server on :8000

poetry run pytest                                    # full suite
poetry run pytest tests/test_app_stream.py           # one module
poetry run pytest tests/test_tool_packs.py::test_is_tool_enabled
```

Use pytest, not `unittest discover`: tests mix `unittest.TestCase` classes with plain `def test_*` functions, and `unittest` silently skips the latter.

RAG indexing is a separate offline step. Deploying never re-indexes:

```bash
poetry run python data_injection.py --sources pdf,md [--docs-dir .docs] [--md-dir .md_files] [--reset]
```

`.docs/` and `.md_files/` are gitignored and not shipped in the Docker image.

Evaluation benchmark (30 prompts, see `evaluation/README.md`):

```bash
poetry run python evaluation/merge_ui_captures.py
poetry run python evaluation/score.py evaluation/runs/<date>/responses.jsonl
```

Docker publish/deploy: follow `.cursor/skills/humaine-swarm-docker-publish/SKILL.md` (Hub repo `gfatouros/humaine-swarm`, production tag `:amd64`; on Apple Silicon, build with `--platform linux/amd64`; build once and retag). Helper script: `bash .cursor/skills/humaine-swarm-docker-publish/scripts/publish-amd64.sh` (`SKIP_PUSH=1`, `PLATFORM=` overrides). Containers need `CHAINLIT_DB=sqlite+aiosqlite:////data/chat_history.db` and a volume on `/data`, or chat history is lost.

## Configuration

- Env vars are read in `utils/config.py`. `utils/dotenv_local.py` loads **`humaine-swarm/.env`** at startup with `setdefault`, so exported shell vars win. A `common-fe/.env` has no effect on Chainlit.
- Templates: `.env-example` (Docker) and `env.sh.example` (shell `source`).
- Keycloak JWKS TLS: set `KEYCLOAK_CA_BUNDLE=/path/ca.pem`. Dev-only escape hatch: `KEYCLOAK_JWKS_SSL_VERIFY=false`.

## Architecture

### Request loop (`app.py`)
`@cl.on_message main()` → append the user turn to the session history → `client.responses.create(stream=True)` → `process_responses_stream()`. That function streams text deltas, accumulates `function_call` args, validates the JSON, runs all valid calls concurrently through `agents.code.function_map`, appends `function_call` / `function_call_output` items, and **recursively** calls itself on a follow-up stream until the model stops calling tools. `plot_data` results become `cl.Plotly` elements on a separate message.

- `message_history` (Responses-API items, system message first) is the single source of truth. `utils/responses_adapter.py` converts tool schemas and splits off `instructions`. `UserSessionManager.set_message_history()` truncates by tiktoken to `MAX_INPUT_TOKENS`.
- Streamed replies **must** be finalized with `finalize_assistant_message()` (`send()` if not yet persisted). An `update()`-only reply is never written to the data layer and disappears on thread resume (see the latest `fix(persistence)` commit and `tests/test_app_finalize.py`).
- `on_chat_resume` rebuilds the LLM history from the persisted Chainlit thread (`utils/chat_persistence.py`). The data layer is SQLite via SQLAlchemy (`utils/chainlit_db.py`).
- **HAIC pre-router:** before calling the LLM, `utils/haic_routing.detect_haic_live_query()` regex-matches the message. If it matches and the HAIC pack is enabled, it calls `query_haic_benchmark` directly and replies with `format_haic_result_markdown` without calling the LLM. Conceptual and documentation HAIC questions are deliberately excluded so they go to `get_docs`.

### Tools: three places to keep in sync
1. `agents/definition.py`: the OpenAI function schemas (`functions`).
2. `agents/code.py`: the async implementation, decorated with `@cl.step(type="tool")` and registered in `function_map`.
3. `agents/tool_packs.py`: `TOOL_PACKS` groups tools into user-selectable "packs" (documentation, haic, manufacturing, kubeflow, storage, smart_cities, …) shown in Chainlit ChatSettings. `filter_functions()` limits the tools sent to the model and `build_system_prompt()` assembles the system prompt from `agents/system.md` plus per-pack guidance. Some packs are gated by pilot authorization (e.g. `smart_cities` requires a pilot context). `plot_data` is always on.

`tests/test_tool_packs.py::test_all_definition_tools_are_covered_by_packs` fails if a defined tool isn't in any pack.
### Auth and per-user clients
- Two login paths: `@cl.oauth_callback` (Keycloak OAuth) and `@cl.header_auth_callback` (`utils/keycloak_header_auth.py`, bearer JWT verified against JWKS when embedded in the Common FE).
- On chat start, `classes/user_handler.UserSessionManager` stores the token, exchanges it for MinIO STS credentials (AssumeRoleWithWebIdentity, with auto-refresh), and pulls the Kubeflow namespace, roles, groups and policies from the token claims.
- Tools never use shared credentials when a user session exists. `get_user_minio_client()` uses the STS creds. `get_user_kubeflow_client()` builds `kfp.Client(existing_token=<Keycloak bearer>)` and raises `ValueError` without a token (there is no password or Dex path). Pipeline listing merges the private namespace with the shared catalog (`namespace=None`).
- HAIC pilot identity comes from email, groups and MinIO policy via the static map in `utils/haic_pilot_map.py`. `query_haic_benchmark` hard-allowlists `configuration_id` to the user's pilot. Keep this isolation intact.

### Other integrations
- `get_docs`: LlamaIndex retriever over the Pinecone index `humaine` (top_k=10). `optimize_query()` exists but is unused.
- `utils/semantic_inference_client.py`: JSI Smart Manufacturing knowledge-graph HTTP client (`SEMANTIC_INFERENCE_*` env).
- `utils/langfuse_setup.py`: wraps the OpenAI client for Langfuse tracing (`get_openai_client`, `trace_chat_turn`). It is optional and is configured by env.
- `kubeflow/`: standalone demo pipelines and KFP YAMLs. They are not imported by the app.

## Conventions

- Commit messages: Conventional Commits (`type(scope): summary`). Existing scopes include `persistence`, `kubeflow`, `config` and `docker`.
- Tests mock Chainlit and external services. Keep them hermetic (no network, Keycloak, MinIO or OpenAI).
