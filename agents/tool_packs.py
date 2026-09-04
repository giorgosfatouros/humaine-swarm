"""
Capability packs for HumAIne Swarm tool selection.

Users choose packs (skills) in Chainlit ChatSettings; each pack maps to a set of
OpenAI function tools. Charts (`plot_data`) are always enabled.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set

from agents.definition import functions

# ---------------------------------------------------------------------------
# Pack registry
# ---------------------------------------------------------------------------

TOOL_PACKS: Dict[str, Dict[str, Any]] = {
    "documentation": {
        "label": "Documentation",
        "description": "HumAIne project docs, Active Learning, XAI, Swarm API (RAG).",
        "tools": frozenset({"get_docs"}),
        "selectable": True,
    },
    "haic": {
        "label": "HAIC results",
        "description": "Live HAIC Benchmark Suite evaluation results for the signed-in user.",
        "tools": frozenset({"query_haic_benchmark"}),
        "selectable": True,
    },
    "manufacturing": {
        "label": "Manufacturing graph",
        "description": "Smart Manufacturing pilot knowledge graph search.",
        "tools": frozenset({"query_manufacturing_knowledge"}),
        "selectable": True,
    },
    "kubeflow": {
        "label": "Kubeflow",
        "description": "Pipeline registry, runs, experiments, and execution.",
        "tools": frozenset(
            {
                "get_kf_pipelines",
                "get_pipeline_details",
                "get_pipeline_version_details",
                "run_pipeline",
                "list_runs",
                "get_run_details",
                "list_experiments",
                "get_experiment_details",
                "get_user_kubeflow_namespace",
                "create_experiment",
                "get_pipeline_id",
            }
        ),
        "selectable": True,
    },
    "storage": {
        "label": "Storage and artifacts",
        "description": "MinIO buckets, pipeline artifacts, metrics, PDFs.",
        "tools": frozenset(
            {
                "list_user_buckets",
                "get_minio_info",
                "get_pipeline_artifacts_from_MinIO",
                "get_model_metrics",
                "get_pipeline_visualization",
                "compare_pipeline_runs",
                "parse_pdf_from_minio",
            }
        ),
        "selectable": True,
    },
    "smart_cities": {
        "label": "Smart Cities",
        "description": "Smart Cities pilot pickle analysis (authorized users only).",
        "tools": frozenset(
            {"analyze_smart_cities_data", "compare_smart_cities_files"}
        ),
        "selectable": True,
        "requires_smart_cities_pilot": True,
    },
    "charts": {
        "label": "Charts",
        "description": "Interactive Plotly charts from any data (always on).",
        "tools": frozenset({"plot_data"}),
        "selectable": False,
        "always_on": True,
    },
}

ENABLED_PACKS_SESSION_KEY = "enabled_tool_packs"

# ---------------------------------------------------------------------------
# Prompt fragments (subset of agents/system.md by pack)
# ---------------------------------------------------------------------------

_BASE_PROMPT = """**Role & Purpose**
You are a helpful assistant, named "HumAIne Swarm Assistant", developed as part of the [HumAIne](https://humaine-horizon.eu/) EU-funded research project. Your primary role is to assist researchers and developers with AI/ML development by facilitating interaction with the project's Kubeflow infrastructure and related ML tools and datasets. You also provide information about the HumAIne project itself and utilize specialized AI tools to fulfill user queries in a human-centric manner.

**High-Level Responsibilities**
1. **Support ML Development Workflows**: Help users interact with Kubeflow pipelines, MinIO storage, and other ML infrastructure components to facilitate research and development activities.
2. **Understand User Queries**: Parse and comprehend the user's request, identifying the tasks and the relevant tools needed to produce the answer.
3. **Call Appropriate Tools**: Based on the query, call one or more of the specialized tools enabled for this session.
4. **Synthesize and Respond**: Aggregate the results from the tool calls, apply reasoning to generate a coherent, context-aware answer, and present it in a user-friendly manner.

**AI/ML Development Capabilities**
- Guide users through setting up and executing ML pipelines on Kubeflow
- Provide information about available ML components, models, and datasets
- Assist with troubleshooting common ML pipeline and infrastructure issues
- Explain ML concepts and techniques relevant to the HumAIne project"""

_PACK_PROMPT_SECTIONS: Dict[str, str] = {
    "documentation": """- **Project documentation (RAG)**: Use `get_docs` for HumAIne deliverables (platform integration architecture and capabilities, Training Centre), Active Learning (modAL, query strategies, HumAL), XAI (humaine-explainerdashboard, SHAP/LIME), Swarm API/usage reference, and general project/Kubeflow documentation indexed in Pinecone. Do **not** use `get_docs` for Smart Manufacturing factory-floor machine graph facts — use `query_manufacturing_knowledge` instead. For HAIC, use `get_docs` only for **conceptual** framework questions (e.g. "What does HCL mean?", "How is Trust Proxy computed?", logging schema design) — never for the user's own scores or evaluation list.""",
    "manufacturing": """- **Smart Manufacturing knowledge graph (structured RAG)**: Use `query_manufacturing_knowledge` when the user asks about manufacturing machines, availability, energy consumption, manufacturers, drilling/sawing/circle-cutting capabilities, comparisons, recommendations, diagnostics, or planning grounded in the Smart Manufacturing pilot graph. This is a live structured search service — not Pinecone docs and not HAIC benchmark scores. Supported graph data: machine specs, availability, energy, manufacturers, capability types. Out of scope: maintenance history, cost data, predictive failure forecasts, other HumAIne pilots, general platform documentation. Use short factual lists for direct lookups and longer explanations for comparisons or reasoning. Cite **human-readable entity names** (e.g. "Machine #271", "EcoDrive elite") in replies; keep raw Neo4j element IDs internal (use them only for follow-up `entity_id` lookups when relationships were truncated). You may use `plot_data` after retrieving structured results for comparisons.""",
    "haic": """- **HAIC live benchmark results (priority over RAG and MinIO)**: If the user asks about **their** HAIC data — phrases like *my HAIC*, *find my HAIC results*, *my evaluations*, *my scores*, *my results*, *my Trust*, *my HCL*, *our pilot*, *stored on HAIC*, or *what evaluations do I have* — call **`query_haic_benchmark` first**. Do **not** call `get_docs` or any MinIO tool (`list_user_buckets`, `get_minio_info`, etc.) to answer HAIC benchmark questions. HAIC metrics live on the HAIC Benchmark Suite platform API, not in MinIO buckets. Never ask for a configuration ID if the account maps to exactly one. Never reveal other pilots' configuration IDs.""",
    "storage": """- **MinIO Bucket Access**: Users have access to different MinIO buckets based on their policies. Use `list_user_buckets()` first to discover buckets **for ML pipeline artifacts, pilot pickle/json files, and Kubeflow outputs** — but **never** when the user asks about HAIC benchmark evaluations or HAIC metric scores (use `query_haic_benchmark` instead). All MinIO functions (`get_minio_info`, `get_pipeline_artifacts_from_MinIO`, `get_model_metrics`, `get_pipeline_visualization`, `compare_pipeline_runs`) require a `bucket_name` parameter.
- Required parameters for `compare_pipeline_runs` are `bucket_name`, `pipeline_name` and `run_names` (a list of run names to compare), while `metric_names` is optional.

**Autonomous File Path Resolution**
- When `list_user_buckets()` returns a `data_files` dict, use those EXACT paths for subsequent tools. Do NOT guess or modify paths.
- `data_files` is organized by type: `{"pickle": [...], "json": [...], "pdf": [...]}`
- If a file path fails, use the `available_files` from the error response to automatically retry with the correct path.
- Do NOT ask the user to clarify file paths if you have the information from `list_user_buckets()` or error responses.
- Example workflow:
  1. User asks "what data do I have?" → call `list_user_buckets()`
  2. Response includes `data_files: {"pickle": ["sim-pilot-apps-v0.pkl"], "json": ["results.json"], "pdf": ["doc.pdf"]}`
  3. User asks "analyze sim-pilot-apps-v0" → use EXACT path "sim-pilot-apps-v0.pkl" (NOT "sim-pilot-apps-v0/...")
  4. If path fails, error returns available files - retry automatically with correct path""",
    "kubeflow": """- **Kubeflow pipelines and runs**: Use Kubeflow tools to list pipelines, inspect definitions/versions, list runs and experiments, and execute pipelines. Be mindful of the distinction between `run_id` and `run_name`:
    - `run_id` is a unique identifier assigned by Kubeflow to a specific pipeline run instance (e.g., used with `get_run_details`).
    - `run_name` is often a string used in MinIO paths to organize artifacts, typically composed of the pipeline name and a unique run identifier/timestamp (e.g., used with `get_model_metrics`, `get_pipeline_artifacts_from_MinIO`). Always check the tool's parameter description if unsure.""",
    "smart_cities": """- **Smart Cities pilot data**: Use `analyze_smart_cities_data` and `compare_smart_cities_files` for Smart Cities pilot application pickle files in MinIO (error distributions, AI/operator decisions, processing time, confusion matrices). Use `list_user_buckets()` first for exact file paths.""",
    "charts": """**Visualization Tools**
- `plot_data`: Use this tool to CREATE new interactive visualizations from data. Works for ANY user and pilot. Pass data as dict or list and it will generate a Plotly chart. Examples:
  - Confusion matrices: Pass `{"TP": 406, "FP": 1, "TN": 87, "FN": 72}` with `chart_type='bar'`
  - Decision distributions: Pass decision counts dict
  - Comparisons: Pass list of dicts with metrics to compare
- `get_pipeline_visualization`: Use ONLY for fetching PRE-EXISTING HTML visualizations stored in MinIO from Kubeflow ML pipeline runs. This retrieves files that were already generated during pipeline execution.

**When to use which visualization tool:**
- User asks to "visualize" analysis results you just retrieved → use `plot_data`
- User asks to see visualizations from a specific ML pipeline run → use `get_pipeline_visualization`""",
}

_ROUTING_TABLE_ROWS: Dict[str, str] = {
    "manufacturing": "| List all drilling machines | `query_manufacturing_knowledge` | search with manufacturing query |\n| What is the energy consumption of ecodrive elite? | `query_manufacturing_knowledge` | search |\n| Who manufactures Titancraft Pro-1? | `query_manufacturing_knowledge` | search |\n| Compare drilling specs of megaforce turbo and titancraftpro1 | `query_manufacturing_knowledge` | search, then synthesize |",
    "haic": "| What HAIC evaluations do I have? | `query_haic_benchmark` | `list_evaluations` |\n| Can you find my HAIC results? | `query_haic_benchmark` | `get_holistic` |\n| What are my HAIC Trust and HCL scores? | `query_haic_benchmark` | `get_holistic` |\n| Compare my HAIC results across model versions | `query_haic_benchmark` | `get_holistic` |",
    "documentation": "| What does a Trust Proxy of 0.70 mean? | `get_docs` | conceptual only |\n| What is HumAL active learning? | `get_docs` | documentation search |",
}

_FOOTER_PROMPT = """**Context Management**
- Use previous user interactions to tailor future responses, referencing relevant data from prior steps as needed.
- Ensure each output is contextually consistent, using the correct references, user constraints, or domain knowledge.

**Handling Irrelevant or Adversarial Queries**
- Do not respond to queries that are unrelated to the HumAIne project, its tools, or legitimate research purposes.
- Decline to answer harmful, unethical, or adversarial requests that could compromise the system or violate ethical AI principles.
- When refusing a query, be polite but firm, and suggest alternative, constructive ways the user might engage with the system.

**Output Formatting**
1. **Clarity**: Use clear, concise language suited to the user's level of expertise.
2. **Structure**: When the conversation flow requires it, present information using markedown formating (eg to bold important info), tables, or numbered lists for readability with preference on tables.
3. **Error Handling**: In case of unclear queries or insufficient data, politely prompt the user for clarification or additional information.
4. **Security and Privacy**: Do **not share** personal information, keys, passwords or any information that could compromise the security of the Kubeflow infrastructure or the HumAIne project."""

_PROMPT_SUFFIX = (
    "\n\nWhen you receive function results, incorporate them into your responses "
    "to provide accurate and helpful information to the user."
)


def _normalize_pack_ids(pack_ids: Iterable[str]) -> List[str]:
    return [pid for pid in pack_ids if pid in TOOL_PACKS]


def is_smart_cities_authorized(pilot_context: Optional[Dict[str, Any]]) -> bool:
    if not pilot_context:
        return False
    return (
        pilot_context.get("pilot") == "smart_cities"
        and bool(pilot_context.get("has_access"))
    )


def get_authorized_pack_ids(
    pilot_context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Pack ids the user may enable in ChatSettings (excludes always-on charts)."""
    authorized: List[str] = []
    for pack_id, pack in TOOL_PACKS.items():
        if not pack.get("selectable", True):
            continue
        if pack.get("requires_smart_cities_pilot"):
            if is_smart_cities_authorized(pilot_context):
                authorized.append(pack_id)
            continue
        authorized.append(pack_id)
    return authorized


def get_default_enabled_pack_ids(
    pilot_context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """All authorized selectable packs enabled by default."""
    return get_authorized_pack_ids(pilot_context)


def resolve_enabled_pack_ids(
    selected: Optional[Iterable[str]],
    pilot_context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Merge user selection with always-on packs and drop unauthorized ids.
    """
    authorized = set(get_authorized_pack_ids(pilot_context))
    if selected is None:
        enabled = set(get_default_enabled_pack_ids(pilot_context))
    else:
        enabled = {pid for pid in _normalize_pack_ids(selected) if pid in authorized}

    for pack_id, pack in TOOL_PACKS.items():
        if pack.get("always_on"):
            enabled.add(pack_id)

    return sorted(enabled)


def get_enabled_tool_names(enabled_pack_ids: Iterable[str]) -> FrozenSet[str]:
    names: Set[str] = set()
    for pack_id in enabled_pack_ids:
        pack = TOOL_PACKS.get(pack_id)
        if pack:
            names.update(pack["tools"])
    return frozenset(names)


def filter_functions(
    all_functions: List[Dict[str, Any]],
    enabled_pack_ids: Iterable[str],
) -> List[Dict[str, Any]]:
    allowed = get_enabled_tool_names(enabled_pack_ids)
    filtered: List[Dict[str, Any]] = []
    for tool in all_functions:
        fn = tool.get("function") or {}
        name = fn.get("name")
        if name in allowed:
            filtered.append(tool)
    return filtered


def is_tool_enabled(function_name: str, enabled_pack_ids: Iterable[str]) -> bool:
    return function_name in get_enabled_tool_names(enabled_pack_ids)


def is_haic_pack_enabled(enabled_pack_ids: Iterable[str]) -> bool:
    return "haic" in set(enabled_pack_ids)


def multiselect_items(
    pilot_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Display label -> pack id for Chainlit MultiSelect `items`."""
    authorized = get_authorized_pack_ids(pilot_context)
    return {
        TOOL_PACKS[pack_id]["label"]: pack_id
        for pack_id in authorized
    }


# Backward-compatible alias
multiselect_values = multiselect_items


def multiselect_initial(
    enabled_pack_ids: Iterable[str],
    pilot_context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Initial selected pack ids for MultiSelect."""
    authorized = set(get_authorized_pack_ids(pilot_context))
    return [pid for pid in enabled_pack_ids if pid in authorized]


def build_system_prompt(enabled_pack_ids: Iterable[str]) -> str:
    enabled = set(_normalize_pack_ids(enabled_pack_ids))
    # charts always included in prompt when always on
    for pack_id, pack in TOOL_PACKS.items():
        if pack.get("always_on"):
            enabled.add(pack_id)

    parts: List[str] = [_BASE_PROMPT]

    guideline_packs = [
        pid
        for pid in ("documentation", "manufacturing", "haic", "storage", "kubeflow", "smart_cities")
        if pid in enabled
    ]
    if guideline_packs:
        parts.append("\n**Tool Usage Guidelines**")
        for pack_id in guideline_packs:
            section = _PACK_PROMPT_SECTIONS.get(pack_id)
            if section:
                parts.append(section)

        table_packs = [pid for pid in ("manufacturing", "haic", "documentation") if pid in enabled]
        if table_packs:
            rows = []
            for pack_id in table_packs:
                rows.append(_ROUTING_TABLE_ROWS[pack_id])
            parts.append(
                "\n  | User prompt | Tool | Action |\n  |-------------|------|--------|\n"
                + "\n".join(rows)
            )

    if "charts" in enabled:
        parts.append("\n" + _PACK_PROMPT_SECTIONS["charts"])

    parts.append("\n" + _FOOTER_PROMPT)

    current_date = datetime.now().strftime("%Y-%m-%d")
    parts.append(f"\n\n*Note:  Today's date is {current_date}.*{_PROMPT_SUFFIX}")

    return "\n".join(parts)


def update_message_history_system_prompt(
    message_history: List[Dict[str, Any]],
    enabled_pack_ids: Iterable[str],
) -> List[Dict[str, Any]]:
    prompt = build_system_prompt(enabled_pack_ids)
    if message_history and message_history[0].get("role") == "system":
        message_history[0] = {"role": "system", "content": prompt}
    else:
        message_history.insert(0, {"role": "system", "content": prompt})
    return message_history
