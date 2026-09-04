from agents.definition import functions
from agents.tool_packs import (
    TOOL_PACKS,
    build_system_prompt,
    filter_functions,
    get_authorized_pack_ids,
    get_default_enabled_pack_ids,
    get_enabled_tool_names,
    is_haic_pack_enabled,
    is_smart_cities_authorized,
    is_tool_enabled,
    multiselect_items,
    multiselect_values,
    resolve_enabled_pack_ids,
)
from utils.responses_adapter import convert_tools_for_responses


def _function_names(tool_list):
    return {t["function"]["name"] for t in tool_list}


def test_get_authorized_pack_ids_excludes_smart_cities_without_pilot():
    authorized = get_authorized_pack_ids(pilot_context=None)
    assert "smart_cities" not in authorized
    assert "documentation" in authorized
    assert "kubeflow" in authorized


def test_smart_cities_authorized_for_pilot_user():
    pilot_context = {
        "pilot": "smart_cities",
        "has_access": True,
        "user_email": "cities-pilot@humaine.com",
    }
    assert is_smart_cities_authorized(pilot_context)
    authorized = get_authorized_pack_ids(pilot_context)
    assert "smart_cities" in authorized


def test_default_enabled_matches_authorized():
    pilot_context = {"pilot": "smart_cities", "has_access": True, "user_email": "x"}
    assert get_default_enabled_pack_ids(pilot_context) == get_authorized_pack_ids(
        pilot_context
    )


def test_resolve_enabled_pack_ids_always_includes_charts():
    enabled = resolve_enabled_pack_ids(["documentation"], pilot_context=None)
    assert "charts" in enabled
    assert "documentation" in enabled
    assert "kubeflow" not in enabled


def test_resolve_enabled_pack_ids_drops_unauthorized_smart_cities():
    enabled = resolve_enabled_pack_ids(
        ["documentation", "smart_cities"], pilot_context=None
    )
    assert "smart_cities" not in enabled
    assert "documentation" in enabled


def test_filter_functions_respects_enabled_packs():
    enabled = resolve_enabled_pack_ids(["documentation", "haic"])
    filtered = filter_functions(functions, enabled)
    names = _function_names(filtered)
    assert names == {"get_docs", "query_haic_benchmark", "plot_data"}
    assert "get_kf_pipelines" not in names


def test_filter_functions_kubeflow_pack():
    enabled = resolve_enabled_pack_ids(["kubeflow"])
    filtered = filter_functions(functions, enabled)
    names = _function_names(filtered)
    assert "get_kf_pipelines" in names
    assert "run_pipeline" in names
    assert "list_user_buckets" not in names
    assert "plot_data" in names


def test_is_tool_enabled():
    enabled = resolve_enabled_pack_ids(["storage"])
    assert is_tool_enabled("list_user_buckets", enabled)
    assert not is_tool_enabled("get_docs", enabled)


def test_is_haic_pack_enabled():
    assert is_haic_pack_enabled(resolve_enabled_pack_ids(["haic"]))
    assert not is_haic_pack_enabled(resolve_enabled_pack_ids(["documentation"]))


def test_convert_tools_for_responses_on_filtered_subset():
    enabled = resolve_enabled_pack_ids(["documentation"])
    filtered = filter_functions(functions, enabled)
    converted = convert_tools_for_responses(filtered)
    names = {t["name"] for t in converted}
    assert names == {"get_docs", "plot_data"}


def test_build_system_prompt_omits_disabled_pack_sections():
    full = build_system_prompt(
        resolve_enabled_pack_ids(get_default_enabled_pack_ids())
    )
    docs_only = build_system_prompt(resolve_enabled_pack_ids(["documentation"]))
    assert "query_haic_benchmark" in full
    assert "query_haic_benchmark" not in docs_only
    assert "get_docs" in docs_only
    assert "HumAIne Swarm Assistant" in docs_only


def test_multiselect_items_excludes_smart_cities_without_access():
    items = multiselect_items(pilot_context=None)
    assert "Smart Cities" not in items
    pilot_context = {"pilot": "smart_cities", "has_access": True, "user_email": "x"}
    items_with_pilot = multiselect_items(pilot_context)
    assert items_with_pilot["Smart Cities"] == "smart_cities"
    assert multiselect_values(pilot_context) == items_with_pilot


def test_all_definition_tools_are_covered_by_packs():
    pack_tools = get_enabled_tool_names(TOOL_PACKS.keys())
    defined = _function_names(functions)
    assert pack_tools == defined
