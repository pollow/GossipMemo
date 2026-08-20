from __future__ import annotations

import pathlib

import pytest

from gossipmemo.config import ConfigurationError, Settings
from gossipmemo.models import ModelMessage
from gossipmemo.prompts import PromptLibrary, defaults
from gossipmemo.prompts.library import PLACEHOLDERS
from gossipmemo.reasoners import (
    ReasoningSettings,
    continuity_prompt,
    coverage_audit_prompt,
    extraction_prompt,
    goal_candidate_prompt,
)


def write(tmp_path, body: str) -> pathlib.Path:
    path = tmp_path / "prompts.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_reasoning_settings_default_to_the_shipped_prompt_text():
    prompts = ReasoningSettings().prompts

    assert prompts.extraction_system == defaults.EXTRACTION_SYSTEM_PROMPT
    assert prompts.query_synthesis_system == defaults.QUERY_SYNTHESIS_SYSTEM_PROMPT
    assert prompts.coverage_root_viewpoints == defaults.COVERAGE_ROOT_VIEWPOINTS
    assert prompts.coverage_root_blind_spots == defaults.COVERAGE_ROOT_BLIND_SPOTS


def test_override_file_replaces_exactly_the_named_fragment(tmp_path):
    path = write(tmp_path, 'extraction_system = "Extract nothing at all."\n')

    prompts = PromptLibrary.from_file(path)

    assert prompts.extraction_system == "Extract nothing at all."
    assert prompts.continuity_system == defaults.CONTINUITY_SYSTEM_PROMPT
    assert prompts.coverage_method == defaults.COVERAGE_METHOD
    assert prompts.projection_stage == defaults.PROJECTION_STAGE_PROMPT


def test_coverage_root_override_merges_instead_of_replacing_the_table(tmp_path):
    path = write(
        tmp_path,
        "[coverage_root_viewpoints]\nM1 = \"Which chapters are legible?\"\n"
        "[coverage_root_blind_spots]\nP3 = \"local override\"\n",
    )

    prompts = PromptLibrary.from_file(path)

    assert prompts.coverage_root_viewpoints["M1"] == "Which chapters are legible?"
    assert prompts.coverage_root_viewpoints["M2"] == defaults.COVERAGE_ROOT_VIEWPOINTS["M2"]
    assert len(prompts.coverage_root_viewpoints) == len(defaults.COVERAGE_ROOT_VIEWPOINTS)
    assert prompts.coverage_root_blind_spots["P3"] == "local override"
    assert prompts.coverage_root_blind_spots["P11"] == defaults.COVERAGE_ROOT_BLIND_SPOTS["P11"]


def test_overridden_text_reaches_the_prompt_builders(tmp_path):
    path = write(
        tmp_path,
        'coverage_method = "Scan only what the user volunteered."\n'
        "[coverage_root_viewpoints]\nM1 = \"Which chapters are legible?\"\n",
    )
    prompts = PromptLibrary.from_file(path)

    assert "Which chapters are legible?" in coverage_audit_prompt("M1", [], [], prompts=prompts)
    candidates = goal_candidate_prompt("M1", [], [], prompts=prompts)
    assert "Which chapters are legible?" in candidates
    assert "Scan only what the user volunteered." in candidates


def test_overridden_instruction_fragments_reach_the_assembled_builders(tmp_path):
    """A fragment cut out of a builder is only configurable if the builder reads it."""

    path = write(
        tmp_path,
        'extraction_retention_rule = "Keep only what recurs."\n'
        'extraction_person_identity_rule = "$user_name is never in `people`."\n'
        'continuity_rebuild_rule = "Summarize the thread."\n'
        'coverage_audit_folding_rule = "Fold it in."\n',
    )
    prompts = PromptLibrary.from_file(path)

    extraction = extraction_prompt([], user_name="Deus", prompts=prompts)
    assert extraction.startswith("Keep only what recurs.\n")
    assert "Deus is never in `people`." in extraction
    assert "Keep explicit durable information" not in extraction
    assert "Summarize the thread." in continuity_prompt(None, [], prompts=prompts)
    assert "Fold it in." in coverage_audit_prompt("M1", [], [], prompts=prompts)


def test_placeholder_fragment_is_filled_with_the_configured_user_name():
    message = ModelMessage(
        id="m-1", space_id="space", author="user", content="Hi.",
        occurred_at="2026-08-14T12:00:00+00:00", source_provider="test",
    )

    prompt = extraction_prompt([message], user_name="Deus", prompts=PromptLibrary())

    assert "The fixed current user is named 'Deus'." in prompt
    assert "$user_name" not in prompt and "$quoted_user_name" not in prompt


def test_shipped_defaults_use_exactly_their_declared_placeholders():
    prompts = PromptLibrary()

    for field, names in PLACEHOLDERS.items():
        text: str = getattr(prompts, field)
        for name in names:
            assert f"${name}" in text


def test_override_using_an_undeclared_placeholder_is_rejected_at_load(tmp_path):
    path = write(
        tmp_path,
        'extraction_person_identity_rule = "$user_name lives in $city."\n',
    )

    with pytest.raises(ConfigurationError, match=r"extraction_person_identity_rule: \$city"):
        PromptLibrary.from_file(path)


def test_override_dropping_a_required_placeholder_is_rejected_at_load(tmp_path):
    path = write(tmp_path, 'extraction_user_evidence_rule = "The user is the user."\n')

    with pytest.raises(ConfigurationError, match=r"must still use the placeholder \$quoted"):
        PromptLibrary.from_file(path)


def test_stray_dollar_sign_in_a_placeholder_fragment_is_rejected_at_load(tmp_path):
    path = write(tmp_path, 'extraction_person_identity_rule = "$user_name owes $ 5."\n')

    with pytest.raises(ConfigurationError, match="stray"):
        PromptLibrary.from_file(path)


def test_fragments_without_placeholders_keep_a_dollar_sign_verbatim(tmp_path):
    path = write(tmp_path, 'continuity_rebuild_rule = "Costs are in $ only."\n')

    prompts = PromptLibrary.from_file(path)

    assert "Costs are in $ only." in continuity_prompt(None, [], prompts=prompts)


def test_unknown_override_key_is_rejected_rather_than_ignored(tmp_path):
    path = write(tmp_path, 'extraction_sytsem = "typo"\n')

    with pytest.raises(ConfigurationError, match="extraction_sytsem"):
        PromptLibrary.from_file(path)


def test_non_string_override_value_is_rejected(tmp_path):
    path = write(tmp_path, "extraction_system = 3\n")

    with pytest.raises(ConfigurationError, match="extraction_system"):
        PromptLibrary.from_file(path)


def test_non_table_and_unknown_root_overrides_are_rejected(tmp_path):
    with pytest.raises(ConfigurationError, match="coverage_root_viewpoints"):
        PromptLibrary.from_file(write(tmp_path, 'coverage_root_viewpoints = "text"\n'))
    with pytest.raises(ConfigurationError, match="M99"):
        PromptLibrary.from_file(
            write(tmp_path, '[coverage_root_viewpoints]\nM99 = "no such root"\n')
        )


def test_configured_but_missing_prompts_file_fails_at_startup(tmp_path):
    with pytest.raises(ConfigurationError, match="prompts file does not exist"):
        Settings(
            llm_base_url="http://model.test/v1",
            llm_api_key="secret",
            llm_model="model-a",
            prompts_path=tmp_path / "absent.toml",
        )


def test_prompts_path_is_read_from_the_environment(tmp_path, monkeypatch):
    path = write(tmp_path, 'extraction_system = "Extract nothing at all."\n')
    monkeypatch.setenv("GOSSIPMEMO_LLM_BASE_URL", "http://model.test/v1")
    monkeypatch.setenv("GOSSIPMEMO_LLM_API_KEY", "secret")
    monkeypatch.setenv("GOSSIPMEMO_LLM_MODEL", "model-a")
    monkeypatch.setenv("GOSSIPMEMO_PROMPTS_PATH", str(path))

    assert Settings.from_env().prompts_path == path


def test_shipped_example_override_file_loads():
    example = pathlib.Path(__file__).resolve().parents[1] / "prompts.example.toml"

    prompts = PromptLibrary.from_file(example)

    assert isinstance(prompts, PromptLibrary)


def test_no_production_code_reads_the_default_constants_directly():
    """An override is only real if every use site goes through the library."""

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "gossipmemo"
    names = [name for name in defaults.__all__]
    offenders = [
        f"{path.relative_to(root)}:{name}"
        for path in root.rglob("*.py")
        if path.parent.name != "prompts"
        for name in names
        if name in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
