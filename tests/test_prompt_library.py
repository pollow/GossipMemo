from __future__ import annotations

import pathlib

import pytest

from gossipmemo.config import ConfigurationError, Settings
from gossipmemo.prompts import PromptLibrary, defaults
from gossipmemo.reasoners import ReasoningSettings, coverage_audit_prompt, goal_candidate_prompt


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
