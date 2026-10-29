from pathlib import Path

import pytest

from cryoid_pipeline.create_conda_env import select_environment


@pytest.mark.parametrize("platform", ["linux-64", "osx-64", "osx-arm64", "win-64"])
def test_shared_environment_selects_a_supported_parasail_install(platform):
    source = (Path(__file__).parents[1] / "environment.yml").read_text(encoding="utf-8")
    selected = select_environment(source, platform)
    assert ("parasail-python=1.3.4" in selected) == (platform != "win-64")
    assert ("  - bioconda" in selected) == (platform != "win-64")
    assert "-e .[gui,dev]" in selected
    assert "# [unix]" not in selected
    assert source.count("parasail-python=1.3.4") == 1


def test_unknown_selector_fails_instead_of_silently_dropping_dependencies():
    with pytest.raises(ValueError, match="Unsupported environment selector"):
        select_environment("dependencies:\n  - example  # [typo]\n", "win-64")
