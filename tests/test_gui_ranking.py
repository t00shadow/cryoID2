from __future__ import annotations

from pathlib import Path

import pytest

from G_GUI.settings_dialogs import OptionsWindow
from G_GUI.main_gui import SequenceRankingDialog, _load_sequence_ranking_display
from G_GUI.qt_compat import QtCore, QtWidgets


def test_sequence_ranking_display_keeps_only_full_id_length_and_score(
    tmp_path: Path,
) -> None:
    ranking = tmp_path / "sequence_ranking.tsv"
    full_candidate_id = "candidate_with_a_name_long_enough_to_have_been_elided"
    ranking.write_text(
        "rank\tmode\tcandidate_id\tmembers\tsequence\tlength\tprimary_score_name\t"
        "primary_score\tdescription\n"
        f"1\tprotein\t{full_candidate_id}\tmember_a,member_b\tMAGST\t605\t"
        "negative_log10_composite_evalue\t188.3767507096021\tdescription\n",
        encoding="utf-8",
    )

    headers, rows = _load_sequence_ranking_display(ranking)

    assert headers == ["Candidate ID", "Sequence Length", "Score"]
    assert rows == [[full_candidate_id, "605", "188.3767507096021"]]
    assert "..." not in rows[0][0]


def test_sequence_ranking_dialog_disables_candidate_id_elision() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = SequenceRankingDialog(None)
    rows = [
        ["a_very_long_candidate_identifier", "605", "188.4"]
        for _index in range(20)
    ]
    dialog._loaded(
        ["Candidate ID", "Sequence Length", "Score"],
        rows,
    )
    dialog.show()
    app.processEvents()

    text_elide_mode = getattr(QtCore.Qt, "TextElideMode", QtCore.Qt)
    assert dialog.table.textElideMode() == text_elide_mode.ElideNone
    assert dialog.model.data(dialog.model.index(0, 0)) == "a_very_long_candidate_identifier"
    assert dialog.status_label.text() == "20 ranked sequences"
    assert " - " not in dialog.status_label.text()
    assert dialog.width() < 1100
    assert dialog.minimumWidth() == dialog.maximumWidth() == dialog.width()
    assert dialog.table.viewport().width() <= sum(
        dialog.table.columnWidth(index) for index in range(dialog.model.columnCount())
    ) + 2

    dialog.close()
    app.processEvents()


def test_sequence_ranking_dialog_defaults_to_latest_and_emits_saved_run_choice(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    older = tmp_path / "runs" / "run_1" / "sequence_ranking.tsv"
    newer = tmp_path / "runs" / "run_2" / "sequence_ranking.tsv"
    for ranking in (older, newer):
        ranking.parent.mkdir(parents=True)
        ranking.write_text(
            "rank\tmode\tcandidate_id\tlength\tprimary_score\n"
            "1\trna\tSEQ1\t10\t0.9\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(SequenceRankingDialog, "_start_loading", lambda self: None)
    dialog = SequenceRankingDialog(None, ranking_paths=[older, newer])
    selected: list[str] = []
    dialog.run_selected.connect(selected.append)

    assert dialog.path == newer.resolve()
    assert dialog.run_combo.count() == 2
    assert Path(str(dialog.run_combo.currentData())) == newer.resolve()

    dialog.run_combo.setCurrentIndex(1)
    app.processEvents()

    assert dialog.path == older.resolve()
    assert selected == [str(older.parent.resolve())]
    dialog.close()


def test_options_describes_current_rna_structure_selection_default() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = OptionsWindow(sequence_mode="rna")

    assert "RNA 0.5 (50%)" in dialog.structure_selection_edit.placeholderText()
    assert dialog.structure_selection_edit.toolTip() == "Candidate count or fraction"
    dialog.close()
    app.processEvents()
