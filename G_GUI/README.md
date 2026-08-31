# G - Desktop GUI

Launch the installed GUI with:

```bash
cryoid-gui
```

For source-checkout development:

```bash
python -m G_GUI.launcher
```

The GUI Run button launches the same `python -m cryoid_pipeline run ...` CLI
used by command-line runs and streams its structured progress events back into
the window. It does not call scientific stage implementations directly. The
primary form stays mode-neutral. Sequence
mode, ranked-sequence structure selection, protein trace length, weights, RCL,
temporary-file retention, BLAST+, and ChimeraX configuration are in Advanced
Settings. Sequence rankings are not included in the default results list. The
dedicated **View > Sequence Ranking** action opens a separate modal dialog
and shows only the first 20 rows even though the TSV contains the complete
input pool. The table displays the full candidate ID without elision, sequence
length, and primary score; all other TSV fields remain saved but are hidden
from this compact view. The action remains available before a run and shows an
empty state when the current project has no ranking. The dialog defaults to the
newest available run and includes a run selector; an explicit selection is
saved with the other GUI configuration and reused until another run is chosen.
The ranking table is display-only, has no buttons, and is closed with the
window X. No ranking toolbar or ranking table is shown at startup. Advanced
Settings uses one flat form with path
configuration first. Protein minimum trace length, structure candidates, and
RCL use matching text fields with explicit default placeholders and no arrow
controls; RCL is formatted to three decimal places. **Help > GitHub** opens the
CryoIDv2 repository in the system browser. At startup, the log contains two
lines only: CUDA availability followed by `CryoID version 1.0.0`. Run and
progress logs appear normally after a command starts.

The folder selected in the Results browser is treated as a parent directory.
The GUI creates or reuses `<selected folder>/cryoid_results/` and places every
project-level and run-level output below it. Selecting `cryoid_results` itself
does not create another nested copy.

Important modules:

- `launcher.py`: installed entry point.
- `main_gui.py`: window behavior, run worker, results, and ranking table.
- `advanced_settings.py`: shared TOML-backed settings dialog.
- `local_config.py`: GUI adapter for `cryoid_pipeline.user_config`.
- `qt_compat.py`: PyQt6/PyQt5 compatibility layer.
- `gui_skin.ui` and `gui_skin.py`: Designer source and generated form.

When regenerating `gui_skin.py`, use the matching `pyuic` command for the Qt
binding and preserve the package-relative SVG resource import.
