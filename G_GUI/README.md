# Desktop GUI

After installation, launch `cryoid-gui` (or `python -m G_GUI.launcher` from an
installed source checkout). Start with **File → New** to select a project, or
**File → Open** to resume one.

- **Settings → Configurations:** weights, BLAST+, ChimeraX, and results paths.
- **Options:** sequence mode, trace length, structure candidates, RCL, and
  temporary-output retention.
- Select the density map and FASTA candidate pool, then **Run**. The same
  button becomes **Abort** while running.
- **Results → Sequence ranking** requires a saved ranking; **Results → Model**
  requires a fitted model and opens it with the input map in ChimeraX.
- **View → Sequence Ranking** also works without results, showing an empty
  viewer. **View → Model** launches ChimeraX without project files.
- **Logs** is collapsible. It starts with the version and records execution
  messages; loading files or changing settings does not add entries.
- **Settings → Theme** switches light/dark colors without resizing controls.

Outputs live under the selected folder's `cryoid_results/`; selecting that
folder directly does not nest another copy. The location is remembered and
can be changed in Configurations. Fresh source checkouts without saved
settings default to `<repository>/cryoid_results/`.

Rankings preview the first 20 rows while the complete TSV remains on disk.
The viewer selects the latest run by default and remembers explicit run
selections. The GUI uses the same pipeline CLI as terminal runs.

The layout is Python code in `gui_skin.py`; Qt Designer and separate Qt tools
are not required. `main_gui.py` handles behavior, `settings_dialogs.py` handles
Configurations/Options, and `qt_compat.py` supports PyQt6/PyQt5.
