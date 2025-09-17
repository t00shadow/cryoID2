# cryoID_GUI

## TLDR, to run the GUI: 
```
cd G_GUI
python3 main_gui.py
```
or
```
python3 G_GUI/main_gui.py
```

## Required packages (at least these):
mrcfile==1.5.4,
numpy==2.2.5,
pandas==2.2.3,
PyQt5==5.15.11

## TLDR of dev commands: 
3 files of interest. `G_GUI/gui_skin.ui` and `G_GUI/gui_skin.py` are for the look of the GUI only. `G_GUI/main_gui.py` is where all the functionality lies.

0. Edit `G_GUI/gui_skin.ui` in Qt Designer
1. Convert the .ui file to a .py file:
```
pyuic5 -x "G_GUI/gui_skin.ui" -o "G_GUI/gui_skin.py"
```
2. Change the last line in `G_GUI/gui_skin.py` from "import file_upload_rc" to "from svg_assets import file_upload_rc"

This is all you need if you only want to change the **LOOK** of the GUI.

3. If you want to change functionality, edit `main_gui.py`

# ====================================
# _Ignore the rest of the README for now, outdated in some places._
# ====================================

## Notes
- You can use the two sidebar file viewers interchangeably and for any folder (i.e. you can ignore the labels). It basically just lets you open any two folders. It does NOT search thru subdirectories (intentional).

## What files are what?
- **main_cryoID_developmental.py** is what you actually run (or main_cryoID.py, whatever the current name is)
- **test.py** was already in this folder, just ignore it
- **.ui** files are for Designer (QtDesigner). Only need them for making changes to the GUI's skin, don't need them to run the GUI.
- **.py** files with the same name are python code generated from the .ui files using pyuic5 (like so: ```pyuic5 placeholder.ui -o placeholder.py```)
    - _optionally in the imports, you could skip using the .py file and just use the .ui file (there's a python package that does that)_
- **svg assets** is a folder for any button icons, other images, etc.
    - currently only has a file upload icon svg
    - **file_upload.qrc** is just something generated in Designer to add the svg in
- **chimerax_launcher.py** is imported as a module into main, but also works as a standalone script.
- **misc_pdbs** is just for testing opening chimerax with files. will remove once finish implementing file fetching.

## Editing the GUI
- To change buttons and other visual stuff about the GUI, edit the .ui file, and generate new python code with pyuic5.
    - **change the last line in the generated .py file to "from svg_assets import file_upload_rc"** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; just add "from svg_assets " in front of "import file_upload_rc"
- Then make any necessary changes in main_cryoID_developmental.py (such as updating button names if you changed them in Designer, attaching functions to buttons, etc.)
- **DON'T directly edit the .py files generated from pyuic5**
    - well actually you can if you want, but your changes will get overwritten if you run pyuic5 again with the **same** output file name
    - also if you change the name of the output file (the generated python file), make sure to update it in the import statement in the main python file. <br>
    ```line 11: from cryoID_v2_exp_FINALPRESVERSION import Ui_cryoID``` <-- this line <br>
    cryoID_v2_exp_FINALPRESVERSION refers to cryoID_v2_exp_FINALPRESVERSION.py, so if you save the pyuic5 output file as a different name, change this. <br>
    Ui_cryoID comes from the object name of the QMainWindow (or whatever template/form you built your GUI on). If you change the name of this in Designer, also update it here.
- **TLDR: edit the .ui file in Designer and then generate the .py file again with pyuic5**
    - *Alternatively, you could hand code the whole GUI and you wouldn't need Designer at all.*

## _======DONT HAVE TO READ THE REST======_

## Future Features/TODOs
- Add keyboard shortcuts
- Add option to set and save a home folder.
- Save user settings, like the last selected folders, options/settings they toggled, etc.
    - maybe have session files like chimerax? or save workspaces like vscode
- Maybe allow users to type file paths everywhere, not just in the lineedit boxes? Esp when saving a home path.
- Launch chimerax with some custom default settings. Maybe let users launch with settings from their session files?
- Clean up the code. separate functions into modules. remove nonessential comments.
    - reorder functions based on the order in the GUI. ie. main window functions, menubar functions, sidebar functions.
- Polish the look (aesthetics)
    - some parts are rly ugly
- Implement the status bar (depends on the rest of the code, so do this at the end)
- Implement drag and drop: (1) drop and drop from file explorer to input file box. (2) drag and drop from sidebar file viewer to input file box


## Maybe features (leaning towards no):
1. Filter files in the sidebar file viewers based on file type (pdb, mrc, xyz, etc.)
    - set some default file types, but allow users to change them
    - doable, tested it earlier, had it hard coded tho so just have to attach some user input
        - could use a textbox, checkbox, popup dialog window, etc. Typing is prob the best. The other ideas add more menus to click through (kinda annoying). For typing, just add a reset button to restore the default options.
    - IMO, it'd be better to display everything (aka dont add this functionality) and leave it up to users to select the correct files. Chimerax can handle wrong file types anyways. More annoying for certain files to be hidden when you don't want them to. But on the other hand, if you have a lot of different files in one folder, this would keep things cleaner.
2. "Search" bar in the sidebar file viewers in case you have a folder with like 50 files for some reason
3. Add option (or make it the default) to search subdirectories too
    - Intentionally didnt do this since if you accidentally selected your root directory, it'd search through your whole filesystem (might take a while). But alternatively, it could be great if you have your results folder and you want to see all the pdb, xyz, etc. files in the various subdirectories.
    - prob should be an option. Either put it as a small checkbox next to the select folder button, or hide it as a setting in the menu bar.
