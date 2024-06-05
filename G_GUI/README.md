# cryoID_GUI

## TLDR, to run the GUI: 
```
python3 main_cryoID_developmental.py
```


## what files are what?
- **main_cryoID_developmental.py** is what you actually run (or main_cryoID.py, whatever the current name is)
- **.ui** files are for Designer (QtDesigner). Only need them for making changes to the GUI's skin, don't need them to run the GUI.
- **.py** files with the same name are python code generated from the .ui files using pyuic5 (like so: ```pyuic5 placeholder.ui -o placeholder.py```)
    - _optionally in the imports, you could skip using the .py file and just use the .ui file (there's a python package that does that)_
- **svg assets** is a folder for any button icons, other images, etc.
    - currently only has a file upload icon svg
    - **file_upload.qrc** is just something generated in Designer to add the svg in
- **chimerax_launcher.py** is imported as a module into main, but also works as a standalone script.
- **misc_pdbs** is just for testing opening chimerax with files. will remove once finish implementing file fetching.

## notes about editing the GUI
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


## TODOs for myself
- finish implementing file fetching, just list the input files and the output files (results)

- maybe rename the files
    - change main_cryoID.py to main_cryoID_gui.py
    - rename results to outputs?
- add shortcuts
- polish the look (aesthetics)
- launch chimerax with some custom default settings, like set models as spheres or ball and stick, or set transparency, so it's nice to look at out of the box
    - or let users open stuff with a session file that has their preferred settings (like mouse settings too)
    - also let users open the file explorer from the side bar, similar to vscode. Like just set a shortcut (shift+alt+R) and open file explorer from the current folder or the folder where the generated files are. Could also implement a file explorer side tab ilke VScode or Pycharm (I already have code from a tutorial that does that. it's the set_up_body function, still needs some tweaking to align it properly tho. But would need to code my own functions to select these files to be opened in chimerax. Actualyl could just display these files with 0 functionality, purely visual, and then the only function would be highlight/selecting files and then that adds it to the list of files to open in chimerax)
- clean up the code. some stuff can be put in separate files as modules. and a lot of nonessential comments can be removed

note to self: All my other developmental code is in C:\Users\noelu\Python Projects\PyQt GUI practice\QtDesigner_practice\draftGUI. There's a bunch of examples and other junk in there. Copy what's needed to here and make the final touches here (edit: slightly cleaned up now, so edit it there and bring it over when done). And then maybe copy it back over to that clusterfuck folder (just make a new folder under backups with the date as the name).