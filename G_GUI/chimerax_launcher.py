'''
    Delete the extra comments later
'''

import os
import subprocess
# import platform          # if no longer using platform, delete these 3 lines (lines 3-5)
# operating_system = platform.system()
# print("Operating System:", operating_system)

# Helper function for finding location of ChimeraX executable on user's machine.
def find_chimerax():
    # Common default installation locations for ChimeraX
    # TODO: allow users to manually specify a location if not in default locations, need to implement this in the GUI tho probably
    default_locations = [
        "/usr/bin/chimerax",                                     # Linux
        "/Applications/ChimeraX.app/Contents/MacOS/ChimeraX",    # macOS
        "C:\\Program Files\\ChimeraX\\bin\\chimerax.exe"         # Windows
    ]

    # Check for ChimeraX executable in default locations
    #   just loops thru the default_locations list, simple solution
    #   Originally was going to use platform to first detect the operating system, but might be overkill, would need to store locations in a dictionary or use a switch statement
    #   Tho if used switch statements, could list various locations for each OS, like from more specific to broad locations like up to the whole file system. And then search in a while loop.
    #   One issue tho is have to include all operating system names. Prob would be fine for most cases: Linux, windows, Darwin (macOS). But users might have other OS's like Cygwin, Java, OpenBSD, FreeBSD, SunOS, etc.
    #   There's also 2 other ways of finding the OS: os.name, sys.platform (https://stackoverflow.com/questions/4553129/when-to-use-os-name-sys-platform-or-platform-system)
    #       sys.platform allows you to do stuff like "if sys.platform.startswith('darwin'):"   Tho I think startswith might just be a default python string method
    #   Apparently the output of platform.system is determined at run time, while the other two are at compile time, so that'd matter if you're packaging and distributing your code as an executable.
    for location in default_locations:
        if os.path.exists(location):
            return location
    return None

# Function for launching ChimeraX
def launch_chimerax(files=None):
    # Find ChimeraX executable
    chimerax_path = find_chimerax()
    if chimerax_path is None:
        print("ChimeraX executable not found.")
        return

    # Command to launch ChimeraX
    command = [chimerax_path]

    # Append files to the command if provided
    if files:
        command.extend(files)

    try:
        # Launch ChimeraX
        subprocess.Popen(command)          # NOTE: need to read more to see if call, check_call, or Popen is best for our use case
        print("ChimeraX launched successfully.")
        # print("launched with these files: ", command[1::])    # for debugging
    except Exception as e:
        print("Error launching ChimeraX:", e)

if __name__ == "__main__":
    # Launch ChimeraX
    # Example usage(s):
    #   no files
    launch_chimerax()
    #   with files
    files_to_open = [r"C:\Users\noelu\BE 177A Capstone\CryoREAD\zzzz_SANDBOX\cryoRead_files\4_26_24\Files for Qibo\clustering_results\all_points_threshold_0_5.pdb", r"C:\Users\noelu\BE 177A Capstone\CryoREAD\zzzz_SANDBOX\cryoRead_files\4_26_24\Files for Qibo\clustering_results\meanshift_clusters_threshold_0_5.pdb"]
    launch_chimerax(files_to_open)
