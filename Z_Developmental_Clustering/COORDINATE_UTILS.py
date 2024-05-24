import os
import numpy as np

def extract_phosphorus_atoms(input_pdb_path, output_pdb):
    with open(input_pdb_path, 'r') as infile, open(os.path.join(os.path.dirname(input_pdb_path),output_pdb), 'w') as outfile:
        for line in infile:
            if line.startswith(("ATOM", "HETATM")):
                atom_name = line[76:78].strip()  # Extracting atom name (P is at columns 77-78)
                if atom_name == 'P':
                    outfile.write(line)

    return

def write_as_xyz(final_coords, output_file_name, title):
    with open(f"{output_file_name}/{title}.xyz", 'w') as xyz_file:
        xyz_file.write(f"{len(final_coords)}\n{title}\n")
        for i in final_coords:
            xyz_file.write("{:4} {:11.6f} {:11.6f} {:11.6f}\n".format(
                "C", i[2], i[1], i[0]))
        
    return

def plot_in_3D(final_coords_phosphate, final_coords_sugar, title):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(final_coords_phosphate[:, 0], final_coords_phosphate[:, 1], final_coords_phosphate[:, 2], c='r', marker='o')
    ax.scatter(final_coords_sugar[:, 0], final_coords_sugar[:, 1], final_coords_sugar[:, 2], c='b', marker='o')
    #add title
    ax.set_title(title)
    #add legend
    ax.legend(['Phosphate', 'Sugar'])
    #Incrase the size of the graph 1080x1080
    fig.set_size_inches(10.80, 10.80)
    
    plt.show()

    return