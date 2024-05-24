import numpy as np
import mrcfile
from tqdm import tqdm, trange
from scipy.spatial import KDTree
import os
from collections import deque

class points(object):
    def __init__(self):
                
        #uNet Points, Densities and Counts
        self.pseudo_coords = 0
        self.ps_dens = 0
        self.ps_count = 0
        
        self.COM_points = 0
        self.COM_density = 0
        
        self.final_coords = 0
        self.final_dens = 0
        self.final_coord_count = 0

    def UNet2Points(self, mrc: mrcfile, params1: str, params2: bool , b_bone_threshold: float=0.5, keyAtom_threshold: float=0.80):
        """
        Converts a UNet map to a list of points based on specified parameters.

        Args:
            mrc (MRC): The MRC object containing the UNet maps and Original Density Map.
            params1 (str): The parameter to consider for converting the UNet map to points.
                          Possible values: 'PHOSPHATE_SS', 'PHOSPHATE', 'SUGAR', 'SUGAR_SS'.
            params2 (bool): Setting true filters uNet_map by secondary structure so that only points with both key atoms and secondary structure are considered.
            b_bone_threshold (float, optional): The threshold value for secondary structure classification.
                                                Default is 0.5.
            threshold (float, optional): The threshold value for converting the UNet map to points.
                                         Default is 0.80.

        Returns:
            None

        Raises:
            None

        """
        # Read in the MRC File, Original Density
        og_dens = np.copy(mrc.originial_density)
        

        # Read in the UNet Map
        if params1 == 'PHOSPHATE_SS':
            uNet_map_KA = np.copy(mrc.uNet_map_KA)
            uNet_map_SS = np.copy(mrc.uNet_map_SS)
            uNet_map_KA[uNet_map_SS < b_bone_threshold] = 0
            
            if params2 == True:
                uNet_map = np.copy(mrc.uNet_map_KA * mrc.uNet_map_SS)
            else:                        
                uNet_map = np.copy(uNet_map_KA)  
                  
            print("Secondary Structure Considered")
        elif params1 == 'PHOSPHATE':
            uNet_map = np.copy(mrc.uNet_map_KA)
            print("Only Phosphate UNet Considered")
        elif params1 == 'SUGAR':
            uNet_map = np.copy(mrc.uNet_map_SA)
            print("Only Sugar UNet Considered")
        elif params1 == 'SUGAR_SS':
            uNet_map_SA = np.copy(mrc.uNet_map_SA)
            uNet_map_SS = np.copy(mrc.uNet_map_SS)
            uNet_map_SA[uNet_map_SS < b_bone_threshold] = 0
            uNet_map = np.copy(uNet_map_SA)
            if params2 == True:
                uNet_map = np.copy(mrc.uNet_map_SA * mrc.uNet_map_SS)
            else:
                uNet_map = np.copy(uNet_map_SA)

            print("Secondary Structure Considered w/ Sugar UNet")

        # Threshold the UNet Map, and find the indices of the points above the threshold
        uNet_map[uNet_map <= keyAtom_threshold] = 0
        self.uNet_map = uNet_map
        uNet_map_threshold_indices = np.nonzero(uNet_map)
        
        

        pseudo_coordinates = []
        pseudo_density = []

        for i, j, k in zip(uNet_map_threshold_indices[0], uNet_map_threshold_indices[1], uNet_map_threshold_indices[2]):
            pseudo_coordinates.extend([float(i), float(j), float(k), 0])
            pseudo_density.append(og_dens[i][j][k])

        
        self.ps_count = len(pseudo_density)
        self.pseudo_coords = np.array(pseudo_coordinates).reshape(self.ps_count, 4)
        self.ps_dens = np.array(pseudo_density)
        
        print(f"Method:{params1}\nThere were {self.ps_count} points found, above the threshold!\n"
              "the threshold was {keyAtom_threshold} and the density threshold was {b_bone_threshold}!")
         
    def COM_Clustering_Simple(self, dist_threshold: float=3):
        """
        Calculates the center of mass for a set of points using a simple algorithm.

        Args:
            threshold (float): The distance threshold for merging points. Points within this distance will be merged.

        Returns:
            None
        """
        ps_coord_count = np.copy(self.ps_count)
        pseudo_coords = np.copy(self.pseudo_coords)
        ps_dens = np.copy(self.ps_dens)
        
        # Sort the points by density
        sort_indices = np.flip(np.argsort(ps_dens))
        pseudo_coords = pseudo_coords[sort_indices] 
        ps_dens = ps_dens[sort_indices]
        
        # Create a Dictionary which contains the merged points and respective groupings/Clusters       
        self.merged_points = {}
        
        # These are the massweight the coordinates
        self.mw_merge_dens = []
        self.mw_coords = []

        print(f"Running COM_Clustering_Simple() with a distance threshold of {dist_threshold} Angstroms")
        pseudo_coords = np.array(pseudo_coords)
        tree = KDTree(pseudo_coords[:,:3]) # Create a KDTree for the coordinates

        for i in tqdm(range(ps_coord_count)):
            if pseudo_coords[i,3] != 0 or ps_dens[i] == 0: # If the 4th column is 1, then it has already been merged
                                                                 # If the density is 0, then it is not a valid point
                continue

            # List is ordered by density, so we can use the first point as the test point
            test_coord = pseudo_coords[i,:3]
            
            # Total Mass From this Point each iter starts at 0
            d_total = 0
            
            # Lists to store the merged coordinates and the mass weighted coordinates
            merge_coords = []
            mass_weighted_coords = []
            
            # Find points within the threshold distance
            indices = tree.query_ball_point(test_coord, dist_threshold)
            for j in indices:
                if pseudo_coords[i,3] == 0:
                    merge_coords.append(pseudo_coords[i,:3])
                    mass_weighted_coords.append(test_coord*10.0*ps_dens[i])
                    d_total += ps_dens[i]*10 # Weight by Density
                    pseudo_coords[i,3] = 1    # Change the 4th column to 1 to indicate that it has been merged
                    
                if pseudo_coords[j,3] != 0 or ps_dens[j] == 0: # If we have already merged continue
                    continue

                merge_coords.append(pseudo_coords[j,:3])
                mass_weighted_coords.append(pseudo_coords[j,:3]*ps_dens[j]*10)
                d_total += ps_dens[j]*10
                pseudo_coords[j,3] = 1

            if len(merge_coords) > 0:
                self.merged_points[i] = merge_coords
                mass_weighted_coords = np.array(mass_weighted_coords).reshape(-1, 3)
                self.mw_coords.extend([np.sum(mass_weighted_coords, axis = 0) / d_total])
                self.mw_merge_dens.append(d_total/float(len(merge_coords))) # Store the total mass per coordinate
                
        
        self.mw_coords = np.array(self.mw_coords).reshape(-1, 3)
            
                
        print(f"From {self.pseudo_coords.shape[0]} points, we haved reduced it to {len(self.merged_points)} coordinates!\
        \nThat means {self.pseudo_coords.shape[0] - len(self.merged_points)} have been merged!")
        
        # self.merged_plot_coords = []
        
        # for iterEnum, i in enumerate(self.merged_points.keys()):            
        #     # Merged Points
        #     coords_centroid = np.mean(self.merged_points[i], axis=0)
        #     self.merged_plot_coords.extend([coords_centroid[0], coords_centroid[1], coords_centroid[2]])
        
        # self.mw_coords = np.array(self.mw_coords).reshape(-1, 3)
        # self.merged_plot_coords = np.array(self.merged_plot_coords).reshape(-1, 3)
       
        return
    
    def Merge_Coord(self, rcut: float = .15, d_cut:float = 3):
        """
        Merge coordinates based on distance and densities.

        Parameters:
        - rcut (float): The cutoff value for density fraction. Default is 0.15.
        - d_cut (int): The cutoff value for distance. Default is 3.

        Returns:
        - merged_coords (list): The final merged coordinates.

        """
        # First Determine Fraction Density
        max_dens = np.max(self.mw_merge_dens)
        min_dens = np.min(self.mw_merge_dens)

        print(f"Max Dens: {max_dens} and Min Dens: {min_dens}")
        frac_dens = 1 / (max_dens - min_dens)

        stock_indexes = np.ones(len(self.mw_merge_dens))
        member_indexes = np.arange(len(self.mw_merge_dens))
        stock_indexes[((self.mw_merge_dens - min_dens) * frac_dens) < rcut] = 0

        # Note, Before Starting we have each center of mass and the associated density with it
        # We will now continue merging the coordinates, via the distance and densities
        print("Merging Coordinates...")
        for i in tqdm(range(0, len(self.mw_merge_dens))):
            for j in range(i + 1, len(self.mw_merge_dens)):
                if np.linalg.norm(self.mw_coords[i] - self.mw_coords[j]) < d_cut:
                    if self.mw_merge_dens[i] > self.mw_merge_dens[j]:
                        member_indexes[j] = i
                        stock_indexes[j] = 0
                    else:
                        member_indexes[i] = j
                        stock_indexes[i] = 0
                        break

        print("Checking for relevant parent child relationships...")
        for i in tqdm(range(0, len(self.mw_merge_dens))):
            child = int(member_indexes[i])
            # If the child (point) is the same as the parent (point), then we have reached the end of the line
            related_coords = []
            related_dens = []
            while child != member_indexes[i]:

                # If the child is not same as the parent (Member_Indexes[i]), then we have a parent child relationship
                # We will store the coordinates and densities of the child for this value of i
                # Then we will update the child to the value it currently stores (Parent of the current child and trace up the
                # Ancestoral tree until we reach the main parent (oldest Ancestor))
                child = int(member_indexes[i])
                related_coords.append(self.mw_coords[i, :3])
                related_dens.append(self.mw_merge_dens[i])

            # Update parent child relationship so that it reflects the child is part of this parent
            member_indexes[i] = child

        num_merge2 = np.count_nonzero(stock_indexes)
        merged_coords = []

        print("Preparing Final Coordinates...")
        for i in tqdm(range(0, len(stock_indexes))):
            if stock_indexes[i] == 0:  # Stock Indexes of 0 are smaller densities that will be merged
                continue

            COM_MergedCoords = self.determine_COM(np.array(related_coords).reshape(-1,3), np.array(related_dens))

            if COM_MergedCoords == 10:
                continue
            # merged_coords.extend(COM_MergedCoords[i, :3])
            merged_coords = np.array(merged_coords).reshape(-1, 3)

        return merged_coords
            
    def Find_COM(coordinates, densities):
        """
        Determines the center of mass of a set of coordinates and densities.

        Args:
            coordinates (np.array): The coordinates of the points.
            densities (np.array): The densities of the points.

        Returns:
            np.array: The center of mass of the points.

        Raises:
            None

        """
        return np.sum(coordinates * densities[:, np.newaxis] * 10, axis=0) / np.sum(densities)

    def filter_data_by_threshold(data: mrcfile, threshold:float=0.5):
    
        uNet_filtrered_map = np.copy(data)
        uNet_filtrered_map[uNet_filtrered_map <= threshold] = 0
        indexes_nonZero = np.nonzero(uNet_filtrered_map)
        
        coordinate_output = np.transpose(indexes_nonZero)

        return uNet_filtrered_map, coordinate_output
    
    def bfs(self, search:int=1): #Breadth First Search
        
        uNet_filtrered_map = np.copy(self.uNet_map)
        indexes_nonZero = np.nonzero(uNet_filtrered_map)            
        coordinate_output = np.transpose(indexes_nonZero)
        
        
        indices_2d = coordinate_output
        data = uNet_filtrered_map
        print(indices_2d)
        
        # #Lets make the array a 2D array, axis 0 is each Nonzero point, axis 1 is the x,y,z coordinates
        # indices = np.nonzero(filtered_uNet_Map) #Data is the map file with the treshold applied indexes are where it is non zero
        # indices_2d = np.transpose(indices) #Make this into an array of 3D coordinates
        
        
        # #Lets make the array a 2D array, axis 0 is each Nonzero point, axis 1 is the x,y,z coordinates
        # indices = np.nonzero(data) #Data is the map file with the treshold applied indexes are where it is non zero
        # indices_2d = np.transpose(indices) #Make this into an array of 3D coordinates
        # print(indices_2d)
        
        print("Number of Points to Traverse", indices_2d.shape[0]) 
        
        visited = np.zeros(data.shape, dtype=bool)
        print("Visited",visited.shape)
        blob = {} #Each key represents a grouping found within a distance

        # Define the directions for up, down, left, right, front, center, back
        directions = np.array([(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)])
        if search > 1:
            directions = directions*int(search)
            
        

        for start in tqdm(indices_2d, desc='BFS'):
            
            start = tuple(start)
                    
            if visited[start]:
                continue

            queue = deque([start])
            visited[start] = True
            blob[start] = [start]
            
            new_blob_start = 1

            while queue:
                
                x, y, z = queue.popleft()
                neighbors = np.array([x, y, z]) + directions
                
                #Ensure neighbors are within boundaries
                valid_neighbors = np.all((neighbors >= 0) & (neighbors < np.array(data.shape)), axis=1)
                
                unvisited_neighbors = neighbors[valid_neighbors]
                unvisited_neighbors = unvisited_neighbors[~visited[tuple(unvisited_neighbors.T)]]
                visited[tuple(unvisited_neighbors.T)] = True
                
                
                for point in unvisited_neighbors:
                    x,y,z = point[0], point[1], point[2]
                    if data[x,y,z] != 0:
                        blob[start].append(tuple([x,y,z]))
                        queue.append(tuple(point))
                
        
        self.bfs_keyAtoms_dict = blob
      
        print("Size of Blob", len(blob))
        print("Number of New Blob Starts", new_blob_start)
        
        

        return blob
    
    def center_of_mass_from_dict(self, blob, data):
        total_density = []
        com_coords = []
        for key in blob.keys():
            
            current_blob = np.array(blob[key])
            com_x, com_y, com_z = current_blob[:,0].astype(float), current_blob[:,1].astype(float), current_blob[:,2].astype(float)
            
            mass_of_blob = data[tuple(np.array(blob[key]).T)]*10
            sum_mass = np.sum(mass_of_blob)
            
            if(sum_mass != 0):
                com_x = np.sum(np.dot(com_x,mass_of_blob)) / sum_mass
                com_y = np.sum(np.dot(com_y,mass_of_blob)) / sum_mass
                com_z = np.sum(np.dot(com_z,mass_of_blob)) / sum_mass
                
                #print("Center of Mass", com_x, com_y, com_z)
                total_density.append(sum_mass)
                com_coords.extend([com_x, com_y, com_z])
                
        return np.array(com_coords).reshape(-1, 3), total_density
    
    #This function will determine the center of mass of a set of coordinates and densities
    def determine_COM(self, coordinates, densities):
        """
        Determines the center of mass of a set of coordinates and densities.

        Args:
            coordinates (np.array): The coordinates of the points.
            densities (np.array): The densities of the points.

        Returns:
            np.array: The center of mass of the points.

        Raises:
            None

        """
        if(np.sum(densities) == 0):
            return (10)
        # if(sum_mass != 0):
        #         #com_x = 
        #         com_x = np.sum(np.dot(com_x,mass_of_blob)) / sum_mass
        #         com_y = np.sum(np.dot(com_y,mass_of_blob)) / sum_mass
        #         com_z = np.sum(np.dot(com_z,mass_of_blob)) / sum_mass
                
        return np.sum(coordinates * densities[:, np.newaxis] * 10, axis=0) / np.sum(densities)
    
    # def filter_data_by_threshold(data1, data2, threshold):
    
    #     keyAtom_MRC = np.copy(data1)
    #     data1[data2 <= threshold] = 0
    #     indexes_nonZero = np.nonzero(data1)
        
    #     coordinate_output = np.transpose(indexes_nonZero)

    #     return data1, coordinate_output

    # def center_of_mass_from_dict(self, blob, data):
    #     total_density = []
    #     com_coords = []
    #     for key in blob.keys():
            
    #         current_blob = np.array(blob[key])
    #         com_x, com_y, com_z = current_blob[:,0].astype(float), current_blob[:,1].astype(float), current_blob[:,2].astype(float)
            
    #         mass_of_blob = data[tuple(np.array(blob[key]).T)]*10
    #         sum_mass = np.sum(mass_of_blob)
            
    #         if(sum_mass != 0):
    #             com_x = np.sum(np.dot(com_x,mass_of_blob)) / sum_mass
    #             com_y = np.sum(np.dot(com_y,mass_of_blob)) / sum_mass
    #             com_z = np.sum(np.dot(com_z,mass_of_blob)) / sum_mass
                
    #             print("Center of Mass", com_x, com_y, com_z)
    #             total_density.append(sum_mass)
    #             com_coords.extend([com_x, com_y, com_z])
                
    #     return np.array(com_coords).reshape(-1, 3), total_density
    
    def merge_coord_bfs(self, mw_coords, mw_merge_dens, rcut=.15, d_cut=3):
            #First Determine Fraction Density
            max_dens = np.max(mw_merge_dens)
            min_dens = np.min(mw_merge_dens)
            
            print(f"Max Dens: {max_dens} and Min Dens: {min_dens}")
            frac_dens = 1/(max_dens - min_dens)
            
            stock_indexes = np.ones(len(mw_merge_dens))
            member_indexes = np.arange(len(mw_merge_dens))
            stock_indexes[((mw_merge_dens - min_dens)*frac_dens) < rcut] = 0
            
            
            print("Merging Coordinates...")
            for i in tqdm(range(0, len(mw_merge_dens))):
                for j in range(i+1, len(mw_merge_dens)):
                    if np.linalg.norm(mw_coords[i] - mw_coords[j]) < d_cut:
                        if(mw_merge_dens[i] > mw_merge_dens[j]):
                            member_indexes[j] = i
                            stock_indexes[j] = 0
                        else:
                            member_indexes[i] = j
                            stock_indexes[i] = 0
                            break
            
            print("Checking for relevant parent child relationships...")           
            
            for i in tqdm(range(0, len(mw_merge_dens))):
                child = int(member_indexes[i])
                related_coords = []
                related_dens = []
                while child != member_indexes[i]:                
                    child = int(member_indexes[i])
                    related_coords.append(mw_coords[i,:3])
                    related_dens.append(mw_merge_dens[i])
                
                # #Determine new COM from the related coords
                # related_coords = np.resize()
                
                member_indexes[i] = child
                
            num_merge2 = np.count_nonzero(stock_indexes)
            merged_coords = []
            
            print("Preparing Final Coordinates...")
            for i in tqdm(range(0,len(stock_indexes))):
                if stock_indexes[i] == 0:
                    continue
                
                merged_coords.extend(mw_coords[i,:3])
                
            merged_coords = np.array(merged_coords)
            merged_coords = merged_coords.reshape(-1,3)
            return merged_coords    
        

         
            
            
