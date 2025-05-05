"""
AGV Pathfinding and Animation Script

This script implements an Automated Guided Vehicle (AGV) navigation system using the ant colon optimisation
algorithm to find the shortest path on a grid-based map. The AGV navigates from a start position
to a target block of a specified colour, avoiding obstacles. The script also includes functionality
to animate the AGV's movement and generate a sequence of commands for the AGV to execute.

The script is divided into three main classes:
1. Database: Handles loading and updating block locations and the AGV's start position from a CSV
    file.
2. Navigation: Implements the ACO algorithm with both Manhattan and diagonal heuristics for
    pathfinding.
3. AGV: Manages the visualization of the grid, animation of the AGV's movement, and generation of
    movement commands.

The script supports command-line arguments for customization, such as specifying the target block
colour and allowing diagonal movement.

Usage:
    python aco.py <colour> [--diagonal <True/False>]

Arguments:
    colou (str): The target block colour (e.g., 'green'). Defaults to 'green'.
    --diagonal (bool): Whether to allow diagonal movement. Defaults to True.

Example:
    python aco.py green --diagonal True 

Dependencies:
    - argparse: For parsing command-line arguments.
    - heapq: For implementing the priority queue in ACO.
    - numpy: For handling grid operations.
    - pandas: For reading and updating the CSV file.
    - matplotlib: For visualizing the grid and animating the AGV's movement.

Raises:
    ValueError: If an invalid movement between non-adjacent cells is detected.

Notes:
    - The grid is represented as a 2D numpy array, where 0 indicates free space and 1 indicates an
        obstacle.
    - The AGV's movement is animated using matplotlib's FuncAnimation.
    - The script assumes the AGV starts facing 'upwards' and generates movement commands
        accordingly.
"""

import argparse
import heapq
import subprocess
import time
import tempfile
import math
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from matplotlib.animation import animation,FuncAnimation, PillowWriter
import matplotlib.animation as animation
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict

class Database:
    """
    Handles the database interactions for storing and retrieving AGV start positions
    and LEGO block locations from a CSV file.
    """

    # def __init__(self, file_path="database/location.csv"):
    def __init__(self, file_path="C:/Users/timri\OneDrive - University of Bath/Documents/Year_4_modules/fyp/code/database/location.csv"):
        self.file_path = file_path
        self.grid = Navigation.map_grid
        self.block_locations, self.start, self.warehouse_dimensions, self.row_scale, self.column_scale = self.load_from_csv()

    def load_from_csv(self):
        """
        Load block locations and start position from a CSV file.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            dict: Block locations with colour as key and (y, x) coordinates as value.
            tuple: Start position (y, x).
        """
        ## read CSV into DataFrame
        df = pd.read_csv(self.file_path)

        ## extract the postion of each LEGO block
        block_locations = {
            row["colour"]: (row["location_y"], row["location_x"])
            for _, row in df.iterrows()
            if row["colour"] not in ("start", "original_width_height", "warehouse_dimensions")
        }

        ## load start position from database
        start_row = df[df["colour"] == "start"]
        if not start_row.empty:
            start = (start_row.iloc[0]["location_y"], start_row.iloc[0]["location_x"])

        ## start position defaults to 7,7 if not found in CSV
        else:
            start = (7, 7)

        ## load grid size from database
        grid_size_row = df[df["colour"] == "original_width_height"]
        if not grid_size_row.empty:
            database_grid_size = (
                grid_size_row.iloc[0]["location_y"],
                grid_size_row.iloc[0]["location_x"],
            )

        ## database grid size defaults to 16 x 16 if not found in CSV
        else:
            database_grid_size = (16, 16)

        ## load warehouse dimensions from database 
        warehouse_dimensions_row = df[df["colour"] == "warehouse_dimensions"]
        if not warehouse_dimensions_row.empty:
            warehouse_dimensions = (
                warehouse_dimensions_row.iloc[0]["location_y"],
                warehouse_dimensions_row.iloc[0]["location_x"],
            )
        ## warehouse dimensions default to 5000x5000                                    ######### add unit#######################    
        else:
            warehouse_dimensions = (5000, 5000)

        ## determine scale factors
        row_scale = self.grid.shape[0] / database_grid_size[0]
        col_scale = self.grid.shape[1] / database_grid_size[1]

        ## scale all block locations to new grid
        def scale_position(pos):
            row, col = pos
            scaled_row = min(int(round(row * row_scale)), self.grid.shape[0] - 1)
            scaled_col = min(int(round(col * col_scale)), self.grid.shape[1] - 1)
            return (scaled_row, scaled_col)

        ## scale coloured blocks
        scaled_locations = {
            colour: scale_position(loc)
            for colour, loc in block_locations.items()
        }

        ## sclae start
        scaled_start = scale_position(start)

        ## scale warehouse dimensions
        scaled_warehouse_dimensions = scale_position(warehouse_dimensions)

        return (
            scaled_locations,
            scaled_start,
            scaled_warehouse_dimensions,
            row_scale,
            col_scale,
        )
    

    def update_start_position(self, new_start):
        """
        Update the start position in the CSV file.

        Args:
            new_start (tuple): new agv start position
        """
        scaled_new_start = (
        max(0, (new_start[0]) // Database().row_scale),  # Adjust for boundary and scale down
        max(0, (new_start[1]) // Database().column_scale)
        )
        ## read CSV into dataframe
        df = pd.read_csv(self.file_path)

        ## update the start position
        df.loc[df["colour"] == "start", ["location_x", "location_y"]] = [
            scaled_new_start[1],
            scaled_new_start[0],
        ]

        ## save the updated DataFrame back to the CSV file
        df.to_csv(self.file_path, index=False)

        print(f"Updated start position to: {scaled_new_start}")


class Navigation:
    """
    Provides pathfinding algorithms (Ant Colony Optimization) 
    and manages the grid-based navigation of the AGV.
    """

    ## define grid-based map (0 = free space, 1 = obstacle)
    map_grid = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
        ]
    )
    # map_grid = np.array(
    #     [
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [1, 1, 1, 0, 0, 1, 0],
    #         [0, 0, 1, 0, 0, 1, 0],
    #         [0, 0, 1, 0, 0, 0, 0],
    #         [0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 1, 1, 1, 1],
    #         [0, 0, 0, 0, 0, 0, 0],
    #     ]
    # )
    map_grid = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    # map_grid = np.load("ML_navigation/map_creation/bw_array.npy")
    # map_grid = np.load("ML_navigation/map_creation/map_grid.npy")
    # map_grid = np.load("map_creation/map_grid.npy")

    CARDINAL_NEIGHBOURS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  ## Up, Down, Left, Right
    DIAGONAL_NEIGHBOURS = [
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 0),  ## Cardinal directions
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ]  ## Diagonal directions

    def __init__(self):
        self.grid = Navigation.map_grid
        self.block_locations = Database().block_locations
        self.goal = None   ## will be set during search
        self.exploration_counts = np.zeros_like(self.grid, dtype=int)
        self.non_optimal_paths = []

        # ACO parameters
        # self.num_ants = 2000
        # self.max_iterations = 1000
        # self.alpha = 4.0  # Pheromone importance
        # self.beta = 4.0   # Heuristic importance
        # self.rho = 0.1    # Evaporation rate
        # self.delta_t = 3.0      # Pheromone deposit constant
        # self.pheromone = np.ones_like(self.grid, dtype=float) * 0.1
        # self.pheromone[self.grid == 1] = 0  # No pheromone on obstacles

        
        # ACO parameters
        self.num_ants = 100
        self.max_iterations = 20
        self.alpha = 1.0  # Pheromone importance
        self.beta = 1.0   # Heuristic importance
        self.rho = 0.1    # Evaporation rate
        self.delta_t = 2.0      # Pheromone deposit constant
        self.pheromone = np.ones_like(self.grid, dtype=float) * 0.1
        self.pheromone[self.grid == 1] = 0  # No pheromone on obstacles

        
        # self.num_ants = 1000
        # self.max_iterations = 1000
        # self.alpha = 2.0  # Pheromone importance
        # self.beta = 2.0   # Heuristic importance
        # self.rho = 0.2    # Evaporation rate
        # self.delta_t = 2.0      # Pheromone deposit constant
        # self.pheromone = np.ones_like(self.grid, dtype=float) * 0.1
        # self.pheromone[self.grid == 1] = 0  # No pheromone on obstacles


    @staticmethod
    def heuristic(node, goal):
        """
        Calculate the Manhattan distance heuristic between two nodes.

        Args:
            node (tuple): Current node coordinates (y, x).
            goal (tuple): Goal node coordinates (y, x).

        Returns:
            int: Manhattan distance.
        """
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    @staticmethod
    def heuristic_diagonal(node, goal):
        """
        Calculate the diagonal distance heuristic between two nodes.

        Args:
            node (tuple): Current node coordinates (y, x).
            goal (tuple): Goal node coordinates (y, x).

        Returns:
            int: Diagonal distance.
        """
        dx = abs(node[0] - goal[0])
        dy = abs(node[1] - goal[1])
        return max(dx, dy)
    
    def aco_search_2(self, start, goal, diagonal=False):
        """
        Perform Ant Colony Optimization to find the shortest path from start to goal.
        
        Args:
            start (tuple): Start position (y, x).
            goal (tuple): Goal position (y, x).
            diagonal (bool): Whether to allow diagonal movement.
            
        Returns:
            list: The shortest path found, or None if no path exists.
            numpy.ndarray: Exploration heatmap
        """
        self.goal = goal
        best_path = None
        best_path_length = float('inf')
        best_path_cost = float('inf')
        
        for iteration in range(self.max_iterations):
            all_paths = []
            
            for ant in range(self.num_ants):
                path, path_cost = self.construct_ant_path(start, diagonal)
                if path:
                    path_length = len(path)
                    all_paths.append((path, path_length))
                    
                    if path_length < best_path_length:
                        best_path = path
                        best_path_length = path_length
                        best_path_cost = path_cost
            
            self.update_pheromone(all_paths)
            self.evaporate_pheromone()
            
            ## early stopping if path found is not improving
            if best_path and iteration > 200 and len(set([p[1] for p in all_paths[-50:]])) == 1:
                break        

        if best_path:
            print(f"Total cost of ACO path: {best_path_cost}")
            return best_path, self.create_exploration_heatmap_aco()
        return None

    def aco_search(self, start, goal, diagonal=False):
        """
        Perform Ant Colony Optimization to find the shortest path from start to goal.

        Args:
            start (tuple): Start position (y, x).
            goal (tuple): Goal position (y, x).
            agv_size (int): AGV radius in cells.
            diagonal (bool): Whether to allow diagonal movement.

        Returns:
            list: The shortest path found, or raises ValueError if not possible.
            numpy.ndarray: Exploration heatmap.
        """
        self.goal = goal
        ## validate start and goal positions
        ## start position out of bounds
        if not (
            0 <= start[0] < self.grid.shape[0] and 0 <= start[1] < self.grid.shape[1]
        ):
            raise ValueError(f"Start position {start} is out of map bounds.")

        ## goal position out of bounds
        if not (
            0 <= goal[0] < self.grid.shape[0] and 0 <= goal[1] < self.grid.shape[1]
        ):
            raise ValueError(f"Goal position {goal} is out of map bounds.")

        ## start position is on an obstacle
        if self.grid[start[0], start[1]] == 1:
            raise ValueError(f"Start position {start} is on an obstacle.")

        ## goal position is on an obstacle
        if self.grid[goal[0], goal[1]] == 1:
            raise ValueError(f"Goal position {goal} is on an obstacle.")

        ## start and goal position are within the same cell
        if self.goal == start:
            raise ValueError(
                f"Start position {start} and goal position {goal} are located within the same cell."
            )

        ## run ACO
        path_result = self.run_aco(start, diagonal)
        if path_result:
            return path_result

        ## if no path is found
        raise ValueError("No valid path found: ACO could not reach the goal.")

    def run_aco(self, start, diagonal):
        """
        Internal ACO run with agv_size-aware navigation.
        """
        best_path = None
        best_path_length = float('inf')
        best_path_cost = float('inf')

        for iteration in range(self.max_iterations):
            all_paths = []

            for ant in range(self.num_ants):
                path, path_cost = self.construct_ant_path(start, diagonal)
                if path:
                    path_length = len(path)
                    all_paths.append((path, path_length))

                    if path_length < best_path_length:
                        best_path = path
                        best_path_length = path_length
                        best_path_cost = path_cost

            self.update_pheromone(all_paths)
            self.evaporate_pheromone()

            if best_path and iteration > 200 and len(set([p[1] for p in all_paths[-50:]])) == 1:
                break

        if best_path:
            print(f"Total cost of ACO path: {best_path_cost}")
            return best_path, self.create_exploration_heatmap_aco()

        return None

    def construct_ant_path(self, start, diagonal):
        """
        Construct a path for a single ant using pheromone trails.
        """
        path = [start]
        current = start
        visited = set([start])
        total_cost = 0.0
        
        while current != self.goal:
            neighbours = self.get_valid_neighbours(current, diagonal, visited)
            if not neighbours:
                return None, float('inf')  # Dead end
            
            next_node = self.select_next_node(current, neighbours, diagonal)
            if next_node is None:
                return None, float('inf')
            
            ## determine cost for each move
            dx = next_node[0] - current[0]
            dy = next_node[1] - current[1]
            move_cost = np.sqrt(2) if dx != 0 and dy != 0 else 1

            total_cost += move_cost
            path.append(next_node)
            visited.add(next_node)
            current = next_node
            
        return path, total_cost

    def get_valid_neighbours(self, node, diagonal, visited):
        """
        Get valid neighbouring nodes that can be moved to.
        """
        neighbours = []
        x, y = node
        
        ## precompute blocked positions (excluding goal)
        blocked_positions = {
            (by, bx) for colour, (by, bx) in self.block_locations.items()
            if (by, bx) != self.goal
        }

        for dx, dy in (self.DIAGONAL_NEIGHBOURS if diagonal else self.CARDINAL_NEIGHBOURS):
            nx, ny = x + dx, y + dy
            
            ## check bounds
            if not (0 <= nx < len(self.grid) and 0 <= ny < len(self.grid[0])):
                continue
                
            ## check obstacle
            if self.grid[nx][ny] == 1:
                continue
                
            ## check if already visited
            if (nx, ny) in visited:
                continue

            ## check for LEGO blocks in the way (unless it's the target)
            if (nx, ny) in blocked_positions:
                continue
            
            ## additional checks for diagonal moves
            if dx != 0 and dy != 0:
                ## check if diagonal path is blocked by obstacles
                if self.grid[x + dx][y] == 1 or self.grid[x][y + dy] == 1:
                    continue

                ## check if diagonal path is blocked by LEGO blocks
                if (nx, y) in blocked_positions or (x, ny) in blocked_positions:
                    continue

            neighbours.append((nx, ny))
            
        return neighbours

    def select_next_node(self, current, neighbours, diagonal):
        """
        Select the next node based on pheromone and heuristic information.
        """
        if not neighbours:
            return None
            
        probabilities = []
        total = 0.0
        
        ## choose heuristic based on diagonal movement setting
        heuristic_func = self.heuristic_diagonal if diagonal else self.heuristic

        for neighbour in neighbours:
            pheromone = self.pheromone[neighbour[0], neighbour[1]]
            heuristic = 1 / (1 + heuristic_func(neighbour, self.goal))
            
            probability = (pheromone ** self.alpha) * (heuristic ** self.beta)
            probabilities.append(probability)
            total += probability
            
        if total == 0:
            return neighbours[np.random.randint(0, len(neighbours))]
            
        probabilities = [p / total for p in probabilities]
        return neighbours[np.random.choice(range(len(neighbours)), p=probabilities)]

    def update_pheromone(self, paths):
        """
        Update pheromone trails based on ant paths.
        """
        for path, cost in paths:
            if not path or cost == 0 or cost == float('inf'):
                continue
                
            pheromone_deposit = self.delta_t / cost
            
            for node in path:
                self.pheromone[node[0], node[1]] += pheromone_deposit

    def evaporate_pheromone(self):
        """
        Evaporate pheromone trails.
        """
        self.pheromone *= (1 - self.rho)
        self.pheromone[self.pheromone < 0.1] = 0.1  # Minimum pheromone level

    def create_exploration_heatmap_aco(self):
        """
        Create exploration heatmap based on pheromone trails.
        """
        heatmap = self.pheromone.copy()
        heatmap[self.grid == 1] = np.nan   ## Mark obstacles as not an number
        return heatmap


class AGV:
    """
    Plot grid and animate AGV movement. Plot heatmap of cells visited during ACO.
    Generate movement pattern for agv from start to goal.
    """

    GRID_SIZE_CM = 10  ## each grid cell represents 10 cm

    def __init__(self):
        """
        Initialize the AGV with database and navigation instances.

        Args:
            database (Database): Database instance to access block locations and start position.
            navigation (Navigation): Navigation instance for pathfinding.
        """
        self.database = Database()
        self.navigation = Navigation()
        self.combined_counts = np.zeros_like(
            self.navigation.grid, dtype=int
        )  ## 2D array
        self.last_velocity = 0.01

    def animate_agv(
        self,
        grid: np.ndarray,
        path: List[Tuple[int, int]],
        block_locations: Dict[str, Tuple[int, int]],
        start: Tuple[int, int],
        warehouse_dimensions: Tuple[int, int],
    ):
        """
        Animate the AGV following the calculated path using matplotlib.

        Args:
            grid (np.ndarray): The map grid with obstacles.
            path (list): List of coordinates representing the AGV's path.
            block_locations (dict): Dictionary containing the locations of blocks,
                where the key is the block colour and the value is its (y, x) coordinates.
            start (tuple): The starting position (y, x) of the AGV.
        """
        ## start timer before animation begins
        start_time = time.time()

        ## plot grid, obstacles, LEGO blocks and AStar path
        ax, fig = self.plot_map(grid, start, block_locations, path)

        ax.set_aspect('equal', adjustable='box')
        plt.show()
        ## load AGV image
        agv_image = mpimg.imread("ML_navigation/LEGO_agv.png")
        agv_img = ax.imshow(
            agv_image, extent=(start[1], start[1] + 1, start[0] + 1, start[0])
        )

        ## create smooth path using interpolation
        smoothed_path = self.smooth_path(path, steps = 10)

        ## time in ms between each frame
        interval = 50

        ## start agv animation
        # self.agv_animation(ax, fig, smoothed_path, agv_img, interval, start_time, warehouse_dimensions)

    def agv_animation(self, ax, fig, smoothed_path, agv_img, interval, start_time, warehouse_dimensions):
        """
        Animate the AGV's movement along the smoothed path.

        Args:
            ax (matplotlib.axes.Axes): The axes object for plotting.
            fig (matplotlib.figure.Figure): The figure object for the plot.
            smoothed_path (list): The smoothed path as a list of (x, y) coordinates.
            agv_img (matplotlib.image.AxesImage): The AGV image object.
            interval (int): Time in milliseconds between each frame.
            start_time (float): The start time of the animation for calculating velocity and ETA.
        """

        ## add text annotations for velocity and ETA
        ## dynamic text positioning based on grid size
        grid_height, grid_width = self.navigation.map_grid.shape
        
        ## position text relative to grid size
        text_x = grid_width * 0.1
        text_y = -grid_height * 0.15
        x_shift = grid_width * 0.5
        
        ## create text annotations with bounding boxes for better visibility
        velocity_text = ax.text(text_x, text_y, "", 
                            fontsize=12, color="blue")
        eta_text = ax.text(text_x + x_shift, text_y, "",  # 5 units spacing
                     fontsize=12, color="red")
        eta_seconds, last_eta_seconds = 0, 0

        ## figure out step size to determine how many frames can be skipped to still make a smooth
        ## animation
        step = math.ceil(min(warehouse_dimensions[0], warehouse_dimensions[1])  
                    / (self.GRID_SIZE_CM * 60))
        
        ## figure out the size of the animated AGV
        width, length = Navigation().map_grid.shape[0], Navigation().map_grid.shape[1]
        radius = max(0, min(10, (min(width, length) // 60)))
        
        ## initialize tqdm progress bar
        progress_bar = tqdm(total=len(smoothed_path), desc="AGV Animation Progress", unit="frame")

        def update(frame):
            """
            Update function for smooth animation of the AGV's movement.

            This function is called for each frame of the animation. It updates the AGV's position,
            calculates its velocity and estimated time of arrival (ETA), and checks if the AGV has
            reached the goal. It also updates the start position in the database once the goal is
            reached.

            Args:
                frame (int): The current frame number in the animation.

            Returns:
                tuple: A tuple containing the AGV image object, velocity text, and ETA text for
                rendering.
            """
            global last_eta_seconds
            ## every step frame is animated
            actual_frame = min(frame * step, len(smoothed_path) - 1)
            x, y = smoothed_path[actual_frame]

            ## update progress bar correctly
            progress_bar.n = actual_frame + 1
            progress_bar.refresh()

            ## ensure last frame completes 100%
            if actual_frame >= len(smoothed_path) - 1:
                progress_bar.n = progress_bar.total
                progress_bar.refresh()
                progress_bar.close()
                    
            ## update AGV's position by setting its image extent
            agv_img.set_extent((x - radius, x + 1 + radius, y + 1 + radius, y - radius))

            ## calculate velocity (V = distance / time)
            if actual_frame > 0:
                prev_x, prev_y = smoothed_path[actual_frame - 1]
                distance_m = np.sqrt((x - prev_x)**2 + (y - prev_y)**2) * self.GRID_SIZE_CM / 100
                time_elapsed = time.time() - start_time
                

                ## calculate instantaneous velocity (m/s)
                if distance_m == 0:
                    velocity_mps = self.last_velocity
                elif time_elapsed > 0:
                    velocity_mps = distance_m / (interval/1000)
                    self.last_velocity = velocity_mps
                else:
                    velocity_mps = 0
                velocity_text.set_text(f"Vel: {velocity_mps:.2f} m/s")

                ## calculate ETA
                remaining_frames = len(smoothed_path) - actual_frame
                
                if velocity_mps > 0:
                    eta_seconds = (remaining_frames * distance_m) / (velocity_mps * step)
                    if eta_seconds == 0.0:
                        eta_seconds = last_eta_seconds
                else:
                    eta_seconds = 0
                
                last_eta_seconds = eta_seconds
                eta_text.set_text(f"ETA: {eta_seconds:.1f} sec")
                

            ## check if the AGV has reached the goal (last frame of the animation)
            if actual_frame == len(smoothed_path) - 1:
                print("Simulated AGV has reached the target block!")
                plt.close(fig)
                velocity_text.set_text(f"Vel: 0 m/s")
                eta_text.set_text(f"ETA: 0 sec")
                ## close progress bar
                progress_bar.close()

                ## update the start position in the CSV file to the current (y, x) position
                new_start = (int(y), int(x))
                # Database().update_start_position(new_start)
                
            ## return the updated AGV image, velocity text, and ETA text for rendering
            return agv_img, velocity_text, eta_text

        time_a1 = time.time()
        ## create the animation using FuncAnimation
        ani = animation.FuncAnimation(
            fig, update, frames=int(len(smoothed_path)/step), interval=1, blit=False
        )

        ## display animation until "ESC" is pressed
        def on_key(event):
            if event.key == 'escape':
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()

        # ax.legend(loc="best", labelspacing=0.5)
        
        ## save the animation before showing it
        try:
            ani.save('ML_navigation/animations/agv_animation_aco.mp4', writer='ffmpeg', fps=1000/interval, 
                    dpi=300, bitrate=1800)
            print("ACO animation saved as agv_animation_aco.mp4")
        except Exception as e:
            print(f"Could not save animation: {e}")


        time_a2 = time.time()
        print(f"Time taken to save and show animation: {time_a2 - time_a1}s")


        # ## save animation as a temporary GIF
        # with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as temp_file:
        #     ani.save(temp_file.name, writer=animation.PillowWriter(fps=5))
        #     temp_filename = temp_file.name

        # ## display in streamlit
        # st.title("🚀 AGV Path Animation")
        # st.image(temp_filename)
        # st.pyplot(fig)

    def plot_map(self, grid, start, block_locations, path):
        """
        Plot grid, obstacles, LEGO blocks and path to end position.

        Args:
            grid (numpy.ndarray): 2D array representing the grid, where 1 indicates an obstacle
                and 0 indicates a free space.
            start (tuple): (row, column) coordinates of the start position.
            block_locations (dict): Dictionary where keys are colour names (str) and values are
                (row, column) coordinates.
            path (list of tuples): List of (row, column) coordinates representing a path to be
                drawn (optional).

        Returns:
            (matplotlib.axes.Axes): The plot's axes object.
            (matplotlib.figure.Figure): The plot's figure object.
        """
        ## create subplot, maintaining full grid visibility
        fig, ax = plt.subplots(figsize=(7, 7))

        ## create new grid full of obstacles and then paste the actual grid inside to add obstacle
        ## boundary
        new_grid = np.ones((grid.shape[0] + 2, grid.shape[1] + 2), dtype=int)
        new_grid[1:-1, 1:-1] = grid

        ## set up grid lines
        ax.set_xticks(np.arange(-0.5, new_grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, new_grid.shape[0], 1), minor=True)
        # ax.grid(which="minor", color="gray", linestyle="-", linewidth=1)
        ax.set_xlim(-0.5, new_grid.shape[1] - 1.5)
        ax.set_ylim(new_grid.shape[0] - 1.5, -0.5)

        ## plot coloured blocks
        for colour, (y, x) in block_locations.items():
            ax.add_patch(
                plt.Circle(
                    (x + 0.5, y + 0.5),
                    0.3,
                    color=colour,
                    label=f"{colour.capitalize()} Block",
                )
            )

        ## plot obstacles
        for y in range(new_grid.shape[0]):
            for x in range(new_grid.shape[1]):
                if new_grid[y, x] == 1:
                    ax.add_patch(plt.Rectangle((x - 1, y - 1), 1, 1, color="black"))

        ## plot start position as a square
        ax.add_patch(
            plt.Rectangle(
                (start[1], start[0]), 1, 1, color="green", label="Start", alpha=0.6
            )
        )

        ## plot path
        if path:
            ax.plot(
                [p[1] + 0.5 for p in path],
                [p[0] + 0.5 for p in path],
                color="blue",
                linestyle="--",
                label="Path",
                zorder=1,
            )

        return ax, fig

    def smooth_path(self, path, steps):
        """
        Generates a smoothed version of the given path using linear interpolation. Needed for
        animation so that AGV also move inbetween path points and doesn't teleport from one to
        another.

        Args:
            path (list of tuples): List of (row, column) coordinates representing the path
            steps (int): number of in-between steps per movement

        Returns:
            smooth_path (list of tuples): List of interpolated (x, y) coordinates for a smoother
                transition.
        """
        smooth_path = []

        ## extract (x, y) coordinates and generate interpolated points between consecutive paths
        ## points
        for i in range(len(path) - 1):
            x1, y1 = path[i][1], path[i][0]
            x2, y2 = path[i + 1][1], path[i + 1][0]

            ## more interpolated points are needed for diagonal movement
            if ((x1 - x2) != 0) and ((y1 - y2) != 0):
                movement_steps = int(steps * np.sqrt(2))
            else:
                movement_steps = steps

            ## interpolate movement
            smooth_x = np.linspace(x1, x2, movement_steps)
            smooth_y = np.linspace(y1, y2, movement_steps)
            smooth_path.extend(zip(smooth_x, smooth_y))

        return smooth_path

    def plot_exploration_heatmap(self, heatmap_data):
        """
        Plot a heatmap of the exploration counts during ACO.

        Args:
            exploration_counts (numpy.ndarray): A 2D array containing the number of times each cell was explored.
        """
        plt.figure(figsize=(10, 10))

        ## plot heatmap (excluding obstacles)
        ax = sns.heatmap(
            heatmap_data,
            annot=False,        ## show values
            fmt=".0f",          ## integer format without decimals
            cmap="YlOrRd",      ## heatmap colour
            cbar=True,          ## show colour bar
        )

        ## overlay obstacles as black squares
        for i in range(heatmap_data.shape[0]):
            for j in range(heatmap_data.shape[1]):
                if heatmap_data[i, j] == -1.0:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, color="black"))

        ## add labels and title
        plt.title("Exploration Heatmap of ACO")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")

        ## show the heatmap
        plt.show()

    def get_movement_sequence(self, path, initial_start_position, agv_orientation):
        """
        Converts a path of grid coordinates into a sequence of movements for the LEGO AGV to
        execute. Returns a list of commands: ["move_straight", "move_straight_diagonally",
        "left_turn", "left_turn_45", "right_turn", "right_turn_45", "stop"].
        Once movement sequence has been created, subprocess is started, which connects to the AGV
        and performs movement sequence generated previously.

        Args:
            path (list of tuples): List of (row, column) coordinates representing the path.

        Raises:
            ValueError: Invalid movement between non-adjacent cells.
        """
        if not path:
            return []
        movement_sequence = []

        ## tracks the current facing direction
        current_direction = None

        for i in range(len(path) - 1):

            ## current and next cells
            current = path[i]
            next_cell = path[i + 1]

            ## calculate the direction to the next cell
            dx = next_cell[1] - current[1]  ## change in x (columns)
            dy = next_cell[0] - current[0]  ## change in y (rows)

            ## determine if cardinal and diagonal movement is needed to arive at the next cell
            if dx == 1 and dy == 1:
                next_direction = "right down"
            elif dx == -1 and dy == 1:
                next_direction = "left down"
            elif dx == 1 and dy == -1:
                next_direction = "right up"
            elif dx == -1 and dy == -1:
                next_direction = "left up"
            elif dx == 1:
                next_direction = "right"
            elif dx == -1:
                next_direction = "left"
            elif dy == 1:
                next_direction = "down"
            elif dy == -1:
                next_direction = "up"
            else:
                raise ValueError("Invalid movement between cells")

            ## determine the LEGO AGV command based on the current and next direction
            if current_direction is None:
            # if agv_orientation
                start_movement = self.get_required_movements_enhanced(agv_orientation, next_direction)
                for movement in start_movement:
                    movement_sequence.append(movement)
                    
                ## assuming LEGO AGV is always placed on physical map, facing 'upwards',
                ## the AGV need to be repositioned first, depending on directional movement
            else:

                ## directional instructions based on current and next direction
                if (current_direction, next_direction) in [
                    ("right up", "right up"),
                    ("left up", "left up"),
                    ("left down", "left down"),
                    ("right down", "right down"),
                ]:
                    movement_sequence.append("move_straight_diagonally")
                elif next_direction == current_direction:
                    movement_sequence.append("move_straight")
                elif (current_direction, next_direction) in [
                    ("up", "right"),
                    ("right", "down"),
                    ("down", "left"),
                    ("left", "up"),
                ]:
                    movement_sequence.append("right_turn")
                    movement_sequence.append("move_straight")
                elif (current_direction, next_direction) in [
                    ("up", "left"),
                    ("left", "down"),
                    ("down", "right"),
                    ("right", "up"),
                ]:
                    movement_sequence.append("left_turn")
                    movement_sequence.append("move_straight")
                elif (current_direction, next_direction) in [
                    ("up", "left up"),
                    ("left", "left down"),
                    ("down", "right down"),
                    ("right", "right up"),
                ]:
                    movement_sequence.append("left_turn_45")
                    movement_sequence.append("move_straight_diagonally")
                elif (current_direction, next_direction) in [
                    ("up", "right up"),
                    ("left", "left up"),
                    ("down", "left down"),
                    ("right", "right down"),
                ]:
                    movement_sequence.append("right_turn_45")
                    movement_sequence.append("move_straight_diagonally")
                elif (current_direction, next_direction) in [
                    ("right up", "up"),
                    ("left up", "left"),
                    ("left down", "down"),
                    ("right down", "right"),
                ]:
                    movement_sequence.append("left_turn_45")
                    movement_sequence.append("move_straight")
                elif (current_direction, next_direction) in [
                    ("right up", "right"),
                    ("left up", "up"),
                    ("left down", "left"),
                    ("right down", "down"),
                ]:
                    movement_sequence.append("right_turn_45")
                    movement_sequence.append("move_straight")
                elif (current_direction, next_direction) in [
                    ("right up", "left up"),
                    ("left up", "left down"),
                    ("left down", "right down"),
                    ("right down", "right up"),
                ]:
                    movement_sequence.append("left_turn")
                    movement_sequence.append("move_straight_diagonally")
                elif (current_direction, next_direction) in [
                    ("right up", "right down"),
                    ("left up", "right up"),
                    ("left down", "left up"),
                    ("right down", "left down"),
                ]:
                    movement_sequence.append("right_turn")
                    movement_sequence.append("move_straight_diagonally")
                else:
                    raise ValueError("Invalid direction change")

            ## update the current direction
            current_direction = next_direction

        ## add a stop command at the end
        movement_sequence.append("stop")

        ## convert the movement sequence to a single string
        movement_sequence_str = " ".join(movement_sequence)
        print(movement_sequence_str)
        ## run the script with the movement sequence as an argument
        subprocess.run(["python", "ML_navigation/skeleton.py", "--movement", movement_sequence_str,
                        "--start", str(initial_start_position)])
        

    def get_required_movements_enhanced(self, current_orientation, next_direction):
        """
        Handles both cardinal and diagonal directions.
        next_direction can be in ['up', 'right', 'down', 'left',
                                'right_up', 'left_up', 'right_down', 'left_down']
        """
        cardinal_map = {
            'up': 0, 'right': 1, 'down': 2, 'left': 3,
            'right_up': 0.5, 'left_up': -0.5,
            'right_down': 1.5, 'left_down': 2.5
        }
        
        current_angle = cardinal_map[current_orientation]
        target_angle = cardinal_map[next_direction]
        
        angle_diff = (target_angle - current_angle) % 4
        
        movements = []
        
        ## handle diagonal turns
        if angle_diff % 1 != 0:
            if angle_diff in (0.5, 1.5):
                movements.append("right_turn_45")
                angle_diff -= 0.5
            elif angle_diff in (2.5, 3.5):
                movements.append("left_turn_45")
                angle_diff += 0.5
            
        ## handle cardinal turns
        turns = round(angle_diff)
        turns = angle_diff
        if turns == 1:
            movements.append("right_turn")
        elif turns == 2:
            movements.extend(["right_turn", "right_turn"])
        elif turns == 3:
            movements.append("left_turn")
        
        ## add appropriate movement
        if any(d in next_direction for d in ['up', 'down', 'left', 'right']):
            if '_' in next_direction:
                movements.append("move_straight_diagonally")
            else:
                movements.append("move_straight")
        return movements

def main():
    """
    Main function to navigate the AGV to a target block using ACO.

    - Loads block locations from a database.
    - Parses command-line arguments to determine the target block colour.
    - Executes ACO to find the optimal path to the target block.
    - Animates the AGV movement and generates the movement sequence.
    """
    ## initialise instances
    map_instance = Database()
    nav = Navigation()
    agv = AGV()
    initial_start_position = Database().start

    ## setup argparse for command line argument parsing
    parser = argparse.ArgumentParser(description="Navigate AGV to a target block.")
    parser.add_argument(
        "--diagonal",
        type=lambda x: (str(x).lower() == "true"),
        default="True",
        help="Wheather or not diagonal movement for ACO is allowed. Defaults to 'True'.",
    )
    parser.add_argument(
        "--agv_orientation",
        type=str,
        default="up",
        help="AGV orientation. Defaults to 'up'.",
    )
    parser.add_argument(
        "--target_colour",
        type=str,
        default="green",
        help="Target colour AGV navigates to. Defaults to 'green'",
    )

    args = parser.parse_args()

    ## set goal for AGV
    selected_colour = args.target_colour.strip().lower()
    if selected_colour in map_instance.block_locations:
        goal = map_instance.block_locations[selected_colour]
    else:
        ## default to green block if missing
        default_colour = "red"
        print(f"Invalid colour '{selected_colour}', defaulting to {default_colour} object.")
        goal = map_instance.block_locations[f"{default_colour}"]

    ## execute ACO to find optimal path to the goal
    ## "diagonal = True" allows for diagonal movements
    ## "diagonal = False" only moves cardinally
    # print(nav.map_grid.shape, map_instance.start, goal, map_instance.block_locations)
    # print(map_instance.start)#
    
    time_s = time.time()
    path, exploration_counts = nav.aco_search(
        map_instance.start, goal, diagonal=True
    )
    
    time_e = time.time()
    print(f"Time taken for ACO to find optimal path: {time_e - time_s:.9f}s")

    ## plot and animate AGV path, get agv exploration heatmap and get movement sequence for LEGO
    ## AGV
    if path:
        agv.animate_agv(
            nav.map_grid, path, map_instance.block_locations, map_instance.start,
            map_instance.warehouse_dimensions
        )

        # agv.plot_exploration_heatmap(exploration_counts)

        agv.get_movement_sequence(path, initial_start_position, args.agv_orientation)
    else:
        print("No path found!")


if __name__ == "__main__":
    main()
