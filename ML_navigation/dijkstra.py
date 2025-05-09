"""
AGV Pathfinding and Animation Script

This script implements an Automated Guided Vehicle (AGV) navigation system using the Dijkstra search
algorithm to find the shortest path on a grid-based map. The AGV navigates from a start position
to a target block of a specified colour, avoiding obstacles. The script also includes functionality
to animate the AGV's movement and generate a sequence of commands for the AGV to execute.

The script is divided into three main classes:
1. Database: Handles loading and updating block locations and the AGV's start position from a CSV
   file.
2. Navigation: Implements the Dijkstra search algorithm with both Manhattan and diagonal heuristics for
   pathfinding.
3. AGV: Manages the visualization of the grid, animation of the AGV's movement, and generation of
   movement commands.

The script supports command-line arguments for customization, such as specifying the target block
colour, AGV orientation, AGV size, and whether diagonal movement is allowed.

Usage:
    python astar.py --target_colour <colour> [--diagonal <True/False>] [--agv_size <float>]
    [--agv_orientation <direction>]

Arguments:
    --target_colour (str): The target block colour (e.g., 'green'). Defaults to 'green'.
    --diagonal (bool): Whether to allow diagonal movement. Defaults to True.
    --agv_size (float): Relative size of the AGV (0.0 to 1.0). Defaults to 0.001.
    --agv_orientation (str): Initial AGV orientation ('up', 'down', 'left', 'right'). Defaults to
        'up'.

Example:
    python astar.py --target_colour green --diagonal True --agv_size 0.05

Dependencies:
    - argparse: For parsing command-line arguments.
    - heapq: For implementing the priority queue in Dijkstra search.
    - numpy: For handling grid operations.
    - pandas: For reading and updating the CSV file.
    - matplotlib: For visualizing the grid and animating the AGV's movement.
    - seaborn: For rendering the exploration heatmap.
    - PIL, tqdm, cv2, streamlit: For enhanced image processing and UI features.

Raises:
    ValueError: If the AGV's start or goal position is invalid or if movement between non-adjacent
    cells is detected.

Notes:
    - The grid is represented as a 2D NumPy array, where 0 indicates free space and 1 indicates an
      obstacle.
    - The AGV's movement is animated using matplotlib's FuncAnimation.
    - The AGV is assumed to initially face 'upward' for orientation-aware movement generation.

Author: Tim Riekeles
Date: 2025-06-05
"""

import argparse
import heapq
import subprocess
import time
import tempfile
import math
from PIL import Image
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

    def __init__(self, file_path="database/location.csv"):
        self.file_path = file_path
        self.grid = Navigation.map_grid
        (
            self.block_locations,
            self.start,
            self.warehouse_dimensions,
            self.row_scale,
            self.column_scale,
        ) = self.load_from_csv()

    def load_from_csv(self):
        """
        Load block locations, start position, grid size, and warehouse dimensions from the CSV file.
        Applies scaling based on the grid to ensure correct placement on the map.

        Returns:
            dict: Scaled block locations with colours as keys and (y, x) coordinates as values.
            tuple: Scaled start position (y, x).
            tuple: Scaled warehouse dimensions.
            float: Row scale factor.
            float: Column scale factor.
        """
        ## read CSV into DataFrame
        df = pd.read_csv(self.file_path)

        ## extract the postion of each LEGO block
        block_locations = {
            row["colour"]: (row["location_y"], row["location_x"])
            for _, row in df.iterrows()
            if row["colour"]
            not in ("start", "original_width_height", "warehouse_dimensions")
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
        ## needed when displaying AGV speed and ETA during animation
        warehouse_dimensions_row = df[df["colour"] == "warehouse_dimensions"]
        if not warehouse_dimensions_row.empty:
            warehouse_dimensions = (
                warehouse_dimensions_row.iloc[0]["location_y"],
                warehouse_dimensions_row.iloc[0]["location_x"],
            )
        ## warehouse dimensions default to 5000x5000, ie 50mx50m
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
        Update the AGV start position in the CSV file based on the scaled coordinates.

        Args:
            new_start (tuple): New start position in (y, x) format.
        """
        ## adjust for boundary and scale down
        scaled_new_start = (
            max(
                0, (new_start[0]) // Database().row_scale
            ),
            max(0, (new_start[1]) // Database().column_scale),
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
    Provides pathfinding algorithms (Dijkstra with Manhattan and diagonal heuristics)
    and manages the grid-based navigation of the AGV.
    """

    ## define grid-based map (0 = free space, 1 = obstacle)
    ## example grid
    # map_grid = np.array(
    #     [
    #         [0, 0, 0, 0, 0],
    #         [1, 1, 0, 1, 0],
    #         [0, 0, 0, 0, 0],
    #         [0, 1, 0, 0, 0],
    #     ]
    # )
    map_grid = np.load("ML_navigation/map_creation/map_grid.npy")

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
        self.exploration_counts = np.zeros_like(self.grid, dtype=int)
        self.non_optimal_paths = []
        self.block_locations = Database().block_locations
        self.goal = None

    def dijkstra_search(self, start, goal, agv_size, diagonal=False):
        """
        Execute Dijkstra search with optional diagonal movement and AGV size consideration.

        Args:
            start (tuple): Start coordinates (y, x).
            goal (tuple): Goal coordinates (y, x).
            agv_size (int): Radius of the AGV used for obstacle avoidance.
            diagonal (bool): Whether to allow diagonal movement.

        Returns:
            tuple: (List of path coordinates, exploration heatmap).

        Raises:
            ValueError: If the path is invalid, blocked, or if AGV size prevents navigation.
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

        ## run normal Dijkstra and return path if path was found
        path_result = self.run_dijkstra(start, agv_size, diagonal)
        if path_result:
            return path_result

        ## retry with agv size of 1 to check if agv size was the issue
        fallback_result = self.run_dijkstra(start, 1, diagonal)
        if fallback_result is not None:
            raise ValueError(
                f"The AGV size ({agv_size}) is too large for the current map layout. Try reducing"
                    "it."
            )

        ## otherwise, no path even with agv size of 1
        raise ValueError(
            "No valid path found: Dijkstra could not reach the goal from the given start."
        )


    def run_dijkstra(self, start, agv_size, diagonal):
        """
        Classic Dijkstra algorithm implementation.
        
        Args:
            start (tuple): Start position (y, x)
            diagonal (bool): Whether to allow diagonal movement
        
        Returns:
            tuple: (path, exploration_counts) if path found, else None
        """
        ## exploration count keeping track of which nodes are explored how often
        exploration_counts = np.zeros_like(self.grid, dtype=int)
        ## mark obstacles in the grid as -1 in the exploration counts
        exploration_counts[self.grid == 1] = -1

        ## convert numpy arrays to tuples if needed
        start = tuple(start) if isinstance(start, np.ndarray) else start
        self.goal = tuple(self.goal) if isinstance(self.goal, np.ndarray) else self.goal
        
        ## priority queue: (cumulative_cost, y, x)
        heap = [(0, start[0], start[1])]
        
        ## visited dictionary: {node: (cost, previous_node)}
        visited = {start: (0, None)}
        
        ## movement directions
        directions = self.DIAGONAL_NEIGHBOURS if diagonal else self.CARDINAL_NEIGHBOURS
        
        ## keep track of visited nodes
        closed_set = set()

        while heap:
            current_cost, y, x = heapq.heappop(heap)
            current_node = (y, x)

            ## ensure no nodes are processed multiple times
            if current_node in closed_set:
                continue
            closed_set.add(current_node)

            exploration_counts[y,x] += 1
            
            ## early exit if goal is reached
            if current_node == self.goal:
                break

            
            ## explore neighbours
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                neighbour = (ny, nx)
                
                # #check if move is valid (bounds, obstacles, AGV size)
                if self.is_valid_move(agv_size, y, x, dy, dx, first_iteration=False):

                    ## calculate move cost (sqrt(2) for diagonal, 1 for cardinal)
                    move_cost = math.sqrt(dy**2 + dx**2)
                    new_cost = current_cost + move_cost
                    
                    ## if neighbour not visited or found cheaper path
                    if neighbour not in visited or new_cost < visited[neighbour][0]:
                        visited[neighbour] = (new_cost, current_node)
                        heapq.heappush(heap, (new_cost, ny, nx))
        
        ## reconstruct path if goal was reached
        if self.goal in visited:
            path = self.reconstruct_path(visited, self.goal, start)

            total_cost = visited[self.goal][0]
            print(f"Total Cost of Dijkstra path with Radius = {agv_size}: {total_cost}")
            return path, exploration_counts
        
        return None

    def is_valid_move(self, radius, x, y, dx, dy, first_iteration):
        """
        Check if a move is valid considering the AGV radius and surrounding obstacles.

        Args:
            radius (int): AGV radius.
            x, y (int): Current grid cell.
            dx, dy (int): Movement direction.
            first_iteration (bool): Whether it's the first iteration (full check needed).

        Returns:
            bool: True if the move is valid, else False.
        """

        ## target postions of next move
        target_y, target_x = x + dx, y + dy

        ## precompute blocked positions (excluding goal)
        blocked_positions = {
            (by, bx)
            for colour, (by, bx) in self.block_locations.items()
            if (by, bx) != self.goal
        }

        ## 1. check that obstacle is not outside of bounds
        if not (0 <= target_y < len(self.grid) and 0 <= target_x < len(self.grid[0])):
            return False

        ## 2. check for blocks in the way (unless it's the target)
        if (target_y, target_x) in blocked_positions:
            return False
        
        ## 3. additional check for diagonal moves - look at the "corner" between current and target
        if dx != 0 and dy != 0:  ## if it is a diagonal move
            if (x + dx, y) in blocked_positions or (x, y + dy) in blocked_positions:
                return False
            if self.grid[x + dx][y] == 1 or self.grid[x][y + dy] == 1:
                return False

        def get_iteration_offsets(radius, first_iteration):
            ## if radius is smaller than 3 or if it's the first iteration
            ## a thorough radius check needs to be conducted
            if radius < 3 or first_iteration:
                ## return list of all (i,j) within square radius
                return [
                    (i, j)
                    for i in range(-radius + 1, radius)
                    for j in range(-radius + 1, radius)
                ]
            else:
                ## if it's not the first iteration and radius is larger than 2, offsets are defined
                offsets = []

                ## loop through the range of the radius to find the valid (i, j) offsets
                for i in range(-radius + 1, radius):
                    for j in range(-radius + 1, radius):

                        ## calculate the Chebyshev distance, which is the max of the absolute
                        ## x or y distance
                        dist = max(abs(i), abs(j))

                        ## only add offsets where the distance is radius - 1
                        ## this represents the outer layer of the radius square
                        if dist == radius - 1:
                            offsets.append((i, j))
                return offsets

        ## 4. add safety radius -> enables making the agv bigger
        if self.grid[target_y][target_x] == 0:
            ## check for obstacles in the vicinity
            for i, j in get_iteration_offsets(radius, first_iteration):
                check_y, check_x = target_y + i, target_x + j

                ## check if new location is outside if bounds
                if not (
                    0 <= check_y < len(self.grid) and 0 <= check_x < len(self.grid[0])
                ):
                    return False

                ## if within bounds check for obstacle presence
                else:
                    if self.grid[check_y][check_x] == 1:  ## obstacle detected
                        return False

            return True  ## no obstacles found in vicinity

    @staticmethod
    def reconstruct_path(visited, goal, start):
        """
        Reconstruct path from visited dictionary for classic Dijkstra.
        
        Args:
            visited (dict): Visited nodes with costs and predecessors
            goal (tuple): Goal position
            start (tuple): Start position
        
        Returns:
            list: Path from start to goal
        """
        path = []
        current = goal
        
        while current != start:
            path.append(current)
            ## get predecessor
            current = visited[current][1]
        
        path.append(start)
        path.reverse()
        return path


class AGV:
    """
    Plot grid and animate AGV movement. Plot heatmap of cells visited during Dijkstra search.
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
        self.object = None

    def animate_agv(
        self, grid, path, block_locations, start, warehouse_dimensions, object, radius
    ):
        """
        Animate the AGV following the computed path using matplotlib.

        Args:
            grid (np.ndarray): Navigation grid.
            path (list): Computed path from start to goal.
            block_locations (dict): Target block positions.
            start (tuple): Start coordinates.
            warehouse_dimensions (tuple): Physical map size.
            object (str): Description of the object.
            radius (int): AGV size in cells.
        """
        ## start timer before animation begins
        start_time = time.time()

        self.object = object

        ## plot grid, obstacles, LEGO blocks and AStar path
        ax, fig = self.plot_map(grid, start, block_locations, path, radius)
        ax.set_aspect("equal", adjustable="box")

        ## load AGV image
        agv_image = mpimg.imread("ML_navigation/LEGO_agv.png")

        ## resize the raw image
        agv_image_pil = (
            Image.fromarray((agv_image * 255).astype(np.uint8))
            if agv_image.max() <= 1
            else Image.fromarray(agv_image)
        )
        agv_image_resized = agv_image_pil.resize((70, 70))
        agv_image_resized = np.array(agv_image_resized)

        agv_img = ax.imshow(
            agv_image_resized, extent=(start[1], start[1] + 0.0001, start[0] + 0.0001, start[0]),
        )
        # plt.show()
        ## create smooth path using interpolation
        smoothed_path = self.smooth_path(path, steps=10)

        ## time in ms between each frame
        interval = 50

        ## start agv animation
        self.agv_animation(
            ax,
            fig,
            smoothed_path,
            agv_img,
            interval,
            start_time,
            warehouse_dimensions,
            radius,
        )

    def agv_animation(
        self,
        ax,
        fig,
        smoothed_path,
        agv_img,
        interval,
        start_time,
        warehouse_dimensions,
        radius,
    ):
        """
        Run real-time animation of AGV movement with dynamic velocity and ETA.

        Args:
            ax (Axes): Matplotlib axes.
            fig (Figure): Matplotlib figure.
            smoothed_path (list): Path interpolated for animation.
            agv_img (AxesImage): AGV image.
            interval (int): Frame interval in ms.
            start_time (float): Time when animation starts.
            warehouse_dimensions (tuple): Physical warehouse size.
            radius (int): AGV size.
        """

        ## add text annotations for velocity and ETA
        ## dynamic text positioning based on grid size
        grid_height, grid_width = self.navigation.map_grid.shape

        ## position text relative to grid size
        text_x = grid_width * 0.1
        text_y = -grid_height * 0.15
        x_shift = grid_width * 0.5
        
        ## create text annotations with bounding boxes for better visibility
        velocity_text = ax.text(text_x, text_y, "", fontsize=12, color="blue")
        eta_text = ax.text(text_x + x_shift, text_y, "", fontsize=12, color="red")
        last_eta_seconds = 0

        ## figure out step size to determine how many frames can be skipped to still make a smooth
        ## animation
        step = math.ceil(
            min(warehouse_dimensions[0], warehouse_dimensions[1])
            / (self.GRID_SIZE_CM * 60)
        )

        ## initialize tqdm progress bar
        progress_bar = tqdm(
            total=len(smoothed_path), desc="AGV Animation Progress", unit="frame"
        )

        ## animation flag
        global animation_done
        animation_done = False

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

            ## update AGV's position by setting its image extent
            agv_img.set_extent((x + 1 - radius, x + radius, y + radius, y + 1 - radius))

            ## calculate velocity (V = distance / time)
            if actual_frame > 0:
                prev_x, prev_y = smoothed_path[actual_frame - 1]
                distance_m = (
                    np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                    * self.GRID_SIZE_CM
                    / 100
                )
                time_elapsed = time.time() - start_time

                ## calculate instantaneous velocity (m/s)
                ## if no distance was covered, retain the previous velocity
                if distance_m == 0:
                    velocity_mps = self.last_velocity

                ## compute velocity using distance and time interval (converted from ms to seconds)
                elif time_elapsed > 0:
                    velocity_mps = distance_m / (interval / 1000)
                    self.last_velocity = velocity_mps

                ## default to zero velocity if no time has passed
                else:
                    velocity_mps = 0
                velocity_text.set_text(f"Vel: {velocity_mps:.2f} m/s")

                ## estimate remaining time to reach the goal
                remaining_frames = len(smoothed_path) - actual_frame

                if velocity_mps > 0:
                    ## calculate ETA in seconds based on remaining distance and current velocity
                    eta_seconds = (remaining_frames * distance_m) / (
                        velocity_mps * step
                    )

                    ## avoid displaying zero when the value is temporarily unstable
                    if eta_seconds == 0.0:
                        eta_seconds = last_eta_seconds
                else:
                    eta_seconds = 0

                last_eta_seconds = eta_seconds
                eta_text.set_text(f"ETA: {eta_seconds:.1f} sec")

            ## check if the AGV has reached the goal (last frame of the animation)
            if actual_frame >= len(smoothed_path) - 2 * step:

                ## ensure progress bar is 100% completed
                progress_bar.n = progress_bar.total
                progress_bar.refresh()
                progress_bar.close()

                ## ensuring text is only printed once, as this is called again, when animation
                ## is saved
                global animation_done
                if not animation_done:
                    print(f"Simultated AGV has reached the target {self.object}!")
                    print("Press Escape (ESC) to stop animation.")

                animation_done = True

                ## ensure text display correct end results
                velocity_text.set_text("Vel: 0 m/s")
                eta_text.set_text("ETA: 0 sec")

            ## return the updated AGV image, velocity text, and ETA text for rendering
            return (agv_img, velocity_text, eta_text)

        time_a1 = time.time()
        ## create the animation using FuncAnimation
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=int(len(smoothed_path) / step),
            interval=1,
            blit=False,
            ## blit=False, when showing animation in pop up window
            ## blit=True, speeds pop up window up, when showing animation in saved version
            init_func=lambda: (agv_img, velocity_text, eta_text),
        )

        ax.legend(loc="best", labelspacing=0.5)

        ## display animation until "ESC" is pressed
        def on_key(event):
            if event.key == "escape":
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()

        ## save the animation
        try:
            ani.save(
                "ML_navigation/animations/dijkstra_animation_astar_search.mp4",
                writer="ffmpeg",
                fps=30,  # Lower FPS reduces file size & encoding time
                dpi=100,  # Lower DPI speeds up rendering
                bitrate=1000,  # Lower bitrate = faster encoding
                extra_args=[
                    "-preset",
                    "ultrafast",  # Fastest encoding (lower quality)
                    "-crf",
                    "28",  # Higher CRF = faster but slightly worse quality
                    "-movflags",
                    "+faststart",  # Optimize for web streaming
                ],
            )
            print("Dijkstra animation saved as dijkstra_animation_astar_search.mp4")
            time_a2 = time.time()
            print(f"Time taken to save and show animation: {time_a2 - time_a1}s")
        except Exception as e:
            print(f"Could not save animation: {e}")

    def plot_map(self, grid, start, block_locations, path, radius):
        """
        Visualize the grid, blocks, start position, and path.

        Args:
            grid (np.ndarray): Grid data.
            start (tuple): Start coordinates.
            block_locations (dict): Coloured block locations.
            path (list): Navigation path.
            radius (int): AGV radius.

        Returns:
            tuple: (Axes, Figure), The plot's axes and figure object.
        """
        ## create subplot, maintaining full grid visibility
        fig, ax = plt.subplots(figsize=(7, 7))

        ## create new grid full of obstacles and then paste the actual grid inside to add obstacle
        ## boundary
        new_grid = np.ones((grid.shape[0] + 2, grid.shape[1] + 2), dtype=int)
        new_grid[1:-1, 1:-1] = grid

        ## set up grid lines
        # ax.set_xticks(np.arange(-0.5, new_grid.shape[1], 1), minor=True)
        # ax.set_yticks(np.arange(-0.5, new_grid.shape[0], 1), minor=True)
        # ax.grid(which="minor", color="gray", linestyle="-", linewidth=1)
        ax.set_xlim(-0.5, new_grid.shape[1] - 1.5)
        ax.set_ylim(new_grid.shape[0] - 1.5, -0.5)

        ## plot coloured blocks
        for colour, (y, x) in block_locations.items():
            ax.add_patch(
                plt.Circle(
                    (x + 0.5, y + 0.5),
                    0.7 * radius - 1,
                    color=colour,
                    label=f"{colour.capitalize()} {self.object}",
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
                (start[1] - radius + 1, start[0] - radius + 1),
                2 * radius - 1,
                2 * radius - 1,
                color="green",
                label="Start",
                alpha=0.6,
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
        Create interpolated path points for smooth animation. Needed for animation so that AGV also
        move inbetween path points and doesn't teleport from one to another.

        Args:
            path (list): Raw list of path coordinates.
            steps (int): Number of interpolation steps between points.

        Returns:
            list: Smoothed path coordinates.
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

    def plot_exploration_heatmap(self, exploration_counts):
        """
        Plot a heatmap of the exploration counts during Dijkstra search.

        Args:
            exploration_counts (numpy.ndarray): A 2D array containing the number of times each cell
            was explored.
        """
        plt.figure(figsize=(10, 10))

        ## create a copy of the exploration data
        heatmap_data = exploration_counts.copy()

        ## set obstacle cells (-1) to NaN so they are ignored in the heatmap
        heatmap_data = exploration_counts.astype(float)  ## convert to float
        heatmap_data[heatmap_data == -1] = np.nan  ## set obstacles to NaN

        ## plot heatmap (excluding obstacles)
        ax = sns.heatmap(
            heatmap_data,
            annot=False,        ## show values
            fmt=".0f",          ## integer format without decimals
            cmap="YlOrRd",      ## heatmap colour
            cbar=True,          ## show colour bar
        )

        ## overlay obstacles as black squares
        for i in range(exploration_counts.shape[0]):
            for j in range(exploration_counts.shape[1]):
                if exploration_counts[i, j] == -1:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, color="black"))

        ## add labels and title
        plt.title("Exploration Heatmap of Dijkstra Search")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")

        ## get the current figure object
        fig = ax.get_figure()

        ## show the heatmap until "ESC" is pressed
        def on_key(event):
            if event.key == "escape":
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()

    def get_movement_sequence(
        self, path, initial_start_position, selected_colour, agv_orientation
    ):
        """
        Converts a path of grid coordinates into a sequence of movements for the LEGO AGV to
        execute. Returns a list of commands: ["move_straight", "move_straight_diagonally",
        "left_turn", "left_turn_45", "right_turn", "right_turn_45", "stop"].
        Once movement sequence has been created, subprocess is started, which connects to the AGV
        and performs movement sequence generated previously.

        Args:
            path (list): Grid coordinates from start to goal.
            initial_start_position (tuple): Starting position of AGV.
            selected_colour (str): Target block colour.
            agv_orientation (str): Initial facing direction of the AGV.

        Raises:
            ValueError: On invalid cell transitions.
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
                next_direction = "right_down"
            elif dx == -1 and dy == 1:
                next_direction = "left_down"
            elif dx == 1 and dy == -1:
                next_direction = "right_up"
            elif dx == -1 and dy == -1:
                next_direction = "left_up"
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
                start_movement = self.get_required_movements_enhanced(
                    agv_orientation, next_direction
                )
                for movement in start_movement:
                    movement_sequence.append(movement)

                ## assuming LEGO AGV is always placed on physical map, facing 'upwards',
                ## the AGV need to be repositioned first, depending on directional movement
            else:

                ## directional instructions based on current and next direction
                ## move straight diagonally
                if (current_direction, next_direction) in [
                    ("right_up", "right_up"),
                    ("left_up", "left_up"),
                    ("left_down", "left_down"),
                    ("right_down", "right_down"),
                ]:
                    movement_sequence.append("move_straight_diagonally")

                ## move straight cardinally
                elif (current_direction, next_direction) in [
                    ("up", "up"),
                    ("left", "left"),
                    ("down", "down"),
                    ("right", "right"),
                ]:
                    movement_sequence.append("move_straight")

                ## right turn
                elif (current_direction, next_direction) in [
                    ("up", "right"),
                    ("right", "down"),
                    ("down", "left"),
                    ("left", "up"),
                ]:
                    movement_sequence.append("right_turn")
                    movement_sequence.append("move_straight")

                ## left turn
                elif (current_direction, next_direction) in [
                    ("up", "left"),
                    ("left", "down"),
                    ("down", "right"),
                    ("right", "up"),
                ]:
                    movement_sequence.append("left_turn")
                    movement_sequence.append("move_straight")

                ## 45° left turn into diagonal movement
                elif (current_direction, next_direction) in [
                    ("up", "left_up"),
                    ("left", "left_down"),
                    ("down", "right_down"),
                    ("right", "right_up"),
                ]:
                    movement_sequence.append("left_turn_45")
                    movement_sequence.append("move_straight_diagonally")

                ## 45° right turn into diagonal movement
                elif (current_direction, next_direction) in [
                    ("up", "right_up"),
                    ("left", "left_up"),
                    ("down", "left_down"),
                    ("right", "right_down"),
                ]:
                    movement_sequence.append("right_turn_45")
                    movement_sequence.append("move_straight_diagonally")

                ## 45° left turn into cardinal movement
                elif (current_direction, next_direction) in [
                    ("right_up", "up"),
                    ("left_up", "left"),
                    ("left_down", "down"),
                    ("right_down", "right"),
                ]:
                    movement_sequence.append("left_turn_45")
                    movement_sequence.append("move_straight")

                ## 45° right turn into cardinal movement
                elif (current_direction, next_direction) in [
                    ("right_up", "right"),
                    ("left_up", "up"),
                    ("left_down", "left"),
                    ("right_down", "down"),
                ]:
                    movement_sequence.append("right_turn_45")
                    movement_sequence.append("move_straight")

                ## left turn into diagonal movement
                elif (current_direction, next_direction) in [
                    ("right_up", "left_up"),
                    ("left_up", "left_down"),
                    ("left_down", "right_down"),
                    ("right_down", "right_up"),
                ]:
                    movement_sequence.append("left_turn")
                    movement_sequence.append("move_straight_diagonally")

                ## right turn into diagonal movement
                elif (current_direction, next_direction) in [
                    ("right_up", "right_down"),
                    ("left_up", "right_up"),
                    ("left_down", "left_up"),
                    ("right_down", "left_down"),
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

        ## run the script with the movement sequence as an argument
        subprocess.run(
            [
                "python",
                "ML_navigation/skeleton.py",
                "--movement",
                movement_sequence_str,
                "--start",
                str(initial_start_position),
                "--target_colour",
                selected_colour,
            ]
        )

    def get_required_movements_enhanced(self, current_orientation, next_direction):
        """
        Determine movement commands needed to face and move toward a new direction.
        next_direction can be one of ['up', 'right', 'down', 'left', 'right_up', 'left_up',
        'right_down', 'left_down']

        Args:
            current_orientation (str): Current AGV direction.
            next_direction (str): Desired direction to move.

        Returns:
            list: List of movement commands to align and move.

        Handles both cardinal and diagonal directions.

        """
        cardinal_map = {
            "up": 0,
            "right": 1,
            "down": 2,
            "left": 3,
            "right_up": 0.5,
            "left_up": -0.5,
            "right_down": 1.5,
            "left_down": 2.5,
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
        if any(d in next_direction for d in ["up", "down", "left", "right"]):
            if "_" in next_direction:
                movements.append("move_straight_diagonally")
            else:
                movements.append("move_straight")
        return movements


def main():
    """
    Main entry point: parses CLI arguments, runs Dijkstra search, animates AGV,
    and sends the movement sequence to the robot.

    Steps:
    - Load map and target.
    - Parse command-line flags.
    - Compute path with Dijkstra.
    - Animate and visualize path.
    - Generate and execute LEGO AGV movement.
    """
    ## initialise instances
    map_instance = Database()
    nav = Navigation()
    agv = AGV()
    initial_start_position = Database().start

    ## setup argparse for command line argument parsing
    parser = argparse.ArgumentParser(description="Navigate AGV to a target object.")
    parser.add_argument(
        "--diagonal",
        type=lambda x: (str(x).lower() == "true"),
        default="True",
        help="Wheather or not diagonal movement for Dijkstra search is allowed. Defaults to 'True'.",
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
    parser.add_argument(
        "--target_object",
        type=str,
        default="block",
        help="Target object AGV navigates to. Defaults to 'block'",
    )
    parser.add_argument(
        "--agv_size",
        default=0.000001,
        help="Size of AGV in map scenario. Defaults to 2% of the map size.",
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

    ## set agv size based on user input
    agv_size = math.ceil(
        min(nav.map_grid.shape[0], nav.map_grid.shape[1]) * args.agv_size
    )

    ## execute Dijkstra search to find optimal path to the goal
    ## "diagonal = True" allows for diagonal movements
    ## "diagonal = False" only moves cardinally
    time_s = time.time()
    try:
        path, exploration_counts = nav.dijkstra_search(
            map_instance.start, goal, int(agv_size), diagonal=args.diagonal
        )

    except ValueError as e:
        print(f"Navigation Error: {e}")
        return
    
    time_e = time.time()
    print(f"Time taken for Dijkstra to find optimal path: {time_e - time_s:.9f}s")

    ## plot and animate AGV path, get agv exploration heatmap and get movement sequence for LEGO
    ## AGV
    if path:
        agv.plot_exploration_heatmap(exploration_counts)
        agv.animate_agv(
            nav.map_grid,
            path,
            map_instance.block_locations,
            map_instance.start,
            map_instance.warehouse_dimensions,
            args.target_object,
            agv_size,
        )


        agv.get_movement_sequence(
            path, initial_start_position, selected_colour, args.agv_orientation
        )
    else:
        print("No path found!")


if __name__ == "__main__":
    main()
