"""
Upgrade Map grid and restart AGV Navigation

This script is called when an obstacle has been detected during the LEGO AGV navigation.
It is used to update the detected obstalce in the map, calculate the new start postion
and restart the pathfinding process using A*.

Command Line Arguments:
    --movement (str): Movement sequence (default is 'move_straight right_turn move_straight').
    --interrupt_index (int): The movement step where interruption occurs (default is 0).
    --start (str): Starting position of the AGV (default is '(0, 0)').
    --target_colour (str): Colour of the target AGV is navigating to.

Usage:
    python movement_obstacle_detection.py --movement 'move_straight right_turn move_straight' 
    --interrupt_index 2 --start '(0, 0)' --target_colour 'red'

Author: Tim Riekeles
Date: 2025-06-05    
"""

import argparse
import numpy as np
import ast
import pandas as pd
import subprocess

parser = argparse.ArgumentParser(
    description="Generate a Pybricks script based on a movement sequence."
)
parser.add_argument(
    "--movement",
    type=str,
    default="move_straight right_turn move_straight",
    help="Movement sequence as a space-separated string. Defaults to 'move_straight right_turn move_straight'.",
)
parser.add_argument(
    "--interrupt_index",
    type=int,
    default=0,
    help="Index to which the movement sequence was executed. Defaults to 0.",
)
parser.add_argument(
    "--start",
    type=str,
    default="(0, 0)",
    help="Start for movement sequence. Defaults to '(0, 0)'.",
)
parser.add_argument("--target_colour", type=str, help="Target colour AGV navigates to.")
parser.add_argument(
    "--agv_orientation",
    type=str,
    default="up",
    help="AGV orientation. Defaults to 'up'.",
)

## parse the arguments
args = parser.parse_args()


def obstacle_location(grid, start, movements, interrupt_index, initial_orientation):
    ## 8 possible directions (Up, Up-Right, Right, Down-Right, Down, Down-Left, Left, Up-Left)
    directions = [
        (-1, 0),  ## Up
        (-1, 1),  ## Up-Right
        (0, 1),  ## Right
        (1, 1),  ## Down-Right
        (1, 0),  ## Down
        (1, -1),  ## Down-Left
        (0, -1),  ## Left
        (-1, -1),  ## Up-Left
    ]

    ## map turns to index changes
    turn_map = {
        "right_turn": 2,  ## 90 degrees right
        "left_turn": -2,  ## 90 degrees left
        "right_turn_45": 1,  ## 45 degrees right
        "left_turn_45": -1,  ## 45 degrees left
    }

    direction_index = initial_orientation
    row, col = start

    for i, move in enumerate(movements):
        if i == interrupt_index:
            break

        ## handle all types of turns
        if move in turn_map:
            direction_index = (direction_index + turn_map[move]) % 8

        ## move straight in the current direction
        elif move == "move_straight" or move == "move_straight_diagonally":
            dr, dc = directions[direction_index]
            new_row, new_col = row + dr, col + dc

            ## ensure the new position is within bounds and not an obstacle
            if (
                0 <= new_row < len(grid)
                and 0 <= new_col < len(grid[0])
                and grid[new_row][new_col] == 0
            ):
                row, col = new_row, new_col

        ## stop the movement if the AGV explicitly stops
        elif move == "stop":
            break

    return (row, col)


def update_database_and_grid(map_grid):
    ## read CSV into dataframe
    file_path = "database/location.csv"
    df = pd.read_csv(file_path)

    ## load database grid size
    database_grid_size_row = df[df["colour"] == "original_height_width"]
    if not database_grid_size_row.empty:
        database_grid_size = (
            database_grid_size_row.iloc[0]["location_y"],
            database_grid_size_row.iloc[0]["location_x"],
        )
    ## database grid size defaults to 100 x 100 if not found in CSV
    else:
        database_grid_size = (100, 100)

    ## determine scale factors
    row_scale = map_grid.shape[0] / database_grid_size[0]
    col_scale = map_grid.shape[1] / database_grid_size[1]

    ## scale up/down new AGV starting position
    scaled_new_start = (
        max(0, (agv_location[0]) // row_scale),
        max(0, (agv_location[1]) // col_scale),
    )

    ## update the start position
    df.loc[df["colour"] == "start", ["location_x", "location_y"]] = [
        scaled_new_start[1],
        scaled_new_start[0],
    ]

    ## save the updated DataFrame back to the CSV file
    df.to_csv(file_path, index=False)

    print(
        f"Obstacle detected at {obstacle}. Updated AGV start position to {agv_location}"
    )


def determine_orientation(movements, interrupt_index, initial_orientation):
    """
    Determines the AGV's orientation after interrupted movement sequence.
    Initial orientation: 0° (facing up/north)
    Right turn: +90°
    Left turn: -90°

    Returns: String representing diagonal direction ('up', 'right', 'down', 'left')
    """
    ## 8 directions indexed 0-7 starting from 'up' and going clockwise
    direction_index = initial_orientation

    ## map of movement commands to direction index changes
    turn_map = {
        "right_turn": 2,  # 90 degrees right
        "left_turn": -2,  # 90 degrees left
        "right_turn_45": 1,  # 45 degrees right
        "left_turn_45": -1,  # 45 degrees left
    }

    ## only process movements up to the interruption point
    for i in range(interrupt_index):
        move = movements[i]
        if move in turn_map:
            direction_index = (direction_index + turn_map[move]) % 8

    ## map to diagonal directions
    direction_map = {
        0: "up",
        1: "right_up",
        2: "right",
        3: "right_down",
        4: "down",
        5: "left_down",
        6: "left",
        7: "left_up",
    }

    return direction_map[direction_index]


## convert data in correct format
start = ast.literal_eval(args.start)
movements = args.movement.split()
interrupt_index = args.interrupt_index

## load map
map_grid = np.load("ML_navigation/map_creation/map_grid.npy")

diagonal_map = {
    "up": 0,
    "right_up": 1,
    "right": 2,
    "right_down": 3,
    "down": 4,
    "left_down": 5,
    "left": 6,
    "left_up": 7,
}

# Make sure to handle default/fallback if the orientation string is invalid
initial_orientation = diagonal_map.get(args.agv_orientation)

## find new orientation of agv
orientation = determine_orientation(movements, interrupt_index, initial_orientation)

## find where obstacle is
obstacle = obstacle_location(
    map_grid, start, movements, interrupt_index, initial_orientation
)

## update map with obstacle and save the map
map_grid[obstacle[0]][obstacle[1]] = 1
np.save("ML_navigation/map_creation/map_grid.npy", map_grid)

## ensure start postion is not the same block as the detected obstacle
if movements[interrupt_index - 1] in (
    "left_turn",
    "right_turn",
    "left_turn_45",
    "right_turn_45",
):
    interrupt_index -= 2
else:
    interrupt_index -= 1

## find where AGV was when obstacle was detected
agv_location = obstacle_location(
    map_grid, start, movements, interrupt_index, initial_orientation
)

## update start position
update_database_and_grid(map_grid)

subprocess.run(
    [
        "python",
        "ML_navigation/astar..py",
        "--agv_orientation",
        orientation,
        "--target_colour",
        args.target_colour,
    ]
)
