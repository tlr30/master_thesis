"""
Convolutional Neural Network (CNN) for Maze Pathfinding

This module implements a CNN-based approach to learn and predict optimal paths in mazes.
The system generates training data using A* algorithm on structured and unstructured mazes,
trains a CNN model, and then uses the trained model to predict paths in new mazes.

Key Features:
- Maze generation (both structured and unstructured)
- Training data generation using A* algorithm
- CNN model with multiple convolutional layers
- Path prediction visualization

Author: Tim Riekeles
Date: 2025-06-05
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import random
import time
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from collections import deque


def astar(grid, start, end):
    """
    A* pathfinding algorithm implementation.

    Args:
        grid: 2D numpy array representing the maze (0=free, 1=wall)
        start: Tuple (x,y) representing start position
        end: Tuple (x,y) representing goal position

    Returns:
        List of tuples representing the path from start to end, or None if no path exists
    """
    rows, cols = grid.shape
    open_set = deque([start])
    came_from = {}
    g_score = {start: 0}

    def heursitic(a, b):
        """Manhattan distance heuristic for A*"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    while open_set:
        ## get node with lowest f-score (g + h)
        current = min(open_set, key=lambda x: g_score[x] + heursitic(x, end))
        open_set.remove(current)

        if current == end:
            ## reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()

            ## validate path doesn't go through a wall
            for x, y in path:
                if grid[x, y] == 1:
                    return None
            return path

        ## explore neighbours
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:

                ## only move through free space
                if grid[neighbor] == 0:
                    tentative_g = g_score[current] + 1
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        if neighbor not in open_set:
                            open_set.append(neighbor)
    return None  ## no path found


# Maze Generation (with random start and goal)
def generate_structured_maze(width, height):
    """
    Generates a structured maze using recursive division algorithm.

    Args:
        width: Width of the maze
        height: Height of the maze

    Returns:
        tuple: (maze_grid, start_position, goal_position)
    """
    maze = np.ones((height, width), dtype=int)  # All walls to start

    def in_bounds(x, y):
        """Check if coordinates are within maze boundaries"""
        return 0 <= x < width and 0 <= y < height

    def add_loops(maze, loop_chance):
        """
        Adds loops to the maze by randomly removing some walls.

        Args:
            maze: The maze grid to modify
            loop_chance: Probability of adding a loop at a valid position

        Returns:
            Modified maze grid
        """
        height, width = maze.shape

        ## list of possible indicies
        height_list = list(range(0, height - 1))
        width_list = list(range(0, width - 1))

        ## randomise order for walls to be processed
        random.shuffle(height_list)
        random.shuffle(width_list)

        ## process cells in random order
        for y in height_list:
            for x in width_list:
                ## random chance to potentially add a loop/passage
                if random.random() < loop_chance:
                    ## only modify walls (1s), not existing passages (0s)
                    if maze[y][x] == 1:
                        ## check for vertical wall between horizontal passages
                        ## (open spaces left/right, walls above/below)
                        if (
                            maze[y][x - 1] == 0
                            and maze[y][x + 1] == 0
                            and maze[y - 1][x] == 1
                            and maze[y + 1][x] == 1
                        ):
                            maze[y][x] = 0

                        ## check for horizontal wall between vertical passages
                        ## (open spaces above/below, walls left/right)
                        elif (
                            maze[y - 1][x] == 0
                            and maze[y + 1][x] == 0
                            and maze[y][x - 1] == 1
                            and maze[y][x + 1] == 1
                        ):
                            maze[y][x] = 0

        return maze

    def carve(x, y):
        """Recursive function to carve out maze passages"""
        ## mark current cell as traversable
        maze[y][x] = 0

        ## possible directions to move (in steps of 2 cells)
        directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]

        ## process in random order
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            ## check if neighbour is in bounds and still a wall
            if in_bounds(nx, ny) and maze[ny][nx] == 1:
                ## calculate wall position between current and neighbour
                wall_x, wall_y = x + dx // 2, y + dy // 2

                ## remove wall
                maze[wall_y][wall_x] = 0

                ## recursively remove walls from current position
                carve(nx, ny)

    ## ensure odd dimensions for proper carving (so we always have walls between cells)
    if width % 2 == 0:
        width -= 1
    if height % 2 == 0:
        height -= 1

    ## start carving from a random position (better than always starting at 0,0)
    carve(random.randint(0, height - 1), random.randint(0, width - 1))

    ## add some loops/alternate paths to make maze more interesting
    maze = add_loops(maze, loop_chance=0.4)

    def get_random_open_cell():
        """Find a random cell that isn't a wall (value = 0)"""
        while True:
            y, x = random.randint(0, height - 1), random.randint(0, width - 1)
            if maze[y][x] == 0:
                return (y, x)

    ## select random start and goal positions in open areas
    start = get_random_open_cell()
    goal = get_random_open_cell()

    return maze, start, goal


# ========== Training Data Generator ==========
def generate_structured_training_data(grid_size, num_samples):
    """
    Generates training data using structured mazes and A* paths.

    Args:
        grid_size: Tuple (width, height) of maze dimensions
        num_samples: Number of training samples to generate

    Returns:
        List of tuples (grid, start, goal, path)
    """
    training_data = []
    current_sample = 0
    for _ in range(num_samples):
        while True:
            ## generate maze and positions
            grid, start, goal = generate_structured_maze(grid_size[0], grid_size[1])

            path = astar(grid, start, goal)

            ## ensuring a valid path is present, that is longer than 1.5x the grid width/length
            # if path is not None and len(path) > 3.0 * max(grid_size[0], grid_size[1]):
            if path is not None and len(path) > 2.0 * max(grid_size[0], grid_size[1]):
                training_data.append((grid, start, goal, path))
                current_sample += 1
                break
    return training_data


def generate_unstructured_maze(width, height):
    """
    Generates an unstructured maze by randomly placing walls.
    Faster but less structured than recursive division method.

    Args:
        width: Width of the maze
        height: Height of the maze

    Returns:
        tuple: (maze_grid, start_position, goal_position)
    """
    ## vectorized obstacle generation
    obstacle_chance = 0.3
    maze = (np.random.random((height, width))) < obstacle_chance
    maze = maze.astype(np.int8)

    ## find all open cells in one operation
    open_cells = np.argwhere(maze == 0)

    ## randomly select start and goal from open cells
    if len(open_cells) >= 2:
        start_idx, goal_idx = np.random.choice(len(open_cells), 2, replace=False)
        start = tuple(open_cells[start_idx])
        goal = tuple(open_cells[goal_idx])

    return maze, start, goal


def generate_unstructured_training_data(grid_size, num_samples):
    """
    Generates training data using unstructured mazes and A* paths.
    Faster than structured generation but produces less regular mazes.

    Args:
        grid_size: Tuple (width, height) of maze dimensions
        num_samples: Number of training samples to generate

    Returns:
        List of tuples (grid, start, goal, path)
    """
    training_data = []
    min_path_length = 2.0 * max(grid_size)

    for _ in range(num_samples):
        while True:
            ## generate maze and positions
            grid, start, goal = generate_unstructured_maze(*grid_size)

            ## only use maze for testing/training if there a valid path is found by A* and the
            ## path length is long enough
            path = astar(grid, start, goal)
            if path is not None and len(path) >= min_path_length:
                training_data.append((grid, start, goal, path))
                break

    return training_data


def preprocess_data(training_data):
    """
    Preprocesses training data into format suitable for CNN.

    Args:
        training_data: List of (grid, start, goal, path) tuples

    Returns:
        tuple: (X, Y) where X is input data and Y is target paths
    """
    X, Y = [], []

    for grid, start, goal, path in training_data:
        ## create start and goal position maps
        start_map = np.zeros_like(grid, dtype=np.bool_)
        start_map[start] = 1
        goal_map = np.zeros_like(grid, dtype=np.bool_)
        goal_map[goal] = 1

        ## stack as 3 channels: [grid, start_map, goal_map]
        input_data = np.stack([grid, start_map, goal_map], axis=-1)
        X.append(input_data.astype(np.float32))

        ## create path map (binary matrix indicating path cells)
        path_map = np.zeros_like(grid, dtype=np.bool_)
        for x, y in path:
            path_map[x, y] = 1
        Y.append(path_map)

    X = np.array(X)
    Y = np.array(Y)

    ## flatten for dense layer
    Y_flat = Y.reshape(Y.shape[0], -1).astype(np.float32)
    return X, Y_flat


def create_cnn_model(input_shape, output_units):
    """
    Creates and compiles a CNN model for path prediction.

    Args:
        input_shape: Shape of input data (height, width, channels)
        output_units: Number of output units (height * width for flattened maze)

    Returns:
        Compiled Keras model
    """
    model = models.Sequential()

    ## convolutional layers with increasing filters
    model.add(
        layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=input_shape, padding="same"
        )
    )
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same"))

    ## flatten and dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(output_units, activation="sigmoid"))

    return model


def train_model(grid_size, structured_mazes, num_samples=2000):
    """
    Main training function that generates data and trains the CNN model.

    Args:
        grid_size: Tuple (width, height) of maze dimensions
        num_samples: Number of training samples to generate

    Returns:
        tuple: (trained_model, grid_size)
    """
    start_time = time.time()

    ## generate training data
    training_func = (
        generate_structured_training_data
        if structured_mazes
        else generate_unstructured_training_data
    )
    training_data = training_func(num_samples=num_samples, grid_size=grid_size)

    X, y = preprocess_data(training_data)

    ## split into training and testing sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    ## save some test samples for visualization
    test_samples = {
        "X": X_test[:3],  ## first 3 test samples
        "y": y_test[:3].reshape(-1, *grid_size),  ## reshaped to original grid
    }

    ## create and compile model
    model = create_cnn_model(X.shape[1:], grid_size[0] * grid_size[1])
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"]
    )
    model.summary()

    ## callbacks for better training
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-4
    )
    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    ## train model
    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
    )

    ## evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    end_time = time.time()
    print(f"Time taken for A Star to find optimal path: {end_time - start_time:.20f} s")

    ## visualize test samples
    visualize_test_predictions(model, test_samples, grid_size)

    ## save the trained model
    model.save(
        f"ML_navigation/other_techniques/maze_solver_model_{grid_size[0]}x{grid_size[1]}.keras"
    )
    print("Model saved to maze_solver_model.keras")

    return model, grid_size


def visualize_test_predictions(model, test_samples, grid_size):
    """
    Visualizes model predictions on test samples.

    Args:
        model: Trained Keras model
        test_samples: Dictionary containing test samples and true paths
        grid_size: Dimensions of the maze grid
    """
    X_test_samples = test_samples["X"]
    y_test_samples = test_samples["y"]

    ## get predictions
    preds = model.predict(X_test_samples)
    preds = preds.reshape(-1, *grid_size)

    ## plot each test sample
    for i in range(len(X_test_samples)):
        plt.figure(figsize=(12, 4))

        ## input grid
        plt.subplot(1, 3, 1)
        plt.imshow(X_test_samples[i][:, :, 0], cmap="binary")  ## maze layer
        plt.title(f"Test Sample {i+1}\nInput Maze")
        plt.colorbar()

        ## true path
        plt.subplot(1, 3, 2)
        plt.imshow(y_test_samples[i], cmap="binary")
        plt.title("True Path")
        plt.colorbar()

        ## predicted path
        plt.subplot(1, 3, 3)
        plt.imshow(preds[i], cmap="coolwarm", vmin=0, vmax=1)
        plt.title("Predicted Path")
        plt.colorbar()

        plt.tight_layout()
        plt.show()


def predict_path(model, grid, start, goal, grid_size):
    """
    Uses trained model to predict path through a maze.

    Args:
        model: Trained Keras model
        grid: 2D numpy array representing the maze
        start: Tuple (x,y) of start position
        goal: Tuple (x,y) of goal position
        grid_size: Dimensions of the maze
        threshold: Confidence threshold for path prediction

    Returns:
        2D numpy array of predicted path probabilities
    """
    ## create start and goal maps
    start_map = np.zeros_like(grid)
    start_map[start] = 1
    goal_map = np.zeros_like(grid)
    goal_map[goal] = 1

    ## stack as 3 channels and add batch dimension
    grid_input = np.stack([grid, start_map, goal_map], axis=-1)
    grid_input = np.expand_dims(grid_input, axis=0)

    ## predict and reshape to original grid size
    pred = model.predict(grid_input)[0]
    return pred.reshape(grid_size)


def main():
    """Main execution function for training and testing the model"""

    ##----------------structured-----------------
    ## set structured_mazes=True
    grid = np.array(
        [
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [1, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0],
        ]
    )
    start = (0, 0)  # start position (row, column)
    goal = (5, 5)  ## goal position (row, column)

    ##----------------unstructured-----------------
    ## set structured_mazes=False
    grid = np.array(
        [
            [0, 0, 1, 1, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 0, 1],
            [1, 0, 0, 0, 0],
        ]
    )
    start = (1, 1)  ## start position (row, column)
    goal = (3, 0)  ## goal position (row, column)

    ## train model
    model, grid_size = train_model(grid_size=grid.shape, structured_mazes=False)

    ## predict and visualize path
    predicted_path = predict_path(model, grid, start, goal, grid.shape)

    ## create visualization
    fig, ax = plt.subplots(figsize=(6, 6))

    ## show maze as background
    maze = grid  # Maze layout (0 = free, 1 = wall)
    ax.imshow(maze, cmap="binary", alpha=1.0)

    ## overlay predicted path confidence heatmap
    heatmap = ax.imshow(predicted_path, cmap="coolwarm", alpha=0.6, vmin=0, vmax=1)

    ## annotate each cell with predicted value
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            conf = predicted_path[x, y]
            ax.text(
                y,
                x,
                f"{conf:.2f}",
                ha="center",
                va="center",
                color="white" if conf > 0.6 else "black",
                fontsize=8,
            )

    ## mark start and goal
    ax.plot(start[1], start[0], marker="o", color="lime", label="Start")
    ax.plot(goal[1], goal[0], marker="X", color="red", label="Goal")

    plt.title("Predicted Path Heatmap with Maze Overlay")
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label("Path Confidence")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
