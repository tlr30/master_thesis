import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import heapq
import random
import time
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.colors as mcolors
from collections import deque


# ========== A* Algorithm ==========
def astar(grid, start, end):
    rows, cols = grid.shape
    open_set = deque([start])
    came_from = {}
    g_score = {start: 0}
    
    def heursitic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    while open_set:
        current = min(open_set, key=lambda x: g_score[x] + heursitic(x, end))
        open_set.remove(current)
        
        if current == end:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()

            ## validate path doesn't go through a wall
            for x, y in path:
                if grid[x, y] == 1:  # Path goes through wall
                    return None
            return path

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            neighbor = (current[0]+dx, current[1]+dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor] == 0:
                    tentative_g = g_score[current] + 1
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        if neighbor not in open_set:
                            open_set.append(neighbor)
    return None  # No path found


# Maze Generation (with random start and goal)
def generate_structured_maze(width, height):
    maze = np.ones((height, width), dtype=int)  # All walls to start

    def in_bounds(x, y):
        return 0 <= x < width and 0 <= y < height

    def add_loops(maze, loop_chance):
        height, width = maze.shape
        # print('before',maze)
        height_list = list(range(0, height - 1))
        width_list = list(range(0, width - 1))
        random.shuffle(height_list)
        random.shuffle(width_list)

        for y in height_list:
            for x in width_list:
                if random.random() < loop_chance:
                    if maze[y][x] == 1:
                        # Look for walls between two open spaces
                        if maze[y][x-1] == 0 and maze[y][x+1] == 0 and maze[y-1][x] == 1 and maze[y+1][x] == 1:
                            maze[y][x] = 0
                            # print('after',maze)
                        elif maze[y-1][x] == 0 and maze[y+1][x] == 0 and maze[y][x-1] == 1 and maze[y][x+1] == 1:
                            maze[y][x] = 0
                            # print('after',maze)
                        
        return maze

    def carve(x, y):
        maze[y][x] = 0
        # print(maze)
        directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if in_bounds(nx, ny) and maze[ny][nx] == 1:
                wall_x, wall_y = x + dx // 2, y + dy // 2
                maze[wall_y][wall_x] = 0  # Knock down wall
                # print(maze)
                carve(nx, ny)

    # Ensure odd dimensions for proper carving
    if width % 2 == 0: width -= 1
    if height % 2 == 0: height -= 1

    # print("start")
    # carve(0, 0)
    carve(random.randint(0, height - 1), random.randint(0, width - 1))
    
    maze = add_loops(maze, loop_chance=0.4)

    def get_random_open_cell():
        while True:
            y, x = random.randint(0, height - 1), random.randint(0, width - 1)
            if maze[y][x] == 0:
                return (y, x)

    start = get_random_open_cell()
    goal = get_random_open_cell()

    return maze, start, goal



# ========== Training Data Generator ==========
def generate_structured_training_data(grid_size, num_samples):
    training_data = []
    current_sample = 0
    for _ in range(num_samples):
        # print(_)
        while True:
            print(_)
            # grid, start, goal = generate_structured_maze(grid_size[0], grid_size[1])
            grid, start, goal = generate_structured_maze(grid_size[0], grid_size[1])
            # print(grid)
            path = astar(grid, start, goal)
            ## ensuring a valid path is present, that is longer than 1.5x the grid width/length
            # if path is not None and len(path) > 3.0 * max(grid_size[0], grid_size[1]):
            if path is not None and len(path) > 2.0 * max(grid_size[0], grid_size[1]):
                training_data.append((grid, start, goal, path))
                current_sample += 1
                break
    print(training_data[:5])
    return training_data

# ========== Preprocess ==========
def preprocess_data(training_data):
    X, Y = [], []
    # print(training_data)
    for grid, start, goal, path in training_data:
        # print('new\n', grid, start, goal, path)
        # Create start and goal position maps
        start_map = np.zeros_like(grid, dtype=np.bool_)
        start_map[start] = 1
        goal_map = np.zeros_like(grid, dtype=np.bool_)
        goal_map[goal] = 1
        
        # Stack as 3 channels: [grid, start_map, goal_map]
        input_data = np.stack([grid, start_map, goal_map], axis=-1)
        X.append(input_data.astype(np.float32))
        
        # Create path map (same as before)
        path_map = np.zeros_like(grid, dtype=np.bool_)
        for x, y in path:
            path_map[x, y] = 1
        Y.append(path_map)
    
    X = np.array(X)
    Y = np.array(Y)
    Y_flat = Y.reshape(Y.shape[0], -1).astype(np.float32)
    return X, Y_flat



# ========== CNN Model ==========
def create_cnn_model(input_shape, output_units):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    # model.add(layers.MaxPooling2D((2, 2), padding='same'))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    # model.add(layers.MaxPooling2D((2, 2), padding='same'))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    # model.add(layers.MaxPooling2D((2, 2), padding='same'))


    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(output_units, activation='sigmoid'))


    return model

def generate_unstructured_maze(width, height):
    """Generate unstructured mazes much faster by optimizing all operations"""
    # Vectorized obstacle generation
    obstacle_chance = 0.3
    maze = (np.random.random((height, width))) < obstacle_chance
    maze = maze.astype(np.int8)
    
    # Find all open cells in one operation
    open_cells = np.argwhere(maze == 0)
    
    # Randomly select start and goal from open cells
    if len(open_cells) >= 2:
        start_idx, goal_idx = np.random.choice(len(open_cells), 2, replace=False)
        start = tuple(open_cells[start_idx])
        goal = tuple(open_cells[goal_idx])
    else:
        # Fallback if not enough open cells
        maze[:,:] = 0
        start = (0, 0)
        goal = (height-1, width-1)
    
    return maze, start, goal



def generate_unstructured_training_data(grid_size, num_samples):
    """Optimized training data generator"""
    training_data = []
    min_path_length = 2.0 * max(grid_size)
    
    for _ in range(num_samples):
        while True:
            print(_)
            # Generate maze and positions
            grid, start, goal = generate_unstructured_maze(*grid_size)
            

                
            path = astar(grid, start, goal)
            if path is not None and len(path) >= min_path_length:
                training_data.append((grid, start, goal, path))
                break
                
    return training_data
    
#     return model, grid_size
def train_model(grid_size, num_samples=50000):
    start_time = time.time()
    # training_data = generate_structured_training_data(num_samples=num_samples, grid_size=grid_size)
    training_data = generate_unstructured_training_data(num_samples=num_samples, grid_size=grid_size)
    print(training_data[:5])
    X, y = preprocess_data(training_data)
    
    # Split into training and testing sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save some test samples for visualization
    # global test_samples
    test_samples = {
        'X': X_test[:3],  # First 5 test samples
        'y': y_test[:3].reshape(-1, *grid_size)  # Reshaped to original grid
    }
    
    model = create_cnn_model(X.shape[1:], grid_size[0] * grid_size[1])
    # model = create_cnn_model(X.shape[1:], grid_size)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

    model.summary()

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-4)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train, 
                       epochs=30, 
                       batch_size=32, 
                       validation_split=0.2,
                       callbacks=[early_stop, reduce_lr])
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    end_time = time.time()
    print(f"Time taken for A Star to find optimal path: {end_time - start_time:.20f} s")
    # Visualize test samples
    visualize_test_predictions(model, test_samples, grid_size)

    # Save the trained model
    model.save(f'ML_navigation/other_techniques/maze_solver_model_{grid_size[0]}x{grid_size[1]}.keras')  # or use .h5 for older TF versions
    print("Model saved to maze_solver_model.keras")
    
    return model, grid_size

def visualize_test_predictions(model, test_samples, grid_size):
    X_test_samples = test_samples['X']
    y_test_samples = test_samples['y']
    
    # Get predictions
    preds = model.predict(X_test_samples)
    preds = preds.reshape(-1, *grid_size)
    
    # Plot each test sample
    for i in range(len(X_test_samples)):
        plt.figure(figsize=(12, 4))
        
        # Input grid
        plt.subplot(1, 3, 1)
        plt.imshow(X_test_samples[i][:, :, 0], cmap='binary')  # Maze layer
        plt.title(f"Test Sample {i+1}\nInput Maze")
        plt.colorbar()
        
        # True path
        plt.subplot(1, 3, 2)
        plt.imshow(y_test_samples[i], cmap='binary')
        plt.title("True Path")
        plt.colorbar()
        
        # Predicted path
        plt.subplot(1, 3, 3)
        plt.imshow(preds[i], cmap='coolwarm', vmin=0, vmax=1)
        plt.title("Predicted Path")
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()


# ========== Predict ==========
def predict_path(model, grid, start, goal, grid_size, threshold=0.5):
    # Create start and goal maps
    start_map = np.zeros_like(grid)
    start_map[start] = 1
    goal_map = np.zeros_like(grid)
    goal_map[goal] = 1
    
    # Stack as 3 channels
    grid_input = np.stack([grid, start_map, goal_map], axis=-1)
    grid_input = np.expand_dims(grid_input, axis=0)  # Add batch dimension
    
    pred = model.predict(grid_input)[0]
    return pred.reshape(grid_size)

# ========== Main ==========
def main():

    # grid = np.array(                ## start,0,0    red,0,100           1
    #     [
    #         [0, 0, 0, 0, 0],
    #         [1, 1, 0, 1, 0],
    #         [0, 0, 0, 1, 0],
    #         [0, 1, 0, 0, 0],
    #     ]
    # )
    # start = (0, 0)  # Start position (row, column)
    # goal = (0, 3)

    # grid = np.array(                ## start,14,14    red,85,85         3
    #     [
    #         [0, 1, 1, 1, 1, 1, 0],
    #         [0, 0, 0, 0, 1, 1, 0],
    #         [1, 1, 1, 0, 1, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0],
    #         [0, 1, 1, 1, 1, 1, 0],
    #         [0, 1, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 1, 1, 1, 0],
    #     ]
    # )
    # start = (0, 0)  # Start position (row, column)
    # goal = (5, 5)

    # grid = np.array([
    #     [0, 0, 0, 1, 0, 0, 0],
    #     [1, 1, 0, 1, 0, 1, 0],
    #     [0, 0, 0, 1, 0, 1, 0],
    #     [0, 1, 1, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 1, 0],
    #     [0, 1, 0, 1, 1, 1, 0],
    #     [0, 0, 0, 1, 0, 0, 0],
    # ])
    # start = (2, 2)  # Start position (row, column)
    # goal = (6, 6)   # Goal position (row, column)

    # grid = np.array([
    #     [0, 1, 0, 0, 0, 1, 0, 0, 0],
    #     [0, 1, 0, 1, 0, 1, 0, 1, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 1, 0],
    #     [1, 1, 0, 1, 1, 1, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 1, 0, 0, 0],
    #     [0, 1, 1, 1, 0, 1, 1, 1, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 1, 0],
    #     [1, 1, 0, 1, 1, 1, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    # ])
    # start = (2, 0)  # Start position (row, column)
    # goal = (8, 8)   # Goal position (row, column)

    # grid = np.array([
    #     [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
    #     [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0],
    #     [1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    #     [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1],
    #     [1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    #     [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    #     [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    #     [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
    #     [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
    #     [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1],
    #     [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    # ])

    # start = (1, 1)  # Start position (row, column)
    # goal = (10, 10)   # Goal position (row, column)
    


    ##---------------- unstructured-----------------
    # grid = np.array([
    #     [0, 0, 1, 1, 0],
    #     [1, 0, 0, 0, 0],
    #     [0, 1, 0, 0, 1],
    #     [0, 0, 1, 0, 1],
    #     [1, 0, 0, 0, 0],
    # ])
    # start = (1, 1)  # Start position (row, column)
    # goal = (3, 0)   # Goal position (row, column)


    # grid = np.array([
    #     [0, 0, 0, 1, 0, 0, 0],
    #     [1, 1, 0, 0, 0, 1, 0],
    #     [0, 0, 1, 1, 0, 0, 0],
    #     [0, 1, 1, 0, 1, 0, 0],
    #     [1, 0, 0, 0, 0, 1, 0],
    #     [0, 0, 1, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 1],
    # ])
    # start = (0, 0)  # Start position (row, column)
    # goal = (4, 3)   # Goal position (row, column)

    # grid = np.array([
    #     [0, 0, 0, 0, 0, 1, 0, 0, 0],
    #     [0, 1, 0, 1, 0, 1, 1, 1, 0],
    #     [0, 0, 1, 1, 0, 0, 1, 1, 0],
    #     [1, 0, 0, 1, 1, 0, 0, 0, 0],
    #     [0, 1, 0, 0, 1, 1, 0, 0, 1],
    #     [0, 1, 0, 0, 1, 0, 0, 1, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 0, 1],
    #     [1, 1, 0, 1, 0, 1, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 1],
    # ])
    # start = (2, 5)  # Start position (row, column)
    # goal = (6, 2)   # Goal position (row, column)

    grid = np.array([
        [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
    ])

    start = (1, 1)  # Start position (row, column)
    goal = (9, 9)   # Goal position (row, column)
    
    model, grid_size = train_model(grid_size=grid.shape)
    # model = train_model(grid_size=grid.shape)

    predicted_path = predict_path(model, grid, start, goal, grid.shape)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    maze = grid  # Maze layout (0 = free, 1 = wall)

    # Show maze as background
    ax.imshow(maze, cmap='binary', alpha=1.0)

    # Overlay predicted path confidence heatmap
    heatmap = ax.imshow(predicted_path, cmap='coolwarm', alpha=0.6, vmin=0, vmax=1)

    # Annotate each cell with predicted value
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            conf = predicted_path[x, y]
            ax.text(y, x, f"{conf:.2f}", ha='center', va='center',
                    color='white' if conf > 0.6 else 'black', fontsize=8)

    # Mark start and goal
    ax.plot(start[1], start[0], marker='o', color='lime', label='Start')
    ax.plot(goal[1], goal[0], marker='X', color='red', label='Goal')

    plt.title("Predicted Path Heatmap with Maze Overlay")
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label("Path Confidence")
    ax.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
