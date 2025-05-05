import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ========== Model Loading ==========
def load_model(model_path='ML_navigation/other_techniques/maze_solver_model_11x11.keras'):
    """Load a previously trained model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please train a model first.")
    return tf.keras.models.load_model(model_path)

# ========== Prediction ==========
def predict_path(model, grid, start, goal):
    """Predict path through maze using loaded model"""
    # Create start and goal position maps
    start_map = np.zeros_like(grid)
    start_map[start] = 1
    goal_map = np.zeros_like(grid)
    goal_map[goal] = 1
    
    # Prepare input (combine grid, start map, and goal map)
    grid_input = np.stack([grid, start_map, goal_map], axis=-1)
    grid_input = np.expand_dims(grid_input, axis=0)  # Add batch dimension
    
    # Make prediction
    pred = model.predict(grid_input, verbose=0)[0]
    return pred.reshape(grid.shape)

# ========== Visualization ==========
def visualize_results(grid, start, goal, predicted_path):
    """Visualize the maze and predictions"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Show maze background
    ax.imshow(grid, cmap='binary', alpha=0.7)
    
    # Show predicted path heatmap
    heatmap = ax.imshow(predicted_path, cmap='coolwarm', alpha=0.5, vmin=0, vmax=1)
    
    # Annotate each cell with prediction confidence
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            conf = predicted_path[x, y]
            ax.text(y, x, f"{conf:.2f}", ha='center', va='center',
                    color='white' if conf > 0.6 else 'black', fontsize=10)

    # Mark start and goal positions
    ax.plot(start[1], start[0], marker='o', markersize=15, color='lime', label='Start')
    ax.plot(goal[1], goal[0], marker='X', markersize=15, color='red', label='Goal')
    
    # Add colorbar and legend
    plt.colorbar(heatmap, ax=ax, label="Path Confidence")
    plt.title("Maze Path Prediction", pad=20)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ========== Main ==========
def main():
    # Load the trained model
    model = load_model()
    
    # Define the test maze
    # grid = np.array([
    #     [0, 0, 1, 1, 0],
    #     [1, 0, 0, 0, 0],
    #     [0, 1, 0, 0, 1],
    #     [0, 0, 1, 0, 1],
    #     [1, 0, 0, 0, 0],
    # ])
    # start = (1, 1)  # (row, column)
    # goal = (3, 0)   # (row, column)
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
    # Make prediction
    predicted_path = predict_path(model, grid, start, goal)
    
    # Visualize results
    visualize_results(grid, start, goal, predicted_path)

if __name__ == "__main__":
    main()