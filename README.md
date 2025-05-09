# Voice-Controlled LEGO AGV Navigation System
A voice-controlled navigation system for a LEGO Autonomous Guided Vehicle (AGV), using Whisper for voice recognition and Pybricks for movement execution.

## Features
- Real-time voice command recognition using OpenAI Whisper
- Autonomous obstacle detection and avoidance
- Dynamic path updates based on AGV progress
- Visual grid-based path plotting

## Installation
git clone https://github.com/tlr30/master_thesis.git
cd master_thesis
pip install -r requirements.txt

## Usage
- Voice command via Open WebUI: Install Open WebUI (https://github.com/open-webui/open-webui) for seamless user interface and create a Open WebUI tool and paste ML_navigation/web_ui_tool.py in. Run ML_navigation/script_runner.py locally and open a new chat on Open WebUI. Activate the created tool and use the inbuilt speech-to-text feature (Open AI's Whisper) to navigate the LEGO AGV. Once sent, the navigation command is recieved by the locally running flask server and starts the LEGO navigation workflow.
- Voice command via local development environment (VS code): Execute speech/speech_to_text.py to use Open AI's Whisper to convert speech into text. This is achieved by creating record windows during which the command is extracted. The extracted navigation command then starts the LEGO navigation workflow.
- Custom navigation maps: Users have the ability to hand-draw custom warehouse and experience how different warehouse layouts affect overall AGV routes and efficiency. For this ML_navigation/get_map.py is exected alongside with the to be converted map [e.g.: python .\ML_navigation\get_map.py --drawing warehouse].

## Workflow
1. Voice Command Processing
    speech/command_execution.py
    - normalises user commands to provide strict AGV commands
    - Starts ML_navigation/astar.py via a subprocess
2. Pathfinding and Map Initialization
    ML_navigation/astar.py
    - loads the critial locations from the database and initialises the navigation map
    - Finds the optimal path from a start to a goal position [stated in database/location.csv]
    - Produces an animation visualizing the pathfinding process, along with a heatmap showing explored grid cells
    - Converts the optimal path (e.g., [(0, 1), (1, 1)]) into a movement sequence (e.g., [move_straight, left_turn])
    - Passes this movement sequence to ML_navigation/skeleton.py via a subprocess.
3. Pybricks Script Generation
    ML_navigation/skeleton.py
    - Generates a custom Pybricks script (ML_navigation/pybricks/lego_navigation.py) based on the given movement sequence
    - Starts the LEGO AGV navigation by executing the generated script
4. LEGO-based AGV Navigation Execution
    ML_navigation/pybricks/lego_navigation.py
    - Executes the AGV movement on LEGO hardware
    - Logs execution details to ML_navigation/pybricks/log.txt
    - Halts navigation when the sequence is completed or interrupted by a previously unknown obstacle
5. Post-Execution Validation
    ML_navigation/skeleton.py
    - Checks log.txt to determine whether the destination was reached or if navigation was interrupted
    - If successful, the process terminates
    - If an unknown obstacle was detected, it identifies the interrupt location using log data
6. Map Update and Replanning
    ML_navigation/update_map_grid.py
    - Updates the original map with the new obstacle and current AGV position in database/location.csv
    - Restarts ML_navigation/astar.py to attempt pathfinding from the new position
    - Repeats Steps 2–5 until a valid path is navigated or no further path can be found