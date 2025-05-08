"""
Generate custom Pybricks file

This script dynamically generates and executes a Pybricks-compatible Python script for controlling a LEGO AGV 
(Autonomous Guided Vehicle) based on a sequence of movement commands. It also monitors real-time output 
from the robot to detect goals or obstacles and updates the environment accordingly.

Main Features:
- Generates movement logic for Pybricks.
- Runs the generated script on the LEGO AGV over BLE.
- Logs movement progress and handles interruptions due to obstacles.
- Triggers an update of the environment state if navigation is interrupted.

Dependencies:
- pybricksdev
- argparse
- subprocess
- re

Author: Tim Riekeles
Date: 2025-08-05
"""
import argparse
import subprocess
import re
def generate_pybricks_script(movement_sequence):
    """
    Generates a Pybricks-compatible Python script for executing a given movement sequence on a LEGO AGV.

    Args:
        movement_sequence (list of str): Sequence of movement function names to execute.

    Returns:
        str: Full Pybricks script as a string.
    """
    skeleton = """
from pybricks.hubs import PrimeHub
from pybricks.pupdevices import Motor, ColorSensor, UltrasonicSensor
from pybricks.parameters import Port
from pybricks.tools import wait

## initialize the motors and sensors
motor_a = Motor(Port.A)
motor_b = Motor(Port.B)
colour_motor = Motor(Port.F)
distance = UltrasonicSensor(Port.C)
colour = ColorSensor(Port.D)

first_turn = True
obstacle_detected = False

def distance_check(repetitions):
    global obstacle_detected
    ## checks for obstacles every 10ms
    ## if obstacle detected set flag to stop everything and output colour
    for reps in range(repetitions):
        if distance.distance() < 45 and distance.distance() !=  -1:
            obstacle_detected = True

            ## stop motor
            motor_a.stop()
            motor_b.stop()

            ## point colour sensor to obstacle
            colour_motor.run_target(200, -100)
            wait(500)
            print(f"Obstacle detected is {colour.color()}")

            ## drive back to middle of previous square
            if reps > 10:
                for _ in range(reps):
                    motor_a.run(300)
                    motor_b.run(-300)
                    wait(10)
            motor_a.stop()
            motor_b.stop()
            break
        wait(10)


def move_straight():
    ## move straight one cell
    global obstacle_detected
    if obstacle_detected == False:
        motor_a.run(-300)
        motor_b.run(300)
        distance_check(150)

def move_straight_diagonally():
    ## move straight diagonally one cell
    global obstacle_detected
    if obstacle_detected == False:
        motor_a.run(-300)
        motor_b.run(300)
        distance_check(212)

def left_turn():
    ## turn left
    global first_turn
    global obstacle_detected
    if obstacle_detected == False:
        motor_a.run(250)
        motor_b.run(250)
        if first_turn == True:
            wait(640)
            first_turn = False
        else:
            wait(730)

def right_turn():
    ## turn right
    global first_turn
    global obstacle_detected
    if obstacle_detected == False:
        motor_a.run(-250)
        motor_b.run(-250)
        if first_turn == True:
            wait(640)
            first_turn = False
        else:
            wait(730)

def left_turn_45():
    ## turn left 45°
    global obstacle_detected
    if obstacle_detected == False:
        motor_a.run(250)
        motor_b.run(250)
        wait(400)

def right_turn_45():
    ## turn right 45°
    global obstacle_detected
    if obstacle_detected == False:
        motor_a.run(-250)
        motor_b.run(-250)
        wait(400)

def stop():
    ## stop AGV
    global obstacle_detected
    if obstacle_detected == False:
        motor_a.stop()
        motor_b.stop()
        print("Goal has been reached. Stopping AGV.")


## ensure colour sensor points down
colour_motor.run_target(200, 0)

## movement sequence
movements = [
"""
    ## add the movement sequence to the script
    script = skeleton
    for command in movement_sequence:
        script += f"    {command},\n"
    script += f"]\n"
    script += f"## execute movements until obstacle is detected\n"
    script += f"for idx, movement in enumerate(movements):\n"
    script += f"    ## stop executing further movements if obstacle is detected\n"
    script += f"    if obstacle_detected:\n"
    script += f"        break\n"
    script += f"    movement()\n"
    script += f"print(f'MOVEMENT_PROGRESS: Movement index: {{idx}}, Total: {{len(movements) - 1}}')\n"
    
    return script

parser = argparse.ArgumentParser(description="Generate a Pybricks script based on a movement sequence.")
parser.add_argument("--movement", type=str, default='move_straight right_turn move_straight', help="Movement sequence as a space-separated string. Defaults to 'move_straight right_turn move_straight'.")
parser.add_argument("--start", type=str, default='(0, 0)', help="Start for movement sequence. Defaults to '(0, 0)'.")
parser.add_argument("--target_colour", type=str, help="Target colour AGV navigates to.")

## parse the arguments
args = parser.parse_args()

## convert the movement argument into a list of commands
movement_sequence = args.movement.split()

## generate the Pybricks script
pybricks_script = generate_pybricks_script(movement_sequence)

# save the generated script to a file
with open("ML_navigation/pybricks/lego_navigation.py", "w") as f:
    f.write(pybricks_script)

print("Script generated successfully!")
print("LEGO AGV navigation starts.")
command = ["pybricksdev", "run", "ble", "--name", "FYP", "ML_navigation/pybricks/lego_navigation.py"]
# subprocess.run(command, check=True)
process = subprocess.Popen(
    command,
    stdout=subprocess.PIPE,
    text=True,
    bufsize = 1
)

## save lines printed from sensor_motor.py to log.txt file
with open("ML_navigation/pybricks/log.txt", "w") as file:

    for line in iter(process.stdout.readline, ''):

        ## show live output
        print(line.strip())

        ## save to file
        file.write(line)

        ## ensure real-time writing
        file.flush()

process.stdout.close()
process.wait()


with open("ML_navigation/pybricks/log.txt", "r") as log_file:
    lines = log_file.readlines()

    ## check if the goal message exists in the file
    goal_message = "Goal has been reached. Stopping AGV."
    goal_reached = any(goal_message in line for line in lines)

    ## obstacle detected
    obstacle_message = "Obstacle detected"
    obstacle_detected = any(goal_message in line for line in lines)
    
    # Output the result
    if not goal_reached:
        print("Goal has not been reached")
        for line in lines:
            if "Movement index:" in line:
                
                match = re.search(r"Movement index: (\d+)", line)
                if match:
                    # Return the extracted number as an integer
                    movement_index =  int(match.group(1))
                    break


        subprocess.run(["python", "ML_navigation/update_map_grid.py", "--movement", args.movement,
                        "--interrupt_index", str(movement_index),
                        "--start", args.start,
                        "--target_colour", args.target_colour])
    else:
        print("LEGO AGV arrived at destination.")