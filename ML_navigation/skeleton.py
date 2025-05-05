import argparse
import subprocess
import re
# Define the skeleton script
def generate_pybricks_script(movement_sequence):
    """
    Generates a Pybricks script based on the movement sequence.
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
            ## slowly approach obstacle to enable better colour identification
            approach = True
            # while approach:
            #     # print(distance.distance())
            #     if distance.distance() > 45:
            #         motor_a.run(-200)
            #         motor_b.run(200)
            #         wait(10)
            #         reps += 1
            #         print("closer", reps)
            #     else:
            #         motor_a.stop()
            #         motor_b.stop()
            #         approach = False
            motor_a.stop()
            motor_b.stop()
            colour_motor.run_target(200, -100)
            wait(500)
            print(f"Obstacle detected is {colour.color()}")
            print(reps)
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
    global obstacle_detected
    if obstacle_detected == False:
        motor_a.run(-300)
        motor_b.run(300)
        # distance_check(75)
        distance_check(150)
        #print("move straight")

def move_straight_diagonally():
    global obstacle_detected
    if obstacle_detected == False:
        motor_a.run(-300)
        motor_b.run(300)
        # distance_check(106)
        distance_check(212)
        #print("move straight diagonally")

def left_turn():
    global first_turn
    global obstacle_detected
    if obstacle_detected == False:
        motor_a.run(250)
        motor_b.run(250)
        if first_turn == True:
            # distance_check(64)
            wait(640)
            first_turn = False
        else:
            # distance_check(73)
            wait(730)
        #print("left turn")

def right_turn():
    global first_turn
    global obstacle_detected
    if obstacle_detected == False:
        motor_a.run(-250)
        motor_b.run(-250)
        if first_turn == True:
            # distance_check(64)
            wait(640)
            first_turn = False
        else:
            # distance_check(73)
            wait(730)
        #print("right turn")

def left_turn_45():
    global obstacle_detected
    if obstacle_detected == False:
        motor_a.run(250)
        motor_b.run(250)
        # distance_check(40)
        wait(400)
        #print("45 degree left turn")

def right_turn_45():
    global obstacle_detected
    if obstacle_detected == False:
        motor_a.run(-250)
        motor_b.run(-250)
        # distance_check(40)
        wait(400)
        #print("45 degree right turn")

def stop():
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