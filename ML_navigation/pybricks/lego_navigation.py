
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
    right_turn,
    move_straight,
    move_straight,
    right_turn,
    move_straight,
    move_straight,
    right_turn,
    move_straight,
    move_straight,
    left_turn,
    move_straight,
    stop,
]
## execute movements until obstacle is detected
for idx, movement in enumerate(movements):
    ## stop executing further movements if obstacle is detected
    if obstacle_detected:
        break
    movement()
print(f'MOVEMENT_PROGRESS: Movement index: {idx}, Total: {len(movements) - 1}')
