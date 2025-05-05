from pybricks.pupdevices import Motor, ColorSensor, UltrasonicSensor
from pybricks.parameters import Port
from pybricks.tools import wait
from pybricks.parameters import Color


## initialize the motor connected to port A, B and F
motor_a = Motor(Port.A)
motor_b = Motor(Port.B)
colour_motor = Motor(Port.F)

## initialize the distance sensor connected to port C
distance_sensor = UltrasonicSensor(Port.C)

## initialize the color sensor connected to port D
colour_sensor = ColorSensor(Port.D)
wait(2000)

def continue_movement():
    while True:

        ## go straight if white colour detected
        if colour_sensor.color() == (Color.WHITE or Color.NONE):
            motor_a.run(-200)
            motor_b.run(200)

        ## left turn if yellow colur detected
        elif colour_sensor.color() == Color.YELLOW:
            motor_a.run(100)
            motor_b.run(100)

        ## right turn if green colour detected
        elif colour_sensor.color() == Color.GREEN:
            motor_a.run(-100)
            motor_b.run(-100)
        
        ## stop motors
        else:
            motor_a.stop()
            motor_b.stop()
        wait(10)

        ## additional code to check if obstacle is present
        if distance_sensor.distance() < 45 and distance_sensor.distance() !=  -1:
            
            ## stop motors
            motor_a.stop()
            motor_b.stop()

            ## turn colour motor towards obstacle and print colour of detected obstacle
            colour_motor.run_target(200, -100)
            wait(500)
            print(f"Obstacle detected is {colour_sensor.color()}")

            break


## reposition colour sensor to face down
colour_motor.run_target(200, 0)

## move along the green and yellow tape, until obstacle is detected
continue_movement()

