from pybricks.pupdevices import Motor, ColorSensor, UltrasonicSensor
from pybricks.parameters import Port
from pybricks.tools import wait

## initialize the motor connected to port A and B
motor_a = Motor(Port.A)
motor_b = Motor(Port.B)

## initialize the distance sensor connected to port C
distance_sensor = UltrasonicSensor(Port.C)

## initialize the color sensor connected to port D
color_sensor = ColorSensor(Port.D)

## check if LEGO AGV is close to obstacles
## if obstacle detected within 10cm of sensor turn to the left
## otherwise drive straight forward
while True:

    ## current distance
    distance = distance_sensor.distance()
    
    ## drive forward
    motor_a.run(-500)
    motor_b.run(500)

    ## distance check to obstacles
    ## briefly drive back to avoid obstacle and then turn to the left
    if distance < 100:
        ## drive back
        motor_a.run(100)
        motor_b.run(-100)
        wait(200)

        ## turn to the left
        motor_a.run(160)
        motor_b.run(160)
        wait(400)

        print("LEGO AGV has detected an obstacle within its vicinity. Initiating left turn")