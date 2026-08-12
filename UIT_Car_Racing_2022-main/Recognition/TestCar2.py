import UITCar
import time
Car = UITCar.UITCar()

Car.setMotorMode(0)

Car.setSpeed_rad(28)
Car.setAngle(4)

time.sleep(5)

Car.setSpeed_rad(0)
Car.setAngle(4)

# CurrentPos = Car.getPosition_rad()
# print("Start Pos ", CurrentPos)
# while(Car.getPosition_rad() - CurrentPos < 30):
#     # time.sleep(0.1)
#     print("Car Pos ", Car.getPosition_rad())

# start_time = time.time()
# while(time.time() - start_time < 200000):
#     pass

# Car.setSpeed_rad(0)
# Car.setAngle(0)