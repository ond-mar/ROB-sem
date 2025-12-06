from ctu_bosch_sr450 import RobotBosch

robot = RobotBosch()
robot.initialize(True)
robot.soft_home()
robot.close()