from ctu_bosch_sr450 import RobotBosch

robot = RobotBosch(tty_dev=None)

q = robot.ik(0.3, 0.34, 0.3, 0)
print(q)