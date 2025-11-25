from ctu_bosch_sr450 import RobotBosch

robot = RobotBosch()
robot.initialize()

Z_INCREMENT = -0.03
Y_INCREMENT = -0.1
X_INCREMENT = -0.1

start_pose = robot.fk(robot.get_q())

def find_closest_q(c, all_q):
    current_dif = float("inf")
    current_q = None

    for q in all_q:
        abs_dif = abs(q[0] - c[0]) + abs(q[1] - c[1])
        if abs_dif < current_dif:
            current_q = q

    return current_q
    

for i in range(10):
    newq = robot.ik_xyz(start_pose[0] + i * X_INCREMENT, start_pose[1] + i * Y_INCREMENT, start_pose[2])
    
    if len(newq) > 0:
        selected = find_closest_q(robot.get_q(), newq)
        robot.move_to_q(selected)
        robot.wait_for_motion_stop()

    pose = robot.fk(robot.get_q())

    for z in range(10):
        new_z_q = robot.ik_xyz(pose[0], pose[1], pose[2] + z * Z_INCREMENT)
        if len(new_z_q) > 0:
            selected = find_closest_q(robot.get_q(), new_z_q)
            robot.move_to_q(selected)
            robot.wait_for_motion_stop()
