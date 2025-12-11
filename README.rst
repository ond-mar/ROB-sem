================
ROBOTICS course – semester project
================
Navigating SCARA robot through a 3D maze.

How to Run
-----------
Make the robot perform the assigned task by running the **main.py** file.

Other scripts
-----------
**camera_robot_pnp.py**: calculate camera2robot matrix using PnP algorithm

**camera.py**: helper class Camera for capturing hoop_images

**hoop_images2.py**: generating PnP callibration data
**images.py**: helper script for simple image capture
**k_matrix.py**: calculate K and distortions based on ChAruCo board images
**maze_pose_solver.py**: class MazePoseSolver finds labyrinth base location and rotation in robot coordinates using camera image and callibration data
**maze_visualizer.py**: script to test and visualize maze points transformations
**mazes.py**: file listing maze types and their coordinates
**planner.py**: class Planner finds robot configurations
**pnp_eval.py, pnp_test.py**: PnP testing scripts
**robot_park.py**: simple script to home robot
**utils.py**: helper constants and methods
