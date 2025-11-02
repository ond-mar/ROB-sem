#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-08-21
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing planar manipulator."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from shapely import MultiPolygon, LineString, MultiLineString

from robotics_toolbox.core import SE2, SE3
from robotics_toolbox.robots.robot_base import RobotBase


class PlanarManipulator(RobotBase):
    def __init__(
        self,
        link_parameters: ArrayLike | None = None,
        structure: list[str] | str | None = None,
        base_pose: SE2 | None = None,
        gripper_length: float = 0.2,
    ) -> None:
        """
        Creates a planar manipulator composed by rotational and prismatic joints.

        The manipulator kinematics is defined by following kinematics chain:
         T_flange = (T_base) T(q_0) T(q_1) ... T_n(q_n),
        where
         T_i describes the pose of the next link w.r.t. the previous link computed as:
         T_i = R(q_i) Tx(l_i) if joint is revolute,
         T_i = R(l_i) Tx(q_i) if joint is prismatic,
        with
         l_i is taken from @param link_parameters;
         type of joint is defined by the @param structure.

        Args:
            link_parameters: either the lengths of links attached to revolute joints
             in [m] or initial rotation of prismatic joint [rad].
            structure: sequence of joint types, either R or P, [R]*n by default
            base_pose: mounting of the robot, identity by default
            gripper_length: length of the gripper measured from the flange
        """
        super().__init__()
        self.link_parameters: np.ndarray = np.asarray(
            [0.5] * 3 if link_parameters is None else link_parameters
        )
        n = len(self.link_parameters)
        self.base_pose = SE2() if base_pose is None else base_pose
        self.structure = ["R"] * n if structure is None else structure
        assert len(self.structure) == len(self.link_parameters)
        self.gripper_length = gripper_length

        # Robot configuration:
        self.q = np.array([np.pi / 8] * n)
        self.gripper_opening = 0.2

        # Configuration space
        self.q_min = np.array([-np.pi] * n)
        self.q_max = np.array([np.pi] * n)

        # Additional obstacles for collision checking function
        self.obstacles: MultiPolygon | None = None

    @property
    def dof(self):
        """Return number of degrees of freedom."""
        return len(self.q)

    def sample_configuration(self):
        """Sample robot configuration inside the configuration space. Will change
        internal state."""
        return np.random.uniform(self.q_min, self.q_max)

    def set_configuration(self, configuration: np.ndarray | SE2 | SE3):
        """Set configuration of the robot, return self for chaining."""
        self.q = configuration
        return self

    def configuration(self) -> np.ndarray | SE2 | SE3:
        """Get the robot configuration."""
        return self.q

    def flange_pose(self) -> SE2:
        """Return the pose of the flange in the reference frame."""

        T = SE2()
        T.set_from(self.base_pose)

        angles = self.get_angles()
        lengths = self.get_lengths()

        for angle, length in zip(angles, lengths):            
            theta = angle
            t = np.array([length, 0])

            R = SE2(rotation=theta)
            T_x = SE2(translation=t)
            T_i = R * T_x
            T = T * T_i
        return T

    def fk_all_links(self) -> list[SE2]:
        """Compute FK for frames that are attached to the links of the robot.
        The first frame is base_frame, the next frames are described in the constructor.
        """
        # todo HW02: implement fk
        frames = []
        T = SE2()
        T.set_from(self.base_pose)
        frames.append(T)

        angles = self.get_angles()
        lengths = self.get_lengths()

        for angle, length in zip(angles, lengths):            
            theta = angle
            t = np.array([length, 0])
            
            R = SE2(rotation=theta)
            T_x = SE2(translation=t)
            T_i = R * T_x
            T = T * T_i
            frames.append(T)

        return frames

    def _gripper_lines(self, flange: SE2):
        """Return tuple of lines (start-end point) that are used to plot gripper
        attached to the flange frame."""
        gripper_opening = self.gripper_opening / 2.0
        return (
            (
                (flange * SE2([0, -gripper_opening])).translation,
                (flange * SE2([0, +gripper_opening])).translation,
            ),
            (
                (flange * SE2([0, -gripper_opening])).translation,
                (flange * SE2([self.gripper_length, -gripper_opening])).translation,
            ),
            (
                (flange * SE2([0, +gripper_opening])).translation,
                (flange * SE2([self.gripper_length, +gripper_opening])).translation,
            ),
        )

    def jacobian(self) -> np.ndarray:
        """Computes jacobian of the manipulator for the given structure and
        configuration."""
        jac = np.zeros((3, len(self.q)))

        lengths = self.get_lengths()
        angles = self.get_angles()

        n = len(self.q)

        # Compute theta* for every joint
        theta_stars = np.zeros(n)
        theta_star = self.base_pose.rotation.angle
        for j in range(0, n):
            theta_star += angles[j]
            theta_stars[j] = theta_star

        for i, link in enumerate(self.structure): # compute i-th column of Jacobian
            dx = 0
            dy = 0
            dtheta = 0

            if link == "R":
                dx = 0                              
                for j in range(i, n):                    
                    dx += -np.sin(theta_stars[j]) * lengths[j]

                dy = 0
                for j in range(i, n):
                    dy += np.cos(theta_stars[j]) * lengths[j]
                
                dtheta = 1

            elif link == "P":
                dx = np.cos(theta_stars[i])
                dy = np.sin(theta_stars[i])
                dtheta = 0
            else:
                # Unsupported structural element
                pass
            
            jac[0][i] = dx
            jac[1][i] = dy
            jac[2][i] = dtheta

        return jac

    def jacobian_finite_difference(self, delta=1e-5) -> np.ndarray:
        jac = np.zeros((3, len(self.q)))
        
        # original flange pose
        T_orig = self.flange_pose()
        pos_orig = T_orig.translation
        rot_orig = T_orig.rotation.angle
        # save original q vector
        q_orig = self.q.copy()

        for i, q_i in enumerate(q_orig):
            # pertrube q
            self.q[i] = q_i + delta
            # compute pertrubed flange pose
            T_p = self.flange_pose()
            pos_p = T_p.translation
            rot_p = T_p.rotation.angle
            # restore original q
            self.q = q_orig.copy()
            # compute Jacobian column
            d_pos = (pos_p - pos_orig)/delta            
            d_rot = ((rot_p - rot_orig + np.pi) % (2*np.pi) - np.pi) / delta

            jac[0][i] = d_pos[0]
            jac[1][i] = d_pos[1]
            jac[2][i] = d_rot

        return jac

    def ik_numerical(
        self,
        flange_pose_desired: SE2,
        max_iterations=1000,
        acceptable_err=1e-4,
    ) -> bool:
        """Compute IK numerically. Value self.q is used as an initial guess and updated
        to solution of IK. Returns True if converged, False otherwise."""
        # todo: HW05 implement numerical IK

        return False

    def ik_analytical(self, flange_pose_desired: SE2) -> list[np.ndarray]:
        """Compute IK analytically, return all solutions for joint limits being
        from -pi to pi for revolute joints -inf to inf for prismatic joints."""
        assert self.structure in (
            "RRR",
            "PRR",
        ), "Only RRR or PRR structure is supported"

        # todo: HW05 implement analytical IK for RRR manipulator
        # todo: HW05 optional implement analytical IK for PRR manipulator
        if self.structure == "RRR":
            pass
        return []

    def in_collision(self) -> bool:
        """Check if robot in its current pose is in collision."""
        frames = self.fk_all_links()
        points = [f.translation for f in frames]
        gripper_lines = self._gripper_lines(frames[-1])

        links = [LineString([a, b]) for a, b in zip(points[:-2], points[1:-1])]
        links += [MultiLineString((*gripper_lines, (points[-2], points[-1])))]
        for i in range(len(links)):
            for j in range(i + 2, len(links)):
                if links[i].intersects(links[j]):
                    return True
        return MultiLineString(
            (*gripper_lines, *zip(points[:-1], points[1:]))
        ).intersects(self.obstacles)
    
    def get_angles(self) -> np.ndarray:
        angles = np.zeros(len(self.structure))
        
        for i, link in enumerate(self.structure):
            if link == "R":
                angles[i] = self.q[i]
            elif link == "P":
                angles[i] = self.link_parameters[i]
            else:
                # Unsupported structural element
                pass

        return angles
    
    def get_lengths(self) -> np.ndarray:
        lenghts = np.zeros(len(self.structure))
        
        for i, link in enumerate(self.structure):
            if link == "R":
                lenghts[i] = self.link_parameters[i]
            elif link == "P":
                lenghts[i] = self.q[i]
            else:
                # Unsupported structural element
                pass

        return lenghts