#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-07-4
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing 3D rotation."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


class SO3:
    """This class represents an SO3 rotations internally represented by rotation
    matrix."""

    def __init__(self, rotation_matrix: ArrayLike | None = None) -> None:
        """Creates a rotation transformation from rot_vector."""
        super().__init__()
        self.rot: np.ndarray = (
            np.asarray(rotation_matrix) if rotation_matrix is not None else np.eye(3)
        )

    @staticmethod
    def exp(rot_vector: ArrayLike) -> SO3:
        """Compute SO3 transformation from a given rotation vector, i.e. exponential
        representation of the rotation."""
        v = np.asarray(rot_vector) # create np array from input
        assert v.shape == (3,)

        theta = np.linalg.norm(v) # compute the angle
        v_hat = v/theta # normalize the vector
        v_matrix = np.array([[0, -v_hat[2], v_hat[1]],
                             [v_hat[2], 0, -v_hat[0]],
                             [-v_hat[1], v_hat[0], 0]]) # skew symmetric matrix
        R = np.eye(3) + np.sin(theta)*v_matrix + (1-np.cos(theta))*(v_matrix @ v_matrix) # Rodrigues' formula

        t = SO3()
        t.rot = R
        return t

    def log(self) -> np.ndarray:
        """Compute rotation vector from this SO3"""
        v = np.zeros(3)

        if np.allclose(self.rot, np.eye(3)):
            return v # zero rotation vector for zero rotation

        tr_R = np.trace(self.rot)
        if(tr_R == -1): # 180 degree rotation
            # find the axis of rotation
            if(not np.isclose(self.rot[0,0], -1)):
                v = 1/np.sqrt(2*(1 + self.rot[0,0])) * np.array([1 + self.rot[0,0], self.rot[1,0], self.rot[2,0]])
            elif(not np.isclose(self.rot[1,1], -1)):
                v = 1/np.sqrt(2*(1 + self.rot[1,1])) * np.array([self.rot[0,1], 1 + self.rot[1,1], self.rot[2,1]])
            else:
                v = 1/np.sqrt(2*(1 + self.rot[2,2])) * np.array([self.rot[0,2], self.rot[1,2], 1 + self.rot[2,2]])
            v = np.pi * v # rotation vector is angle * axis
            return v

        theta = np.arccos((tr_R - 1)/2) # compute the angle
        if theta == 0:
            return v # zero rotation vector for zero rotation
        v_matrix = (self.rot - self.rot.T)/(2*np.sin(theta)) # skew symmetric matrix
        v = theta * np.array([v_matrix[2,1], v_matrix[0,2], v_matrix[1,0]]) # rotation vector

        return v

    def __mul__(self, other: SO3) -> SO3:
        """Compose two rotations, i.e., self * other"""        
        composed = self.rot @ other.rot
        return SO3(composed)

    def inverse(self) -> SO3:
        """Return inverse of the transformation."""
        transposed = self.rot.T
        return SO3(transposed)

    def act(self, vector: ArrayLike) -> np.ndarray:
        """Rotate given vector by this transformation."""
        v = np.asarray(vector)
        assert v.shape == (3,)
        return self.rot @ v

    def __eq__(self, other: SO3) -> bool:
        """Returns true if two transformations are almost equal."""
        return np.allclose(self.rot, other.rot)

    @staticmethod
    def rx(angle: float) -> SO3:
        """Return rotation matrix around x axis."""        
        R = np.array([[1, 0, 0],
                      [0, np.cos(angle), -np.sin(angle)],
                      [0, np.sin(angle), np.cos(angle)]])
        return SO3(R)        

    @staticmethod
    def ry(angle: float) -> SO3:
        """Return rotation matrix around y axis."""       
        R = np.array([[np.cos(angle), 0, np.sin(angle)],
                      [0, 1, 0],
                      [-np.sin(angle), 0, np.cos(angle)]])
        return SO3(R)

    @staticmethod
    def rz(angle: float) -> SO3:
        """Return rotation matrix around z axis."""
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]])
        return SO3(R)
   
    @staticmethod
    def from_quaternion(q: ArrayLike) -> SO3:
        """Compute rotation from quaternion in a form [qx, qy, qz, qw]."""
        # todo: HW1opt: implement from quaternion

        raise NotImplementedError("From quaternion needs to be implemented")

    def to_quaternion(self) -> np.ndarray:
        """Compute quaternion from self."""
        # todo: HW1opt: implement to quaternion
        raise NotImplementedError("To quaternion needs to be implemented")

    @staticmethod
    def from_angle_axis(angle: float, axis: ArrayLike) -> SO3:
        """Compute rotation from angle axis representation."""
        # todo: HW1opt: implement from angle axis
        raise NotImplementedError("Needs to be implemented")

    def to_angle_axis(self) -> tuple[float, np.ndarray]:
        """Compute angle axis representation from self."""
        # todo: HW1opt: implement to angle axis
        raise NotImplementedError("Needs to be implemented")

    @staticmethod
    def from_euler_angles(angles: ArrayLike, seq: list[str]) -> SO3:
        """Compute rotation from euler angles defined by a given sequence.
        angles: is a three-dimensional array of angles
        seq: is a list of axis around which angles rotate, e.g. 'xyz', 'xzx', etc.
        """
        # todo: HW1opt: implement from euler angles
        raise NotImplementedError("Needs to be implemented")

    def __hash__(self):
        return id(self)
