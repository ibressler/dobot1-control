"""
open-dobot inverse kinematics.

Implements inverse and forward kinematics functions.

Find firmware, driver, and SDK at https://github.com/maxosprojects/open-dobot

Author: maxosprojects (March 18, 2016)
Additional Authors: <put your name here>

Version 1.2.2

License: MIT
"""

import math
from dobot.DobotBase import DobotBase

# Dimensions in mm
lengthRearArm = 135.0
lengthFrontArm = 160.0
# Joint1 height.
heightFromGround = 80.0 + 23.0

lengthRearSquared = pow(lengthRearArm, 2)
lengthFrontSquared = pow(lengthFrontArm, 2)


class DobotKinematics(DobotBase):
    def __init__(self, endEffectorOffset=(50.9, 15.)):
        """
        Manages the Dobot geometry configuration with an end effector offset and an optional debug mode.

        :param endEffectorOffset: Offset or distance (horizontal, vertical) of end effector tool from joint 3 in mm.
            Defaults to (50.9, 15.), the horizontal distance from Joint3 to the center of the tool mounting position
            on the standard end effector, see `docs/img/dobot-geometry.png`.
        :param debug: Indicates whether the debug mode is enabled. Pass True to enable
            debugging or False to disable it.
        """
        self._endEffectorOffset = endEffectorOffset

    def coordinatesFromAngles(self, baseAngle, rearArmAngle, frontArmAngle):
        radius = lengthRearArm * math.cos(rearArmAngle) + lengthFrontArm * math.cos(frontArmAngle) + self._endEffectorOffset[0]
        x = radius * math.cos(baseAngle)
        y = radius * math.sin(baseAngle)
        z = heightFromGround - lengthFrontArm * math.sin(frontArmAngle) + lengthRearArm * math.sin(rearArmAngle) - self._endEffectorOffset[1]
        return x, y, z

    def anglesFromCoordinates(self, xyz, debug=False):
        """
        https://www.learnaboutrobots.com/inverseKinematics.htm
        """
        if debug:
            self._debug("anglesFromCoordinates", xyz, level=0)
        # Radius to the center of the tool.
        radiusTool = math.sqrt(xyz[0]**2 + xyz[1]**2)
        if debug:
            self._debug("radiusTool", radiusTool)
        # Radius to joint3.
        radius = radiusTool - self._endEffectorOffset[0]
        if debug:
            self._debug("radius", radius)
        baseAngle = math.atan2(xyz[1], xyz[0])
        if debug:
            self._debug("ik base angle", baseAngle)
        # X coordinate of joint3.
        jointX = radius * math.cos(baseAngle)
        if debug:
            self._debug("jointX", jointX)
        # Y coordinate of joint3.
        jointY = radius * math.sin(baseAngle)
        if debug:
            self._debug("jointY", jointY)
        actualZ = xyz[2] - heightFromGround + self._endEffectorOffset[1]
        if debug:
            self._debug("actualZ", actualZ)
        # Imaginary segment connecting joint1 with joint2, squared.
        hypotenuseSquared = pow(actualZ, 2) + pow(radius, 2)
        hypotenuse = math.sqrt(hypotenuseSquared)
        if debug:
            self._debug("hypotenuse", hypotenuse)
            self._debug("hypotenuseSquared", hypotenuseSquared)

        q1 = math.atan2(actualZ, radius)
        if debug:
            self._debug("q1", q1)
        q2 = math.acos(
            (lengthRearSquared - lengthFrontSquared + hypotenuseSquared) / (2.0 * lengthRearArm * hypotenuse)
        )
        if debug:
            self._debug("q2", q2)
        rearAngle = .5 * math.pi - (q1 + q2)
        if debug:
            self._debug("ik rear angle", rearAngle)
        frontAngle = .5 * math.pi - (
            math.acos(
                (lengthRearSquared + lengthFrontSquared - hypotenuseSquared) / (2.0 * lengthRearArm * lengthFrontArm)
            )
            - rearAngle
        )
        if debug:
            self._debug("ik front angle", frontAngle)

        return baseAngle, rearAngle, frontAngle

    # angles passed as arguments here should be real world angles (horizontal = 0, below is negative, above is positive)
    # i.e., they should be set up the same way as the unit circle is
    def check_for_angle_limits_is_valid(self, baseAngle, rearArmAngle, foreArmAngle):
        ret = True
        # implementing limit switches and IMUs will make this function more accurate and allow the user to calibrate the limits
        # necessary for this function.
        # Not currently checking the base angle

        # check the rearArmAngle
        # max empirically determined to be around 107 - 108 degrees. Using 105.
        # min empirically determined to be around -23/24 degrees. Using -20.
        if -20 > rearArmAngle > 105:
            print("Rear arm angle out of range")
            ret = False

        # check the foreArmAngle
        # the valid forearm angle is dependent on the rear arm angle. The real-world angle of the forearm
        # (0 degrees = horizontal) needs to be evaluated.
        # min empirically determined to be around -105 degrees. Using -102.
        # max empirically determined to be around 21 degrees. Using 18.
        if -102 > foreArmAngle > 18:
            print("Fore arm angle out of range")
            ret = False

        return ret
