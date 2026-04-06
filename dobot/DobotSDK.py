"""
open-dobot SDK.

SDK provides high-level functions to control Dobot via the driver to open firmware, which, in turn, controls Dobot FPGA.
Abstracts specifics of commands sent to FPGA.
Find firmware and driver at https://github.com/maxosprojects/open-dobot

It is assumed that upon SDK initialization the arms are between 0 and 90 degrees - between their normal
horizontal and vertical positions.
Upon initialization, accelerometers are read to figure out current arms' configuration. Accelerometers get confused
when the rear arm leans backwards from the dobot base or when the front arm bends towards the base.
Also, Inverse Kinematics at the moment don't account for when the front arm is looking up (higher than its
normal horizontal position). So be gentle and give dobot some reasonable initial configuration in case it happens
to be beyond the mentioned limits.
Refer to docs/images/ to find more about reference frame, arm names, and more.

SDK keeps track of the current end effector pose, thus in case the arm slips or motors are disabled while
in move (with the "Laser Adjustment" button), it has to be re-initialized and SDK re-initialized.

Author: maxosprojects (March 18, 2016)
Additional Authors: <put your name here>

Version: 1.2.2

License: MIT
"""

import sys
import math
import numpy as np
from dobot.DobotDriver import DobotDriver
from dobot.DobotKinematics import DobotKinematics, piHalf, piTwo

# Workaround to support Python 2/3
if sys.version_info > (3,):
    long = int

# See calibrate-accelerometers.py for details
accelOffsets = (1024, 1024)

# Backlash in the motor reduction gears is actually 22 steps, but 5 is visually unnoticeable.
# It is a horrible thing to compensate a bad backlash in software, but the only other
# option is to physically rebuild Dobot to fix this problem.
backlash = 5

# The NEMA 17 stepper motors that Dobot uses are 200 steps per revolution.
stepperMotorStepsPerRevolution = 200.0
# FPGA board has all stepper drivers' stepping pins set to microstepping.
baseMicrosteppingMultiplier = 16.0
rearArmMicrosteppingMultiplier = 16.0
frontArmMicrosteppingMultiplier = 16.0
# The NEMA 17 stepper motors Dobot uses are connected to a planetary gearbox, the black cylinders
# with 10:1 reduction ratio
stepperPlanetaryGearBoxMultiplier = 10.0

# calculate the actual number of steps it takes for each stepper motor to rotate 360 degrees
baseActualStepsPerRevolution = (
    stepperMotorStepsPerRevolution * baseMicrosteppingMultiplier * stepperPlanetaryGearBoxMultiplier
)
rearArmActualStepsPerRevolution = (
    stepperMotorStepsPerRevolution * rearArmMicrosteppingMultiplier * stepperPlanetaryGearBoxMultiplier
)
frontArmActualStepsPerRevolution = (
    stepperMotorStepsPerRevolution * frontArmMicrosteppingMultiplier * stepperPlanetaryGearBoxMultiplier
)


class DobotPlotter:
    def __init__(self):
        import matplotlib.pyplot as plt

        self._plt = plt
        # Lists for detailed coordinate tracking in MoveWithSpeed
        self.reset_move_plots()

    def reset_move_plots(self):
        self._coords = {'x': [], 'y': [], 'z': []}
        self._next = {'x': [], 'y': [], 'z': []}
        self._diff = {'x': [], 'y': [], 'z': []}
        self._slice_diff = {'base': [], 'rear': [], 'front': []}
        self._slice_actual = {'base': [], 'rear': [], 'front': []}

    def add_slice_data(self, base_diff, actual_steps_base, rear_diff, actual_steps_rear, front_diff, actual_steps_front):
        self._slice_diff['base'].append(base_diff)
        self._slice_diff['rear'].append(rear_diff)
        self._slice_diff['front'].append(front_diff)
        self._slice_actual['base'].append(actual_steps_base)
        self._slice_actual['rear'].append(actual_steps_rear)
        self._slice_actual['front'].append(actual_steps_front)

    def add_move_data(self, cx, cy, cz, nx, ny, nz):
        self._coords['x'].append(cx)
        self._coords['y'].append(cy)
        self._coords['z'].append(cz)
        self._next['x'].append(nx)
        self._next['y'].append(ny)
        self._next['z'].append(nz)
        self._diff['x'].append(cx - nx)
        self._diff['y'].append(cy - ny)
        self._diff['z'].append(cz - nz)

    def show(self):
        linewidth = 1.0
        colors = dict(x='darkred', y='darkgreen', z='lightblue',
                      base='lightblue', rear='darkgreen', front='darkred')

        def get_kwargs(axis):
            return dict(color=colors[axis], linewidth=linewidth, label=axis)

        plt = self._plt
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 3, 1)
        plt.title("Current Coords")
        for axis in ['x', 'y', 'z']:
            plt.plot(self._coords[axis], **get_kwargs(axis))
        plt.legend()

        plt.subplot(2, 3, 2)
        plt.title("Next Coords")
        for axis in ['x', 'y', 'z']:
            plt.plot(self._next[axis], **get_kwargs(axis))
        plt.legend()

        plt.subplot(2, 3, 3)
        plt.title("Diff Coords")
        for axis in ['x', 'y', 'z']:
            plt.plot(self._diff[axis], **get_kwargs(axis))
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.title("Slice Data (Actual Steps)")
        for axis in ['base', 'rear', 'front']:
            plt.plot(self._slice_actual[axis], **get_kwargs(axis))
        plt.legend()

        # make the y ticks integers, not floats
        yint = []
        locs, _ = plt.yticks()
        for each in locs:
            yint.append(int(each))
        plt.yticks(yint)

        plt.tight_layout()
        plt.show()


class Dobot:
    pos = None

    def __init__(self, port, rate=115200, timeout=0.025, debug=False, plot=False, fake=False):
        self._debugOn = debug
        self._fake = fake
        self._driver = DobotDriver(port, rate)
        if fake:
            self._driver._ramps = True
            self._driver._stepCoeff = 20000
            self._driver._stopSeq = 0
            self._driver._stepCoeffOver2 = self._driver._stepCoeff / 2
            self._driver._freqCoeff = self._driver._stepCoeff * 25
        else:
            self._driver.Open(timeout)
        self._plotter = DobotPlotter() if plot else None
        self._kinematics = DobotKinematics(debug=debug)
        self._toolRotation = 0
        self._gripper = 480
        # Last directions to compensate for backlash.
        self._lastBaseDirection = 0
        self._lastRearDirection = 0
        self._lastFrontDirection = 0
        # Initialize arms current configuration from accelerometers
        if fake:
            self._baseSteps = long(0)
            self._rearSteps = long(0)
            self._frontSteps = long(0)
        else:
            self.InitializeAccelerometers()

    def _debug(self, *args):
        if self._debugOn:
            print(*args)

    def InitializeAccelerometers(self):
        print("--=========--")
        print("Initializing accelerometers")
        if self._driver.isFpga():
            # In FPGA v1.0 SPI accelerometers are read only when Arduino boots. The readings
            # are already available, so read once.
            ret = (0, 0, 0, 0, 0, 0, 0)
            while not ret[0]:
                ret = self._driver.GetAccelerometers()
            accelRearX = ret[1]
            accelFrontX = ret[4]
            rearAngle = piHalf - self._driver.accelToRadians(accelRearX, accelOffsets[0])
            frontAngle = self._driver.accelToRadians(accelFrontX, accelOffsets[1])
        else:
            # In RAMPS accelerometers are on I2C bus and can be read at any time. We need to
            # read them multiple times to get average as MPU-6050 has greater resolution but is noisy.
            # However, due to the interference from the way A4988 holds the motors, if none of the
            # recommended measures to suppress interference are in place (see open-dobot wiki), or
            # in case accelerometers are not connected, we need to give up and assume some predefined pose.
            accelRearX = 0
            accelRearY = 0
            accelRearZ = 0
            accelFrontX = 0
            accelFrontY = 0
            accelFrontZ = 0
            successes = 0
            for _ in range(20):
                ret = (0, 0, 0, 0, 0, 0, 0)
                attempts = 10
                while attempts:
                    ret = self._driver.GetAccelerometers()
                    if ret[0]:
                        successes += 1
                        accelRearX += ret[1]
                        accelRearY += ret[2]
                        accelRearZ += ret[3]
                        accelFrontX += ret[4]
                        accelFrontY += ret[5]
                        accelFrontZ += ret[6]
                        break
                    attempts -= 1
            if successes > 0:
                divisor = float(successes)
                rearAngle = piHalf - self._driver.accel3DXToRadians(
                    accelRearX / divisor, accelRearY / divisor, accelRearZ / divisor
                )
                frontAngle = -self._driver.accel3DXToRadians(
                    accelFrontX / divisor, accelFrontY / divisor, accelFrontZ / divisor
                )
            else:
                print("Failed to read accelerometers. Make sure they are connected and interference is suppressed.")
                print("See open-dobot wiki")
                print("Assuming rear arm vertical and front arm horizontal")
                rearAngle = 0
                frontAngle = -piHalf
        self._baseSteps = long(0)
        self._rearSteps = long((rearAngle / piTwo) * rearArmActualStepsPerRevolution + 0.5)
        self._frontSteps = long((frontAngle / piTwo) * frontArmActualStepsPerRevolution + 0.5)
        self._driver.SetCounters(self._baseSteps, self._rearSteps, self._frontSteps)
        print(
            "Initializing with steps:",
            self._baseSteps,
            self._rearSteps,
            self._frontSteps,
        )
        print("Reading back what was set:", self._driver.GetCounters())
        currBaseAngle = piTwo * self._baseSteps / baseActualStepsPerRevolution
        currRearAngle = piHalf - piTwo * self._rearSteps / rearArmActualStepsPerRevolution
        currFrontAngle = piTwo * self._frontSteps / frontArmActualStepsPerRevolution
        self.pos = self._kinematics.coordinatesFromAngles(currBaseAngle, currRearAngle, currFrontAngle)
        print("Current estimated coordinates:", self.pos)
        print("--=========--")

    def _moveToAnglesSlice(self, baseAngle, rearArmAngle, frontArmAngle, toolRotation):
        angles = np.array([baseAngle, rearArmAngle, frontArmAngle])
        multipliers = np.array([
            baseActualStepsPerRevolution,
            rearArmActualStepsPerRevolution,
            frontArmActualStepsPerRevolution
        ])
        stepLocations = angles * multipliers / piTwo
        # rear and front are absolute in the original code
        stepLocations[1:] = np.abs(stepLocations[1:])

        self._debug("Step Locations", *stepLocations)
        self._debug("Current Steps", self._baseSteps, self._rearSteps, self._frontSteps)

        currSteps = np.array([self._baseSteps, self._rearSteps, self._frontSteps])
        diffs = stepLocations - currSteps

        self._debug("Diffs", *diffs)

        dirs = np.ones(3, dtype=int)  # get numpy.int64 somewhere, fix?
        signs = np.array([1, 1, -1])
        base, rear, front = range(3)

        if diffs[base] < 1:
            dirs[base] = 0
            signs[base] = -1
        if diffs[rear] < 1:
            dirs[rear] = 0
            signs[rear] = -1
        if diffs[front] > 1:
            dirs[front] = 0
            signs[front] = 1

        diffsAbs = np.abs(diffs)

        # We still need to call stepsToCmdValFloat for each, as it returns a tuple
        resBase = self._driver.stepsToCmdValFloat(diffsAbs[base])
        resRear = self._driver.stepsToCmdValFloat(diffsAbs[rear])
        resFront = self._driver.stepsToCmdValFloat(diffsAbs[front])

        cmdVals = [resBase[0], resRear[0], resFront[0]]
        actualSteps = np.array([resBase[1], resRear[1], resFront[1]])
        leftSteps = np.array([resBase[2], resRear[2], resFront[2]])

        # Compensate for backlash.
        # For now compensate only backlash in the base motor as the backlash in the arm motors depends on a specific task (a laser/brush or push-pull tasks).
        if self._lastBaseDirection != dirs[base] and actualSteps[base] > 0:
            cmdVals[base], _ignore, _ignore = self._driver.stepsToCmdValFloat(diffsAbs[base] + backlash)
            self._lastBaseDirection = dirs[base]
        # if self._lastRearDirection != rearDir and actualStepsRear > 0:
        # 	cmdRearVal, _ignore, _ignore = self._driver.stepsToCmdValFloat(rearDiffAbs + backlash)
        # 	self._lastRearDirection = rearDir
        # if self._lastFrontDirection != frontDir and actualStepsFront > 0:
        # 	cmdFrontVal, _ignore, _ignore = self._driver.stepsToCmdValFloat(frontDiffAbs + backlash)
        # 	self._lastFrontDirection = frontDir

        if not self._fake:
            # Repeat until the command is queued. May not be queued if the queue is full.
            ret = (0, 0)
            while not ret[1]:
                self.steps = self._driver.Steps(cmdVals[base], cmdVals[rear], cmdVals[front], dirs[base], dirs[rear],
                                                dirs[front], self._gripper, int(toolRotation), )
                ret = self.steps

        actualSteps *= signs
        leftSteps *= signs
        if self._plotter:
            self._plotter.add_slice_data(diffs[base], actualSteps[base],
                                         diffs[rear], actualSteps[rear],
                                         diffs[front], actualSteps[front])

        return actualSteps, leftSteps

    def freqToCmdVal(self, freq):
        """
        See DobotDriver.freqToCmdVal()
        """
        return self._driver.freqToCmdVal(freq)

    def MoveWithSpeed(self, x, y, z, maxSpeed, accel=None, toolRotation=None):
        """
        For toolRotation see DobotDriver.Steps() function description (servoRot parameter).
        """

        if self._plotter:
            self._plotter.reset_move_plots()

        maxVel = float(maxSpeed)
        xx = float(x)
        yy = float(y)
        zz = float(z)

        if toolRotation is None:
            toolRotation = self._toolRotation
        elif toolRotation > 1024:
            toolRotation = 1024
        elif toolRotation < 0:
            toolRotation = 0

        accelf = None
        # Set 100% acceleration to equal maximum velocity if it wasn't provided
        if accel is None:
            accelf = maxVel
        else:
            accelf = float(accel)

        self._debug("--=========--")
        self._debug("maxVel", maxVel)
        self._debug("accelf", accelf)

        currBaseAngle = piTwo * self._baseSteps / baseActualStepsPerRevolution
        currRearAngle = piHalf - piTwo * self._rearSteps / rearArmActualStepsPerRevolution
        currFrontAngle = piTwo * self._frontSteps / frontArmActualStepsPerRevolution
        currPos = np.array(self._kinematics.coordinatesFromAngles(currBaseAngle, currRearAngle, currFrontAngle))
        targetPos = np.array([xx, yy, zz])

        vect = targetPos - currPos
        self._debug("moving from", *currPos)
        self._debug("moving to", *targetPos)
        self._debug("moving by", *vect)

        distance = np.linalg.norm(vect)
        self._debug("distance to travel", distance)
        if distance == 0.0:
            return  # nothing to do, avoid div-by-zero below

        # If half the distance is reached before reaching maxSpeed with the given acceleration, then actual
        # maximum velocity will be lower; the total number of slices is determined from half the distance
        # and acceleration.
        distToReachMaxSpeed = pow(maxVel, 2) / (2.0 * accelf)
        if distToReachMaxSpeed * 2.0 >= distance:
            timeToAccel = math.sqrt(distance / accelf)
            accelSlices = timeToAccel * 50.0
            timeFlat = 0
            flatSlices = 0
            maxVel = math.sqrt(distance * accelf)
        # Or else the number of slices when velocity does not change is greater than zero.
        else:
            timeToAccel = maxVel / accelf
            accelSlices = timeToAccel * 50.0
            timeFlat = (distance - distToReachMaxSpeed * 2.0) / maxVel
            flatSlices = timeFlat * 50.0

        slices = accelSlices * 2.0 + flatSlices
        self._debug("slices to do", slices)
        self._debug("accelSlices", accelSlices)
        self._debug("flatSlices", flatSlices)

        # Acceleration/deceleration in respective axes
        accelVect = (accelf * vect) / distance
        self._debug("accelXYZ", *accelVect)

        # Vectors in respective axes to complete acceleration/deceleration
        segmentAccel = accelVect * pow(timeToAccel, 2) / 2.0
        self._debug("segmentAccelXYZ", *segmentAccel)

        # Maximum velocity in respective axes for the segment with constant velocity
        maxVelVect = (maxVel * vect) / distance
        self._debug("maxVelXYZ", *maxVelVect)

        # Vectors in respective axes for the segment with constant velocity
        segmentFlat = maxVelVect * timeFlat
        self._debug("segmentFlatXYZ", *segmentFlat)

        segmentToolRotation = (toolRotation - self._toolRotation) / slices
        self._debug("segmentToolRotation", segmentToolRotation)

        commands = 1

        while commands < slices:
            self._debug("==============================")
            self._debug("slice #", commands)
            # If accelerating
            if commands <= accelSlices:
                t2half = pow(commands / 50.0, 2) / 2.0
                nextPos = currPos + accelVect * t2half
            # If decelerating
            elif commands >= accelSlices + flatSlices:
                t2half = pow((slices - commands) / 50.0, 2) / 2.0
                nextPos = currPos + segmentAccel * 2.0 + segmentFlat - accelVect * t2half
            # Or else moving at maxSpeed
            else:
                t = abs(commands - accelSlices) / 50.0
                nextPos = currPos + segmentAccel + maxVelVect * t
            self._debug("moving to", *nextPos)

            nextToolRotation = self._toolRotation + (segmentToolRotation * commands)
            self._debug("nextToolRotation", nextToolRotation)

            baseAngle, rearAngle, frontAngle = self._kinematics.anglesFromCoordinates(*nextPos)

            movedSteps, leftSteps = self._moveToAnglesSlice(baseAngle, rearAngle, frontAngle, nextToolRotation)

            self._debug("moved", *movedSteps, "steps")
            self._debug("leftovers", *leftSteps)

            commands += 1

            self._baseSteps += movedSteps[0]
            self._rearSteps += movedSteps[1]
            self._frontSteps += movedSteps[2]

            currBaseAngle = piTwo * self._baseSteps / baseActualStepsPerRevolution
            currRearAngle = piHalf - piTwo * self._rearSteps / rearArmActualStepsPerRevolution
            currFrontAngle = piTwo * self._frontSteps / frontArmActualStepsPerRevolution
            cX, cY, cZ = self._kinematics.coordinatesFromAngles(currBaseAngle, currRearAngle, currFrontAngle)
            if self._plotter:
                self._plotter.add_move_data(cX, cY, cZ, *nextPos)

        self._toolRotation = toolRotation

        if self._plotter:
            self._plotter.show()

    def Gripper(self, value):
        if value > 480:
            self._gripper = 480
        elif value < 208:
            self._gripper = 208
        else:
            self._gripper = value

        self._driver.Steps(0, 0, 0, 0, 0, 0, self._gripper, self._toolRotation)

    def Wait(self, waitTime):
        """
        See description in DobotDriver.Wait()
        """
        self._driver.Wait(waitTime)

    def CalibrateJoint(self, joint, forwardCommand, backwardCommand, direction, pin, pinMode, pullup):
        """
        See DobotDriver.CalibrateJoint()
        """
        return self._driver.CalibrateJoint(joint, forwardCommand, backwardCommand, direction, pin, pinMode, pullup)

    def EmergencyStop(self):
        """
        See DobotDriver.EmergencyStop()
        """
        return self._driver.EmergencyStop()

    def LaserOn(self, on):
        return self._driver.LaserOn(on)

    def PumpOn(self, on):
        return self._driver.PumpOn(on)

    def ValveOn(self, on):
        return self._driver.ValveOn(on)
