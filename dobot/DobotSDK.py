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

BASE, REAR, FRONT = range(3)

class DobotPlotter:
    def __init__(self):
        import matplotlib.pyplot as plt

        self._plt = plt
        # Lists for detailed coordinate tracking in MoveWithSpeed
        self.reset_move_plots()

    def reset_move_plots(self):
        self._coords = []
        self._next = []
        self._diff = []
        self._slice_diff = []
        self._slice_actual = []

    def add_slice_data(self, diffs, actual_steps):
        #, rear_diff, actual_steps_rear, front_diff, actual_steps_front):
        self._slice_diff.append(diffs)
        self._slice_actual.append(actual_steps)

    def add_move_data(self, coord:np.ndarray, nextPos:np.ndarray):
        self._coords.append(coord)
        self._next.append(nextPos)
        self._diff.append(coord - nextPos)

    def show(self):
        linewidth = 1.0
        colors = dict(x='darkred', y='darkgreen', z='lightblue',
                      base='lightblue', rear='darkgreen', front='darkred')

        def get_kwargs(axis):
            return dict(color=colors[axis], linewidth=linewidth, label=axis)

        plt = self._plt
        plt.figure(figsize=(12, 8))

        plt.subplot(3, 3, 1)
        plt.title("Current Coords")
        data = np.stack(self._coords)
        for i, axis in enumerate(['x', 'y', 'z']):
            plt.plot(data[:,i], **get_kwargs(axis))
        plt.legend()

        plt.subplot(3, 3, 2)
        plt.title("Next Coords")
        data = np.stack(self._next)
        for i, axis in enumerate(['x', 'y', 'z']):
            plt.plot(data[:,i], **get_kwargs(axis))
        plt.legend()

        plt.subplot(3, 3, 3)
        plt.title("Diff Coords")
        data = np.stack(self._diff)
        for i, axis in enumerate(['x', 'y', 'z']):
            plt.plot(data[:,i], **get_kwargs(axis))
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.title("Slice Data (Actual Steps)")
        data = np.stack(self._slice_actual)
        for i, axis in enumerate(['base', 'rear', 'front']):
            plt.plot(data[:,i], **get_kwargs(axis))
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.title("Slice Data (Diff Steps)")
        data = np.stack(self._slice_diff)
        for i, axis in enumerate(['base', 'rear', 'front']):
            plt.plot(data[:,i], **get_kwargs(axis))
        plt.legend()

        # make the y ticks integers, not floats
        yint = []
        locs, _ = plt.yticks()
        for each in locs:
            yint.append(int(each))
        plt.yticks(yint)

        plt.tight_layout()
        plt.show()


def print_arr(prefix, *args):
    print(f"{prefix:>11s}", *[f"{v: 7.1f}" for arr in args for v in arr])

class Dobot:
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
        print("Current estimated coordinates:", self.pos)
        print("--=========--")

    @property
    def pos(self):
        currBaseAngle = piTwo * self._baseSteps / baseActualStepsPerRevolution
        currRearAngle = piHalf - piTwo * self._rearSteps / rearArmActualStepsPerRevolution
        currFrontAngle = piTwo * self._frontSteps / frontArmActualStepsPerRevolution
        return np.array(self._kinematics.coordinatesFromAngles(currBaseAngle, currRearAngle, currFrontAngle), dtype=float)

    def _moveToAnglesSlice(self, angles, toolRotation, debug=False):
        multipliers = np.array([
            baseActualStepsPerRevolution,
            rearArmActualStepsPerRevolution,
            frontArmActualStepsPerRevolution
        ])
        stepLocations = angles * multipliers / piTwo
        # rear and front are absolute in the original code
        stepLocations[1:] = np.abs(stepLocations[1:])
        currSteps = np.array([self._baseSteps, self._rearSteps, self._frontSteps])
        diffs = stepLocations - currSteps

        if debug:
            print_arr("currSteps:", currSteps)
            print_arr("stepLocs:", stepLocations)
            print_arr("diffs:", diffs)

        dirs = np.ones(3, dtype=int)
        signs = np.array([1, 1, -1])

        if diffs[BASE] < 1:
            dirs[BASE] = 0
            signs[BASE] = -1
        if diffs[REAR] < 1:
            dirs[REAR] = 0
            signs[REAR] = -1
        if diffs[FRONT] > 1:
            dirs[FRONT] = 0
            signs[FRONT] = 1

        diffsAbs = np.abs(diffs)

        # We still need to call stepsToCmdValFloat for each, as it returns a tuple
        resBase = self._driver.stepsToCmdValFloat(diffsAbs[BASE])
        resRear = self._driver.stepsToCmdValFloat(diffsAbs[REAR])
        resFront = self._driver.stepsToCmdValFloat(diffsAbs[FRONT])

        cmdVals = [resBase[0], resRear[0], resFront[0]]
        actualSteps = np.array([resBase[1], resRear[1], resFront[1]])
        leftSteps = np.array([resBase[2], resRear[2], resFront[2]])

        # Compensate for backlash.
        # For now compensate only backlash in the base motor as the backlash in the arm motors depends on a specific task (a laser/brush or push-pull tasks).
        if self._lastBaseDirection != dirs[BASE] and actualSteps[BASE] > 0:
            cmdVals[BASE], _ignore, _ignore = self._driver.stepsToCmdValFloat(diffsAbs[BASE] + backlash)
            self._lastBaseDirection = dirs[BASE]
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
                self.steps = self._driver.Steps(cmdVals, dirs, self._gripper, int(toolRotation))
                ret = self.steps

        actualSteps *= signs
        leftSteps *= signs
        if self._plotter:
            self._plotter.add_slice_data(diffs, actualSteps)

        return actualSteps, leftSteps

    def freqToCmdVal(self, freq):
        """
        See DobotDriver.freqToCmdVal()
        """
        return self._driver.freqToCmdVal(freq)

    @staticmethod
    def _axis_sign(value):
        if value > 0.0:
            return 1
        if value < 0.0:
            return -1
        return 0

    def _normalize_move_targets(self, targets):
        if isinstance(targets, np.ndarray) and targets.ndim == 1:
            return [targets.astype(float)]
        return [np.array(t, dtype=float) for t in targets]

    def _segment_axis_reversal(self, seg_a, seg_b):
        """
        Return True if any axis changes the direction between two consecutive segments.
        """
        if seg_b is None or seg_a is None:
            return False

        reversal = [False] * 3
        for axis in range(3):
            a = self._axis_sign(seg_a[axis])
            b = self._axis_sign(seg_b[axis])
            if a != 0 and b != 0 and a != b:
                reversal[axis] = True
        return any(reversal)

    def _plan_waypoint_speeds(self, segments, maxVel, accel):
        """
        Two-pass lookahead planner.

        Returns a list of waypoint speeds with:
          speeds[0] == 0.0
          speeds[-1] == 0.0
        """
        n = len(segments)
        speeds = [0.0] * (n + 1)
        if n == 0:
            return speeds

        distances = [float(np.linalg.norm(seg)) for seg in segments]

        # Waypoints that require a stop because the direction reverses in the next segment.
        waypoint_limit = [maxVel] * (n + 1)
        waypoint_limit[0] = 0.0
        waypoint_limit[-1] = 0.0

        for i in range(1, n):
            if self._segment_axis_reversal(segments[i - 1], segments[i]):
                waypoint_limit[i] = 0.0

        # Forward pass: accelerate as much as possible from the start.
        speeds[0] = 0.0
        for i in range(n):
            speeds[i + 1] = min(
                waypoint_limit[i + 1],
                math.sqrt(max(speeds[i] * speeds[i] + 2.0 * accel * distances[i], 0.0)),
            )

        # Backward pass: ensure we can decelerate to the next waypoint speed in time.
        speeds[-1] = 0.0
        for i in range(n - 1, -1, -1):
            speeds[i] = min(
                speeds[i],
                math.sqrt(max(speeds[i + 1] * speeds[i + 1] + 2.0 * accel * distances[i], 0.0)),
            )

        # One more forward pass to keep the profile consistent after the backward clamp.
        speeds[0] = 0.0
        for i in range(n):
            speeds[i + 1] = min(
                waypoint_limit[i + 1],
                speeds[i + 1],
                math.sqrt(max(speeds[i] * speeds[i] + 2.0 * accel * distances[i], 0.0)),
            )

        speeds[-1] = 0.0
        return speeds, distances

    def MoveWithSpeed(self, targetPos, maxSpeed, accel=None, toolRotation=None):
        """
        For toolRotation see DobotDriver.Steps() function description (servoRot parameter).

        targetPos may be either:
          - a single 3-element position, or
          - a list/array of 3-element positions to follow with lookahead
        The current position is used as the starting point.
        """

        if self._plotter:
            self._plotter.reset_move_plots()

        # Set 100% acceleration to equal maximum velocity if it wasn't provided
        maxVel = float(maxSpeed)
        accelf = maxVel if accel is None else float(accel)

        if toolRotation is None:
            toolRotation = self._toolRotation
        toolRotation = float(np.clip(toolRotation, 0, 1024))

        self._debug("--=========--")
        self._debug(f"{maxVel=}")
        self._debug(f"{accelf=}")

        targets = self._normalize_move_targets(targetPos)
        if len(targets) == 0:
            return

        # Build a full path including the current position as the starting point.
        currPos = self.pos
        points = [currPos] + targets
        segments = [points[i + 1] - points[i] for i in range(len(points) - 1)]
        waypoint_speeds, distances = self._plan_waypoint_speeds(segments, maxVel, accelf)
        self._debug(f"{points=}")
        self._debug(f"{segments=}")
        self._debug(f"{distances=}")
        self._debug(f"{waypoint_speeds=}")

        for seg_index, seg_vect in enumerate(segments):
            target = points[seg_index + 1]
            distance = distances[seg_index]

            if distance == 0.0:
                currPos = target
                continue

            self._debug("-- segment", seg_index)
            self._debug("from", *points[seg_index])
            self._debug("to", *target)
            self._debug("vect", *seg_vect)
            self._debug("distance to travel", distance)

            v_start = float(waypoint_speeds[seg_index])
            v_end = float(waypoint_speeds[seg_index + 1])

            # Compute the peak speed possible for this segment.
            # If the segment is too short for a flat section, v_peak is reduced.
            v_peak_sq = accelf * distance + 0.5 * (v_start * v_start + v_end * v_end)
            v_peak = min(maxVel, math.sqrt(max(v_peak_sq, 0.0)))

            d_accel = max((v_peak * v_peak - v_start * v_start) / (2.0 * accelf), 0.0)
            d_decel = max((v_peak * v_peak - v_end * v_end) / (2.0 * accelf), 0.0)
            d_flat = max(distance - d_accel - d_decel, 0.0)

            t_accel = (v_peak - v_start) / accelf if v_peak > v_start else 0.0
            t_decel = (v_peak - v_end) / accelf if v_peak > v_end else 0.0
            t_flat = d_flat / v_peak if v_peak > 0.0 else 0.0

            slices_accel = int(round(t_accel * 50.0))
            slices_flat = int(round(t_flat * 50.0))
            slices_decel = int(round(t_decel * 50.0))
            totalSlices = max(slices_accel + slices_flat + slices_decel, 1)

            self._debug(f"{v_start=}", f"{v_end=}", f"{v_peak=}")
            self._debug("slices", slices_accel, slices_flat, slices_decel)
            self._debug("seg.dist.", d_accel, d_flat, d_decel)
            self._debug("seg.times", t_accel, t_flat, t_decel)

            dirVect = seg_vect / distance

            commands = 1
            while commands <= totalSlices:
                if commands <= slices_accel and slices_accel > 0:
                    # accelerate from v_start to v_peak
                    t = commands / 50.0
                    self._debug(f"  accelerating… {t=}")
                    dist_now = v_start * t + 0.5 * accelf * t * t
                    nextPos = currPos + dirVect * dist_now

                elif commands <= slices_accel + slices_flat:
                    # constant velocity
                    flat_cmd = commands - slices_accel
                    t = flat_cmd / 50.0
                    self._debug(f"  constant velocity… {t=}")
                    nextPos = currPos + dirVect * d_accel + dirVect * (v_peak * t)

                else:
                    # decelerate from v_peak to v_end
                    dec_cmd = commands - slices_accel - slices_flat
                    t = dec_cmd / 50.0
                    self._debug(f"  decelerating… {t=}")
                    dist_back = v_peak * t - 0.5 * accelf * t * t
                    nextPos = currPos + dirVect * d_accel + dirVect * d_flat + dirVect * dist_back

                if False:
                    print_arr("currPos:", currPos)
                    print_arr("nextPos:", nextPos, target)
                    print_arr("diff:", nextPos-currPos)

                nextToolRotation = self._toolRotation + ((toolRotation - self._toolRotation) * (commands / float(totalSlices)))
                self._debug("nextToolRotation", nextToolRotation)

                angles = self._kinematics.anglesFromCoordinates(nextPos, debug=False)
                movedSteps, leftSteps = self._moveToAnglesSlice(np.array(angles), nextToolRotation, debug=False)

                self._debug("moved", *movedSteps, "steps")
                self._debug("leftovers", *leftSteps)

                commands += 1

                self._baseSteps += movedSteps[0]
                self._rearSteps += movedSteps[1]
                self._frontSteps += movedSteps[2]

                if self._plotter:
                    self._plotter.add_move_data(self.pos, nextPos)

            currPos = self.pos

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

        self._driver.Steps([0, 0, 0], [0, 0, 0], self._gripper, self._toolRotation)

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
