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
from tornado.log import access_log

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
    print(f"{str(prefix):>15s}", *[f"({",".join([f"{v: 7.4f}" if isinstance(v, (float, np.floating)) else f"{v:5d}" for v in arr])})" for arr in args])

class Dobot:
    def __init__(self, port, rate=115200, timeout=0.025, debug=False, plot=False, fake=False,
                 jointMaxAccelerations=None):
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
        # Per-joint acceleration limits in joint units per second^2.
        # These are used as the baseline, and the MoveWithSpeed() accel argument
        # is interpreted as a percentage of these maxima.
        if jointMaxAccelerations is None:
            jointMaxAccelerations = (1.0, 1.0, 1.0)  # deg/sec-squared
        self._jointMaxAccelerations = np.array(jointMaxAccelerations, dtype=float)
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

    @staticmethod
    def _normalize_move_targets(targets):
        if isinstance(targets, np.ndarray) and targets.ndim == 1:
            return [targets.astype(float)]
        return [np.array(t, dtype=float) for t in targets]

    @staticmethod
    def _unwrap_angles(angles_list):
        """
        Keep angle sequences continuous across +/-pi boundaries.
        """
        if len(angles_list) == 0:
            return angles_list

        out = [np.array(angles_list[0], dtype=float)]
        for ang in angles_list[1:]:
            prev = out[-1].copy()
            cur = np.array(ang, dtype=float)
            for axis in range(3):
                while cur[axis] - prev[axis] > math.pi:
                    cur[axis] -= 2.0 * math.pi
                while cur[axis] - prev[axis] < -math.pi:
                    cur[axis] += 2.0 * math.pi
            out.append(cur)
        return out

    @staticmethod
    def _signf(value):
        if value > 0.0:
            return 1
        if value < 0.0:
            return -1
        return 0

    def _plan_joint_waypoint_speeds(self, joint_points):
        """
        Compute per-waypoint joint velocities with lookahead.

        Returns:
          waypoint_speeds: list of 3-element speed vectors, one per waypoint
          waypoint_limits:  list of 3-element max speed vectors per waypoint
        """
        n = len(joint_points) - 1
        waypoint_speeds = [np.zeros(3, dtype=float) for _ in range(n + 1)]
        waypoint_limits = [np.full(3, np.inf, dtype=float) for _ in range(n + 1)]

        if n <= 0:
            return waypoint_speeds, waypoint_limits

        waypoint_limits[0][:] = 0.0
        waypoint_limits[-1][:] = 0.0

        # Corner handling: if a joint changes direction, that joint must stop at the waypoint.
        for i in range(1, n):
            prev_delta = joint_points[i] - joint_points[i - 1]
            next_delta = joint_points[i + 1] - joint_points[i]
            for axis in range(3):
                a = self._signf(prev_delta[axis])
                b = self._signf(next_delta[axis])
                if a != 0 and b != 0 and a != b:
                    waypoint_limits[i][axis] = 0.0

        # Forward pass: accelerate as much as possible.
        for i in range(n):
            delta = joint_points[i + 1] - joint_points[i]
            dist = np.abs(delta)

            for axis in range(3):
                amax = max(self._jointMaxAccelerations[axis], 1e-9)
                s0 = waypoint_speeds[i][axis]
                vmax = waypoint_limits[i + 1][axis]

                # Speed reachable over this axis segment.
                if dist[axis] <= 0.0:
                    reachable = s0
                else:
                    reachable = math.sqrt(max(s0 * s0 + 2.0 * amax * dist[axis], 0.0))

                waypoint_speeds[i + 1][axis] = min(vmax, reachable)

        # Backward pass: ensure deceleration to the next waypoint is possible.
        for i in range(n - 1, -1, -1):
            delta = joint_points[i + 1] - joint_points[i]
            dist = np.abs(delta)

            for axis in range(3):
                amax = max(self._jointMaxAccelerations[axis], 1e-9)
                s1 = waypoint_speeds[i + 1][axis]

                if dist[axis] <= 0.0:
                    reachable = s1
                else:
                    reachable = math.sqrt(max(s1 * s1 + 2.0 * amax * dist[axis], 0.0))

                waypoint_speeds[i][axis] = min(waypoint_speeds[i][axis], reachable, waypoint_limits[i][axis])

        # Final forward pass for consistency.
        for i in range(n):
            delta = joint_points[i + 1] - joint_points[i]
            dist = np.abs(delta)

            for axis in range(3):
                amax = max(self._jointMaxAccelerations[axis], 1e-9)
                s0 = waypoint_speeds[i][axis]
                vmax = waypoint_limits[i + 1][axis]

                if dist[axis] <= 0.0:
                    reachable = s0
                else:
                    reachable = math.sqrt(max(s0 * s0 + 2.0 * amax * dist[axis], 0.0))

                waypoint_speeds[i + 1][axis] = min(waypoint_speeds[i + 1][axis], vmax, reachable)

        waypoint_speeds[-1][:] = 0.0
        return waypoint_speeds, waypoint_limits

    def MoveWithSpeed(self, targetPos, maxSpeed, accel=None, toolRotation=None):
        """
        Fully joint-wise motion planning.

        maxSpeed is the path speed between Cartesian waypoints.
        accel is a percentage [0..1] of per-joint max acceleration.
        """

        if self._plotter:
            self._plotter.reset_move_plots()

        maxVel = float(maxSpeed)
        accel_pct = 1.0 if accel is None else float(accel)
        if accel_pct > 1.0:
            accel_pct = accel_pct / 100.0
        accel_pct = 2.
        accel_vec = self._jointMaxAccelerations * accel_pct
        accel_vec = np.where(accel_vec < 1e-9, 1e-9, accel_vec)

        if toolRotation is None:
            toolRotation = self._toolRotation
        toolRotation = float(np.clip(toolRotation, 0, 1024))

        self._debug("--=========--")
        self._debug(f"{maxVel=}")
        self._debug(f"{accel_pct=}")

        targets = self._normalize_move_targets(targetPos)
        if len(targets) == 0:
            return

        # Build a full path including the current position as the starting point.
        currPos = self.pos
        points = [currPos] + targets

        # Convert Cartesian waypoints to joint angles first.
        joint_points = []
        for p in points:
            joint_points.append(np.array(self._kinematics.anglesFromCoordinates(p, debug=False), dtype=float))
        print_arr("unwrap.bef", *joint_points)
        joint_points = self._unwrap_angles(joint_points)

        waypoint_speeds, _ = self._plan_joint_waypoint_speeds(joint_points)

        print_arr("points", *points)
        print_arr("joint_points", *joint_points)
        print_arr("waypoint_speeds", *waypoint_speeds)

        for seg_index in range(len(joint_points) - 1):
            joint_start = joint_points[seg_index]
            joint_end = joint_points[seg_index + 1]
            delta = joint_end - joint_start
            # calc the required time per joint with given max acceleration
            v_start = np.abs(waypoint_speeds[seg_index])
            v_end = np.abs(waypoint_speeds[seg_index + 1])
            # max possible speed for each joint
            v_peak = np.sqrt(accel_vec * np.abs(delta) + 0.5 * (v_start * v_start + v_end * v_end))

            phase_distances = np.zeros((3, 3), dtype=float)
            phase_duration = np.zeros((3, 3), dtype=float)

            accel, flat, decel = 0, 1, 2
            # calc max required time for each joint in each phase of this segment
            phase_distances[accel] = np.maximum((v_peak * v_peak - v_start * v_start) / (2.0 * accel_vec), 0.0)
            phase_distances[flat] = np.maximum(np.abs(delta) - phase_distances[accel] - phase_distances[decel], 0.0)
            phase_distances[decel] = np.maximum((v_peak * v_peak - v_end * v_end) / (2.0 * accel_vec), 0.0)
            phase_duration[accel] = np.where(v_peak > v_start, (v_peak - v_start) / accel_vec, 0.0)
            phase_duration[flat] = np.where(v_peak > 0.0, phase_distances[flat] / v_peak, 0.0)
            phase_duration[decel] = np.where(v_peak > v_end, (v_peak - v_end) / accel_vec, 0.0)
            # Each joint drives its own phase duration, use the maximum.
            # determine the required number of slices in each part across all joints
            # durations for each phase in seconds
            phase_duration = phase_duration.max(axis=1)
            slices = np.round(phase_duration * 50.0).astype(int)
            totalSlices = slices.sum()

            # recalculate v_peak for each joint to synchronize them across the given phase durations
            segment_duration = 0.5 * phase_duration[accel] + phase_duration[flat] + 0.5 * phase_duration[decel]
            joint_distances = np.abs(delta) - 0.5 * (v_start * phase_duration[accel] + v_end * phase_duration[decel])
            v_peak_synced = np.where(segment_duration > 0.0, joint_distances / segment_duration, 0.0)

            # recalculate accelerations for each phase for each joint
            joint_accel = np.where(phase_duration[accel] > 0.0, (v_peak_synced - v_start) / phase_duration[accel], 0.0)
            joint_decel = np.where(phase_duration[accel] > 0.0, (v_peak_synced - v_end) / phase_duration[decel], 0.0)

            # pre-calculate distances for each phase synchronized
            phase_distances[accel] = v_start * phase_duration[accel] + 0.5 * joint_accel * phase_duration[accel] * phase_duration[accel]
            phase_distances[flat] = v_peak_synced * phase_duration[flat]
            phase_distances[decel] = np.zeros(3)

            print("-- segment", seg_index)
            print_arr("start", joint_start)
            print_arr("end", joint_end)
            print_arr("delta", delta)
            print_arr("v_start", v_start)
            print_arr("v_end", v_end)
            print_arr("v_peak", v_peak_synced)
            print_arr("phase_duration", phase_duration)
            print_arr("totalSlices", [totalSlices], slices)

            commands = 1
            while commands <= totalSlices:
                #print(f"{commands=}", f"{slices[accel]=}")
                if commands <= slices[accel] and slices[accel] > 0:
                    t = commands / 50.0
                    s = v_start * t + 0.5 * joint_accel * t * t
                    print_arr("accelerating", [t], s)
                elif commands <= slices[accel] + slices[flat]:
                    flat_cmd = commands - slices[accel]
                    t = flat_cmd / 50.0
                    s = phase_distances[accel] + v_peak_synced * t
                    print_arr("constant", [t], s)
                else:
                    dec_cmd = commands - slices[accel] - slices[flat]
                    t = dec_cmd / 50.0
                    s = phase_distances[accel] + phase_distances[flat] + (v_peak_synced * t - 0.5 * joint_decel * t * t)
                    print_arr("decelerating", [t], s)

                nextJoint = joint_start + np.sign(delta) * s
                print_arr("nextJoint", nextJoint)

                nextToolRotation = self._toolRotation + (
                        (toolRotation - self._toolRotation) * (commands / float(totalSlices))
                )

                movedSteps, leftSteps = self._moveToAnglesSlice(nextJoint, nextToolRotation, debug=False)
                self._debug("moved", *movedSteps, "steps")
                self._debug("leftovers", *leftSteps)

                commands += 1

                self._baseSteps += movedSteps[0]
                self._rearSteps += movedSteps[1]
                self._frontSteps += movedSteps[2]

                if self._plotter:
                    nextPos = np.array(self._kinematics.coordinatesFromAngles(*nextJoint), dtype=float)
                    self._plotter.add_move_data(self.pos, nextPos)

            #currPos = points[seg_index + 1]

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
