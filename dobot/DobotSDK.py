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
Additional Authors: Ingo Breßler (April 12, 2026), <put your name here>

Version: 1.2.2

License: MIT
"""

import sys
import math
import numpy as np

from dobot.DobotDriver import DobotDriver
from dobot.DobotKinematics import DobotKinematics
from dobot.DobotBase import DobotBase, BASE, REAR, FRONT, JOINT_NAME

piTwo = 2. * np.pi

# Workaround to support Python 2/3
if sys.version_info > (3,):
    long = int

# Backlash in the motor reduction gears is actually 22 steps, but 5 is visually unnoticeable.
# It is a horrible thing to compensate for a bad backlash in software, but the only other
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

ACCEL, FLAT, DECEL = range(3)


def valueToStr(v):
    if isinstance(v, (float, np.floating)):
        s = f"{v: 7.4f}"
    elif isinstance(v, (bool, np.bool_)):
        s = f"{str(v):>7s}"
    elif isinstance(v, (int, np.integer)):  # int?
        s = f"{v:>7d}"
    else:
        s = f"{v:>7s}"  # string?
    return s

def arrayToStr(arr):
    if arr is None:
        return "None"
    try:
        return f"({",".join([valueToStr(v) for v in arr])})"
    except TypeError:
        return valueToStr(arr)

def arraysToStr(*args):
    return [arrayToStr(arr) for arr in args]

def print_arr(prefix, *args):
    print(f"{str(prefix):>15s}", *arraysToStr(*args))

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
        self._slice_actual = []

    def add_slice_data(self, actual_steps):
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

        if not len(self._slice_actual):
            return  # nothing to do

        plt = self._plt
        plt.figure(figsize=(12, 8))

        plt.subplot(3, 3, 1)
        plt.title("Current Coords")
        if len(self._coords):
            data = np.stack(self._coords)
            for i, axis in enumerate(['x', 'y', 'z']):
                plt.plot(data[:,i], **get_kwargs(axis))
        plt.legend()

        plt.subplot(3, 3, 2)
        plt.title("Next Coords")
        if len(self._next):
            data = np.stack(self._next)
            for i, axis in enumerate(['x', 'y', 'z']):
                plt.plot(data[:,i], **get_kwargs(axis))
        plt.legend()

        plt.subplot(3, 3, 3)
        plt.title("Diff Coords")
        if len(self._diff):
            data = np.stack(self._diff)
            for i, axis in enumerate(['x', 'y', 'z']):
                plt.plot(data[:,i], **get_kwargs(axis))
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.title("Slice Data (Actual Steps)")
        if len(self._slice_actual):
            data = np.stack(self._slice_actual)
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

class SegmentParams:
    start = None
    end = None
    delta = None
    v_start = None
    v_end = None
    a_max = None
    phase_duration = None
    joint_v_peak = None
    joint_accel = None
    joint_decel = None
    phase_distances = None

    def v_from(self, isforward):
        return self.v_start if isforward else self.v_end
    def set_v_from(self, isforward, v):
        if isforward:
            self.v_start = v
        else:
            self.v_end = v
    def v_to(self, isforward):
        return self.v_end if isforward else self.v_start
    def set_v_to(self, isforward, v):
        if isforward:
            self.v_end = v
        else:
            self.v_start = v

    def __init__(self, start, end, v_start, v_end, v_max, a_max):
        self.start = start
        self.end = end
        self.delta = end - start
        self.v_start = v_start
        self.v_end = v_end
        self.v_max = v_max
        self.a_max = a_max

    def __str__(self):
        return "\n".join((
            ", ".join((f"{name}: {arrayToStr(getattr(self, name))}" for name in ("start", "end", "delta"))),
            ", ".join((f"{name}: {arrayToStr(getattr(self, name))}" for name in ("v_start", "v_end")))))

    @staticmethod
    def get_phase_durations(delta, v_start, v_end, v_max, a_max, debug=False):
        # 1. Calculate theoretical max possible speed for a triangular profile
        v_peak_theory = np.sqrt(a_max * np.abs(delta) + 0.5 * (v_start * v_start + v_end * v_end))
        # 2. Cap this peak speed with the external v_max
        v_peak = np.minimum(v_peak_theory, v_max)
        if debug:
            print_arr("v_peak, theory, v_max", v_peak, v_peak_theory, v_max)
        # calc max required time for each joint in each phase of this segment
        phase_distances = np.zeros((3, 3), dtype=float)
        phase_duration = np.zeros((3, 3), dtype=float)
        # Distance covered during accel/decel at this (possibly capped) v_peak
        phase_distances[ACCEL] = np.maximum((v_peak * v_peak - v_start * v_start) / (2.0 * a_max), 0.0)
        phase_distances[DECEL] = np.maximum((v_peak * v_peak - v_end * v_end) / (2.0 * a_max), 0.0)
        # The remaining distance is covered in the flat phase, if v_peak was capped, this will be > 0
        phase_distances[FLAT] = np.maximum(np.abs(delta) - phase_distances[ACCEL] - phase_distances[DECEL], 0.0)
        # Calculate durations based on the final v_peak and phase distances
        phase_duration[ACCEL] = np.where(v_peak > v_start, (v_peak - v_start) / a_max, 0.0)
        phase_duration[FLAT] = np.divide(
            phase_distances[FLAT],
            v_peak,
            out=np.zeros_like(phase_distances[FLAT]),
            where=v_peak > 0.0,
        )
        phase_duration[DECEL] = np.where(v_peak > v_end, (v_peak - v_end) / a_max, 0.0)
        # Each joint drives its own phase duration, use the maximum.
        phase_duration = phase_duration.max(axis=1)
        #return phase_duration
        return np.ceil(phase_duration*50)/50.  # avoid rounding errors later

    @staticmethod
    def calc_profile(effective_distance, v_start, v_end, phase_duration, debug=False):
        # Reconstruct the synchronized peak velocity from the total segment distance.
        effective_duration = (
                0.5 * phase_duration[ACCEL] + phase_duration[FLAT] + 0.5 * phase_duration[DECEL]
        )
        joint_v_peak = np.maximum(np.divide(
            effective_distance,
            effective_duration,
            out=np.zeros_like(effective_distance),
            where=effective_duration > 0.0,
        ), 0.0)
        joint_accel = np.maximum(np.divide(
            joint_v_peak - v_start,
            phase_duration[ACCEL],
            out=np.zeros_like(joint_v_peak),
            where=phase_duration[ACCEL] > 0.0,
        ), 0.0)
        joint_decel = np.maximum(np.divide(
            joint_v_peak - v_end,
            phase_duration[DECEL],
            out=np.zeros_like(joint_v_peak),
            where=phase_duration[DECEL] > 0.0,
        ), 0.0)
        phase_distances = np.zeros((3, 3), dtype=float)
        phase_distances[ACCEL] = (
                v_start * phase_duration[ACCEL]
                + 0.5 * joint_accel * phase_duration[ACCEL] * phase_duration[ACCEL]
        )
        phase_distances[DECEL] = (
                joint_v_peak * phase_duration[DECEL]
                + 0.5 * (-joint_decel) * phase_duration[DECEL] * phase_duration[DECEL]
        )
        phase_distances[FLAT] = joint_v_peak * phase_duration[FLAT]
        if debug:
            print_arr("phase_distances", *(phase_distances[x] for x in range(3)))
            print_arr("joint_v_peak", joint_v_peak)
            print_arr("joint_accel", joint_accel)
            print_arr("joint_decel", joint_decel)
        return joint_v_peak, joint_accel, joint_decel, phase_distances

    @staticmethod
    def _solve_common(delta, v_start, v_end, v_max, a_max, fix_mismatch=False, debug=False):
        phase_duration = SegmentParams.get_phase_durations(delta, v_start, v_end, v_max, a_max, debug=debug)
        effective_distance = np.maximum(np.abs(delta) - 0.5 * (
                v_start * phase_duration[ACCEL] + v_end * phase_duration[DECEL]
        ), 0.0)  # must not be negative
        if debug:
            print_arr("phase_duration", phase_duration)
            print_arr("effective_distance", effective_distance)

        mismatch_orig, mismatch_new = None, None
        # helpers for calculating an exact mismatch factor
        factors = [0., 1.]
        mismatch = []
        mismatch_largest_idx = 0
        for i in range(3):
            eff_dist = effective_distance
            if mismatch_orig is not None and mismatch_new is not None:
                non_zero_boundary = (np.abs(v_start) > 1e-9) | (np.abs(v_end) > 1e-9)
                needs_closure = np.logical_and(non_zero_boundary, np.logical_not(np.abs(mismatch_new) < 1e-5))
                if np.any(needs_closure):
                    eff_dist = np.maximum(effective_distance + factors[i]*mismatch_orig, 0.0)
                else:  # nothing to do, no closure needed
                    break
            joint_v_peak, joint_accel, joint_decel, phase_distances = SegmentParams.calc_profile(
                eff_dist, v_start, v_end, phase_duration, debug=debug)

            # If a joint has no flat phase and non-zero boundary velocities, force exact closure
            # by letting the decel phase absorb the remaining mismatch.
            reconstructed = np.sign(delta) * phase_distances.sum(axis=0)
            mismatch_new = np.abs(delta) - np.abs(reconstructed)
            if mismatch_orig is None:
                mismatch_orig = mismatch_new
                mismatch_largest_idx = np.argmax(np.abs(mismatch_orig))
            mismatch.append(mismatch_new[mismatch_largest_idx])
            if debug:
                print_arr("reconstructed", reconstructed)
                print_arr("mismatch", [str(mismatch_new), str(mismatch[-1]), factors[i]])
                rel_mismatch = np.divide(
                    np.abs(mismatch_new),
                    np.maximum(np.abs(delta), 1e-9),
                    out=np.zeros_like(mismatch_new),
                    where=np.abs(delta) >= 0.0,
                )
                print_arr("rel_mismatch", (f"{joint_rel * 100.0:.2f}%" for joint_rel in rel_mismatch))
            if len(mismatch) > 1:
                factor = 0.7  # typically between 65 % and 75% of the mismatch
                if mismatch[0]-mismatch[1] != 0:
                    factor = (mismatch[0])*(factors[1]-factors[0]) / (mismatch[0]-mismatch[1])
                factors.append(factor)
                if debug:
                    print(f"Applying {factors[-1]} mismatch factor.", f"({factor})" if factor > 1 else "")
            if not fix_mismatch:
                break

        return phase_duration, joint_v_peak, joint_accel, joint_decel, phase_distances

    def update(self, v_max, isforward=True, **kwargs):
        (self.phase_duration, self.joint_v_peak, self.joint_accel, self.joint_decel, self.phase_distances
         ) = self._solve_common(self.delta, self.v_start, self.v_end, v_max, self.a_max, **kwargs)
        if isforward:
            self.v_end = self.joint_v_peak - self.joint_decel * self.phase_duration[DECEL]
            if kwargs.get("debug", False):
                print_arr("v_end", self.v_end)
        else:
            # Backward-pass variant: keep v_end fixed and solve the start side consistently.
            self.v_start = self.joint_v_peak - self.joint_accel * self.phase_duration[ACCEL]
            if kwargs.get("debug", False):
                print_arr("v_start", self.v_start)

class Dobot(DobotBase):
    # See calibrate-accelerometers.py for details
    _accelOffsetRear = 1024  # FIXME: move to driver? or into configurable DobotConfig obj
    _accelOffsetFront = 1024

    def __init__(self, port, rate=115200, timeout=0.025, debug=False, plot=False, fake=False,
                 jointMaxVelDeg=None, jointMaxAccelDeg=None, sca1000Sensors=False, endEffectorOffset=None,
                 baseLimitDeg=None, rearLimitDeg=None, frontLimitDeg=None):
        """
        Initializes the Dobot control class with parameters for serial communication, debugging,
        plotting options, and maximum joint accelerations. Also initializes internal configurations
        like kinematics, driver setup, and optional fake mode for testing.

        :param port: The serial port to which the Dobot is connected.
        :type port: str
        :param rate: The baud rate for serial communication. Defaults to 115200.
        :type rate: int, optional
        :param timeout: The timeout duration (in seconds) for serial communication. Defaults to 0.025.
        :type timeout: float, optional
        :param debug: Enables debugging mode. Defaults to False.
        :type debug: bool, optional
        :param plot: Enables plotting via the DobotPlotter. Defaults to False.
        :type plot: bool, optional
        :param fake: Enables fake mode for testing without real hardware. Defaults to False.
        :type fake: bool, optional
        :param jointMaxVelDeg: Per-joint velocity maximum limit in degrees per second.
            Defaults to (45.0, 45.0, 45.0).
        :type jointMaxVelDeg: tuple[float, float, float], optional
        :param jointMaxAccelDeg: Per-joint acceleration maximum limit in degrees per second squared.
            Defaults to (90.0, 90.0, 90.0).
        :type jointMaxAccelDeg: tuple[float, float, float], optional
        :param sca1000Sensors: Enable when SCA1000 sensors are installed as accelerometers.
            This changes conversion of sensor values to degrees and thus affects positioning.
            Defaults to False.
        :type sca1000Sensors: bool, optional
        :param endEffectorOffset: Offset or distance (horizontal, vertical) of end effector tool from joint 3 in mm.
            Defaults to (50.9, 15.).
        :type endEffectorOffset: tuple[float, float], optional
        :param baseLimitDeg: Angular limits (min, max) for the base joint in degrees.
            Defaults to (-90.0, 90.0).
        :type baseLimitDeg: tuple[float, float], optional
        :param rearLimitDeg: Angular limits (min, max) for the rear joint in degrees.
            Defaults to (0.0, 105.0).
        :type rearLimitDeg: tuple[float, float], optional
        :param frontLimitDeg: Angular limits (min, max) for the front joint in degrees.
            Defaults to (-102.0, 18.0).
        :type frontLimitDeg: tuple[float, float], optional
        """
        self._debugOn = debug
        self._fake = fake
        self._driver = DobotDriver(port, rate, sca1000Sensors=sca1000Sensors)
        if fake:
            self._driver._ramps = True
            self._driver._stepCoeff = 20000
            self._driver._stopSeq = 0
            self._driver._stepCoeffOver2 = self._driver._stepCoeff / 2
            self._driver._freqCoeff = self._driver._stepCoeff * 25
        else:
            self._driver.Open(timeout)
        self._plotter = DobotPlotter() if plot else None
        self._kinematics = DobotKinematics(endEffectorOffset=endEffectorOffset)
        self._toolRotation = 0
        self._gripper = 480
        # Per-joint velocity limits in joint units per second.
        # The MoveWithSpeed() velocity argument is interpreted as a percentage of these maxima.
        if jointMaxVelDeg is None:
            jointMaxVelDeg = (45.0, 45.0, 45.0)  # fallback deg/sec
        self._jointMaxVelDeg = np.clip(np.array(jointMaxVelDeg, dtype=float), 1e-2, 360.)
        print_arr(f"Maximum joint velocity in degrees/sec:", self._jointMaxVelDeg)
        # Per-joint acceleration limits in joint units per second^2.
        # The MoveWithSpeed() accel argument is interpreted as a percentage of these maxima.
        if jointMaxAccelDeg is None:
            jointMaxAccelDeg = (90.0, 90.0, 90.0)  # fallback deg/sec-squared
        self._jointMaxAccelDeg = np.clip(np.array(jointMaxAccelDeg, dtype=float), 1e-2, 360.)
        print_arr(f"Maximum joint accel. in degrees/sec²: ", self._jointMaxAccelDeg)
        # Per-joint angular limits in degrees, defaults for Dobot with SCA1000 accelerometers
        if baseLimitDeg is None:
            baseLimitDeg = (-90.0, 90.0)
        if rearLimitDeg is None:
            rearLimitDeg = (-15., 90.0)
        if frontLimitDeg is None:
            frontLimitDeg = (-23., 83.)
        self._limitsRad = np.deg2rad(np.array((baseLimitDeg, rearLimitDeg, frontLimitDeg)))
        print_arr(f"Base  joint angular limits in degrees:", np.rad2deg(self._limitsRad[BASE]))
        print_arr(f"Rear  joint angular limits in degrees:", np.rad2deg(self._limitsRad[REAR]))
        print_arr(f"Front joint angular limits in degrees:", np.rad2deg(self._limitsRad[FRONT]))
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
            self._init_accelerometers()

    def _get_accelerometers_raw(self):
        attempts = 10
        ret = (0, 0, 0, 0, 0, 0, 0)
        while attempts:
            ret = self._driver.GetAccelerometers()
            if ret[0]:
                break
            attempts -= 1
        return ret

    def _rear_angle_fpga(self, sensorValue):
        return (math.pi * .5) - self._driver.accelToRadians(sensorValue, self._accelOffsetRear)

    def _front_angle_fpga(self, sensorValue):
        return self._driver.accelToRadians(sensorValue, self._accelOffsetFront)

    def _init_accelerometers(self):
        print("--=========--")
        print("Initializing accelerometers")
        if self._driver.isFpga():
            # In FPGA v1.0 SPI accelerometers are read only when Arduino boots. The readings
            # are already available, so read once.
            _, accelRearX, _, _, accelFrontX, _, _ = self._get_accelerometers_raw()
            rearAngle = self._rear_angle_fpga(accelRearX)
            frontAngle = self._front_angle_fpga(accelFrontX)
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
                ret = self._get_accelerometers_raw()
                successes += 1
                accelRearX += ret[1]
                accelRearY += ret[2]
                accelRearZ += ret[3]
                accelFrontX += ret[4]
                accelFrontY += ret[5]
                accelFrontZ += ret[6]
            if successes > 0:
                divisor = float(successes)
                rearAngle = .5*np.pi - self._driver.accel3DXToRadians(
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
                frontAngle = -.5*np.pi
        print(f"Angles read: rear= {np.rad2deg(rearAngle):.3f}°, front= {np.rad2deg(frontAngle):.3f}°")
        print("    [ expecting: rear @vertical -> 0°, front @horizontal -> 0° ]")
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
    def posAngles(self):
        """
        Calculates and returns the positional angles of the joints in radians.

        The computed angles are derived based on the number of current steps recorded
        for each joint and the values of the actual steps per revolution for each joint.

        It has to return the same angles as the 'Angles read:' output of the constructor __init__().

        :return: Positional angles of the joints in radians as a numpy array.
        :rtype: numpy.ndarray
        """
        currSteps = np.array([self._baseSteps, self._rearSteps, self._frontSteps])
        multipliers = np.array([
            baseActualStepsPerRevolution,
            rearArmActualStepsPerRevolution,
            frontArmActualStepsPerRevolution
        ])
        angles = 2. * np.pi * currSteps / multipliers
        return angles

    @property
    def pos(self):
        return self._kinematics.coordinatesFromAngles(*self.posAngles)

    @staticmethod
    def fmtPos(pos: np.ndarray, prefix="pos"):
        return prefix+": "+", ".join([f"{coord} = {pos[i]:.2f}" for i, coord in enumerate(("x", "y", "z"))])

    def _prepareAnglesSlice(self, angles, debug=False):
        currSteps = np.array([self._baseSteps, self._rearSteps, self._frontSteps])
        multipliers = np.array([
            baseActualStepsPerRevolution,
            rearArmActualStepsPerRevolution,
            frontArmActualStepsPerRevolution
        ])
        angles = np.clip(angles, self._limitsRad[:,0], self._limitsRad[:,1])
        stepLocations = angles * multipliers / piTwo
        diffs = stepLocations - currSteps
        # rear and front are absolute in the original code
        stepLocations[1:] = np.abs(stepLocations[1:])

        if debug:
            # print_arr("angles:", angles)
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

        actualSteps *= signs
        leftSteps *= signs

        return cmdVals, dirs, actualSteps, leftSteps

    def _moveToAnglesSlice(self, cmdVals, dirs, toolRotation):

        if not self._fake:
            # Repeat until the command is queued. May not be queued if the queue is full.
            ret = (0, 0)
            while not ret[1]:
                self.steps = self._driver.Steps(cmdVals, dirs, self._gripper, int(toolRotation))
                ret = self.steps

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

    def MoveWithSpeed(self, targets, vel=0.5, accel=0.5, toolRotation=None):
        """
        Moves the robotic system through a series of target positions while maintaining a specified
        maximum velocity and acceleration. Optionally, the movement can also involve interpolating
        a tool rotation.

        :param targets: A list or NumPy array of Cartesian target coordinates (xyz) to move through.
            If a single NumPy array is provided, it is treated as a single target position.
        :param vel: (Optional) Maximum joint velocity percentage of the maximum allowed joint velocity for the movement.
        :param accel: (Optional) Acceleration percentage of the maximum allowed joint acceleration
            to be applied. If not provided, a default value of 50% is assumed.
        :param toolRotation: (Optional) The desired tool rotation value. If not provided,
            defaults to the current tool rotation. The value is clamped within the range [0, 1024].
        :return: None
        """

        if self._plotter:
            self._plotter.reset_move_plots()

        # translate given percentages for speed & accel to vectors in rad with indiv. values for each joint
        vel = np.clip(vel, 1e-9, 1.)
        v_max = np.deg2rad(self._jointMaxVelDeg) * vel
        accel = np.clip(accel, 1e-9, 1.)
        a_max = np.deg2rad(self._jointMaxAccelDeg) * accel

        if toolRotation is None:
            toolRotation = self._toolRotation
        toolRotation = float(np.clip(toolRotation, 0, 1024))

        if isinstance(targets, np.ndarray) and targets.ndim == 1:
            targets = [targets.astype(float)]  # single coordinate
        targets = [np.array(t, dtype=float) for t in targets]
        if len(targets) == 0:
            return

        debug = self._debugOn
        # Build a full path including the current position as the starting point.
        currPos = self.pos
        points = [currPos] + targets

        # Convert Cartesian waypoints to joint angles first.
        joint_points = []
        for p in points:
            angles = self._kinematics.anglesFromCoordinates(p, debug=debug)
            # clip way points to stay within allowed ranges
            angles = np.clip(angles, self._limitsRad[:,0], self._limitsRad[:,1])
            joint_points.append(angles)

        if debug:
            self._debug("MoveWithSpeed:", level=0)
            print_arr("v_max", v_max)
            print_arr("a_max", a_max)
            print_arr("world points", *points)
            #print_arr("unwrap.bef", *joint_points)
            joint_points = self._unwrap_angles(joint_points)
            print_arr("joint points", *joint_points)

        # create all segments first
        segments = [SegmentParams(joint_points[i], joint_points[i+1], v_max, v_max, v_max, a_max)
                    for i in range(len(joint_points)-1)]

        def apply_static_conditions(segments, idx):
            """
            Apply static conditions to a segment, such as zero velocity at the start and end points.
            To be used before the motion planning updates of each scan forward or backward.
            This makes sure the conditions are met before the next calculation.
            """
            if idx == 0:  # all start with zero
                segments[idx].v_start = np.zeros(3, dtype=float)
            if idx == len(segments) - 1:  # all end with zero
                segments[idx].v_end = np.zeros(3, dtype=float)

        def apply_directional_conditions(segments, idx, isforward=True):
            """
            Apply directional conditions to a segment depending on the scan direction,
            such as that start and end velocities have to match.
            To be used before the motion planning updates of each scan forward or backward.
            This makes sure the conditions are met before the next calculation.
            """
            incr = int(isforward)*2-1
            if 0 <= (idx - incr) < len(segments):
                # continue the next segment with the speed of the previous one
                segments[idx].set_v_from(isforward, segments[idx-incr].v_to(isforward))
            # for no position change within the segment, the velocity stays the same
            segments[idx].set_v_to(isforward, np.where(np.abs(segments[idx].delta) < 1e-5,
                                segments[idx].v_from(isforward), segments[idx].v_to(isforward)))
            if 0 <= (idx + incr) < len(segments):
                # on direction change, set those joints to zero
                # sign change with zero is ok
                sign_change = np.logical_and((np.sign(segments[idx].delta)  * np.sign(segments[idx+incr].delta)) != 0,
                                              np.sign(segments[idx].delta) != np.sign(segments[idx+incr].delta))
                # print_arr("sign_change", sign_change)
                segments[idx].set_v_to(isforward, np.where(sign_change, 0., segments[idx].v_to(isforward)))

        # forward scan for allowed segment velocities
        for seg_index in range(len(segments)):
            apply_static_conditions(segments, seg_index)
            apply_directional_conditions(segments, seg_index, isforward=True)
            if debug:
                print(f"# seg {seg_index}:\n{segments[seg_index]}")
            segments[seg_index].update(v_max, debug=debug)

        # backward scan for allowed segment velocities
        for seg_index in reversed(range(len(segments))):
            apply_static_conditions(segments, seg_index)
            apply_directional_conditions(segments, seg_index, isforward=False)
            if debug:
                print(f"# seg {seg_index}:\n{segments[seg_index]}")
            segments[seg_index].update(v_max, isforward=False, debug=debug)

        # forward scan for allowed segment velocities
        for seg_index in range(len(segments)):
            apply_static_conditions(segments, seg_index)
            apply_directional_conditions(segments, seg_index, isforward=True)
            if debug:
                print(f"# seg {seg_index}:\n{segments[seg_index]}")
            segments[seg_index].update(v_max, fix_mismatch=True, debug=debug)

        prevMovedSteps = None
        for seg_index in range(len(segments)):
            segment = segments[seg_index]
            if debug:
                print(f"# seg {seg_index}:\n{segment}")

            # determine the required number of slices in each part across all joints
            slices = np.ceil(segment.phase_duration * 50.0).astype(int)
            totalSlices = int(slices.sum())
            if debug:
                print_arr("slices, total", segment.phase_duration * 50., (totalSlices,))

            commands = 1
            while commands <= totalSlices:
                #print(f"{commands=}", f"{slices[ACCEL]=}")
                if commands <= slices[ACCEL] and slices[ACCEL] > 0:
                    t = commands / 50.0
                    s = segment.v_start * t + 0.5 * segment.joint_accel * t * t
                    if debug:
                        print_arr("accelerating", [t], s)
                elif commands <= slices[ACCEL] + slices[FLAT]:
                    flat_cmd = commands - slices[ACCEL]
                    t = flat_cmd / 50.0
                    s = segment.phase_distances[ACCEL] + segment.joint_v_peak * t
                    if debug:
                        print_arr("constant", [t], s)
                else:
                    dec_cmd = commands - slices[ACCEL] - slices[FLAT]
                    t = dec_cmd / 50.0
                    s = segment.phase_distances[ACCEL] + segment.phase_distances[FLAT] + (
                            segment.joint_v_peak * t - 0.5 * segment.joint_decel * t * t
                    )
                    if debug:
                        print_arr("decelerating", [t], s)

                next_joint_pos = segment.start + np.sign(segment.delta) * s
                nextToolRotation = self._toolRotation + (
                        (toolRotation - self._toolRotation) * (commands / float(totalSlices))
                )
                cmdVals, dirs, movedSteps, leftSteps = self._prepareAnglesSlice(next_joint_pos, debug=debug)
                stepDelta = 0
                if prevMovedSteps is not None:
                    stepDelta = np.abs(movedSteps - prevMovedSteps).sum()
                skip_this_slice = np.all(np.abs(movedSteps) <= 1) and stepDelta > 20
                if debug:
                    self._debug("steps to move:", *movedSteps, "delta:", stepDelta,
                                "skipped!" if skip_this_slice else "")
                    self._debug("leftovers", *leftSteps)
                commands += 1
                if skip_this_slice:
                    continue

                self._moveToAnglesSlice(cmdVals, dirs, nextToolRotation)
                prevMovedSteps = movedSteps
                if self._plotter:
                    self._plotter.add_slice_data(movedSteps)

                self._baseSteps += movedSteps[0]
                self._rearSteps += movedSteps[1]
                self._frontSteps += movedSteps[2]

                if self._plotter:
                    nextPos = self._kinematics.coordinatesFromAngles(*next_joint_pos)
                    self._plotter.add_move_data(self.pos, nextPos)

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
