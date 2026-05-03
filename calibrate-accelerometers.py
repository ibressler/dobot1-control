#! /usr/bin/env python

"""
open-dobot accelerometer calibration tool.

This tool continuously reports accelerometers and angles from those.

Use this tool to find offsets for your accelerometers.

Follow the procedure below to enable the accelerometer reporting mode on FPGA.
No action is required on the RAMPS as GY-521 accelerometers can be read at any time there.
1. Turn off power on the arm and disconnect USB cable
2. Remove accelerometers from the arm and put them on a flat surface that has no inclination
3. Connect USB cable
4. Enable the accelerometer reporting mode:
   4.1. Press and hold the "Sensor Calibration" button on the FPGA version
   		or ground pin D23 on AUX-4 on the RAMPS version
   4.2. Press and release the "Reset" button
   4.3. Start this tool (still holding the "Sensor Calibration" button on the FPGA version
    	or keeping pin D23 grounded on the RAMPS)
   4.4. Wait for the accelerometer data to start flowing on your console/whatever_you_use_to_start_this_tool
   4.5. Release the "Sensor Calibration" button
5. Gently push down the accelerometers so that they are on the surface evenly. Don't touch any contacts/leads.
	You can push them one by one, not necessary to push both at the same time
6. Note the "Raw" data from accelerometers reported on the console. Those are your accelerometers' offsets
7. Turn off power on the arm, disconnect USB cable, mount accelerometers back onto the arm 

Author: maxosprojects (March 18, 2016)
Additional Authors: Ingo Breßler (April 11, 2026), <put your name here>

Version: 1.2.2

License: MIT
"""

from pathlib import Path
import time
import argparse
import sys

import numpy as np

from dobot import DobotDriver
from dobot import DobotKinematics
from dobot.DobotBase import REAR, FRONT, arrayToStr


def toEndEffectorHeight(kinematics, rear, front):
	_, _, z = kinematics.coordinatesFromAngles(0, rear, front)
	return z

def display_accelerometer_data(driver, kinematics, ret):
	if ret[0]:
		if driver.isFpga():
			rear_rad = driver.accelToRadiansAxis(REAR, ret[1])
			front_rad = driver.accelToRadiansAxis(FRONT, ret[4])
			rear_deg = np.rad2deg(rear_rad)
			front_deg = np.rad2deg(front_rad)
			xyz = kinematics.coordinatesFromAngles(0, rear_rad, front_rad)
			print(f"Angles rear: {rear_deg:10f}°, front: {front_deg:10f}° | "
				  f"Cartesian coord: {arrayToStr(xyz)} mm | "
				  f"Raw rear: {ret[1]:4d}, front: {ret[4]:4d}")
		else:
			rear_rad = driver.accel3DXToRadians(ret[1], ret[2], ret[3])
			front_rad = -driver.accel3DXToRadians(ret[4], ret[5], ret[6])
			rear_deg = np.rad2deg(rear_rad)
			front_deg = np.rad2deg(front_rad)
			xyz = kinematics.coordinatesFromAngles(0, rear_rad, front_rad).tolist()
			print(f"Angles rear: {rear_deg:10f}°, front: {front_deg:10f}° | "
				  f"Cartesian coord: {arrayToStr(xyz)} mm | "
				  f"Raw rear: {ret[1]:6d} {ret[2]:6d} {ret[3]:6d}, "
				  f"front: {ret[4]:6d} {ret[5]:6d} {ret[6]:6d}")
	else:
		print('Error occurred reading data')

def continuous_mode(driver, kinematics):
	while True:
		ts = time.time()
		ret = driver.GetAccelerometers()
		display_accelerometer_data(driver, kinematics, ret)
		# limit queries to 4x per second
		tdelta = time.time() - ts
		if tdelta < 0.25:
			time.sleep(0.25 - tdelta)

def positions_mode(driver, kinematics):
	pos1 = (110,  0, 20)
	pos2 = (320,  0, 20)

	for i, pos in enumerate([pos1, pos2], 1):
		posAngles = tuple(kinematics.anglesFromCoordinates(pos).tolist())
		input(f"Move the arm to cartesian position {i} {pos}, angles {posAngles}, and press <enter>...")
		ret = driver.GetAccelerometers()
		display_accelerometer_data(driver, kinematics, ret)

def main():
	parser = argparse.ArgumentParser(description='open-dobot accelerometer calibration tool')
	parser.add_argument('mode', choices=['continuous', 'positions'], nargs='?', default='continuous',
						help='Calibration mode (default: continuous)')
	args = parser.parse_args()

	if sys.platform.startswith("win"):
		port = "COM4"
	elif sys.platform.startswith("linux"):
		try:
			port = str(next(Path("/dev").glob("ttyACM*")))
		except StopIteration:
			print("Error: Could not find any ttyACM device in /dev")
			return
	elif sys.platform.startswith("darwin"):
		try:
			port = str(next(Path("/dev").glob("tty.usbmodem*")))
		except StopIteration:
			print("Error: Could not find any tty.usbmodem device in /dev")
			return
	else:
		print(f"Unsupported platform: {sys.platform}")
		return

	driver = DobotDriver(port, accelOffset=(997, 1016))
	driver.Open()
	# driver.Open(timeout=0.3)
	kinematics = DobotKinematics(endEffectorOffset=(49., 64.))

	if args.mode == 'positions':
		positions_mode(driver, kinematics)
	elif args.mode == 'continuous':
		continuous_mode(driver, kinematics)

if __name__ == '__main__':
	main()
