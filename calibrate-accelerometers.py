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

Version: 1.3.0

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


positionDefaults = {
	'pos1': (120, 0, 0),
	'pos2': (320, 0, 0),
	'offset': (51, 15),
}

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

def positions_mode(driver, kinematics, positions):
	"""
	Calculates sensor offsets and conversion factors for the accelerometers
	based on measurements at specified cartesian positions. This function requires
	user interaction to move the arm to specified positions and confirm via prompts.
	The values for *accelOffset* and *accelConversion* are finally given for
	the *Dobot()* or *DobotDriver()* constructors.

	For a given pair of positions, *angles_1* and *angles_2* are known.
	This gives four equations. The base equation is::

		angle = math.asin((sensorVal - offset) / accelConv)

	The four angle equations are::

		(1)  angles1.rear  = np.pi * 0.5 - asin((sensorVal1.rear  - offset.rear)  / accelConv)
		(2)  angles1.front = asin((sensorVal1.front - offset.front) / accelConv)
		(3)  angles2.rear  = np.pi * 0.5 - asin((sensorVal2.rear  - offset.rear)  / accelConv)
		(4)  angles2.front = asin((sensorVal2.front - offset.front) / accelConv)

	Rearranging gives::

		(5)  sin(np.pi * 0.5 - angles1.rear) = (sensorVal1.rear  - offset.rear)  / accelConv
		(6)  sin(np.pi * 0.5 - angles2.rear) = (sensorVal2.rear  - offset.rear)  / accelConv
		(7)  sin(angles1.front)              = (sensorVal1.front - offset.front) / accelConv
		(8)  sin(angles2.front)              = (sensorVal2.front - offset.front) / accelConv

	Multiplying by accelConv gives::

		(9)   sin(np.pi * 0.5 - angles1.rear) * accelConv = sensorVal1.rear  - offset.rear
		(10)  sin(np.pi * 0.5 - angles2.rear) * accelConv = sensorVal2.rear  - offset.rear
		(11)  sin(angles1.front)              * accelConv = sensorVal1.front - offset.front
		(12)  sin(angles2.front)              * accelConv = sensorVal2.front - offset.front

	Combine equations (9) and (10), then eliminate offset.rear::

		(13)  (sensorVal1.rear - offset.rear) / sin(np.pi * 0.5 - angles1.rear)
			= (sensorVal2.rear - offset.rear) / sin(np.pi * 0.5 - angles2.rear)
		(14)  offset.rear =
			(
			  sensorVal1.rear / sin(np.pi * 0.5 - angles1.rear)
			- sensorVal2.rear / sin(np.pi * 0.5 - angles2.rear)
			) / (
			  1 / sin(np.pi * 0.5 - angles1.rear)
			- 1 / sin(np.pi * 0.5 - angles2.rear)
			)

	Using::

		sin(np.pi * 0.5 - x) == cos(x)

	Equation (14) simplifies to::

		(15)  offset.rear =
			(
			  sensorVal1.rear * np.cos(angles2.rear)
			- sensorVal2.rear * np.cos(angles1.rear)
			) / (
			  np.cos(angles2.rear)
			- np.cos(angles1.rear)
			)

	Combine equations (11) and (12), then eliminate *offset.front*::

		(16)  (sensorVal1.front - offset.front) / sin(angles1.front)
			= (sensorVal2.front - offset.front) / sin(angles2.front)

		(17)  sensorVal1.front / sin(angles1.front) - offset.front / sin(angles1.front)
			= sensorVal2.front / sin(angles2.front) - offset.front / sin(angles2.front)

		(18)  sensorVal2.front / sin(angles2.front) - sensorVal1.front / sin(angles1.front)
			= offset.front * ( 1 / sin(angles1.front) - 1 / sin(angles2.front) )

		(19)  offset.front =
			  (
				  sensorVal2.front / sin(angles2.front) - sensorVal1.front / sin(angles1.front)
			  ) / (
				  1 / sin(angles1.front) - 1 / sin(angles2.front)
			  )

		(20)  offset.front =
			  (
				  sensorVal1.front * np.sin(angles2.front) - sensorVal2.front * np.sin(angles1.front)
			  ) / (
				  np.sin(angles2.front) - np.sin(angles1.front)
			  )

	Calculate accelConv using equations (9)–(12)::

		(21)  accelConv = (sensorVal1.rear  - offset.rear)  / sin(np.pi * 0.5 - angles1.rear)
		(22)  accelConv = (sensorVal2.rear  - offset.rear)  / sin(np.pi * 0.5 - angles2.rear)
		(23)  accelConv = (sensorVal1.front - offset.front) / sin(angles1.front)
		(24)  accelConv = (sensorVal2.front - offset.front) / sin(angles2.front)

	:param driver: The driver that is responsible for communicating with the robotic arm
	 	and retrieving sensor data.
	:type driver: Any
	:param kinematics: The kinematics module used to calculate angular positions
	    from cartesian coordinates and perform related computations.
	:type kinematics: Any
	:param positions: A pair of target cartesian positions to which the robotic
	    arm should be moved. Each entry specifies 3D coordinates.
	:type positions: tuple | list
	:return: None
	"""
	angles = []
	sensors = []

	for i, pos in enumerate(positions):
		angles.append(kinematics.anglesFromCoordinates(pos))
		input(f"-> Move the arm to cartesian position {i} {pos}, angles {arrayToStr(np.rad2deg(angles[-1]))}, and press <enter>...")
		sensors.append(driver.GetAccelerometers())
		display_accelerometer_data(driver, kinematics, sensors[-1])

	offset_rear = float((sensors[0][1] * np.cos(angles[1][REAR]) - sensors[1][1] * np.cos(angles[0][REAR])) / (
				np.cos(angles[1][REAR]) - np.cos(angles[0][REAR])))
	offset_front = float((sensors[0][4] * np.sin(angles[1][FRONT]) - sensors[1][4] * np.sin(angles[0][FRONT])
					) / (np.sin(angles[1][FRONT]) - np.sin(angles[0][FRONT])))
	accelConv0 = float((sensors[0][1] - offset_rear) / np.sin(np.pi * .5 - angles[0][REAR]))
	accelConv1 = float((sensors[1][4] - offset_front) / np.sin(angles[1][FRONT]))
	print("-> Use the following values in the Dobot or DobotDriver constructor:")
	print(f"   accelOffset=({offset_rear:.2f}, {offset_front:.2f}), "
		  f"accelConversion={.5*(accelConv0+accelConv1):.2f}")
	# print(f"   ({accelConv0=}, {accelConv1=})")  # for debugging
	# no idea why accelConv calculated from rear and front offsets differs

def main():
	parser = argparse.ArgumentParser(description='open-dobot accelerometer calibration tool')
	parser.add_argument('mode', choices=['continuous', 'positions'], nargs='?', default='continuous',
						help='Calibration mode (default: continuous)')
	parser.add_argument('--pos1', type=float, nargs=3, default=positionDefaults['pos1'],
						metavar=('X', 'Y', 'Z'), help=f'First position for calibration, default: {positionDefaults["pos1"]}')
	parser.add_argument('--pos2', type=float, nargs=3, default=positionDefaults['pos2'],
						metavar=('X', 'Y', 'Z'), help=f'Second position for calibration, default: {positionDefaults["pos2"]}')
	parser.add_argument('--offset', type=float, nargs=2, default=positionDefaults['offset'],
						metavar=('H', 'V'), help="End effector offset: (horizontal, vertical) distance "
						"of the mounted tool from joint3 (see 'docs/img/dobot-geometry.png')"
						f", default: {positionDefaults["offset"]}")
	args = parser.parse_args()

	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(0)

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

	driver = DobotDriver(port)
	driver.Open()
	# driver.Open(timeout=0.3)
	kinematics = DobotKinematics(endEffectorOffset=args.offset)

	if args.mode == 'positions':
		positions_mode(driver, kinematics, (args.pos1, args.pos2))
	elif args.mode == 'continuous':
		continuous_mode(driver, kinematics)

if __name__ == '__main__':
	main()
