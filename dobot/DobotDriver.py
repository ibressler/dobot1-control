"""
open-dobot driver.

Implements driver to open firmware that controls Dobot FPGA.
Abstracts communication protocol, CCITT CRC, and commands sent to FPGA.
Find firmware and SDK at https://github.com/maxosprojects/open-dobot

Author: maxosprojects (March 18, 2016)
Additional Authors: Ingo Breßler (April 12, 2026), <put your name here>

Version: 1.2.2

License: MIT
"""
import numpy as np
import serial
import threading
import time
from serial import SerialException
import math
import sys
from dobot.DobotBase import DobotBase, BASE, REAR, FRONT, JOINT_NAME

# Workaround to support Python 2/3
if sys.version_info > (3,):
    long = int

_max_trys = 1

CMD_READY = 0
CMD_STEPS = 1
CMD_EXEC_QUEUE = 2
CMD_GET_ACCELS = 3
CMD_SWITCH_TO_ACCEL_REPORT_MODE = 4
CMD_CALIBRATE_JOINT = 5
CMD_EMERGENCY_STOP = 6
CMD_SET_COUNTERS = 7
CMD_GET_COUNTERS = 8
CMD_LASER_ON = 9
CMD_PUMP_ON = 10
CMD_VALVE_ON = 11
CMD_BOARD_VERSION = 12


class DobotDriver(DobotBase):
    def __init__(self, comport, rate=115200, sca1000Sensors=False):
        """
        Initializes a serial communication object.

        This class is responsible for setting up and managing a serial port
        connection given a communication port and a baud rate. It facilitates
        communication between the Dobot controller and your PC.

        :param comport: The name or path (on Linux) of the communication port (e.g., COM3, ttyUSB0, etc.).
        :type comport: str
        :param rate: The baud rate for the communication. Defaults to 115200.
        :type rate: int
        :param sca1000Sensors: Whether the Dobot has SCA1000-D01 sensors installed. Defaults to False.
        :type sca1000Sensors: bool
        """
        self._accelConversion = 493.56
        if sca1000Sensors:
            # on a Dobot with SCA1000-D01 sensors, the maximum accelerometer value of the rear arm
            # when vertical (90°) is 1538, value at 0° is 1024, so the offset is fine here
            self._accelConversion = 514.
        self._lock = threading.Lock()
        self._comport = comport
        self._rate = rate
        self._port = None
        self._crc = 0xFFFF
        self.FPGA = 0
        self.RAMPS = 1
        self._toolRotation = 0
        self._gripper = 480

    def Open(self, timeout=0.025):
        """
        Opens the serial port and identifies the board version.

        :param timeout: Serial port read timeout in seconds. Defaults to 0.025.
        :type timeout: float
        """
        try:
            self._port = SerialAggregator(
                serial.Serial(self._comport, baudrate=self._rate, timeout=timeout, inter_byte_timeout=0.1)
            )
            # self._port = serial.Serial(self._comport, baudrate=self._rate, timeout=timeout, interCharTimeout=0.1)

            # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2)
            # s.connect(("localhost", 5555))
            # self._port = serial2socket(s)

            # Have to wait for Arduino initialization to finish, or else it doesn't boot.
            time.sleep(2)
        except SerialException as e:
            self._debug(e, level=0)
            sys.exit(1)

        ret = (0, 0)
        i = 200
        while not ret[0] and i > 0:
            ret = self.BoardVersion()
            i -= 1
        if i == 0:
            self._debug("Cannot get board version! Giving up.", level=0)
            sys.exit(1)

        self._ramps = bool(ret[1])
        if self._ramps:
            self._debug("Board: RAMPS", level=0)
        else:
            self._debug("Board: FPGA", level=0)

        if self._ramps:
            self._stepCoeff = 20000
            self._stopSeq = self.reverseBits32(0)
        else:
            self._stepCoeff = 500000
            self._stopSeq = 0x0242F000
        self._stepCoeffOver2 = self._stepCoeff / 2
        self._freqCoeff = self._stepCoeff * 25

    def Close(self):
        """
        Closes the serial port connection.
        """
        self._port.close()

    def _crc_clear(self):
        """
        Resets the CRC value to its initial state (0xFFFF).
        """
        self._crc = 0xFFFF

    def _crc_update(self, data):
        """
        Updates the current CRC value with a new byte of data.

        :param data: The byte of data to include in the CRC calculation.
        :type data: int
        """
        self._crc = self._crc ^ (data << 8)
        for bit in range(0, 8):
            if (self._crc & 0x8000) == 0x8000:
                self._crc = (self._crc << 1) ^ 0x1021
            else:
                self._crc = self._crc << 1

    def _readchecksumword(self):
        """
        Reads a 2-byte checksum (CRC) from the serial port.

        :return: A tuple where the first element is a success flag (1 for success, 0 for failure),
                 and the second element is the 16-bit CRC value.
        :rtype: tuple[int, int]
        """
        data = self._port.read(2)
        if len(data) == 2:
            arr = bytearray(data)
            crc = (arr[0] << 8) | arr[1]
            return 1, crc
        return 0, 0

    def _readbyte(self):
        """
        Reads a single byte from the serial port and updates the CRC.

        :return: A tuple where the first element is a success flag (1 for success, 0 for failure),
                 and the second element is the byte value.
        :rtype: tuple[int, int]
        """
        data = self._port.read(1)
        if len(data):
            val = bytearray(data)[0]
            self._crc_update(val)
            return 1, val
        return 0, 0

    def _readword(self):
        """
        Reads a 2-byte unsigned word from the serial port.

        :return: A tuple where the first element is a success flag (1 for success, 0 for failure),
                 and the second element is the 16-bit unsigned integer value.
        :rtype: tuple[int, int]
        """
        val1 = self._readbyte()
        if val1[0]:
            val2 = self._readbyte()
            if val2[0]:
                return 1, val1[1] << 8 | val2[1]
        return 0, 0

    def _readsword(self):
        """
        Reads a 2-byte signed word from the serial port.

        :return: A tuple where the first element is a success flag (1 for success, 0 for failure),
                 and the second element is the 16-bit signed integer value.
        :rtype: tuple[int, int]
        """
        val = self._readword()
        if val[0]:
            if val[1] & 0x8000:
                return val[0], val[1] - 0x10000
            return val[0], val[1]
        return 0, 0

    def _readlong(self):
        """
        Reads a 4-byte unsigned long from the serial port.

        :return: A tuple where the first element is a success flag (1 for success, 0 for failure),
                 and the second element is the 32-bit unsigned integer value.
        :rtype: tuple[int, int]
        """
        val1 = self._readbyte()
        if val1[0]:
            val2 = self._readbyte()
            if val2[0]:
                val3 = self._readbyte()
                if val3[0]:
                    val4 = self._readbyte()
                    if val4[0]:
                        return 1, val1[1] << 24 | val2[1] << 16 | val3[1] << 8 | val4[1]
        return 0, 0

    def _readslong(self):
        """
        Reads a 4-byte signed long from the serial port.

        :return: A tuple where the first element is a success flag (1 for success, 0 for failure),
                 and the second element is the 32-bit signed integer value.
        :rtype: tuple[int, int]
        """
        val = self._readlong()
        if val[0]:
            if val[1] & 0x80000000:
                return val[0], val[1] - 0x100000000
            return val[0], val[1]
        return 0, 0

    def _read1(self, cmd):
        """
        Sends a command and reads one byte in response.

        :param cmd: The command byte to send.
        :type cmd: int
        :return: A tuple containing success flag and the byte read.
        :rtype: tuple[int, int]
        """
        return self._read(cmd, [self._readbyte])

    def _read22(self, cmd):
        """
        Sends a command and reads two unsigned words in response.

        :param cmd: The command byte to send.
        :type cmd: int
        :return: A tuple containing success flag and two unsigned words.
        :rtype: tuple[int, int, int]
        """
        return self._read(cmd, [self._readword, self._readword])

    def _reads22(self, cmd):
        """
        Sends a command and reads two signed words in response.

        :param cmd: The command byte to send.
        :type cmd: int
        :return: A tuple containing success flag and two signed words.
        :rtype: tuple[int, int, int]
        """
        return self._read(cmd, [self._readsword, self._readsword])

    def _reads222222(self, cmd):
        """
        Sends a command and reads six signed words in response.

        :param cmd: The command byte to send.
        :type cmd: int
        :return: A tuple containing success flag and six signed words.
        :rtype: tuple[int, int, int, int, int, int, int]
        """
        return self._read(
            cmd, [self._readsword, self._readsword, self._readsword, self._readsword, self._readsword, self._readsword]
        )

    def _read4(self, cmd):
        """
        Sends a command and reads one unsigned long in response.

        :param cmd: The command byte to send.
        :type cmd: int
        :return: A tuple containing success flag and the unsigned long read.
        :rtype: tuple[int, int]
        """
        return self._read(cmd, [self._readlong])

    def _read41(self, cmd):
        """
        Sends a command and reads one signed long and one byte in response.

        :param cmd: The command byte to send.
        :type cmd: int
        :return: A tuple containing success flag, one signed long and one byte.
        :rtype: tuple[int, int, int]
        """
        return self._read(cmd, [self._readslong, self._readbyte])

    def _reads444(self, cmd):
        """
        Sends a command and reads three signed longs in response.

        :param cmd: The command byte to send.
        :type cmd: int
        :return: A tuple containing success flag and three signed longs.
        :rtype: tuple[int, int, int, int]
        """
        return self._read(cmd, [self._readslong, self._readslong, self._readslong])

    def _read(self, cmd, read_commands=None):
        """
        Generic read command that sends a command byte and executes a list of read functions.

        :param cmd: The command byte to send.
        :type cmd: int
        :param read_commands: A list of functions to call to read the response data. Defaults to None.
        :type read_commands: list[callable] | None
        :return: A tuple where the first element is success (1 or 0), followed by the values read.
        :rtype: tuple
        """
        if read_commands is None:
            read_commands = []
        trys = _max_trys
        while trys:
            self._sendcommand(cmd)
            self._writechecksum()
            self._port.send()

            ret = [1]
            for c in read_commands:
                val = c()
                if not val[0]:
                    return tuple([0] * (len(read_commands) + 1))
                ret.append(val[1])

            crc = self._readchecksumword()
            if crc[0]:
                if self._crc & 0xFFFF == crc[1] & 0xFFFF:
                    return tuple(ret)
            trys -= 1
        return tuple([0] * (len(read_commands) + 1))

    def _writebyte(self, val):
        """
        Writes a single byte to the serial port and updates the CRC.

        :param val: The byte value to write.
        :type val: int
        """
        self._crc_update(val & 0xFF)
        self._port.write(bytearray([val & 0xFF]))

    def _writeword(self, val):
        """
        Writes a 2-byte unsigned word to the serial port.

        :param val: The word value to write.
        :type val: int
        """
        self._writebyte((val >> 8) & 0xFF)
        self._writebyte(val & 0xFF)

    def _writelong(self, val):
        """
        Writes a 4-byte unsigned long to the serial port.

        :param val: The long value to write.
        :type val: int
        """
        self._writebyte((val >> 24) & 0xFF)
        self._writebyte((val >> 16) & 0xFF)
        self._writebyte((val >> 8) & 0xFF)
        self._writebyte(val & 0xFF)

    def _writechecksum(self):
        """
        Writes the current 16-bit CRC to the serial port.
        """
        self._port.write(bytearray([(self._crc >> 8) & 0xFF]))
        self._port.write(bytearray([self._crc & 0xFF]))

    def _sendcommand(self, command):
        """
        Clears the CRC and sends a command byte.

        :param command: The command byte to send.
        :type command: int
        """
        self._crc_clear()
        self._writebyte(command)

    def _write(self, cmd, write_commands=None):
        """
        Generic write command that sends a command byte and executes a list of write functions.

        :param cmd: The command byte to send.
        :type cmd: int
        :param write_commands: A list of tuples, each containing a write function and a value.
        :type write_commands: list[tuple[callable, int]] | None
        :return: True if the command and data were successfully acknowledged by matching CRC, False otherwise.
        :rtype: bool
        """
        if write_commands is None:
            write_commands = []
        trys = _max_trys
        while trys:
            self._sendcommand(cmd)

            for c in write_commands:
                c[0](c[1])

            self._writechecksum()
            self._port.send()
            crc = self._readchecksumword()
            if crc[0]:
                if self._crc & 0xFFFF == crc[1] & 0xFFFF:
                    return True
            trys -= 1
        return False

    def _write0(self, cmd):
        """
        Sends a command byte with no additional data.

        :param cmd: The command byte to send.
        :type cmd: int
        :return: True if success, False otherwise.
        :rtype: bool
        """
        return self._write(cmd)

    def _write1(self, cmd, val):
        """
        Sends a command byte followed by one byte of data.

        :param cmd: The command byte to send.
        :type cmd: int
        :param val: The byte value to write.
        :type val: int
        :return: True if success, False otherwise.
        :rtype: bool
        """
        return self._write(cmd, [(self._writebyte, val)])

    def _write2(self, cmd, val):
        """
        Sends a command byte followed by one 2-byte word of data.

        :param cmd: The command byte to send.
        :type cmd: int
        :param val: The word value to write.
        :type val: int
        :return: True if success, False otherwise.
        :rtype: bool
        """
        return self._write(cmd, [(self._writeword, val)])

    def _write4(self, cmd, val):
        """
        Sends a command byte followed by one 4-byte long of data.

        :param cmd: The command byte to send.
        :type cmd: int
        :param val: The long value to write.
        :type val: int
        :return: True if success, False otherwise.
        :rtype: bool
        """
        return self._write(cmd, [(self._writelong, val)])

    def _write14(self, cmd, val1, val2):
        """
        Sends a command byte followed by one byte and one 4-byte long of data.

        :param cmd: The command byte to send.
        :type cmd: int
        :param val1: The byte value to write.
        :type val1: int
        :param val2: The long value to write.
        :type val2: int
        :return: True if success, False otherwise.
        :rtype: bool
        """
        return self._write(cmd, [(self._writebyte, val1), (self._writelong, val2)])

    def _write14411(self, cmd, val1, val2, val3, val4):
        """
        Sends a command byte followed by two longs and two bytes of data.

        :param cmd: The command byte to send.
        :type cmd: int
        :param val1: First long value.
        :type val1: int
        :param val2: Second long value.
        :type val2: int
        :param val3: First byte value.
        :type val3: int
        :param val4: Second byte value.
        :type val4: int
        :return: True if success, False otherwise.
        :rtype: bool
        """
        return self._write(
            cmd, [(self._writelong, val1), (self._writelong, val2), (self._writebyte, val3), (self._writebyte, val4)]
        )

    def _write444(self, cmd, val1, val2, val3):
        """
        Sends a command byte followed by three 4-byte longs of data.

        :param cmd: The command byte to send.
        :type cmd: int
        :param val1: First long value.
        :type val1: int
        :param val2: Second long value.
        :type val2: int
        :param val3: Third long value.
        :type val3: int
        :return: True if success, False otherwise.
        :rtype: bool
        """
        return self._write(cmd, [(self._writelong, val1), (self._writelong, val2), (self._writelong, val3)])

    def _write_read(self, cmd, write_commands):
        """
        Generic write-read command that sends data and reads a single byte response.

        :param cmd: The command byte to send.
        :type cmd: int
        :param write_commands: A list of tuples, each containing a write function and a value.
        :type write_commands: list[tuple[callable, int]]
        :return: A tuple where the first element is success (1 or 0), and the second is the byte read.
        :rtype: tuple[int, int]
        """
        tries = _max_trys
        while tries:
            self._sendcommand(cmd)

            for c in write_commands:
                c[0](c[1])

            self._writechecksum()
            self._port.send()
            ret = self._readbyte()
            if ret[0]:
                crc = self._readchecksumword()
                if crc[0]:
                    if self._crc & 0xFFFF != crc[1] & 0xFFFF:
                        # raise Exception('crc differs', self._crc, crc)
                        return 0, 0
                    return 1, ret[1]
            tries -= 1
        return 0, 0

    def _write1read1(self, cmd, val1):
        """
        Sends a command byte and one byte of data, then reads one byte response.

        :param cmd: The command byte to send.
        :type cmd: int
        :param val1: The byte value to write.
        :type val1: int
        :return: A tuple containing success flag and the byte read.
        :rtype: tuple[int, int]
        """
        return self._write_read(cmd, [(self._writebyte, val1)])

    def _write11121read1(self, cmd, val1, val2, val3, val4, val5):
        """
        Sends a command followed by multiple data types, then reads one byte response.

        :param cmd: The command byte to send.
        :type cmd: int
        :param val1: First byte value.
        :type val1: int
        :param val2: Second byte value.
        :type val2: int
        :param val3: Third byte value.
        :type val3: int
        :param val4: One 2-byte word value.
        :type val4: int
        :param val5: Fourth byte value.
        :type val5: int
        :return: A tuple containing success flag and the byte read.
        :rtype: tuple[int, int]
        """
        return self._write_read(
            cmd,
            [
                (self._writebyte, val1),
                (self._writebyte, val2),
                (self._writebyte, val3),
                (self._writeword, val4),
                (self._writebyte, val5),
            ],
        )

    def _write14441read1(self, cmd, val1, val2, val3, val4):
        """
        Sends a command followed by three longs and one byte, then reads one byte response.

        :param cmd: The command byte to send.
        :type cmd: int
        :param val1: First long value.
        :type val1: int
        :param val2: Second long value.
        :type val2: int
        :param val3: Third long value.
        :type val3: int
        :param val4: One byte value.
        :type val4: int
        :return: A tuple containing success flag and the byte read.
        :rtype: tuple[int, int]
        """
        return self._write_read(
            cmd, [(self._writelong, val1), (self._writelong, val2), (self._writelong, val3), (self._writebyte, val4)]
        )

    def _write1444122read1(self, cmd, val1, val2, val3, val4, val5, val6):
        """
        Sends a command followed by three longs, one byte, and two words, then reads one byte response.

        :param cmd: The command byte to send.
        :type cmd: int
        :param val1: First long value.
        :type val1: int
        :param val2: Second long value.
        :type val2: int
        :param val3: Third long value.
        :type val3: int
        :param val4: One byte value.
        :type val4: int
        :param val5: First word value.
        :type val5: int
        :param val6: Second word value.
        :type val6: int
        :return: A tuple containing success flag and the byte read.
        :rtype: tuple[int, int]
        """
        return self._write_read(
            cmd,
            [
                (self._writelong, val1),
                (self._writelong, val2),
                (self._writelong, val3),
                (self._writebyte, val4),
                (self._writeword, val5),
                (self._writeword, val6),
            ],
        )

    def reverseBits32(self, val):
        """
        Reverses the byte order of a 32-bit integer.

        :param val: The 32-bit integer to reverse.
        :type val: int
        :return: The reversed 32-bit integer.
        :rtype: int
        """
        ### return long(bin(val&0xFFFFFFFF)[:1:-1], 2)
        # return int('{0:032b}'.format(val)[::-1], 2)
        # Not reversing bits in bytes anymore as SPI switched to LSB first.
        # But still need to reverse bytes places.
        return (
            ((val & 0x000000FF) << 24)
            | ((val & 0x0000FF00) << 8)
            | ((val & 0x00FF0000) >> 8)
            | ((val & 0xFF000000) >> 24)
        )

    def reverseBits16(self, val):
        """
        Reverses the byte order of a 16-bit integer.

        :param val: The 16-bit integer to reverse.
        :type val: int
        :return: The reversed 16-bit integer.
        :rtype: int
        """
        # Not reversing bits in bytes anymore as SPI switched to LSB first.
        # But still need to reverse bytes places.
        return ((val & 0x00FF) << 8) | ((val & 0xFF00) >> 8)

    def freqToCmdVal(self, freq):
        """
        Converts stepping frequency into a command value that dobot takes.

        :param freq: The stepping frequency in Hz.
        :type freq: float | int
        :return: The command value to send to Dobot.
        :rtype: int
        """
        if freq == 0:
            return self._stopSeq
        return self.reverseBits32(long(self._freqCoeff / freq))

    def stepsToCmdVal(self, steps):
        """
        Converts the number of steps for dobot to do in 20ms into a command value that dobot
        takes to set the stepping frequency.

        :param steps: The number of steps to perform in 20ms.
        :type steps: int
        :return: The command value to send to Dobot.
        :rtype: int
        """
        if steps == 0:
            return self._stopSeq
        return self.reverseBits32(long(self._stepCoeff / steps))

    def stepsToCmdValFloat(self, steps):
        """
        Converts the number of steps for dobot to do in 20ms into a command value that dobot
        takes to set the stepping frequency.

        :param steps: float number of steps; float to minimize error and have finer control
        :type steps: float
        :return: tuple (command_value, actualSteps, leftover), where leftover is the fractioned steps that don't fit
                into the 20ms interval a command runs for
        :rtype: tuple[int, int, float]
        """
        if abs(steps) < 0.01:
            return self._stopSeq, 0, 0.0
        # "round" makes leftover negative in certain cases and causes backlash compensation to oscillate.
        # actualSteps = long(round(steps))
        actualSteps = long(steps)
        if actualSteps == 0:
            return self._stopSeq, 0, steps
        val = long(self._stepCoeff / actualSteps)
        actualSteps = long(self._stepCoeff / val)
        if val == 0:
            return self._stopSeq, 0, steps
        return self.reverseBits32(val), actualSteps, steps - actualSteps

    def accelToRadians(self, val, offset):
        """
        Converts accelerometer raw value to radians.

        :param val: Raw accelerometer value.
        :type val: int
        :param offset: Calibration offset for the accelerometer.
        :type offset: int
        :return: The angle in radians.
        :rtype: float
        """
        try:
            return math.asin(float(val - offset) / self._accelConversion)
        except ValueError:
            return np.pi*.5

    @staticmethod
    def accel3DXToRadians(x, y, z):
        """
        Converts 3-axis accelerometer raw values to radians using atan2.

        :param x: Raw accelerometer X value.
        :type x: int | float
        :param y: Raw accelerometer Y value.
        :type y: int | float
        :param z: Raw accelerometer Z value.
        :type z: int | float
        :return: The angle in radians.
        :rtype: float
        """
        try:
            xf = float(x)
            yf = float(y)
            zf = float(z)
            return math.atan2(xf, math.sqrt(yf * yf + zf * zf))
        except ValueError:
            return np.pi*.5

    def CalibrateJoint(self, joint, forwardCommand, backwardCommand, direction, pin, pinMode, pullup):
        """
        Initiates joint calibration procedure using a limit switch/photointerrupter. Effective immediately.
        The current command buffer is cleared.
        Cancel the procedure by issuing EmergencyStop() is necessary.

        :param joint: which joint to calibrate: 1-3
        :type joint: int
        :param forwardCommand: command to send to the joint when moving forward (towards limit switch);
                use freqToCmdVal()
        :type forwardCommand: int
        :param backwardCommand: command to send to the joint after hitting  (towards limit switch);
                use freqToCmdVal()
        :type backwardCommand: int
        :param direction: direction to move joint towards limit switch/photointerrupter: 0-1
        :type direction: int
        :param pin: firmware internal pin reference number that limit switch is connected to;
                    refer to dobot.h -> calibrationPins
        :type pin: int
        :param pinMode: limit switch/photointerrupter normal LOW = 0, normal HIGH = 1
        :type pinMode: int
        :param pullup: enable pullup on the pin = 1, disable = 0
        :type pullup: int
        :return: True if command successfully received, False otherwise.
        :rtype: bool
        """
        if 1 > joint > 3:
            return False
        control = ((pinMode & 0x01) << 4) | ((pullup & 0x01) << 3) | ((direction & 0x01) << 2) | ((joint - 1) & 0x03)
        self._lock.acquire()
        result = self._write14411(CMD_CALIBRATE_JOINT, forwardCommand, backwardCommand, pin, control)
        self._lock.release()
        return result

    def EmergencyStop(self):
        """
        Stops the arm in case of emergency. Clears command buffer and cancels calibration procedure.

        :return: True if the command was successfully received, False otherwise.
        :rtype: bool
        """
        self._lock.acquire()
        result = self._write0(CMD_EMERGENCY_STOP)
        self._lock.release()
        return result

    def Steps(self, jointCmd, jointDir, servoGrab, servoRot):
        """
        Adds a command to the controller's queue to execute on FPGA.

        :param jointCmd: Joint subcommands as a list of 3 integers
        :type jointCmd: list[int]
        :param jointDir: Direction for each joint as a list of 3 integers: 0-1
        :type jointDir: list[int]
        :param servoGrab: servoGrab position (gripper): 0x00d0-0x01e0 (or 208-480 decimal)
        :type servoGrab: int
        :param servoRot: servoRot position (tool rotation): 0x0000-0x0400 (or 0-1024 decimal)
        :type servoRot: int
        :return: a tuple where the first element tells whether the command has been successfully
            received (1 - received, 0 - timed out), and the second element tells whether the command was added
            to the controller's command queue (1 - added, 0 - not added, as the queue was full).
        :rtype: tuple[int, int]
        """
        control = (int(jointDir[0]) & 0x01) | ((int(jointDir[1]) & 0x01) << 1) | ((int(jointDir[2]) & 0x01) << 2)
        self._lock.acquire()
        if servoGrab > 480:
            servoGrab = 480
        elif servoGrab < 208:
            servoGrab = 208
        if servoRot > 1024:
            servoRot = 1024
        elif servoRot < 0:
            servoRot = 0

        if self._ramps:
            servoRot *= 2
            servoRot += 2000
            servoGrab *= 2
            servoGrab += 2000

        self._toolRotation = servoRot
        self._gripper = servoGrab

        result = self._write1444122read1(
            CMD_STEPS, int(jointCmd[0]), int(jointCmd[1]), int(jointCmd[2]),
            control, self.reverseBits16(servoGrab), self.reverseBits16(servoRot)
        )
        self._lock.release()
        return result

    def GetAccelerometers(self):
        """
        Reads data from accelerometers.

        :return: Data acquired from accelerometers at power on.
            There are 17 reads in an FPGA version and 20 reads in the RAMPS version of each accelerometer
            that the firmware does and then averages the result before returning it here.
            Tuple contains (success, x1, y1, z1, x2, y2, z2).
        :rtype: tuple[int, int, int, int, int, int, int]
        """
        self._lock.acquire()
        result = self._reads222222(CMD_GET_ACCELS)
        self._lock.release()
        return result

    def GetCounters(self):
        """
        Reads the current step counters for all three joints.

        :return: A tuple (success, base_counter, rear_counter, fore_counter).
        :rtype: tuple[int, int, int, int]
        """
        self._lock.acquire()
        result = self._reads444(CMD_GET_COUNTERS)
        self._lock.release()
        return result

    def SetCounters(self, base, rear, fore):
        """
        Sets the step counters for all three joints.

        :param base: Step counter for the base joint.
        :type base: int
        :param rear: Step counter for the rear joint.
        :type rear: int
        :param fore: Step counter for the fore joint.
        :type fore: int
        :return: True if the command was successfully received, False otherwise.
        :rtype: bool
        """
        self._lock.acquire()
        result = self._write444(CMD_SET_COUNTERS, base, rear, fore)
        self._lock.release()
        return result

    def SwitchToAccelerometerReportMode(self):
        """
        Switches dobot to accelerometer report mode.

        Apparently the following won't work because of the way dobot was designed
        and limitations of AVR - cannot switch SPI from Slave to Master back.
        So, as a workaround, hold the "Sensor Calibration" button and start your
        app. Arduino is reset on serial port connection, and it takes about 2 seconds
        for it to start. After that you can release the button. That switches dobot to
        accelerometer reporting mode. To move the arm, turn off the power switch.

        This function is left just in case a proper fix comes up.

        Dobot must be reset to enter normal mode after issuing this command.

        :raises NotImplementedError: Always, as this functionality is not currently working as intended.
        """
        raise NotImplementedError("Read function description for more info")

    def LaserOn(self, on):
        """
        Turns the laser on or off.

        :param on: True to turn the laser on, False to turn it off.
        :type on: bool
        :return: Returns a tuple where the first element tells whether the command has been successfully
            received (0 - yes, 1 - timed out), and the second element tells whether the command was added
            to the controller's command queue (1 - added, 0 - not added, as the queue was full).
        :rtype: tuple[int, int]
        """
        self._lock.acquire()
        if on:
            result = self._write1read1(CMD_LASER_ON, 1)
        else:
            result = self._write1read1(CMD_LASER_ON, 0)
        self._lock.release()
        return result

    def PumpOn(self, on):
        """
        Turn On/Off the pump motor, if you want to actually grip something,
        you also need to turn the valve on otherwise the air flows through it.

        This method interacts with the hardware to enable or disable the pump
        using a specific command. The method ensures thread safety by acquiring
        and releasing a lock during the operation.

        :param on: Specifies the pump state as a boolean. If True, the pump is turned on.
                   If False, the pump is turned off.
        :type on: bool
        :return: Returns a tuple where the first element tells whether the command has been successfully
            received (0 - yes, 1 - timed out), and the second element tells whether the command was added
            to the controller's command queue (1 - added, 0 - not added, as the queue was full).
        :rtype: Any
        """
        self._lock.acquire()
        if on:
            result = self._write1read1(CMD_PUMP_ON, 1)
        else:
            result = self._write1read1(CMD_PUMP_ON, 0)
        self._lock.release()
        return result

    def ValveOn(self, on):
        """
        Turns the valve on or off.
        Does little by itself but work in tandem with the pump, when the pump is on turning the valve
        on will allow you to grab things, turn it off to release the air pressure.

        :param on: True to turn the valve on, False to turn it off.
        :type on: bool
        :return: Returns a tuple where the first element tells whether the command has been successfully
            received (0 - yes, 1 - timed out), and the second element tells whether the command was added
            to the controller's command queue (1 - added, 0 - not added, as the queue was full).
        :rtype: tuple[int, int]
        """
        self._lock.acquire()
        if on:
            result = self._write1read1(CMD_VALVE_ON, 1)
        else:
            result = self._write1read1(CMD_VALVE_ON, 0)
        self._lock.release()
        return result

    def Wait(self, waitTime):
        """
        Makes the arm wait in the current position for the specified period of time.

        The wait period is specified in seconds and can be fractions of seconds.
        The resolution of this command is up to 20ms.

        To make the arm wait, a number of commands are issued to do nothing. Each command takes 20ms
        to execute by the arm.

        :param waitTime: The time to wait in seconds.
        :type waitTime: float | int
        """
        iterations = int(waitTime * 50)
        for i in range(iterations):
            ret = (0, 0)
            # Keep sending until buffered
            while not ret[0] or not ret[1]:
                ret = self.Steps([0, 0, 0], [0, 0, 0], self._gripper, self._toolRotation)

    def BoardVersion(self):
        """
        Checks board version.

        :return: A tuple (success, version), where version=0 is FPGA and version=1 is RAMPS.
        :rtype: tuple[int, int]
        """
        self._lock.acquire()
        result = self._read1(CMD_BOARD_VERSION)
        self._lock.release()
        return result

    def isFpga(self):
        """
        Checks if the board is an FPGA version.

        :return: True if it is FPGA, False otherwise.
        :rtype: bool
        """
        return self._ramps == self.FPGA

    def isRamps(self):
        """
        Checks if the board is a RAMPS version.

        :return: True if it is RAMPS, False otherwise.
        :rtype: bool
        """
        return self._ramps == self.RAMPS

    def isReady(self):
        """
        Checks whether the controller is up and running.

        :return: A tuple containing success flag and the ready status (0x40 if ready).
        :rtype: tuple[int, int]
        """
        self._lock.acquire()
        result = self._read1(CMD_READY)
        self._lock.release()
        # Check for the magic number.
        # return [result[0], result[1] == 0x40]
        return result

    def reset(self):
        """
        Resets the driver state by clearing the CRC and reading any pending data from the port.
        """
        # 		self._lock.acquire()
        i = 0
        while i < 5:
            self._port.read(1)
            i += 1
        self._crc_clear()


# 		self._lock.release()

class SerialAggregator:
    """
    open-dobot serial aggregator.

    This is a workaround to send data in bursts on systems that have slow API
    used by pyserial (e.g. Windows).
    """

    def __init__(self, ser):
        """
        Initializes the SerialAggregator with a serial port object.

        :param ser: The serial port object to wrap.
        :type ser: serial.Serial
        """
        self._ser = ser
        self._buf = bytearray()

    def write(self, data):
        """
        Appends data to the internal buffer instead of writing it immediately.

        :param data: The data to write.
        :type data: bytes | bytearray
        """
        self._buf.extend(data)

    def read(self, size):
        """
        Reads data directly from the serial port.

        :param size: The number of bytes to read.
        :type size: int
        :return: The data read from the serial port.
        :rtype: bytes
        """
        return self._ser.read(size)

    def flushInput(self):
        """
        No-op for compatibility.
        """
        pass

    def flush(self):
        """
        No-op for compatibility.
        """
        pass

    def send(self):
        """
        Sends the buffered data to the serial port in one burst.
        """
        self._ser.write(self._buf)
        self._buf = bytearray()

    def close(self):
        """
        Closes the serial port.
        """
        self._ser.close()
