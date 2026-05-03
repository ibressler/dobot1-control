"""
Base class for Dobot control.

Provides common debugging and utility functions.

Author: Ingo Breßler (April 12, 2026)
Additional Authors: <put your name here>

License: MIT
"""

import numpy as np


BASE, REAR, FRONT = range(3)
JOINT_NAME = ['base', 'rear', 'front']


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

class DobotBase:
    def class_name(self):
        return type(self).__name__

    def _debug(self, *args, level=1):
        indent = level * 2
        prefix = " " * indent + f"{self.class_name()}:" if level == 0 else ""
        print(prefix, *args)
