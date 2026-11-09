# __init__.py

__version__ = "1.5.1"

from dobot1_control.DobotControl import DobotControl
from dobot1_control.DobotDriver import DobotDriver
from dobot1_control.DobotKinematics import DobotKinematics
from dobot1_control.SVGParser import svg_to_polylines

__all__ = ["DobotControl", "DobotDriver", "DobotKinematics", "svg_to_polylines"]
