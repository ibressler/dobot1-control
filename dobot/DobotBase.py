"""
Base class for Dobot control.

Provides common debugging and utility functions.

Author: Ingo Breßler (April 12, 2026)
Additional Authors: <put your name here>

License: MIT
"""

class DobotBase:
    def class_name(self):
        return type(self).__name__

    def _debug(self, *args, level=1):
        indent = level * 2
        prefix = " " * indent + f"{self.class_name()}:" if level == 0 else ""
        print(prefix, *args)
