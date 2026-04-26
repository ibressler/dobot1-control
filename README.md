# Dobot v1 Control

Control a Dobot v1 arm (2016 model) with Python, based on the work (and alternate firmware of
[maxosprojects](maxosprojects/open-dobot)).

It's further development of the [DobotSDK package](maxosprojects/open-dobot/application/python/dobot), adding:
- motion planning, move through a series of points with the highest possible speed allowed by the given maximum velocity and acceleration, full stops on direction changes only.
- SCA1000 accelerometers support (different conversion factors)
- limits (WIP)
