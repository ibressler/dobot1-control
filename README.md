# Dobot1-Control (v1.5.0)

[![PyPI Package latest release](https://img.shields.io/pypi/v/dobot1-control.svg)](https://test.pypi.org/project/dobot1-control)
[![Commits since latest release](https://img.shields.io/github/commits-since/ibressler/dobot1-control/v1.5.0.svg)](https://github.com/ibressler/dobot1-control/compare/v1.5.0...main)
[![License](https://img.shields.io/pypi/l/dobot1-control.svg)](https://en.wikipedia.org/wiki/MIT_license)
[![Supported versions](https://img.shields.io/pypi/pyversions/dobot1-control.svg)](https://test.pypi.org/project/dobot1-control)
[![PyPI Wheel](https://img.shields.io/pypi/wheel/dobot1-control.svg)](https://test.pypi.org/project/dobot1-control#files)
[![Weekly PyPI downloads](https://img.shields.io/pypi/dw/dobot1-control.svg)](https://test.pypi.org/project/dobot1-control/)
[![Continuous Integration and Deployment Status](https://github.com/ibressler/dobot1-control/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/ibressler/dobot1-control/actions/workflows/ci-cd.yml)
[![Coverage report](https://img.shields.io/endpoint?url=https://ibressler.github.io/dobot1-control/coverage-report/cov.json)](https://ibressler.github.io/dobot1-control/coverage-report/)

Control a Dobot v1 arm (2016 model) with Python, based on the work (and requires [the firmware](https://github.com/maxosprojects/open-dobot/releases/tag/1.3.0)) of [open-dobot by maxosprojects](https://github.com/maxosprojects/open-dobot). Now including motion planning, limits and flexible accelerometers support, tested with SCA1000 additionally.

This is a further development of the [DobotSDK package](https://github.com/maxosprojects/open-dobot/tree/master/application/python/dobot). Tested and developed with the original FPGA hardware and the [open-dobot firmware](https://github.com/maxosprojects/open-dobot/tree/master/firmware).

## Installation

    pip install dobot1-control

You can also install the in-development version with:

    pip install git+https://github.com/ibressler/dobot1-control.git@main

## Documentation

https://ibressler.github.io/dobot1-control

## Usage
### Motion planning

The arm should move through a series of points with the highest possible speed allowed by the given maximum velocity and acceleration, full stops on direction changes only. For a circular path like the one below, it should try to maintain a continuous motion.

<img src="docs/img/circle-run_positions.png" alt="Circle positions" width="30%">

This is what the actual motion looks like now, with the actual speed of each joint over time (steps taken per interval of 50 ms):

![Steps plot along circle](docs/img/circle-run_actual-steps.png)

This example can be found in the [test move.ipynb](notebooks/test_move.ipynb) notebook.

### Free Accelerometer Conversion

 As it seems, the dobot arm was shipped with different accelerometer ICs at different production batches. The previously hardcoded conversion factors were not accurate for the SCA1000 accelerometers found in some Dobot v1 arms. Additionally, the sensor offsets need to be determined for each accelerometer anyway.

To move the arm tool as precisely as its mechanical capabilities allow along a straight line parallel to a flat surface, such as a table, calibrate the accelerometer offsets and conversion factor. The determined values need to be provided to the *Dobot()* constructor:

```python
Dobot("/dev/ttyACM0",
      endEffectorOffset=(49., 64.),
      accelOffset=(1016.39, 1009.08),
      accelConversion=509.00)
```

### Accelerometer Calibration

The `scripts/calibrate-accelerometers.py` tool helps in finding offsets and conversion factors for the installed accelerometers. Once the package was installed, the tool can be run as follows:

#### Usage

```bash
dobot-calibrate-accelerometers [mode] [options]
```

#### Modes

- `continuous` (default): Continuously reports accelerometer data and calculated angles.
- `positions`: Calculates sensor offsets and conversion factors based on two measured positions.

#### Positions Mode Options

- `--pos1 X Y Z`: First position for calibration (default: `120 0 0`).
- `--pos2 X Y Z`: Second position for calibration (default: `320 0 0`).
- `--offset H V`: End effector offset: horizontal and vertical distance of the mounted tool from joint 3 (default: `51 15`).


Follow the procedure below to enable the accelerometer reporting mode on FPGA. No action is required on the RAMPS as GY-521 accelerometers can be read at any time there.

1. Turn off power on the arm and disconnect USB cable
2. Connect USB cable
3. Enable the accelerometer reporting mode:
   1. Press and hold the "Sensor Calibration" button on the FPGA version or ground pin D23 on AUX-4 on the RAMPS version
   2. Press and release the "Reset" button
   3. Start this tool (still holding the "Sensor Calibration" button on the FPGA version or keeping pin D23 grounded on the RAMPS)
   4. Once the constructor ran (or the accelerometer data starts flowing on your console in *continuous* mode),
   5. Release the "Sensor Calibration" button
4. Move the arm around to the positions prompted by the tool.
5. Finally, note the accelerometer offsets and conversion factor values reported on the console. Those are your accelerometers' offsets to be supplied to the *Dobot()* constructor.
6. Disconnect USB cable again to reinit with the new values.

## Development

### Testing

This project uses `pytest` for testing and `tox` for running them in a reproducible environment.

See which tests are available (arguments after `--` get passed to `pytest` which runs the tests):

    tox -e py -- --co

Run a specific test only:

    tox -e py -- -k <test_name from listing before>

Run all tests with:

    tox -e py

### Package Version

Get the next version number and how the GIT history would be interpreted for that:

    pip install python-semantic-release
    semantic-release -v version --print

This prints its interpretation of the commits in detail. Make sure to supply the `--print`
argument to not raise the version number which is done automatically by the *release* job
of the GitHub Action Workflows.

### Project template

Update the project configuration from the *copier* template and make sure the required packages
are installed:

    pip install copier jinja2-time
    copier update --trust --skip-answered

## Copyright

* Copyright © 2016 maxosprojects (original).
* Copyright © 2026 Ingo Breßler (fork).
* Licensed under the MIT License — see LICENSE.
