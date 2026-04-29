import unittest
import numpy as np
from dobot.DobotKinematics import DobotKinematics

class TestDobotKinematics(unittest.TestCase):
    def setUp(self):
        self.kinematics = DobotKinematics()

    def test_consistency(self):
        # Define some test angles
        test_cases = [
            (   0,   0,   0),
            (  45 , 30, -20),
            ( -45,  60,  10),
            (  90,  10, -45),
            ( 180,  45,  45),
            (-180,   0,   0),
            (   0,  14,  62),
            (   0,  40, -40),
        ]

        for angles in test_cases:
            angles = np.deg2rad(angles)
            with self.subTest(angles=angles):
                # Forward kinematics: angles -> coordinates
                coords = self.kinematics.coordinatesFromAngles(*angles)
                
                # Inverse kinematics: coordinates -> angles
                calculated_angles = self.kinematics.anglesFromCoordinates(coords)
                
                # Check if calculated angles match original angles
                np.testing.assert_allclose(
                    calculated_angles, 
                    angles,
                    atol=1e-7,
                    err_msg=f"Angles mismatch for input {angles}"
                )

                # Round trip: coordinates -> angles -> coordinates
                calculated_coords = self.kinematics.coordinatesFromAngles(*calculated_angles)
                np.testing.assert_allclose(
                    calculated_coords, 
                    coords, 
                    atol=1e-7,
                    err_msg=f"Coordinates mismatch for input {angles}"
                )

if __name__ == '__main__':
    unittest.main()
