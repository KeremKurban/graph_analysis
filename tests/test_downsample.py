import unittest
import os
from graph_analysis.coordinate_query import CoordinateQuery
from graph_analysis.downsample import RatCA1SubmatrixGenerator

class RatCA1SubmatrixGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.coordinate_query = CoordinateQuery()
        self.ATLAS_DIR = "examples/"
        self.save_dir = "./test_output/"
        self.generator = RatCA1SubmatrixGenerator(self.coordinate_query, self.ATLAS_DIR)

    def tearDown(self):
        # Clean up any temporary files/directories created during testing
        pass

    def test_generate_lon_slices(self):
        # Specify the slice thickness for testing
        slice_thickness = 0.1

        # Generate the lon slices
        self.generator.generate_lon_slices(slice_thickness, self.save_dir)

        # Assert that at least one lon slice is generated
        lon_slice_dir = os.path.join(self.save_dir, "lon")
        self.assertTrue(os.listdir(lon_slice_dir))


    def test_generate_tra_slices(self):
        # Specify the tra_dt for testing
        tra_dt = 0.2

        # Generate the tra slices
        self.generator.generate_tra_slices(tra_dt, self.save_dir)

        # Assert that the tra slices are generated in the save_dir
        expected_num_slices = 5  # Assuming 5 slices with 0.2 dt
        actual_num_slices = len(os.listdir(os.path.join(self.save_dir, "tra")))
        self.assertEqual(actual_num_slices, expected_num_slices)

    def test_generate_lon_tra_intersection_slices(self):
        # Generate the lon slices and tra slices first for testing intersection
        slice_thickness = 0.1
        tra_dt = 0.2
        self.generator.generate_lon_slices(slice_thickness, self.save_dir)
        self.generator.generate_tra_slices(tra_dt, self.save_dir)

        # Generate the lon-tra intersection slices
        self.generator.generate_lon_tra_intersection_slices(self.save_dir)

        # Assert that the lon-tra intersection slices are generated in the save_dir
        expected_num_slices = 10  # Assuming 10 lon slices and 5 tra slices
        actual_num_slices = len(os.listdir(os.path.join(self.save_dir, "lon_tra_intersection")))
        self.assertEqual(actual_num_slices, expected_num_slices)

if __name__ == "__main__":
    unittest.main()
