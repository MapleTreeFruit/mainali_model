import gudhi as gd
import numpy as np
from scipy.stats import norm

from .analysis import PlaceFieldComparison
from .place_fields_1d import PlaceField1D


class PlaceField2D:
    """
    Class for analyzing 2D place fields data with filtering options.

    This class processes a firing rate map for place field analysis in a 2D space.
    It supports filtering of the firing rate map to adjust for noise or thresholds.

    Attributes:
        firing_rate_map (np.ndarray): A processed numpy array representing the place fields.
        x_length (int): The length of the x dimension.
        y_length (int): The length of the y dimension.
    """

    def __init__(self, firing_rate_map, area, filtering=0):
        """
        Initializes the PlaceField2D object with place fields data and dimensions.

        The firing rate map is processed by applying a maximum filter with the specified threshold.

        Args:
            firing_rate_map (np.ndarray): A numpy array of shape (N, x, y) representing the firing rates.
            x_length (int): The length of the x dimension.
            y_length (int): The length of the y dimension.
            filtering (float, optional): The threshold for filtering the firing rate map. Defaults to 0.
        """
        if not isinstance(firing_rate_map, np.ndarray) or firing_rate_map.ndim != 3:
            raise ValueError("firing_rate_map must be a 3D numpy array")

        self.unfiltered_firing_rate_map = firing_rate_map
        self.firing_rate_map = np.maximum(firing_rate_map - filtering, 0)
        self.x_length = self.firing_rate_map.shape[1]
        self.y_length = self.firing_rate_map.shape[2]
        self.theta = norm.ppf(1 - np.count_nonzero(self.firing_rate_map) / self.firing_rate_map.size)
        self.area = area
        self.filtrations = None
        self.ec_mean = None
        self.ec_err = None

    def create_1d_slices_arrays(self, cutoff):
        """
        Creates two large numpy arrays from 1D slices of the 2D firing rate map,
        applying a noise filtering mechanism based on a non-zero value count threshold.

        One array is of size (N*x, y) containing slices along the x axis,
        and the other is of size (N*y, x) containing slices along the y axis.
        Slices are only included if their count of non-zero values is greater than the cutoff.

        Args:
            cutoff (int): The minimum count of non-zero values required for a slice to be included.

        Returns:
            A tuple containing two numpy arrays:
            - The first array is of size (N*x, y) after filtering.
            - The second array is of size (N*y, x) after filtering.
        """
        slices_x = []
        slices_y = []

        for cell in range(self.firing_rate_map.shape[0]):
            for i in range(self.x_length):
                # Slicing along the y axis to create a pseudo 1D field
                slice_y = self.firing_rate_map[cell, i, :]
                if np.count_nonzero(slice_y) > cutoff:
                    slices_y.append(slice_y)

            for j in range(self.y_length):
                # Slicing along the x axis to create a pseudo 1D field
                slice_x = self.firing_rate_map[cell, :, j]
                if np.count_nonzero(slice_x) > cutoff:
                    slices_x.append(slice_x)

        x_slices = PlaceField1D(np.array(slices_x), self.x_length / 10)
        y_slices = PlaceField1D(np.array(slices_y), self.y_length / 10)
        slice_comparison = PlaceFieldComparison(x_slices, y_slices)
        slice_comparison.analyze()
        return slice_comparison

    def _get_euler_characteristic_2d(self, data, filtrations):
        """
        Calculate the Euler characteristic for a given 2D data set.

        Args:
            data (numpy.ndarray): The 2D data for which to calculate the EC.
            filtrations (numpy.ndarray): Filtration values for persistence calculation.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: Euler characteristic and Betti numbers.
        """
        numPoints = filtrations.shape[0]
        cubeplex = gd.CubicalComplex(
            dimensions=np.shape(data),
            top_dimensional_cells=np.ndarray.flatten(data),
        )
        cubeplex.persistence()
        b = np.zeros((numPoints, 3))
        ec = np.zeros(numPoints)

        for i, fval in enumerate(np.flip(filtrations)):
            betti = cubeplex.persistent_betti_numbers(fval, fval)
            b[i] = [betti[0], betti[1], betti[2]]
            ec[i] = betti[0] - betti[1] + betti[2]

        return ec, b

    def analyze_euler_characteristic_2d(self, num_points=1000, filtration_start=-10, filtration_end=10):
        """
        Analyze the Euler characteristic across the 2D firing rate map.

        Args:
            num_points (int): Number of points in the filtration range.
            filtration_start (float): Starting value of the filtration range.
            filtration_end (float): Ending value of the filtration range.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Filtration values, average EC, and standard error of EC.
        """
        filtrations = np.linspace(filtration_start, filtration_end, num_points)
        ec_data, b_data = zip(
            *[self._get_euler_characteristic_2d(cell_map, filtrations) for cell_map in self.unfiltered_firing_rate_map]
        )

        ec_avg_data = np.mean(ec_data, axis=0)
        ec_err_data = np.std(ec_data, axis=0) / np.sqrt(self.firing_rate_map.shape[0])
        b_avg_data = np.mean(b_data, axis=0)

        self.filtrations = filtrations
        self.ec_mean = ec_avg_data
        self.ec_err = ec_err_data

        return filtrations, ec_avg_data, ec_err_data, b_avg_data
