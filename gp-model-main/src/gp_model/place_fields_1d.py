import itertools
import operator

import gudhi as gd
import numpy as np
from scipy.stats import norm


def smooth(y, box_pts):
    """Smooth the input array by convolving with a box filter.

    Args:
        y (numpy.ndarray): Input array to smooth.
        box_pts (int): Number of points for the box filter.

    Returns:
        numpy.ndarray: Smoothed array.
    """
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


class PlaceField1D:
    """Class to handle 1D place fields data and perform various analyses."""

    def __init__(self, firing_rate_map, length, filtering=None):
        """Initialize the PlaceField1D object with optional filtering.

        Args:
            firing_rate_map (numpy.ndarray): NxP array representing firing rates.
            length (float): The physical length of the place field.
            filtering (float): The threshold value to filter the firing_rate_map. Default is 0.
        """
        if filtering != None:
            self.firing_rate_map = np.maximum(firing_rate_map - filtering, 0)
        else:
            self.firing_rate_map = firing_rate_map
        # self.firing_rate_map = np.maximum(firing_rate_map - filtering, 0)
        self.length = length
        self.num_cells = firing_rate_map.shape[0]
        self.resolution = self.length / firing_rate_map.shape[1]
        self.theta = norm.ppf(1 - np.count_nonzero(self.firing_rate_map) / self.firing_rate_map.size)
        self.filtrations = None
        self.ec_mean = None
        self.ec_err = None
        self.scaling = None

    def _get_euler_characteristic(self, data, filtrations):
        """Calculate the Euler characteristic for a given data set.

        Args:
            data (numpy.ndarray): The data for which to calculate the EC.
            filtrations (numpy.ndarray): Filtration values for persistence calculation.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: Euler characteristic and Betti numbers.
        """
        cubical_complex = gd.CubicalComplex(
            dimensions=data.shape,
            top_dimensional_cells=data.flatten(),
        )
        cubical_complex.persistence()
        betti_numbers = np.zeros((len(filtrations), 2))
        euler_characteristic = np.zeros(len(filtrations))

        # for i, fval in enumerate(np.flip(filtrations)):
        for i, fval in enumerate(filtrations):
            betti = cubical_complex.persistent_betti_numbers(fval, fval)
            betti_numbers[i] = [betti[0], betti[1]]
            euler_characteristic[i] = betti[0] - betti[1]

        return euler_characteristic, betti_numbers

    def _segment_field(self, field):
        """Segment the firing field into individual place fields and gaps.

        Args:
            field (numpy.ndarray): The firing rate map of a single neuron.

        Returns:
            Tuple: Segmented fields, widths, peak firing rates, and gap widths.
        """
        original = field.copy()
        field_binary = np.where(field != 0, 1, 0).tolist()
        field_complement = np.where(field != 0, 0, 1).tolist()

        idx = [
            [i for i, value in it]
            for key, it in itertools.groupby(enumerate(field_binary), key=operator.itemgetter(1))
            if key != 0
        ]
        idx_prime = [
            [i for i, value in it]
            for key, it in itertools.groupby(enumerate(field_complement), key=operator.itemgetter(1))
            if key != 0
        ]
        segments = [np.asarray([original[i] for i in index_group]) for index_group in idx]
        segments_width = [len(segment) for segment in segments]
        gaps_width = [len(index_group) for index_group in idx_prime]
        peak_firing_rates = [np.max(original[index_group]) for index_group in idx]

        return segments, np.array(segments_width), np.array(peak_firing_rates), np.array(gaps_width)

    def get_single_fields(self, concatenate=True):
        """Get the segments of all place fields along with their properties.

        Args:
            concatenate (bool): Whether to concatenate the results for all neurons.

        Returns:
            Tuple: List of segments and numpy arrays of widths, peak firing rates, and gap widths.
        """
        all_segments = []
        all_widths = []
        all_peak_firing_rates = []
        all_gaps = []

        for i in range(self.num_cells):
            field_segments, widths, peak_firing_rates, gaps = self._segment_field(self.firing_rate_map[i, :])
            all_segments.extend(field_segments)
            all_widths.append(widths * self.resolution)
            all_gaps.append(gaps * self.resolution)
            all_peak_firing_rates.append(peak_firing_rates)

        if concatenate:
            all_widths = np.concatenate(all_widths)
            all_gaps = np.concatenate(all_gaps)
            all_peak_firing_rates = np.concatenate(all_peak_firing_rates)

        return all_segments, all_widths, all_peak_firing_rates, all_gaps

    def analyze_euler_characteristic(self, num_points=1000, filtration_start=-40, filtration_end=0):
        """Analyze the Euler characteristic across the firing rate map.

        Args:
            num_points (int): Number of points in the filtration range.
            filtration_start (float): Starting value of the filtration range.
            filtration_end (float): Ending value of the filtration range.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Filtration values, average EC, and standard error of EC.
        """
        filtrations = np.linspace(filtration_start, filtration_end, num_points)
        ec_data = [self._get_euler_characteristic(-row, filtrations)[0] for row in self.firing_rate_map]
        ec_avg_data = np.mean(ec_data, axis=0)
        ec_err_data = np.std(ec_data, axis=0) / np.sqrt(self.num_cells)

        self.filtrations = filtrations
        self.ec_mean = ec_avg_data
        self.ec_err = ec_err_data

        return filtrations, ec_avg_data, ec_err_data


if __name__ == "__main__":
    firing_rate_data = np.random.rand(10, 100)
    length_of_field = 100
    place_field = PlaceField1D(firing_rate_data, length_of_field)

    segments, widths, peak_firing_rates, gaps = place_field.get_single_fields()
    print("Segments:", segments)
    print("Widths:", widths)
    print("Peak firing rates:", peak_firing_rates)
    print("Gaps:", gaps)

    filtrations, ec_avg, ec_err = place_field.analyze_euler_characteristic()
    print("Euler characteristic analysis:")
    print("Filtrations:", filtrations)
    print("Average EC:", ec_avg)
    print("EC Standard Error:", ec_err)
