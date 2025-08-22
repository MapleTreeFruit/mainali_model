import gudhi as gd
import numpy as np
from scipy import ndimage
from scipy.ndimage import label
from scipy.spatial.distance import pdist
from scipy.stats import norm
from skimage.measure import marching_cubes


class PlaceField3D:
    """
    Class for analyzing 3D place fields data.

    This class processes a firing rate map for place field analysis in a 3D space.
    It supports filtering of the firing rate map to adjust for noise or thresholds.

    Attributes:
        firing_rate_map (np.ndarray): A processed numpy array representing the place fields.
    """

    def __init__(self, firing_rate_map, filtering=0):
        """
        Initializes the PlaceField3D object with place fields data.

        The firing rate map is processed by applying a maximum filter with the specified threshold.

        Args:
            firing_rate_map (np.ndarray): A numpy array of shape (N, x, y, z) representing the firing rates.
            filtering (float, optional): The threshold for filtering the firing rate map. Defaults to 0.
        """
        if not isinstance(firing_rate_map, np.ndarray) or firing_rate_map.ndim != 4:
            raise ValueError("firing_rate_map must be a 4D numpy array")

        self.unfiltered_firing_rate_map = firing_rate_map
        self.firing_rate_map = np.maximum(firing_rate_map - filtering, 0)
        self.smoothed_firing_rate_map = None
        self.shape = self.firing_rate_map.shape[1:]
        self.x_length = self.firing_rate_map.shape[1]
        self.y_length = self.firing_rate_map.shape[2]
        self.z_length = self.firing_rate_map.shape[3]
        self.size = (self.x_length * self.y_length * self.z_length) / 1e3
        self.theta = norm.ppf(1 - np.count_nonzero(self.firing_rate_map) / self.firing_rate_map.size)
        self.scaling = None
        self.filtrations = None
        self.ec_mean = None
        self.ec_err = None

    def analyze_place_fields(self, filter_size=1, outlier_threshold=0.08, threshold=0):
        """
        Analyzes the 3D place fields by identifying connected volumes, smoothing the data,
        and calculating various properties of the place fields.

        Args:
            filter_size (int): The size of the median filter. Defaults to 1.
            threshold (float): The threshold value for identifying connected volumes. Defaults to 0.

        Returns:
            dict: A dictionary containing various measurements and properties of the place fields.
        """
        # Apply median filter and identify connected volumes
        smoothed_data = [ndimage.median_filter(arr, size=filter_size) for arr in self.firing_rate_map]
        labeled_data = [self._identify_connected_volumes(arr, threshold) for arr in smoothed_data]

        # Initialize lists to store measurements
        counts, volumes, max_vals, volumes_per_cell, max_vals_per_cell, max_val_locs_all = [], [], [], [], [], []
        pairwise_dists_per_cell = []

        cleaned_data = []
        # Process each neuron's data
        for arr, (labels, num_features) in zip(smoothed_data, labeled_data):
            counts.append(num_features)
            tmp_volumes = []
            tmp_max_vals = []

            temp_arr = arr.copy()
            # Process each feature within neuron's data
            for feature_id in range(1, num_features + 1):
                mask = labels == feature_id
                feature_volume = np.sum(mask) / arr.size
                volumes.append(feature_volume)
                tmp_volumes.append(feature_volume)

                max_val = np.max(arr[mask])
                max_vals.append(max_val)
                tmp_max_vals.append(max_val)

                max_val_locs = np.argwhere(arr == max_val)
                max_val_locs_all.extend(max_val_locs)

                # Compute pairwise distances if more than one max location
                if len(max_val_locs) > 1:
                    pairwise_dists = pdist(np.array(max_val_locs))
                    pairwise_dists_per_cell.extend(pairwise_dists)

                if np.sum(mask) / arr.size > outlier_threshold:
                    temp_arr[mask] = 0

            cleaned_data.append(temp_arr)
            # Store per-cell measurements
            volumes_per_cell.append(tmp_volumes)
            max_vals_per_cell.append(tmp_max_vals)

        # Package results in a dictionary
        self.smoothed_firing_rate_map = np.asarray(cleaned_data)
        results = {
            "volumes": np.asarray(volumes),
            "max_firing_rates": np.asarray(max_vals),
            "counts": np.asarray(counts),
            "volumes_per_cell": volumes_per_cell,
            "max_vals_per_cell": max_vals_per_cell,
            "max_val_locs_all": np.asarray(max_val_locs_all),
            "pairwise_dists_per_cell": np.asarray(pairwise_dists_per_cell),
        }

        return results

    def _identify_connected_volumes(self, arr, thresh):
        """
        Identifies connected volumes in an array based on a threshold.

        Args:
            arr (np.ndarray): The input array.
            thresh (float): The threshold value for identifying volumes.

        Returns:
            tuple: A tuple containing the labeled array and the number of features.
        """
        binary_mask = arr > thresh
        labeled, num_features = ndimage.label(binary_mask)
        return labeled, num_features

    def analyze_euler_characteristic_3d(self, num_points=1000, filtration_start=-10, filtration_end=10):
        """
        Analyzes the Euler characteristic across the 3D firing rate map.

        Args:
            num_points (int): Number of points in the filtration range.
            filtration_start (float): Starting value of the filtration range.
            filtration_end (float): Ending value of the filtration range.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Filtration values, average EC, and standard error of EC.
        """
        filtrations = np.linspace(filtration_start, filtration_end, num_points)
        ec_data, b_data = zip(
            *[
                self._get_euler_characteristic_3d(cell_map / self.scaling, filtrations)
                for cell_map in self.firing_rate_map
            ]
        )

        ec_avg_data = np.mean(ec_data, axis=0)
        ec_err_data = np.std(ec_data, axis=0) / np.sqrt(self.firing_rate_map.shape[0])
        b_avg_data = np.mean(b_data, axis=0)

        self.filtrations = filtrations
        self.ec_mean = ec_avg_data
        self.ec_err = ec_err_data

        return filtrations, ec_avg_data, ec_err_data, b_avg_data

    def _get_euler_characteristic_3d(self, data, filtrations):
        """
        Calculate the Euler characteristic for a given 3D data set.

        Args:
            data (numpy.ndarray): The 3D data for which to calculate the EC.
            filtrations (numpy.ndarray): Filtration values for persistence calculation.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: Euler characteristic and Betti numbers.
        """
        numPoints = filtrations.shape[0]
        cubeplex = gd.CubicalComplex(
            # dimensions=np.shape(data),
            dimensions=[np.shape(data)[2], np.shape(data)[1], np.shape(data)[0]],
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

    def analyze_slices(self, active_points_threshold=0):
        """
        Analyzes and normalizes 2D and 1D slices of the 3D smoothed firing rate map to calculate field sizes.

        Args:
            active_points_threshold (int): Minimum number of active points to consider a slice for analysis.

        Returns:
            tuple: Two numpy arrays containing mean-normalized aggregated areas of fields in 2D slices and
                   mean-normalized aggregated lengths of fields in 1D slices, respectively.
        """
        normalized_areas_2d = []
        normalized_lengths_1d = []

        for neuron_map in self.smoothed_firing_rate_map:
            # Process 2D Slices
            for dim in range(3):
                for slice_idx in range(neuron_map.shape[dim]):
                    slice_2d = self._get_slice(neuron_map, dim, slice_idx)
                    if np.count_nonzero(slice_2d) > active_points_threshold:
                        areas = self._calculate_field_sizes(slice_2d)
                        normalized_areas_2d.extend(areas / (self.size ** (2 / 3)))

            # Process 1D Slices
            for dim in range(3):
                for slice_idx in range(neuron_map.shape[dim]):
                    slice_1d = self._get_line(neuron_map, dim, slice_idx)
                    if np.count_nonzero(slice_1d) > active_points_threshold:
                        lengths = self._calculate_line_lengths(slice_1d)
                        normalized_lengths_1d.extend(lengths / (self.size ** (1 / 3)))

        return np.array(normalized_areas_2d), np.array(normalized_lengths_1d)

    def _get_slice(self, data, dim, index):
        """Extracts a 2D slice from a 3D array based on the specified dimension and index."""
        return np.take(data, index, axis=dim)

    def _get_line(self, data, dim, index):
        """Extracts a 1D line from a 3D array based on the specified dimension and combination of other two indexes."""
        if dim == 0:
            return data[index, :, :].flatten()
        elif dim == 1:
            return data[:, index, :].flatten()
        else:
            return data[:, :, index].flatten()

    def _calculate_field_sizes(self, data):
        """Calculates the sizes of connected fields in a 2D slice."""
        labeled, num_features = ndimage.label(data > 0)
        return np.asarray([np.sum(labeled == i) for i in range(1, num_features + 1)])

    def _calculate_line_lengths(self, data):
        """Calculates the lengths of connected fields in a 1D line."""
        labeled, num_features = ndimage.label(data > 0)
        return np.asarray([np.sum(labeled == i) for i in range(1, num_features + 1)])

    def calculate_curvatures(self, K=None):
        """
        Calculates the mean and Gaussian curvatures for each 3D field in the firing rate map.
        An optional parameter K can be provided to process only the first K fields.

        Args:
            K (int, optional): The number of fields from the firing rate map to process. Defaults to None.

        Returns:
            tuple: Two arrays containing the mean and Gaussian curvatures of each 3D field.
        """
        mean_curvatures, gauss_curvatures = [], []

        # If K is provided, use only the first K fields; otherwise, use the entire map.
        fields_to_process = self.firing_rate_map[:K] if K is not None else self.firing_rate_map

        for field_3d in fields_to_process:
            mc, gc = self._process_field_and_calculate_curvatures(field_3d)
            mean_curvatures.extend(mc)
            gauss_curvatures.extend(gc)

        return np.array(mean_curvatures), np.array(gauss_curvatures)

    @staticmethod
    def _process_field_and_calculate_curvatures(field):
        labeled_field, num_features = label(field > 0)
        mean_curvatures, gauss_curvatures = [], []

        for component in range(1, num_features + 1):
            component_mask = labeled_field == component
            verts, faces, _, _ = marching_cubes(component_mask)
            Z = PlaceField3D._create_simple_heatmap(verts, 30, 30)
            H, K = PlaceField3D._mean_curvature_2d(Z)
            H_flat = H.flatten()
            H_flat = H_flat[~np.isnan(H_flat)]  # Remove NaN values

            K_flat = K.flatten()
            K_flat = K_flat[~np.isnan(K_flat)]  # Remove NaN values
            mean_curvatures.extend(H_flat)
            gauss_curvatures.extend(K_flat)

        return mean_curvatures, gauss_curvatures

    @staticmethod
    def _create_simple_heatmap(verts, grid_size_x, grid_size_y):
        heatmap = np.zeros((grid_size_y, grid_size_x))
        x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
        y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
        x_normalized = (verts[:, 0] - x_min) / (x_max - x_min) * (grid_size_x - 1)
        y_normalized = (verts[:, 1] - y_min) / (y_max - y_min) * (grid_size_y - 1)

        for x, y, z in zip(x_normalized.astype(int), y_normalized.astype(int), verts[:, 2]):
            heatmap[y, x] += z

        return heatmap

    @staticmethod
    def _mean_curvature_2d(Z):
        Zy, Zx = np.gradient(Z)
        Zxy, Zxx = np.gradient(Zx)
        Zyy, _ = np.gradient(Zy)
        H = (Zx**2 + 1) * Zyy - 2 * Zx * Zy * Zxy + (Zy**2 + 1) * Zxx
        H = -H / (2 * (Zx**2 + Zy**2 + 1) ** 1.5)
        K = (Zxx * Zyy - (Zxy**2)) / (1 + (Zx**2) + (Zy**2)) ** 2
        return H, K
