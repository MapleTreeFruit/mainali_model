import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


from scipy.ndimage import gaussian_filter


class PlaceFieldComparison:
    """Class to compare properties of place field data and model."""

    def __init__(self, data, model):
        """
        Initialize with instances of PlaceField1D (data) and PlaceFieldModel1D (model).

        Args:
            data (PlaceField1D): An instance of PlaceField1D containing the experimental data.
            model (PlaceFieldModel1D): An instance of PlaceFieldModel1D containing the modeled data.
        """
        self.data = data
        self.model = model
        self.properties = {"data": {}, "model": {}}

    def analyze(self):
        """Perform analysis on both data and model, storing the properties."""
        (
            self.properties["data"]["segments"],
            self.properties["data"]["widths"],
            self.properties["data"]["max_firing_rates"],
            self.properties["data"]["gaps"],
        ) = self.data.get_single_fields()

        (
            self.properties["model"]["segments"],
            self.properties["model"]["widths"],
            self.properties["model"]["max_firing_rates"],
            self.properties["model"]["gaps"],
        ) = self.model.get_single_fields()

    def regress_properties(self, property_x, property_y, transform_x=None, transform_y=None):
        """Perform regression analysis between two properties for both data and model.

        Args:
            property_x (str): Name of the first property to use as the predictor (independent variable).
            property_y (str): Name of the second property to use as the response (dependent variable).
            transform_x (callable, optional): A function to transform property_x before regression.
            transform_y (callable, optional): A function to transform property_y before regression.

        Returns:
            dict: Regression results for both data and model containing slope, intercept, r_value, p_value, std_err.
        """
        x_data, x_model = self(property_x)
        y_data, y_model = self(property_y)

        # Apply the transformations if they are provided
        if transform_x is not None:
            x_data = transform_x(x_data)
            x_model = transform_x(x_model)

        if transform_y is not None:
            y_data = transform_y(y_data)
            y_model = transform_y(y_model)

        # Perform regression analysis
        regression_results = {"data": stats.linregress(x_data, y_data), "model": stats.linregress(x_model, y_model)}

        # Format the results to return
        formatted_results = {}
        for key in regression_results:
            result = regression_results[key]
            formatted_results[key] = {
                "slope": result.slope,
                "intercept": result.intercept,
                "r_value": result.rvalue,
                "p_value": result.pvalue,
                "std_err": result.stderr,
            }

        return formatted_results

    def count_peaks(self, segments, width_filter=None):
        """
        Counts the number of peaks in each segment.

        Args:
            segments (np.array): Array of segments to analyze.
            width_filter (int, optional): The width to use for smoothing during peak finding.

        Returns:
            np.array: Counts of peaks in each segment.
        """
        peak_counts = []
        for segment in segments:
            if width_filter is not None:
                peaks, _ = find_peaks(gaussian_filter(segment, sigma=width_filter))
            else:
                peaks, _ = find_peaks(segment)
            peak_counts.append(len(peaks))
        return np.array(peak_counts)

    def compare_peak_frequencies(self, subsample_size=100000, min_width=6, max_width=20):
        """
        Compares the frequency of peak counts between data and model.

        Args:
            subsample_size (int): Number of subsamples to draw from the model segments.
            min_width (int): Minimum width to use for smoothing the data.
            max_width (int): Maximum width to use for smoothing the data.

        Returns:
            dict: Histogram mean and std frequency of peak counts for data and model.
        """
        # First, analyze peak distribution in the model across all segments
        all_model_peak_counts = self.count_peaks(self.properties["model"]["segments"])
        self.properties["model"]["peak_counts"] = all_model_peak_counts
        valid_model_peak_counts = all_model_peak_counts[(all_model_peak_counts >= 1) & (all_model_peak_counts <= 3)]

        # Subsample from valid peak counts to calculate mean and std frequencies for the model
        model_peak_freqs = np.zeros((subsample_size, 3))
        for i in tqdm(range(subsample_size), desc="Subsampling model segments"):
            subsample = np.random.choice(
                valid_model_peak_counts, size=len(self.properties["data"]["segments"]), replace=False
            )
            model_peak_freqs[i, :] = np.bincount(subsample, minlength=4)[1:4] / len(subsample)

        model_mean_freq = model_peak_freqs.mean(axis=0)
        model_std_freq = model_peak_freqs.std(axis=0)

        # Next, smooth the data and analyze peak distribution
        data_peak_freqs = np.zeros((max_width - min_width + 1, 3))
        for w in tqdm(range(min_width, max_width + 1), desc="Smoothing data segments"):
            smoothed_data_peak_counts = self.count_peaks(self.properties["data"]["segments"], width_filter=w)
            valid_counts = smoothed_data_peak_counts[
                (smoothed_data_peak_counts >= 1) & (smoothed_data_peak_counts <= 3)
            ]
            if len(valid_counts) > 0:
                data_peak_freqs[w - min_width, :] = np.bincount(valid_counts, minlength=4)[1:4] / len(valid_counts)

        self.properties["data"]["peak_counts"] = self.count_peaks(self.properties["data"]["segments"], width_filter=10)
        data_mean_freq = data_peak_freqs.mean(axis=0)
        data_std_freq = data_peak_freqs.std(axis=0)

        return {
            "model": {"mean": model_mean_freq, "std": model_std_freq},
            "data": {"mean": data_mean_freq, "std": data_std_freq},
        }

    def calculate_boundary_angles(self, n_points=2, near_zero_threshold=0.2):
        """
        Analyze the place fields' boundary angles using linear regression for both data and model,
        and return the combined array of angles for both ascending and descending parts.

        Args:
            filtering (float): Extra thresholding to apply to fields to filter noise
            n_points (int): Number of points to use for the regression to determine the slope.
            near_zero_threshold (float): Threshold to determine what is considered near-zero firing rate.

        Returns:
            tuple: Contains two numpy arrays with the combined boundary angles for data and model.
        """

        def process_curves(curves, threshold, points, scaling=1):
            ascending_angles = []
            descending_angles = []

            for curve in curves:
                if len(curve) < 2 * points:  # Ignore small fields
                    continue

                # Prepare data for linear regression
                X = np.arange(points).reshape(-1, 1)

                # Ascending part - first few values
                if curve[0] < threshold:
                    y = scaling * curve[:points]
                    reg = LinearRegression().fit(X, y)
                    ascending_angles.append(np.abs(reg.coef_[0]))

                # Descending part - last few values
                if curve[-1] < threshold:
                    y = scaling * curve[-points:]
                    reg = LinearRegression().fit(X, y)
                    descending_angles.append(np.abs(reg.coef_[0]))

            return np.array(ascending_angles), np.array(descending_angles)

        data_segments = self.properties["data"]["segments"]
        model_segments = self.properties["model"]["segments"]

        # Process curves and calculate angles for data
        data_ascending, data_descending = process_curves(data_segments, near_zero_threshold, n_points)

        # Process curves and calculate angles for model with scaling
        model_ascending, model_descending = process_curves(
            model_segments, near_zero_threshold, n_points, self.model.scaling**0.5
        )

        # Combine and store the results
        self.properties["data"]["boundary_angles"] = np.concatenate((data_ascending, data_descending))
        self.properties["model"]["boundary_angles"] = np.concatenate((model_ascending, model_descending))

        # Return the results as well
        return (self.properties["data"]["boundary_angles"], self.properties["model"]["boundary_angles"])

    def values_and_derivatives(self):
        """
        Aggregates values and derivatives of curve segments for both data and model,
        and updates self.properties with the computed values and derivatives.
        """

        def calculate_values_derivatives(curves):
            """
            Calculates values and their derivatives for a list of curves.

            Args:
                curves (list): A list of curves, each curve being a numpy array.

            Returns:
                tuple: Two numpy arrays, one for values and one for derivatives.
            """
            values, derivatives = [], []
            for curve in curves:
                if len(curve) >= 3:
                    # curve = np.array(curve)
                    values.extend(curve)

                    curve_derivative = np.zeros_like(curve)
                    curve_derivative[1:-1] = (curve[2:] - curve[:-2]) / 2
                    curve_derivative[0] = curve[1] - curve[0]
                    curve_derivative[-1] = curve[-1] - curve[-2]

                    derivatives.extend(curve_derivative)

            return np.array(values), np.array(derivatives)

        # Processing for data_segments
        data_values, data_derivatives = calculate_values_derivatives(self.properties["data"]["segments"])
        self.properties["data"]["value"] = data_values
        self.properties["data"]["derivative"] = data_derivatives

        # Processing for model_segments
        model_values, model_derivatives = calculate_values_derivatives(self.properties["model"]["segments"])
        self.properties["model"]["value"] = model_values
        self.properties["model"]["derivative"] = model_derivatives

    def __call__(self, property_name):
        """Allow the instance to be called as a function to directly compare properties."""
        data_property = self.properties["data"].get(property_name)
        model_property = self.properties["model"].get(property_name)
        if data_property is not None and model_property is not None:
            return self._make_comparable(data_property, model_property)
        else:
            raise ValueError(f"Property '{property_name}' not found in comparison data.")

    def _make_comparable(self, data_array, model_array):
        """Wrap the arrays in a way that allows direct invocation of NumPy methods."""

        class ComparableArrayPair:
            def __init__(self, data, model):
                self.data = data
                self.model = model

            def __getattr__(self, attr):
                data_attr = getattr(self.data, attr)
                model_attr = getattr(self.model, attr)
                if callable(data_attr) and callable(model_attr):

                    def method(*args, **kwargs):
                        return data_attr(*args, **kwargs), model_attr(*args, **kwargs)

                    return method
                else:
                    return data_attr, model_attr

            def __getitem__(self, key):
                # Assuming key is either 0 for data or 1 for model
                if key == 0:
                    return self.data
                elif key == 1:
                    return self.model
                else:
                    raise IndexError("Index out of bounds for ComparableArrayPair.")

            def __repr__(self):
                return f"ComparableArrayPair(data={self.data}, model={self.model})"

            def __iter__(self):
                return iter((self.data, self.model))

        return ComparableArrayPair(data_array, model_array)


class PlaceFieldComparison3D:
    """Class to compare properties of place field data and model."""

    def __init__(self, data, model):
        """
        Initialize with instances of PlaceField3D (data) and PlaceFieldModel3D (model).

        Args:
            data (PlaceField3D): An instance of PlaceField3D containing the experimental data.
            model (PlaceFieldModel3D): An instance of PlaceFieldModel3D containing the modeled data.
        """
        self.data = data
        self.model = model
        self.properties = {"data": {}, "model": {}}

    def analyze(self, smooth=3):
        """
        Compiles properties from analyze_place_fields and analyze_slices methods
        for both data and model.
        """
        # Analyze place fields for data and model
        print("Analysing 3d place fields\n")
        data_place_fields = self.data.analyze_place_fields(filter_size=smooth)  # also tried 7
        model_place_fields = self.model.analyze_place_fields()
        # Store properties separately
        for property_name in data_place_fields:
            self.properties["data"][property_name] = data_place_fields[property_name]
            self.properties["model"][property_name] = model_place_fields[property_name]

        # Analyze slices for data and model
        print("Analysing 3d place field slices\n")
        data_slices = self.data.analyze_slices()
        model_slices = self.model.analyze_slices()

        # Store 2D and 1D slice analysis results
        self.properties["data"]["areas"], self.properties["data"]["lengths"] = data_slices
        self.properties["model"]["areas"], self.properties["model"]["lengths"] = model_slices

        # Analyze curvature for data and model
        # data_mc, data_gc = self.data.calculate_curvatures()
        # model_mc, model_gc = self.model.calculate_curvatures(115)

        # Store 2D and 1D slice analysis results
        print("Analysing 3d place field curvature")
        (
            self.properties["data"]["mean_curvature"],
            self.properties["data"]["gaussian_curvature"],
        ) = self.data.calculate_curvatures()
        (
            self.properties["model"]["mean_curvature"],
            self.properties["model"]["gaussian_curvature"],
        ) = self.model.calculate_curvatures(115)

    def regress_properties(self, property_x, property_y, transform_x=None, transform_y=None):
        """Perform regression analysis between two properties for both data and model.

        Args:
            property_x (str): Name of the first property to use as the predictor (independent variable).
            property_y (str): Name of the second property to use as the response (dependent variable).
            transform_x (callable, optional): A function to transform property_x before regression.
            transform_y (callable, optional): A function to transform property_y before regression.

        Returns:
            dict: Regression results for both data and model containing slope, intercept, r_value, p_value, std_err.
        """
        x_data, x_model = self(property_x)
        y_data, y_model = self(property_y)

        # Apply the transformations if they are provided
        if transform_x is not None:
            x_data = transform_x(x_data)
            x_model = transform_x(x_model)

        if transform_y is not None:
            y_data = transform_y(y_data)
            y_model = transform_y(y_model)

        # Perform regression analysis
        regression_results = {"data": stats.linregress(x_data, y_data), "model": stats.linregress(x_model, y_model)}

        # Format the results to return
        formatted_results = {}
        for key in regression_results:
            result = regression_results[key]
            formatted_results[key] = {
                "slope": result.slope,
                "intercept": result.intercept,
                "r_value": result.rvalue,
                "p_value": result.pvalue,
                "std_err": result.stderr,
            }

        return formatted_results

    def __call__(self, property_name):
        """Allow the instance to be called as a function to directly compare properties."""
        data_property = self.properties["data"].get(property_name)
        model_property = self.properties["model"].get(property_name)
        if data_property is not None and model_property is not None:
            return self._make_comparable(data_property, model_property)
        else:
            raise ValueError(f"Property '{property_name}' not found in comparison data.")

    def _make_comparable(self, data_array, model_array):
        """Wrap the arrays in a way that allows direct invocation of NumPy methods."""

        class ComparableArrayPair:
            def __init__(self, data, model):
                self.data = data
                self.model = model

            def __getattr__(self, attr):
                data_attr = getattr(self.data, attr)
                model_attr = getattr(self.model, attr)
                if callable(data_attr) and callable(model_attr):

                    def method(*args, **kwargs):
                        return data_attr(*args, **kwargs), model_attr(*args, **kwargs)

                    return method
                else:
                    return data_attr, model_attr

            def __getitem__(self, key):
                # Assuming key is either 0 for data or 1 for model
                if key == 0:
                    return self.data
                elif key == 1:
                    return self.model
                else:
                    raise IndexError("Index out of bounds for ComparableArrayPair.")

            def __repr__(self):
                return f"ComparableArrayPair(data={self.data}, model={self.model})"

            def __iter__(self):
                return iter((self.data, self.model))

        return ComparableArrayPair(data_array, model_array)
