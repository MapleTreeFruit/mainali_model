import numpy as np
from scipy import stats
from scipy.stats import norm, skewnorm

from .analysis import PlaceFieldComparison
from .place_fields_1d import PlaceField1D
from .utils import GaussField


class PlaceFieldModel1D(PlaceField1D):
    """Class to handle 1D place field models and perform various analyses.

    This class extends PlaceField1D by adding the capability to create a model
    of place fields based on certain parameters and apply the analytical methods
    from PlaceField1D to the model.
    """

    def __init__(self, data, sigma, multiple=1):
        """Initialize the PlaceFieldModel1D object with parameters from data.

        Args:
            data (PlaceField1D): An instance of PlaceField1D with experimental data.
            sigma (float): Standard deviation for Gaussian filter smoothing.
        """
        # Create a model based on the experimental data
        self.num_cells = int(multiple * data.num_cells)
        self.theta = data.theta
        self.psp = GaussField(self.num_cells, data.firing_rate_map.shape[1], sigma=sigma / data.resolution)
        self.firing_rate_map = np.maximum(self.psp - data.theta, 0)
        self.length = data.length  # Use the physical length from the data
        self.scaling = None

        # Calculate additional theoretical properties
        self.rho = 0.5 / sigma**2  # Variance of the derivative
        self.kac_rice_prefactor = (self.rho**0.5) / (2 * np.pi)  # Kac-Rice equation prefix
        self.EEC = self.length * self.kac_rice_prefactor * np.exp((-data.theta**2) / 2) + (1 - norm.cdf(data.theta))
        self.kac_rice_mean = self.length * (1 - norm.cdf(data.theta)) / self.EEC

        # Initialize the base class with the generated model and length
        super().__init__(self.firing_rate_map, self.length)


class BiDirectionalPlaceField1D:
    """
    A class to handle bi-directional 1D place field models.

    This class creates and manages two models representing place fields for forward and backward movements.
    Each model is an instance of PlaceField1D.
    """

    def __init__(self, model, theta_variance, alpha=-5):
        """
        Initialize the BiDirectionalPlaceField1D object.

        Args:
            data (PlaceField1D): An instance of PlaceField1D with experimental data.
            theta_variance (float): Variance used for generating the theta vector for thresholding.
        """
        self.size = model.num_cells // 2
        # self.theta_vector = np.abs(np.random.normal(model.theta, theta_variance, size=self.size))
        # self.theta_vector = truncnorm.rvs(0, np.inf, loc=0, scale=theta_variance, size=self.size)
        # self.theta_vector = self.theta_vector / np.mean(self.theta_vector) * model.theta
        # self.theta_vector = np.random.exponential(scale=model.theta, size=self.size)
        # alpha = -5
        delta = alpha / np.sqrt(1 + alpha**2)
        xi = model.theta - (delta * np.sqrt(2 / np.pi))
        self.theta_vector = np.abs(skewnorm.rvs(a=alpha, loc=xi, scale=theta_variance, size=self.size))+0.4
        # self.theta_vector = skewnorm.rvs(a=alpha, loc=xi, scale=theta_variance, size=self.size)

        self.forward_model = PlaceField1D(
            np.maximum((model.psp[self.size :, :].T - self.theta_vector).T, 0), model.length
        )
        self.backward_model = PlaceField1D(
            np.maximum((model.psp[: self.size, :].T - self.theta_vector).T, 0), model.length
        )
        self.compare = PlaceFieldComparison(self.forward_model, self.backward_model)
        self.compare.analyze()
        self.theta_vector = np.abs(skewnorm.rvs(a=alpha, loc=xi, scale=theta_variance, size=50 * self.size))

    def analyze_fields(self):
        """
        Analyze and filter the place fields, and compute regression for field counts, mean widths, and mean gaps.

        Returns:
            dict: A dictionary containing filtered and analyzed data.
        """
        forward_analysis = self._analyze_model_fields(self.forward_model)
        backward_analysis = self._analyze_model_fields(self.backward_model)

        # Filter cells with at least two fields in both forward and backward models
        valid_cells = np.where((forward_analysis["field_counts"] >= 2) & (backward_analysis["field_counts"] >= 2))[0]

        # Compute mean widths and gaps for valid cells
        median_widths_forward = np.array([np.median(forward_analysis["widths"][i]) for i in valid_cells])
        median_widths_backward = np.array([np.median(backward_analysis["widths"][i]) for i in valid_cells])
        mean_gaps_forward = np.array([np.mean(forward_analysis["gaps"][i]) for i in valid_cells])
        mean_gaps_backward = np.array([np.mean(backward_analysis["gaps"][i]) for i in valid_cells])
        ratio_forward = np.array(
            [np.max(forward_analysis["widths"][i]) / np.min(forward_analysis["widths"][i]) for i in valid_cells]
        )
        ratio_backward = np.array(
            [np.max(backward_analysis["widths"][i]) / np.min(backward_analysis["widths"][i]) for i in valid_cells]
        )

        # Compute regression for each quantity
        regression_results = {
            "field_counts": stats.linregress(
                forward_analysis["field_counts"][valid_cells], backward_analysis["field_counts"][valid_cells]
            ),
            "median_widths": stats.linregress(median_widths_forward, median_widths_backward),
            "mean_gaps": stats.linregress(mean_gaps_forward, mean_gaps_backward),
            "ratio": stats.linregress(ratio_forward, ratio_backward),
        }

        return {
            "forward": {
                "field_counts": forward_analysis["field_counts"][valid_cells],
                "median_widths": median_widths_forward,
                "mean_gaps": mean_gaps_forward,
                "ratio": ratio_forward,
            },
            "backward": {
                "field_counts": backward_analysis["field_counts"][valid_cells],
                "median_widths": median_widths_backward,
                "mean_gaps": mean_gaps_backward,
                "ratio": ratio_backward,
            },
            "regression": regression_results,
        }

    def _analyze_model_fields(self, model):
        """
        Analyze the place fields of a single model.

        Args:
            model (PlaceField1D): The model to be analyzed.

        Returns:
            dict: A dictionary containing raw arrays of field counts, widths, and gaps.
        """
        segments, widths, peak_firing_rates, gaps = model.get_single_fields(concatenate=False)
        field_counts = np.array([len(w) for w in widths])
        return {"field_counts": field_counts, "widths": widths, "gaps": gaps}

    def __getattr__(self, method_name):
        """
        Override attribute access to apply PlaceField1D methods to both models.

        Args:
            method_name (str): Name of the method.

        Returns:
            Function: A function that when called, applies the method to both models.
        """

        def method(*args, **kwargs):
            forward_result = getattr(self.forward_model, method_name)(*args, **kwargs)
            backward_result = getattr(self.backward_model, method_name)(*args, **kwargs)
            return forward_result, backward_result

        return method


# Example usage (assuming you have a PlaceField1D class and a data instance)
# bidirectional_model = BiDirectionalPlaceField1D(data, sigma=1.0, theta_variance=0.2)
# results = bidirectional_model.some_method_name(arg1, arg2)

# Example usage
if __name__ == "__main__":
    pass
    # Assuming `data` is an instance of PlaceField1D
