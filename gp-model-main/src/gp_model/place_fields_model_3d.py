import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from .place_fields_3d import PlaceField3D


class PlaceFieldModel3D(PlaceField3D):
    def __init__(self, data, sigma, N=3 * 115):
        """
        Initializes the PlaceFieldModel3D object with simulated 3D place fields.

        Args:
            data (PlaceField3D): The PlaceField3D object to base the simulation dimensions and theta on.
            sigma (float): The standard deviation for the Gaussian filter.
            N (int): The number of neurons to simulate. Defaults to 7*115.
        """
        self.sigma = sigma
        self.lamda = 1 / (np.sqrt(2) * self.sigma)
        self.theta = data.theta
        self.scaling = None
        simulated_firing_rate_map = self._create_simulated_data(data.shape, N)
        super().__init__(simulated_firing_rate_map)

    def _create_simulated_data(self, dimensions, N):
        """
        Creates simulated 3D place fields based on Gaussian filtering of random noise.

        Args:
            dimensions (tuple): The dimensions of the 3D space for each neuron's place field.
            N (int): The number of neurons to simulate.

        Returns:
            np.ndarray: A numpy array representing the simulated firing rate maps.
        """
        simulated_data = []
        for _ in tqdm(range(N), desc="Generating simulated fields"):
            field = gaussian_filter(np.random.normal(0, 1, dimensions), sigma=self.sigma, mode="wrap")
            field = field / (np.sqrt(np.mean(field**2)))
            field = field - self.theta
            field[field < 0] = 0
            simulated_data.append(field)

        return np.asarray(simulated_data)
