import numpy as np
from scipy.ndimage import gaussian_filter


def GaussField(N, P, var=1, sigma=1):
    """Generate Gaussian fields for an NxP grid.

    Args:
        N (int): Number of fields to generate.
        P (int): Size of each field.
        var (float): Variance of the Gaussian distribution. Default is 1.
        sigma (float): Standard deviation for Gaussian filter.

    Returns:
        numpy.ndarray: An array of Gaussian fields.
    """
    fields = np.zeros((N, P))
    for i in range(N):
        fields[i, :] = gaussian_filter(
            np.random.normal(0, np.sqrt(var), P), sigma=sigma, mode="reflect"
        )
        fields[i, :] /= np.sqrt(np.mean(fields[i, :] ** 2))
    return fields


def rayleigh(xlim, cst, min_val=0):
    x = np.linspace(0, xlim, 1000)
    if min_val > 0:
        return x, cst * x * np.exp(-(cst / 2) * x**2) / (1 - 0.5 * cst * min_val)
    else:
        return x, cst * x * np.exp(-(cst / 2) * x**2)


def rayleigh_cube(xlim, cst):
    x = np.linspace(0, xlim, 1000)
    return x, (cst / 3) * (x ** (-1 / 3)) * np.exp(-(cst / 2) * x ** (2 / 3))


def exponential(xlim, cst):
    x = np.linspace(0, xlim, 1000)
    return x, cst * np.exp(-cst * x)


def spherical_fields(compare, single_field=False):
    field_volume_list = compare.properties["data"]["volumes_per_cell"]
    firing_rate_list = compare.properties["data"]["max_vals_per_cell"]
    shape = compare.data.shape

    N = len(field_volume_list)
    space = np.zeros((N, *shape), dtype=float)

    # Generate grid
    x, y, z = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )

    # Lists to collect statistics
    total_volume = np.prod(shape)
    mean_size = np.mean(shape)
    # mean_size = total_volume ** (1 / 3)
    x_scale = shape[0] / mean_size
    y_scale = shape[1] / mean_size
    z_scale = shape[2] / mean_size
    x_scale, y_scale, z_scale = 1, 1, 1

    for i in range(N):
        num_fields = len(field_volume_list[i]) if not single_field else 1
        volumes = field_volume_list[i]
        firing_rate = firing_rate_list[i]
        # Keep track of centers to prevent overlap
        existing_centers = []
        # Loop to generate each field
        for j in range(num_fields):
            # Use the provided relative volume for the sphere (as a fraction of the total volume)
            relative_sphere_volume = volumes[j] if not single_field else volumes
            sphere_volume = relative_sphere_volume * total_volume

            # Calculate the radius of the sphere given its volume: V = 4/3 * π * r^3
            sphere_radius = 1.17 * (sphere_volume * 3 / (4 * np.pi)) ** (1 / 3)  # 1.16

            center_firing_rate = firing_rate[j] if not single_field else firing_rate

            # center_x, center_y, center_z = (
            #     np.random.randint(0, shape[0]),
            #     np.random.randint(0, shape[1]),
            #     np.random.randint(0, shape[2]),
            # )

            while True:
                center_x, center_y, center_z = (
                    np.random.randint(0, shape[0]),
                    np.random.randint(0, shape[1]),
                    np.random.randint(0, shape[2]),
                )
                if all(
                    np.sqrt(
                        (center_x - x) ** 2 + (center_y - y) ** 2 + (center_z - z) ** 2
                    )
                    >= 2 * sphere_radius
                    for x, y, z in existing_centers
                ):
                    existing_centers.append((center_x, center_y, center_z))
                    break

            # Generate quadratic field and update the 3D numpy array
            quadratic_field = center_firing_rate * (
                1
                - (
                    ((x - center_x) / (x_scale * sphere_radius)) ** 2
                    + ((y - center_y) / (y_scale * sphere_radius)) ** 2
                    + ((z - center_z) / (z_scale * sphere_radius)) ** 2
                )
            )
            quadratic_field[quadratic_field < 0] = 0
            space[i] = np.maximum(space[i], quadratic_field)

    return space
